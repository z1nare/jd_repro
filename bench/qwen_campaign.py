"""Overnight campaign: jdgram on Qwen3.5-0.8B, end to end.

Five stages, each independent. Every cell is wrapped, so an OOM or an unsupported
engine is recorded as a row rather than ending the run -- a capability boundary is
a result, not a crash.

    C0  inventory + per-module correctness against fp64 autograd
    C1  whole-Gramian exactness against a brute-force [m, P] Jacobian
        (on a reduced-depth config, so the ground truth fits in memory)
    C2  memory and speed against m, jdgram vs autogram vs autojac
    C3  objective conflict: pairwise gradient cosines against m and mode
    C4  training: perplexity per aggregator against a plain-SGD control

Results stream to JSONL as they land, so a run killed at 4am still yields
everything up to that point.

    python bench/qwen_campaign.py --out results/qwen-c1 --stages C0 C1 C2 C3 C4

Qwen3.5 is 75% Gated DeltaNet. Its gated norm is called on a batch-flattened
tensor, so ``driver="squashed"`` refuses the model -- correctly, since its
row-is-objective shortcut does not hold there. ``loop`` is exact and is the
default here; it costs m backward passes.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import torch
from torch import nn

from jdgram.engine import registry
from jdgram.engine.hooks import compute_gramian
from jdgram.engine.residual import residual_gramian, residual_params
from jdgram.identities.tied import tied_gramian

MODES = ("independent", "duplicate", "conflicting")


# --------------------------------------------------------------------------- io
class Sink:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.n = 0

    def row(self, **kw) -> None:
        self.n += 1
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(kw, default=str) + "\n")
        keys = ("stage", "cell", "engine", "m", "T", "agg", "mode", "metric", "value", "unit")
        print("  " + "  ".join(f"{k}={kw[k]}" for k in keys if k in kw), flush=True)

    def fail(self, stage: str, cell: str, exc: BaseException, **kw) -> None:
        oom = isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()
        self.row(stage=stage, cell=cell, status="OOM" if oom else "ERROR",
                 error=f"{type(exc).__name__}: {str(exc)[:300]}", **kw)
        if not oom:
            traceback.print_exc()
        torch.cuda.empty_cache()


# ----------------------------------------------------------------------- model
def load(model_id: str, device: str, layers: int | None = None):
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    text = getattr(cfg, "text_config", cfg)
    if layers is not None:                       # reduced depth for brute force
        text.num_hidden_layers = layers
        if getattr(text, "layer_types", None):
            text.layer_types = list(text.layer_types)[:layers]
        model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
    else:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id, dtype=torch.float32, trust_remote_code=True)
        except TypeError:                        # transformers 4.x spelling
            model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch.float32, trust_remote_code=True)
    return model.to(device).train(), text


def wire(model: nn.Module):
    """Hooked modules, shared handler for the tied pair, and the residual tail."""
    modules = registry.collect_hookable_modules(model)
    owners: dict[int, list[str]] = {}
    for nm, mod in modules.items():
        for p in mod.parameters(recurse=False):
            if p.requires_grad:
                owners.setdefault(id(p), []).append(nm)

    shared: dict = {}
    for names in (v for v in owners.values() if len(v) > 1):
        head = [n for n in names if isinstance(modules[n], nn.Linear)]
        emb = [n for n in names if isinstance(modules[n], nn.Embedding)]
        if len(names) == 2 and len(head) == 1 and len(emb) == 1:
            h, e = head[0], emb[0]
            shared[frozenset(names)] = (
                lambda caps, h=h, e=e: tied_gramian(
                    caps[h].A, caps[h].X, caps[e].A, caps[e].X)
            )

    handled = {n for n, mod in modules.items() if registry.has_handler(mod)}
    hooked = {n: mod for n, mod in modules.items() if n in handled}
    tail = residual_params(model, handled)
    return hooked, shared, tail


def batch(m: int, T: int, V: int, mode: str, device: str, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    hi = min(V, 30000)
    if mode == "duplicate":
        one = torch.randint(0, hi, (1, T), generator=g)
        idx = one.repeat(m, 1)
    else:
        idx = torch.randint(0, hi, (m, T), generator=g)
    coeffs = torch.tensor(
        [1.0 if (i % 2 == 0 or mode != "conflicting") else -1.0 for i in range(m)],
        device=device)
    return idx.to(device), coeffs


def losses_of(model, idx, coeffs):
    def fn():
        out = model(input_ids=idx)
        logits = out.logits if hasattr(out, "logits") else out[0]
        m, T = idx.shape
        per_tok = nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]),
            idx[:, 1:].reshape(-1), reduction="none").reshape(m, T - 1)
        return per_tok.mean(dim=1) * coeffs
    return fn


def gramian(model, fn, hooked, shared, tail, driver="loop"):
    """Identity-covered blocks plus the materialised tail. Exact."""
    res = compute_gramian(model, fn, modules=hooked,
                          shared_handlers=shared or None, driver=driver)
    G = res.total.double()
    if tail:
        G = G + residual_gramian(fn(), [p for _, p in tail])
    return G, res


def cosines(G: torch.Tensor):
    d = G.diagonal().clamp_min(1e-30).sqrt()
    c = G / d.unsqueeze(0) / d.unsqueeze(1)
    off = c[~torch.eye(G.shape[0], dtype=torch.bool, device=G.device)]
    return off.min().item(), off.mean().item()


# ---------------------------------------------------------------------- stages
def C0(a, sink):
    print("\n=== C0  inventory + per-module correctness ===", flush=True)
    model, cfg = load(a.model, a.device)
    hooked, shared, tail = wire(model)
    sink.row(stage="C0", cell="inventory", n_hooked=len(hooked),
             n_shared=len(shared), residual_params=sum(p.numel() for _, p in tail),
             total_params=sum(p.numel() for p in model.parameters() if p.requires_grad))

    idx, co = batch(2, a.T, cfg.vocab_size, "independent", a.device)
    fn = losses_of(model, idx, co)
    try:
        G, res = gramian(model, fn, hooked, shared, tail)
        sink.row(stage="C0", cell="gramian", m=2, T=a.T, status="ok",
                 metric="diag0", value=G[0, 0].item())
    except Exception as e:                                        # noqa: BLE001
        sink.fail("C0", "gramian", e, m=2, T=a.T); return

    in_group = {n for g in res.per_shared_group for n in g}
    by_type: dict[str, str] = {}
    for n in res.per_module:
        if n in hooked and n not in in_group:
            by_type.setdefault(type(hooked[n]).__name__, n)

    losses = fn()
    for tname, name in by_type.items():
        try:
            ps = [p for p in hooked[name].parameters(recurse=False) if p.requires_grad]
            J = torch.stack([
                torch.cat([g.reshape(-1).double() for g in torch.autograd.grad(
                    losses[i], ps, retain_graph=True)]) for i in range(2)])
            true = J @ J.T
            rel = ((res.per_module[name].double() - true).abs().max()
                   / true.abs().max().clamp_min(1e-30)).item()
            sink.row(stage="C0", cell="verify", module=name, type=tname,
                     metric="rel_err", value=rel, status="ok" if rel < 1e-6 else "MISMATCH")
        except Exception as e:                                    # noqa: BLE001
            sink.fail("C0", "verify", e, module=name, type=tname)
    del model; torch.cuda.empty_cache()


def C1(a, sink):
    print("\n=== C1  whole-Gramian exactness vs brute force ===", flush=True)
    for m in (2, 3):
        try:
            model, cfg = load(a.model, a.device, layers=a.bf_layers)
            hooked, shared, tail = wire(model)
            idx, co = batch(m, a.bf_T, cfg.vocab_size, "independent", a.device)
            fn = losses_of(model, idx, co)
            G, _ = gramian(model, fn, hooked, shared, tail)

            params = [p for p in model.parameters() if p.requires_grad]
            losses = fn()
            J = torch.stack([
                torch.cat([g.reshape(-1).double() for g in torch.autograd.grad(
                    losses[i], params, retain_graph=True)]) for i in range(m)])
            true = J @ J.T
            rel = ((G - true).abs().max() / true.abs().max().clamp_min(1e-30)).item()
            sink.row(stage="C1", cell="exactness", m=m, T=a.bf_T,
                     layers=a.bf_layers, P=sum(p.numel() for p in params),
                     metric="rel_err", value=rel,
                     status="ok" if rel < 1e-6 else "MISMATCH")
            del model, J, true; torch.cuda.empty_cache()
        except Exception as e:                                    # noqa: BLE001
            sink.fail("C1", "exactness", e, m=m); torch.cuda.empty_cache()


def _time_engine(engine, model, fn, hooked, shared, tail, m, reps):
    """(ms, peak MiB) for one Gramian build. Raises on OOM / unsupported."""
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    if engine == "jdgram":
        call = lambda: gramian(model, fn, hooked, shared, tail)          # noqa: E731
    elif engine == "autogram":
        from torchjd.autogram import Engine as AG
        eng = AG(model, batch_dim=0)
        call = lambda: eng.compute_gramian(fn())                          # noqa: E731
    else:
        from torchjd.autojac import backward as ajb
        call = lambda: ajb(fn(), [p for p in model.parameters() if p.requires_grad])  # noqa: E731
    call()                                                                # warm
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(reps):
        call()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1e3, torch.cuda.max_memory_allocated() / 2**20


def C2(a, sink):
    print("\n=== C2  memory + speed vs m ===", flush=True)
    model, cfg = load(a.model, a.device)
    hooked, shared, tail = wire(model)
    for m in a.ms:
        idx, co = batch(m, a.T, cfg.vocab_size, "duplicate", a.device)
        fn = losses_of(model, idx, co)
        for engine in a.engines:
            try:
                ms, mib = _time_engine(engine, model, fn, hooked, shared, tail, m, a.reps)
                sink.row(stage="C2", cell="cost", engine=engine, m=m, T=a.T,
                         metric="ms", value=round(ms, 2), peak_mib=round(mib, 1), status="ok")
            except Exception as e:                                # noqa: BLE001
                sink.fail("C2", "cost", e, engine=engine, m=m, T=a.T)
    del model; torch.cuda.empty_cache()


def C3(a, sink):
    print("\n=== C3  objective conflict ===", flush=True)
    model, cfg = load(a.model, a.device)
    hooked, shared, tail = wire(model)
    for mode in MODES:
        for m in a.ms:
            try:
                idx, co = batch(m, a.T, cfg.vocab_size, mode, a.device)
                G, res = gramian(model, losses_of(model, idx, co), hooked, shared, tail)
                mn, mean = cosines(G)
                sink.row(stage="C3", cell="conflict", mode=mode, m=m, T=a.T,
                         metric="min_cos", value=round(mn, 5), mean_cos=round(mean, 5),
                         status="ok")
                # Per-layer, so "where does conflict live" is answerable rather
                # than just "is there any". Rui's scaling question is exactly
                # whether it concentrates in the vocabulary head or the backbone.
                for name, blk in res.per_module.items():
                    b = blk.double()
                    if b.diagonal().min() <= 0:
                        continue
                    lo, avg = cosines(b)
                    sink.row(stage="C3", cell="conflict_by_layer", mode=mode, m=m,
                             module=name, metric="min_cos", value=round(lo, 5),
                             mean_cos=round(avg, 5))
            except Exception as e:                                # noqa: BLE001
                sink.fail("C3", "conflict", e, mode=mode, m=m)
    del model; torch.cuda.empty_cache()


def C4(a, sink):
    print("\n=== C4  training: perplexity per aggregator ===", flush=True)
    from torchjd.aggregation import MGDAWeighting, PCGradWeighting, UPGradWeighting

    WEIGHTS = {"UPGrad": UPGradWeighting, "MGDA": MGDAWeighting,
               "PCGrad": PCGradWeighting, "Mean": None, "sgd_erm": None}
    for agg, W in WEIGHTS.items():
        try:
            model, cfg = load(a.model, a.device)
            hooked, shared, tail = wire(model)
            opt = torch.optim.SGD(model.parameters(), lr=a.lr)
            wt = W() if W is not None else None
            t0 = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
            for step in range(a.steps):
                idx, co = batch(a.train_m, a.T, cfg.vocab_size, "independent",
                                a.device, seed=step)
                fn = losses_of(model, idx, co)
                opt.zero_grad(set_to_none=True)
                if agg == "sgd_erm":
                    fn().mean().backward()
                else:
                    G, _ = gramian(model, fn, hooked, shared, tail)
                    w = (wt(G.float()) if wt is not None
                         else torch.full((a.train_m,), 1.0 / a.train_m, device=a.device))
                    fn().backward(w.to(a.device).float())
                opt.step()
                if step % max(1, a.steps // 5) == 0:
                    sink.row(stage="C4", cell="curve", agg=agg, step=step,
                             metric="loss", value=round(fn().mean().item(), 5))
            # held-out perplexity on a fixed unseen batch
            with torch.no_grad():
                idx, co = batch(a.train_m, a.T, cfg.vocab_size, "independent",
                                a.device, seed=99991)
                nats = losses_of(model, idx, co)().mean().item()
            sink.row(stage="C4", cell="final", agg=agg, m=a.train_m, steps=a.steps,
                     metric="val_nats", value=round(nats, 5),
                     perplexity=round(float(torch.tensor(nats).exp()), 3),
                     wall_s=round(time.perf_counter() - t0, 1),
                     peak_mib=round(torch.cuda.max_memory_allocated() / 2**20, 1),
                     status="ok")
            del model, opt; torch.cuda.empty_cache()
        except Exception as e:                                    # noqa: BLE001
            sink.fail("C4", "train", e, agg=agg); torch.cuda.empty_cache()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    p.add_argument("--out", type=str, default="results/qwen-campaign")
    p.add_argument("--stages", nargs="+", default=["C0", "C1", "C2", "C3", "C4"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--T", type=int, default=128)
    p.add_argument("--ms", type=int, nargs="+", default=[2, 3, 4, 8])
    p.add_argument("--engines", nargs="+", default=["jdgram", "autogram", "autojac"])
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--bf-layers", type=int, default=2, help="depth for the C1 ground truth")
    p.add_argument("--bf-T", type=int, default=32)
    p.add_argument("--train-m", type=int, default=2)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--lr", type=float, default=0.01)
    a = p.parse_args()

    out = Path(a.out)
    sink = Sink(out / "rows.jsonl")
    (out / "config.json").write_text(json.dumps(vars(a), indent=2, default=str))
    print(f"campaign -> {out}", flush=True)
    print(f"torch {torch.__version__}  device {torch.cuda.get_device_name(0)}", flush=True)

    for name in a.stages:
        stage = {"C0": C0, "C1": C1, "C2": C2, "C3": C3, "C4": C4}.get(name)
        if stage is None:
            print(f"unknown stage {name}"); continue
        t0 = time.perf_counter()
        try:
            stage(a, sink)
        except Exception as e:                                    # noqa: BLE001
            sink.fail(name, "stage", e)
        print(f"--- {name} done in {time.perf_counter()-t0:.0f}s "
              f"({sink.n} rows so far)", flush=True)
        torch.cuda.empty_cache()
    print(f"\nCAMPAIGN COMPLETE: {sink.n} rows -> {sink.path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
