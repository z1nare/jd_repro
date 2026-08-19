"""Overnight campaign: jdgram on Qwen3.5-0.8B, end to end.

Five stages, each independent. Every cell is isolated, wrapped, and cleaned, so
an OOM or an unsupported engine is recorded as a row rather than ending the run
-- a capability boundary is a result, not a crash.

    C0  inventory + per-module correctness against fp64 autograd
    C1  whole-Gramian exactness against a brute-force [m, P] Jacobian
    C2  memory and speed against m, jdgram vs autogram vs autojac
    C3  objective conflict: pairwise gradient cosines, overall and per layer
    C4  training: perplexity per aggregator against a plain-SGD control

Rows stream to JSONL as they land, so a run killed at 4am keeps everything up to
that point.

    python bench/qwen_campaign.py --out results/qwen-c0 --stages C0 C1 C2

Measurement discipline, because a benchmark that quietly measures the previous
cell's leftovers is worse than none:

* every cell runs inside :func:`cell`, which collects garbage, empties the
  caching allocator and resets the peak counter *before* the body, so the peak
  it reports belongs to that cell alone;
* timing uses CUDA events, not ``perf_counter``, so host scheduling delay is
  outside the measurement;
* ``autograd.grad`` throughout rather than ``backward``, except where a step is
  deliberately being taken -- nothing accumulates into ``.grad`` behind a
  measurement;
* once an engine OOMs at some m, larger m for that engine are skipped rather
  than retried, and the boundary is recorded.

Qwen3.5 is 75% Gated DeltaNet. Its gated norm is applied to a batch-flattened
tensor, so ``driver="squashed"`` refuses the model -- correctly, since its
row-is-objective shortcut does not hold there. ``loop`` is exact and is the
default; it costs m backward passes, which is *not* like-for-like against
autogram's one, and the rows say so.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import statistics
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
MIB = 2 ** 20


# ------------------------------------------------------------------ bookkeeping
class Sink:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.n = 0

    def row(self, **kw) -> None:
        self.n += 1
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(kw, default=str) + "\n")
        show = ("stage", "cell", "engine", "agg", "mode", "m", "T",
                "metric", "value", "unit", "peak_mib", "status")
        print("  " + "  ".join(f"{k}={kw[k]}" for k in show if k in kw), flush=True)

    def fail(self, stage: str, cell: str, exc: BaseException, **kw) -> None:
        msg = str(exc)
        oom = isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in msg.lower()
        self.row(stage=stage, cell=cell, status="OOM" if oom else "ERROR",
                 error=f"{type(exc).__name__}: {msg[:300]}", **kw)
        if not oom:
            traceback.print_exc()
        return oom


def _scrub() -> None:
    """Drop dead tensors and return cached blocks, so the next peak is honest."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


@contextlib.contextmanager
def cell():
    """Isolate one measurement: clean before and after, and report both memory numbers.

    ``peak_mib`` is the high-water mark of *total* allocation, which is what
    decides whether the card OOMs. ``delta_mib`` subtracts what was already
    resident on entry -- with a 3.6 GiB model live, the total is dominated by
    weights and only the delta says what the engine actually cost.
    """
    _scrub()
    state = {}
    base = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    state["base_mib"] = round(base / MIB, 1)
    try:
        yield state
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated()
            state["peak_mib"] = round(peak / MIB, 1)
            state["delta_mib"] = round((peak - base) / MIB, 1)
        _scrub()


def timed(call, reps: int, warm: int = 2) -> tuple[float, float]:
    """(median ms, min ms) on CUDA events, so host scheduling is not in the number."""
    for _ in range(warm):
        call()
    torch.cuda.synchronize()
    pairs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
             for _ in range(reps)]
    for start, end in pairs:
        start.record()
        call()
        end.record()
    torch.cuda.synchronize()
    ms = [s.elapsed_time(e) for s, e in pairs]
    return statistics.median(ms), min(ms)


# ------------------------------------------------------------------------ model
def load(model_id: str, device: str, *, small: dict | None = None):
    """Pretrained weights, or a small config of the same architecture for ground truth."""
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    text = getattr(cfg, "text_config", cfg)
    if small:
        # Same architecture -- DeltaNet blocks, gated norms, tied embedding -- at a
        # size where a full [m, P] fp64 Jacobian fits. Shrinking the vocabulary is
        # the load-bearing part: at 248320 x 1024 the embedding alone is 254M
        # parameters, so ground truth would be 2 GiB per objective and OOM before
        # testing anything.
        for k, v in small.items():
            setattr(text, k, v)
        if getattr(text, "layer_types", None):
            text.layer_types = list(text.layer_types)[:small.get("num_hidden_layers", 2)]
        model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
    else:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id, dtype=torch.float32, trust_remote_code=True)
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch.float32, trust_remote_code=True)
    model = model.to(device).train()
    model.config.use_cache = False          # a KV cache during training is dead weight
    return model, getattr(model.config, "text_config", model.config)


def wire(model: nn.Module):
    """Hooked modules, the tied-pair handler, and the identity-less tail."""
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
    return hooked, shared, residual_params(model, handled)


def batch(m: int, T: int, V: int, mode: str, device: str, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    hi = max(2, min(V, 30000))
    if mode == "duplicate":
        idx = torch.randint(0, hi, (1, T), generator=g).repeat(m, 1)
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


def gramian(model, fn, hooked, shared, tail, *, driver="loop", with_tail=True,
            workspace_dtype=None):
    """Identity blocks plus the materialised tail.

    Two forwards when ``with_tail``: compute_gramian owns its own graph and frees
    it, so the residual needs a fresh one. Wasteful but honest -- and it is the
    real price of an exact Gramian on this model, so C2 reports it both ways
    rather than quietly charging jdgram the cheaper number.
    """
    res = compute_gramian(model, fn, modules=hooked,
                          shared_handlers=shared or None, driver=driver,
                          workspace_dtype=workspace_dtype)
    G = res.total.double()
    if with_tail and tail:
        G = G + residual_gramian(fn(), [p for _, p in tail])
    return G, res


def cosines(G: torch.Tensor):
    d = G.diagonal().clamp_min(1e-30).sqrt()
    c = G / d.unsqueeze(0) / d.unsqueeze(1)
    off = c[~torch.eye(G.shape[0], dtype=torch.bool, device=G.device)]
    return off.min().item(), off.mean().item()


# ----------------------------------------------------------------------- stages
def C0(a, sink):
    print("\n=== C0  inventory + per-module correctness ===", flush=True)
    model, cfg = load(a.model, a.device)
    try:
        hooked, shared, tail = wire(model)
        total = sum(p.numel() for p in model.parameters() if p.requires_grad)
        sink.row(stage="C0", cell="inventory", n_hooked=len(hooked), n_shared=len(shared),
                 residual_params=sum(p.numel() for _, p in tail), total_params=total,
                 residual_frac=round(sum(p.numel() for _, p in tail) / total, 6))

        with cell() as st:
            idx, co = batch(2, a.T, cfg.vocab_size, "independent", a.device)
            fn = losses_of(model, idx, co)
            G, res = gramian(model, fn, hooked, shared, tail)
            per_mod = {k: v.detach().clone() for k, v in res.per_module.items()}
            groups = {n for g in res.per_shared_group for n in g}
            diag = G[0, 0].item()
        sink.row(stage="C0", cell="gramian", m=2, T=a.T, status="ok",
                 metric="diag0", value=diag, peak_mib=st.get("peak_mib"))

        by_type: dict[str, str] = {}
        for n in per_mod:
            if n in hooked and n not in groups:
                by_type.setdefault(type(hooked[n]).__name__, n)

        for tname, name in by_type.items():
            with cell() as st:
                try:
                    ps = [p for p in hooked[name].parameters(recurse=False) if p.requires_grad]
                    losses = fn()
                    J = torch.stack([
                        torch.cat([g.reshape(-1).double() for g in torch.autograd.grad(
                            losses[i], ps, retain_graph=(i < 1))]) for i in range(2)])
                    true = J @ J.T
                    rel = ((per_mod[name].double() - true).abs().max()
                           / true.abs().max().clamp_min(1e-30)).item()
                    del J, true, losses
                    sink.row(stage="C0", cell="verify", module=name, type=tname,
                             metric="rel_err", value=rel,
                             status="ok" if rel < 1e-6 else "MISMATCH")
                except Exception as e:                            # noqa: BLE001
                    sink.fail("C0", "verify", e, module=name, type=tname)
    finally:
        del model
        _scrub()


def C1(a, sink):
    print("\n=== C1  whole-Gramian exactness vs brute force ===", flush=True)
    small = {"num_hidden_layers": a.bf_layers, "vocab_size": a.bf_vocab,
             "hidden_size": a.bf_hidden}
    for m in a.bf_ms:
        model = None
        with cell() as st:
            try:
                model, cfg = load(a.model, a.device, small=small)
                hooked, shared, tail = wire(model)
                idx, co = batch(m, a.bf_T, cfg.vocab_size, "independent", a.device)
                fn = losses_of(model, idx, co)

                params = [p for p in model.parameters() if p.requires_grad]
                P = sum(p.numel() for p in params)
                losses = fn()
                J = torch.stack([
                    torch.cat([g.reshape(-1).double() for g in torch.autograd.grad(
                        losses[i], params, retain_graph=(i < m - 1))]) for i in range(m)])
                true = J @ J.T
                den = true.abs().max().clamp_min(1e-30)
                del J, losses

                # Both precisions, because the campaign's default is fp32 while the
                # gates that produce the published accuracy figure pin fp64. If the
                # error is round-off it collapses here; if it survives fp64 it is a
                # bug in an identity or in how the blocks are assembled.
                for tag, wd in (("fp32", torch.float32), ("fp64", torch.float64)):
                    G, _ = gramian(model, fn, hooked, shared, tail, workspace_dtype=wd)
                    rel = ((G - true).abs().max() / den).item()
                    sink.row(stage="C1", cell="exactness", m=m, T=a.bf_T, P=P,
                             layers=a.bf_layers, vocab=a.bf_vocab, workspace=tag,
                             residual_params=sum(p.numel() for _, p in tail),
                             metric="rel_err", value=rel,
                             status="ok" if rel < 1e-6 else "MISMATCH")
                    del G

                # Split the discrepancy: identities alone against a ground truth over
                # only the parameters they own. If this is clean, the assembly or the
                # materialised tail is at fault, not the closed forms.
                handled_ps, seen = [], set()
                for n, mod in hooked.items():
                    for p in mod.parameters(recurse=False):
                        if p.requires_grad and id(p) not in seen:
                            seen.add(id(p)); handled_ps.append(p)
                losses = fn()
                Jh = torch.stack([
                    torch.cat([g.reshape(-1).double() for g in torch.autograd.grad(
                        losses[i], handled_ps, retain_graph=(i < m - 1))]) for i in range(m)])
                true_h = Jh @ Jh.T
                Gh, _ = gramian(model, fn, hooked, shared, tail,
                                with_tail=False, workspace_dtype=torch.float64)
                relh = ((Gh - true_h).abs().max()
                        / true_h.abs().max().clamp_min(1e-30)).item()
                sink.row(stage="C1", cell="identities_only", m=m, T=a.bf_T,
                         workspace="fp64", n_params=len(handled_ps),
                         metric="rel_err", value=relh,
                         status="ok" if relh < 1e-6 else "MISMATCH")
                del Jh, true_h, Gh, true, losses
            except Exception as e:                                # noqa: BLE001
                sink.fail("C1", "exactness", e, m=m)
            finally:
                del model
                _scrub()


def C2(a, sink):
    print("\n=== C2  memory + speed vs m ===", flush=True)
    model, cfg = load(a.model, a.device)
    dead: set[str] = set()          # engines that have OOMed; do not retry larger m
    variants = [(e, tail_on) for e in a.engines
                for tail_on in ((True, False) if e == "jdgram" else (True,))]
    try:
        hooked, shared, tail = wire(model)
        for m in a.ms:
            for engine, tail_on in variants:
                key = f"{engine}{'' if tail_on else '-identities-only'}"
                if key in dead:
                    sink.row(stage="C2", cell="cost", engine=key, m=m, T=a.T,
                             status="SKIPPED", note="OOMed at a smaller m")
                    continue
                with cell() as st:
                    try:
                        idx, co = batch(m, a.T, cfg.vocab_size, "duplicate", a.device)
                        fn = losses_of(model, idx, co)
                        if engine == "jdgram":
                            call = lambda: gramian(model, fn, hooked, shared, tail,  # noqa: E731
                                                   with_tail=tail_on)
                        elif engine == "autogram":
                            from torchjd.autogram import Engine as AG
                            eng = AG(model, batch_dim=0)
                            call = lambda: eng.compute_gramian(fn())              # noqa: E731
                        else:
                            from torchjd.autojac import backward as ajb
                            ps = [p for p in model.parameters() if p.requires_grad]
                            def call():                                            # noqa: E306
                                ajb(fn(), ps)
                                for p in ps:                # autojac accumulates into
                                    p.grad = None           # .grad; do not let it pile up
                        med, best = timed(call, a.reps)
                        sink.row(stage="C2", cell="cost", engine=key, m=m, T=a.T,
                                 metric="ms", value=round(med, 2), min_ms=round(best, 2),
                                 status="ok")
                    except Exception as e:                        # noqa: BLE001
                        if sink.fail("C2", "cost", e, engine=key, m=m, T=a.T):
                            dead.add(key)
                if "peak_mib" in st:
                    sink.row(stage="C2", cell="mem", engine=key, m=m, T=a.T,
                             metric="peak_mib", value=st["peak_mib"],
                             delta_mib=st.get("delta_mib"), base_mib=st.get("base_mib"))
                for p in model.parameters():
                    p.grad = None
    finally:
        del model
        _scrub()


def C3(a, sink):
    print("\n=== C3  objective conflict ===", flush=True)
    model, cfg = load(a.model, a.device)
    dead = False
    try:
        hooked, shared, tail = wire(model)
        for mode in MODES:
            for m in a.ms:
                if dead:
                    sink.row(stage="C3", cell="conflict", mode=mode, m=m,
                             status="SKIPPED", note="OOMed at a smaller m")
                    continue
                with cell() as st:
                    try:
                        idx, co = batch(m, a.T, cfg.vocab_size, mode, a.device)
                        G, res = gramian(model, losses_of(model, idx, co),
                                         hooked, shared, tail)
                        mn, mean = cosines(G)
                        per_layer = [(n, b.detach().double()) for n, b in res.per_module.items()]
                        del res
                    except Exception as e:                        # noqa: BLE001
                        dead = sink.fail("C3", "conflict", e, mode=mode, m=m)
                        continue
                sink.row(stage="C3", cell="conflict", mode=mode, m=m, T=a.T,
                         metric="min_cos", value=round(mn, 5), mean_cos=round(mean, 5),
                         peak_mib=st.get("peak_mib"), status="ok")
                # Per layer, so "where does conflict live" is answerable rather than
                # just "is there any" -- the vocabulary-vs-backbone question, asked
                # of the model rather than argued from parameter counts.
                for name, blk in per_layer:
                    if blk.shape[0] < 2 or blk.diagonal().min() <= 0:
                        continue
                    lo, avg = cosines(blk)
                    sink.row(stage="C3", cell="conflict_by_layer", mode=mode, m=m,
                             module=name, metric="min_cos", value=round(lo, 5),
                             mean_cos=round(avg, 5))
                del per_layer
    finally:
        del model
        _scrub()


def C4(a, sink):
    print("\n=== C4  training: perplexity per aggregator ===", flush=True)
    from torchjd.aggregation import MGDAWeighting, PCGradWeighting, UPGradWeighting

    plan = [("sgd_erm", None), ("Mean", None), ("UPGrad", UPGradWeighting),
            ("MGDA", MGDAWeighting), ("PCGrad", PCGradWeighting)]
    for agg, W in plan:
        model = None
        with cell() as st:
            try:
                model, cfg = load(a.model, a.device)
                hooked, shared, tail = wire(model)
                opt = torch.optim.SGD(model.parameters(), lr=a.lr)
                wt = W() if W is not None else None
                t0 = time.perf_counter()
                for step in range(a.steps):
                    idx, co = batch(a.train_m, a.T, cfg.vocab_size,
                                    "independent", a.device, seed=step)
                    fn = losses_of(model, idx, co)
                    opt.zero_grad(set_to_none=True)
                    if agg == "sgd_erm":
                        L = fn()
                        loss = L.mean().item()
                        L.mean().backward()
                    else:
                        G, _ = gramian(model, fn, hooked, shared, tail)
                        if wt is not None:
                            w = wt(G.float()).to(a.device).float()
                        else:
                            w = torch.full((a.train_m,), 1.0 / a.train_m, device=a.device)
                        L = fn()
                        loss = L.mean().item()
                        L.backward(w)
                        del G
                    opt.step()
                    del L
                    if step % max(1, a.steps // 5) == 0:
                        sink.row(stage="C4", cell="curve", agg=agg, step=step,
                                 metric="train_loss", value=round(loss, 5))
                wall = time.perf_counter() - t0
                with torch.no_grad():
                    idx, co = batch(a.train_m, a.T, cfg.vocab_size,
                                    "independent", a.device, seed=99991)
                    nats = losses_of(model, idx, co)().mean().item()
                sink.row(stage="C4", cell="final", agg=agg, m=a.train_m, T=a.T,
                         steps=a.steps, metric="val_nats", value=round(nats, 5),
                         perplexity=round(float(torch.tensor(nats).exp()), 3),
                         ms_per_step=round(wall / max(1, a.steps) * 1e3, 1), status="ok")
                del opt
            except Exception as e:                                # noqa: BLE001
                sink.fail("C4", "train", e, agg=agg)
            finally:
                del model
                _scrub()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    p.add_argument("--out", default="results/qwen-campaign")
    p.add_argument("--stages", nargs="+", default=["C0", "C1", "C2", "C3", "C4"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--T", type=int, default=128)
    p.add_argument("--ms", type=int, nargs="+", default=[2, 3, 4, 8])
    p.add_argument("--engines", nargs="+", default=["jdgram", "autogram", "autojac"])
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--bf-ms", type=int, nargs="+", default=[2, 3])
    p.add_argument("--bf-layers", type=int, default=2)
    p.add_argument("--bf-vocab", type=int, default=2048)
    p.add_argument("--bf-hidden", type=int, default=256)
    p.add_argument("--bf-T", type=int, default=32)
    p.add_argument("--train-m", type=int, default=2)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--mem-fraction", type=float, default=0.92,
                   help="cap the process at this fraction of the card, so an OOM is "
                        "raised cleanly instead of the driver killing a neighbour")
    a = p.parse_args()

    out = Path(a.out)
    sink = Sink(out / "rows.jsonl")
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(a), indent=2, default=str))

    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(a.mem_fraction)
        props = torch.cuda.get_device_properties(0)
        print(f"{props.name}  {props.total_memory/2**30:.1f} GiB  "
              f"capped at {a.mem_fraction:.0%}", flush=True)
    print(f"torch {torch.__version__}  ->  {out}", flush=True)

    for name in a.stages:
        fn = {"C0": C0, "C1": C1, "C2": C2, "C3": C3, "C4": C4}.get(name)
        if fn is None:
            print(f"unknown stage {name!r}")
            continue
        t0 = time.perf_counter()
        try:
            fn(a, sink)
        except Exception as e:                                    # noqa: BLE001
            sink.fail(name, "stage", e)
        _scrub()
        print(f"--- {name} done in {time.perf_counter()-t0:.0f}s "
              f"({sink.n} rows total)", flush=True)
    print(f"\nCAMPAIGN COMPLETE: {sink.n} rows -> {sink.path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
