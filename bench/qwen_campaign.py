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
from jdgram.engine.residual import residual_params
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
        show = ("stage", "cell", "engine", "agg", "mode", "module", "type", "m", "T",
                "metric", "value", "unit", "peak_mib", "status")
        print("  " + "  ".join(f"{k}={kw[k]}" for k in show if k in kw), flush=True)

    def fail(self, stage: str, cell: str, exc: BaseException, **kw) -> None:
        msg = str(exc)
        oom = isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in msg.lower()
        self.row(stage=stage, cell=cell, status="OOM" if oom else "ERROR",
                 error=f"{type(exc).__name__}: {msg[:300]}", **kw)
        if not oom:
            traceback.print_exc()
        # Drop the traceback before returning. It references every frame of the
        # failed call, and those frames hold the hook manager, its captures, and
        # through them the whole forward graph. Keeping it alive is how one
        # failed cell leaks gigabytes into the next cell's baseline -- which is
        # what made the published m=4 OOM a measurement of the harness rather
        # than of the engine.
        exc.__traceback__ = None
        return oom


def _scrub(passes: int = 3) -> None:
    """Drop dead tensors and return cached blocks, so the next peak is honest.

    Collects more than once on purpose. The engine's live structures form
    reference *cycles* -- a capture holds the graph, the graph's nodes hold the
    capture back -- and a cycle is only reclaimed once nothing outside it refers
    in. Freeing one cycle can drop the last reference into another, so a single
    pass reliably leaves some of them standing; three converges in practice.
    """
    for _ in range(passes):
        gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


#: Live bytes on entering a cell, in MiB, from the first cell that recorded one.
#: A cell whose baseline has drifted far above this is measuring leaked tensors
#: from an earlier cell, not the engine.
_BASELINE: dict[str, float] = {}

#: Drift above the first baseline, in MiB, that marks a cell's numbers unusable.
#: Generous on purpose: it should catch gigabyte-scale leaks, not allocator noise.
LEAK_TOLERANCE_MIB = 512.0


@contextlib.contextmanager
def cell(sink: "Sink | None" = None, **tag):
    """Isolate one measurement: clean before and after, and report both memory numbers.

    ``peak_mib`` is the high-water mark of *total* allocation, which is what
    decides whether the card OOMs. ``delta_mib`` subtracts what was already
    resident on entry -- with a 3.6 GiB model live, the total is dominated by
    weights and only the delta says what the engine actually cost.

    Also guards the baseline. Every cell should start with the same bytes live
    (the weights, and nothing else); if one starts far above that, an earlier
    cell leaked into it and both its peak and any OOM it reports are artefacts.
    That failure mode is not hypothetical -- it is what made a previously
    published m=4 OOM meaningless -- so it is recorded in the row rather than
    left for someone to notice in the logs.
    """
    _scrub()
    state: dict = {}
    base = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    base_mib = round(base / MIB, 1)
    state["base_mib"] = base_mib

    first = _BASELINE.setdefault("mib", base_mib)
    drift = base_mib - first
    if drift > LEAK_TOLERANCE_MIB:
        state["baseline_drift_mib"] = round(drift, 1)
        state["leaked"] = True
        print(f"  !! baseline drift {drift:,.1f} MiB above {first:,.1f} MiB "
              f"-- this cell's memory numbers are NOT trustworthy", flush=True)
        if sink is not None:
            sink.row(status="LEAKED", metric="baseline_drift_mib",
                     value=round(drift, 1), base_mib=base_mib,
                     first_base_mib=first, **tag)
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
def load(model_id: str, device: str, *, small: dict | None = None,
         dtype: str = "fp32", grad_ckpt: bool = False):
    """Pretrained weights, or a small config of the same architecture for ground truth.

    ``dtype`` and ``grad_ckpt`` exist to reach objective counts that matter.
    At fp32 with activations kept, m=2 is the ceiling on a 24 GiB card for this
    model -- and m=2 answers nothing, since two objectives is where every engine
    looks similar and where JD has least to do.

    Gradient checkpointing is the honest lever: it recomputes activations in the
    backward instead of storing them, which is exactly the axis that scales with
    m, and it does not perturb the Gramian -- measured at 1.2e-16 against the
    non-checkpointed run. bf16 halves weights and activations again but is a
    *numerical* change, so accuracy claims stay on the fp32 runs.
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    torch_dtype = {"fp32": torch.float32, "bf16": torch.bfloat16,
                   "fp16": torch.float16}[dtype]

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
                model_id, dtype=torch_dtype, trust_remote_code=True)
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch_dtype, trust_remote_code=True)
    model = model.to(device).train()
    model.config.use_cache = False          # a KV cache during training is dead weight
    if grad_ckpt:
        # use_reentrant=False is required: the reentrant implementation does not
        # play with the multiple autograd.grad calls the loop driver makes.
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        # Deliberately NOT enable_input_require_grads(): it calls requires_grad_()
        # on the embedding output, and autogram runs its backward under a functorch
        # transform that forbids that -- so switching checkpointing on was breaking
        # the engine being compared against, not just this one. It is only needed
        # when the inputs would otherwise not require grad (frozen embeddings,
        # LoRA), which is not the case here.
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
            workspace_dtype=None, use_leaf_edges=True):
    """Identity blocks plus the materialised tail.

    ONE forward, tail included. The tail parameters are handed to
    ``compute_gramian`` as ``residual_params=``, so their gradients come out of
    the reverse passes it already runs. This used to take a second forward plus
    ``m`` more backwards -- ``residual_gramian`` on a fresh graph -- to reach
    445,248 parameters (0.059% of the model) that the first traversal already
    walked past on its way to layer 0. Measured at m=2/T=256 that tail was 757 ms
    of a 1,858 ms step. The block is identical either way (gate 5h pins it).

    ``use_leaf_edges`` is a real workaround, not a tuning knob. hooks.py's
    _reverse_loop -- its own docstring: "Legacy m reverse passes -- gate/debug
    path" -- computes the leaf-edge set ONCE from the whole [m] losses vector,
    then reuses that fixed set across each objective's separate
    torch.autograd.grad call. Measured failure: at m=3 on Qwen3.5,
    model.embed_tokens registers zero backward captures on objective 0 even
    though the forward called it exactly once -- the edge-pruned traversal
    for that specific objective's autograd.grad apparently does not reach it.
    m=2 does not exhibit this; root cause in edges.py's get_leaf_edges is not
    yet isolated. use_leaf_edges=False bypasses the optimisation entirely --
    _reverse_loop takes model.zero_grad() + losses[i].backward() per objective
    instead, a full backward against every parameter rather than a pruned
    graph walk. Slower, but sidesteps the pruning path where the bug lives.
    """
    fold = [p for _, p in tail] if (with_tail and tail) else None
    res = compute_gramian(model, fn, modules=hooked,
                          shared_handlers=shared or None, driver=driver,
                          workspace_dtype=workspace_dtype,
                          use_leaf_edges=use_leaf_edges,
                          residual_params=fold)
    return res.total.double(), res


def cosines(G: torch.Tensor):
    d = G.diagonal().clamp_min(1e-30).sqrt()
    c = G / d.unsqueeze(0) / d.unsqueeze(1)
    off = c[~torch.eye(G.shape[0], dtype=torch.bool, device=G.device)]
    return off.min().item(), off.mean().item()


# ----------------------------------------------------------------------- stages
def C0(a, sink):
    print("\n=== C0  inventory + per-module correctness ===", flush=True)
    model, cfg = load(a.model, a.device, dtype=a.dtype, grad_ckpt=bool(a.grad_ckpt))
    try:
        hooked, shared, tail = wire(model)
        total = sum(p.numel() for p in model.parameters() if p.requires_grad)
        sink.row(stage="C0", cell="inventory", n_hooked=len(hooked), n_shared=len(shared),
                 residual_params=sum(p.numel() for _, p in tail), total_params=total,
                 residual_frac=round(sum(p.numel() for _, p in tail) / total, 6))

        with cell(sink, stage="C0", cell="baseline") as st:
            idx, co = batch(2, a.T, cfg.vocab_size, "independent", a.device)
            fn = losses_of(model, idx, co)
            G, res = gramian(model, fn, hooked, shared, tail, use_leaf_edges=bool(a.use_leaf_edges))
            per_mod = {k: v.detach().clone() for k, v in res.per_module.items()}
            groups = {n for g in res.per_shared_group for n in g}
            diag = G[0, 0].item()
        sink.row(stage="C0", cell="gramian", m=2, T=a.T, status="ok",
                 metric="diag0", value=diag, peak_mib=st.get("peak_mib"))

        # One per type is enough to catch a broken identity, but not to catch an
        # identity that is right for one *call shape* and wrong for another. Qwen
        # applies the same Qwen3_5RMSNorm class to rank-3 hidden states and to
        # rank-4 per-head tensors, so sampling by type checks one and never the
        # other. --verify-all walks every hooked module and reports the worst.
        if a.verify_all:
            targets = {n: n for n in per_mod if n in hooked and n not in groups}
        else:
            targets = {}
            for n in per_mod:
                if n in hooked and n not in groups:
                    targets.setdefault(type(hooked[n]).__name__, n)
        by_type = targets

        worst: list[tuple[float, str, str]] = []
        for tname, name in by_type.items():
            with cell(sink, stage="C0", cell="baseline") as st:
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
                    shape = tuple(hooked[name].weight.shape) if hasattr(
                        hooked[name], "weight") else None
                    worst.append((rel, name, type(hooked[name]).__name__))
                    if not a.verify_all or rel >= 1e-6:
                        sink.row(stage="C0", cell="verify", module=name,
                                 type=type(hooked[name]).__name__, wshape=shape,
                                 metric="rel_err", value=rel,
                                 status="ok" if rel < 1e-6 else "MISMATCH")
                except Exception as e:                            # noqa: BLE001
                    sink.fail("C0", "verify", e, module=name, type=tname)
        if a.verify_all and worst:
            worst.sort(reverse=True)
            bad = sum(1 for r, _, _ in worst if r >= 1e-6)
            sink.row(stage="C0", cell="verify_summary", n_checked=len(worst),
                     n_mismatched=bad, metric="max_rel_err", value=worst[0][0],
                     worst_module=worst[0][1], worst_type=worst[0][2],
                     status="ok" if bad == 0 else "MISMATCH")
            print("  worst 8 modules by relative error:", flush=True)
            for rel, nm, tn in worst[:8]:
                print(f"    {rel:.3e}  {tn:<22} {nm}", flush=True)
    finally:
        del model
        _scrub()


def C1(a, sink):
    print("\n=== C1  whole-Gramian exactness vs brute force ===", flush=True)
    # Shrink depth and vocabulary only. hidden_size is deliberately left alone:
    # Qwen3.5 fixes head_dim independently (128 for the DeltaNet value heads, 256
    # for attention), so overriding hidden gives a model whose head arithmetic no
    # longer matches the real one -- and then a mismatch says nothing about the
    # architecture being ported. Vocabulary is what has to shrink for the ground
    # truth to fit: at 248320 x 1024 the embedding alone is 254M parameters, so a
    # brute-force fp64 Jacobian would be 2 GiB per objective.
    small = {"num_hidden_layers": a.bf_layers, "vocab_size": a.bf_vocab}
    if a.bf_hidden:
        small["hidden_size"] = a.bf_hidden
    for m in a.bf_ms:
        model = None
        with cell(sink, stage="C1", cell="baseline") as st:
            try:
                model, cfg = load(a.model, a.device, small=small)
                # The whole model in float64, not just the Gramian workspace.
                # `J` below is built by autograd on this same model, so on an fp32
                # model it is not ground truth at all -- it carries fp32 error of
                # its own, and the comparison measures the difference of two
                # errors. That is why fp32 and fp64 *workspace* gave 1.4076e-05
                # and 1.4101e-05: identical, because the model was fp32 in both.
                # The gates reach 1e-10 by running the model itself in fp64.
                if a.bf_fp64:
                    model = model.double()
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
                    G, _ = gramian(model, fn, hooked, shared, tail, workspace_dtype=wd, use_leaf_edges=bool(a.use_leaf_edges))
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
                                with_tail=False, workspace_dtype=torch.float64, use_leaf_edges=bool(a.use_leaf_edges))
                relh = ((Gh - true_h).abs().max()
                        / true_h.abs().max().clamp_min(1e-30)).item()
                sink.row(stage="C1", cell="identities_only", m=m, T=a.bf_T,
                         workspace="fp64", n_params=len(handled_ps),
                         metric="rel_err", value=relh,
                         status="ok" if relh < 1e-6 else "MISMATCH")
                del Jh, true_h, Gh

                # If the assembled sum is wrong while every block is right, the
                # fault is in assembly, not in a closed form. Localise it on the
                # config that actually fails -- verifying on the full pretrained
                # model answers a different question.
                _, res = gramian(model, fn, hooked, shared, tail,
                                 with_tail=False, workspace_dtype=torch.float64, use_leaf_edges=bool(a.use_leaf_edges))
                groups = {n for g in res.per_shared_group for n in g}
                worst = []
                for name, blk in res.per_module.items():
                    if name in groups or name not in hooked:
                        continue
                    ps = [p for p in hooked[name].parameters(recurse=False)
                          if p.requires_grad]
                    if not ps:
                        continue
                    L = fn()
                    Jm = torch.stack([
                        torch.cat([g.reshape(-1).double() for g in torch.autograd.grad(
                            L[i], ps, retain_graph=(i < m - 1))]) for i in range(m)])
                    tm = Jm @ Jm.T
                    e = ((blk.double() - tm).abs().max()
                         / tm.abs().max().clamp_min(1e-30)).item()
                    worst.append((e, name, type(hooked[name]).__name__))
                    del Jm, tm, L
                worst.sort(reverse=True)
                for e, name, tname in worst[:8]:
                    sink.row(stage="C1", cell="per_module_small", m=m, module=name,
                             type=tname, metric="rel_err", value=e,
                             status="ok" if e < 1e-6 else "MISMATCH")
                sink.row(stage="C1", cell="per_module_summary", m=m,
                         n_checked=len(worst), metric="max_rel_err",
                         value=worst[0][0] if worst else float("nan"),
                         worst_module=worst[0][1] if worst else None,
                         worst_type=worst[0][2] if worst else None,
                         status="ok" if worst and worst[0][0] < 1e-6 else "MISMATCH")
                del res, worst, true, losses
            except Exception as e:                                # noqa: BLE001
                sink.fail("C1", "exactness", e, m=m)
            finally:
                del model
                _scrub()


def C2(a, sink):
    print("\n=== C2  memory + speed vs m ===", flush=True)
    model, cfg = load(a.model, a.device, dtype=a.dtype, grad_ckpt=bool(a.grad_ckpt))
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
                with cell(sink, stage="C2", cell="baseline") as st:
                    try:
                        idx, co = batch(m, a.T, cfg.vocab_size, "duplicate", a.device)
                        fn = losses_of(model, idx, co)
                        if engine == "jdgram":
                            call = lambda: gramian(model, fn, hooked, shared, tail,  # noqa: E731
                                                   with_tail=tail_on, use_leaf_edges=bool(a.use_leaf_edges))
                        elif engine == "autogram":
                            from torchjd.autogram import Engine as AG
                            eng = AG(model, batch_dim=0)
                            call = lambda: eng.compute_gramian(fn())              # noqa: E731
                        else:
                            from torchjd.autojac import backward as ajb
                            ps = [p for p in model.parameters() if p.requires_grad]
                            def call():                                            # noqa: E306
                                # torchjd 0.17 takes only the tensors positionally;
                                # the parameter list is the `inputs` keyword. Passing
                                # it positionally raised "backward() takes 1
                                # positional argument but 2 were given".
                                try:
                                    ajb(fn(), inputs=ps)
                                except TypeError:
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
    model, cfg = load(a.model, a.device, dtype=a.dtype, grad_ckpt=bool(a.grad_ckpt))
    dead = False
    try:
        hooked, shared, tail = wire(model)
        for mode in MODES:
            for m in a.ms:
                if dead:
                    sink.row(stage="C3", cell="conflict", mode=mode, m=m,
                             status="SKIPPED", note="OOMed at a smaller m")
                    continue
                with cell(sink, stage="C3", cell="baseline") as st:
                    try:
                        idx, co = batch(m, a.T, cfg.vocab_size, mode, a.device)
                        G, res = gramian(model, losses_of(model, idx, co),
                                         hooked, shared, tail, use_leaf_edges=bool(a.use_leaf_edges))
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
        with cell(sink, stage="C4", cell="baseline") as st:
            try:
                model, cfg = load(a.model, a.device, dtype=a.dtype, grad_ckpt=bool(a.grad_ckpt))
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
                        G, _ = gramian(model, fn, hooked, shared, tail, use_leaf_edges=bool(a.use_leaf_edges))
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
    p.add_argument("--bf-hidden", type=int, default=0,
                   help="0 keeps the real hidden_size, so head_dim arithmetic "
                        "matches the model being ported")
    p.add_argument("--bf-T", type=int, default=32)
    p.add_argument("--dtype", default="fp32", choices=("fp32", "bf16", "fp16"),
                   help="model weights and activations; accuracy claims stay on fp32")
    p.add_argument("--grad-ckpt", type=int, default=0,
                   help="recompute activations in the backward instead of storing "
                        "them. Costs ~30%% time, and is the axis that scales with m")
    p.add_argument("--bf-fp64", type=int, default=1,
                   help="run the ground-truth model itself in float64 (default). "
                        "0 compares two fp32 computations and calls one of them truth")
    p.add_argument("--train-m", type=int, default=2)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--verify-all", action="store_true",
                   help="C0: check every hooked module, not one per type. Catches "
                        "an identity that is right for one call shape and wrong "
                        "for another; prints the worst offenders.")
    p.add_argument("--use-leaf-edges", type=int, default=1,
                   help="1 (default): compute_gramian's pruned-graph optimisation "
                        "under driver=loop. 0: bypass it (model.zero_grad()+backward() "
                        "per objective) -- the workaround for the m=3 "
                        "'expected 1 backward capture' bug; see gramian()'s docstring.")
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
