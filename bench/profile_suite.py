"""profile_suite.py -- decoupled profiling of the jdgram Gramian engine.

The engine is a pipeline, and a single end-to-end number cannot say which stage
owns the cost. Each level below isolates one stage so the answer is attributable:

  L0  identity kernels alone      no model, no hooks, no autograd
  L1  forward + hook injection    hook overhead against a bare forward
  L2  reverse drivers             squashed vs batched vs loop -- how A is obtained
  L3  accumulation alone          identity dispatch given precomputed (A, X)
  L4  full step, phase by phase   sync-bracketed, retained vs transient
  L5  autogram / autojac A/B      matched model and batch, plus Gramian equivalence
  L6  scaling laws                sweep m, T, P; fit exponents; test m^2T^2 / mP
  L7  accuracy and convergence    loss curves, val CE, Gramian error vs brute force
  L8  CUDA memory snapshot        opt-in; large file, disk-guarded
  L9  torch.profiler capture      feeds bench/profile_stats.py

Design rules, because measurement bugs are worse than no measurement:
  * every timing is sync-bracketed on both ends, after warmup
  * "peak" = max allocated within a phase, with the counter reset at phase entry
  * "retained" = allocated delta across a phase; a transient phase that retains
    is the leak signature
  * OOM is a logged row with oom=True, never a gap and never fatal
  * --isolate runs each config in a fresh subprocess, so one config's allocator
    state cannot pollute the next one's peak (this is a real effect: long-T probes
    inflate every later reading in the same process)
  * every artifact lands under one run tag; see bench/runtag.py

Usage
  python bench/profile_suite.py --version 5 --name baseline --levels L0 L2 L4 L5
  python bench/profile_suite.py --version 5 --name full --levels all --isolate
  python bench/profile_suite.py --version 5 --name trace --levels L9 --trace-raw
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_ROOT), str(_ROOT / "src"), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import torch.nn as nn

from runtag import RunContext

MiB = 1024**2

# ---- project imports. Kept optional so a partial environment still yields data,
# ---- but tracked loudly: a silently-skipped level looks like a clean result.
IMPORT_ERRORS: dict[str, str] = {}
try:
    from jdgram.engine.hooks import compute_gramian
    from jdgram.engine.materialize import materialized_gramian
    from jdgram.engine.registry import (
        collect_hookable_modules,
        positional_embedding_handler,
    )
    from jdgram.engine.router import route as router_route
    from jdgram.identities import embedding as embedding_id
    from jdgram.identities import linear as linear_id
    from jdgram.identities import norm as norm_id
    from jdgram.identities.precision import workspace_dtype as workspace_dtype_ctx
    from jdgram.identities.tied import tied_gramian
    from jdgram.utils.flatten import flatten_grads, param_layout
except Exception as e:  # noqa: BLE001
    IMPORT_ERRORS["jdgram"] = repr(e)
try:
    from models.configs import forward_logits, per_sequence_losses
    from models.nanogpt.model import GPT, GPTConfig
except Exception as e:  # noqa: BLE001
    IMPORT_ERRORS["models"] = repr(e)
try:
    from torchjd.aggregation import UPGrad, UPGradWeighting
    from torchjd.autogram import Engine as AutogramEngine
    from torchjd.autojac import backward as autojac_backward
    from torchjd.autojac import jac_to_grad
except Exception as e:  # noqa: BLE001
    IMPORT_ERRORS["torchjd"] = repr(e)

DRIVERS = ("squashed", "batched", "loop")
ROUTES = (None, "tfirst", "dfirst")


# ============================================================== instrumentation
def sync(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def allocated(dev: torch.device) -> int:
    return torch.cuda.memory_allocated(dev) if dev.type == "cuda" else 0


def peak_allocated(dev: torch.device) -> int:
    return torch.cuda.max_memory_allocated(dev) if dev.type == "cuda" else 0


def reset_peak(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)


def clear(dev: torch.device) -> None:
    """Between configs: drop python refs, return blocks, zero the peak counter.

    ``gc.collect`` matters here specifically -- the hook machinery forms reference
    cycles through autograd nodes, so refcounting alone does not free it.
    """
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev)
        torch.cuda.reset_accumulated_memory_stats(dev)


@dataclass
class Row:
    """One measurement. Long format: one metric per row, so new metrics never
    change the schema and partial runs remain readable."""

    level: str = ""
    test: str = ""
    engine: str = ""
    driver: str = ""
    route: str = "n/a"
    dtype: str = ""
    m: int = -1
    T: int = -1
    V: int = -1
    n_embd: int = -1
    n_layer: int = -1
    n_head: int = -1
    d_in: int = -1
    d_out: int = -1
    P_layer: int = -1
    P_total: int = -1
    tie: str = ""
    metric: str = ""
    value: float = float("nan")
    unit: str = ""
    oom: bool = False
    note: str = ""


class RunLogger:
    """Append-only CSV, flushed per row, resumable by identity key.

    Flushing every row is deliberate: a sweep that OOMs the host or gets killed by
    the scheduler still leaves every measurement it already took.
    """

    # Every field that distinguishes one configuration from another must be here.
    # A field left out silently collapses two configs into one and the second is
    # skipped as "already done" -- which reads as a completed sweep with a
    # missing row rather than as a bug.
    # n_head and P_total earn their place: an attention-head count change leaves
    # the parameter count untouched (c_attn is n_embd -> 3*n_embd either way), so
    # without n_head a 4-head row and a 12-head row are indistinguishable here AND
    # in the CSV, and re-running a corrected config into an existing directory is
    # skipped as "already done".
    KEY = ("level", "test", "engine", "driver", "route", "dtype", "tie",
           "m", "T", "V", "n_embd", "n_layer", "n_head", "d_in", "d_out",
           "P_layer", "P_total", "metric")

    def __init__(self, path: Path, resume: bool = True) -> None:
        self.path = path
        self.fields = list(asdict(Row()).keys())
        self.seen: set[tuple] = set()
        exists = path.exists()
        if exists and resume:
            with open(path, newline="") as f:
                for r in csv.DictReader(f):
                    self.seen.add(tuple(str(r.get(k, "")) for k in self.KEY))
            print(f"[log] resuming: {len(self.seen)} rows already present in {path}")
        self._f = open(path, "a", newline="")
        self._w = csv.DictWriter(self._f, fieldnames=self.fields)
        if not exists:
            self._w.writeheader()
            self._f.flush()
        self.n = 0

    def key_of(self, row: Row) -> tuple:
        d = asdict(row)
        return tuple(str(d[k]) for k in self.KEY)

    @staticmethod
    def _fill_n_head(row: Row) -> Row:
        """Derive n_head from n_embd the same way build_model does.

        Threading a head count through eight level functions to reach one CSV
        column is not worth it; every level builds its model through build_model,
        which derives n_head from the process-wide HEAD_DIM, so the same
        derivation reproduces it exactly here.
        """
        if row.n_head == -1 and row.n_embd > 0:
            row.n_head = max(1, row.n_embd // HEAD_DIM)
        return row

    def done(self, row: Row) -> bool:
        return self.key_of(self._fill_n_head(row)) in self.seen

    def log(self, row: Row) -> None:
        self._fill_n_head(row)
        self._w.writerow(asdict(row))
        self._f.flush()
        self.seen.add(self.key_of(row))
        self.n += 1

    def many(self, rows: list[Row]) -> None:
        for r in rows:
            self.log(r)

    def close(self) -> None:
        self._f.close()


class Violations:
    """Assertions that would invalidate a conclusion if they failed silently."""

    def __init__(self) -> None:
        self.items: list[dict] = []

    def check(self, name: str, ok: bool, detail: str = "", severity: str = "error") -> bool:
        self.items.append(
            {"check": name, "ok": bool(ok), "detail": detail, "severity": severity}
        )
        mark = "PASS" if ok else ("WARN" if severity == "warn" else "FAIL")
        print(f"  [{mark}] {name}  {detail}")
        return ok

    def summary(self) -> str:
        bad = [i for i in self.items if not i["ok"]]
        errs = [i for i in bad if i["severity"] == "error"]
        out = f"{len(self.items) - len(bad)}/{len(self.items)} checks passed"
        if errs:
            out += " -- ERRORS: " + ", ".join(i["check"] for i in errs)
        return out


@dataclass
class PhaseRecord:
    name: str
    ms: float
    retained_mib: float
    peak_mib: float
    note: str = ""


class PhaseTracker:
    def __init__(self, dev: torch.device) -> None:
        self.dev = dev
        self.records: list[PhaseRecord] = []

    @contextlib.contextmanager
    def phase(self, name: str, note: str = ""):
        dev = self.dev
        sync(dev)
        before = allocated(dev)
        reset_peak(dev)
        t0 = time.perf_counter()
        try:
            yield
        finally:
            sync(dev)
            ms = (time.perf_counter() - t0) * 1e3
            after = allocated(dev)
            self.records.append(
                PhaseRecord(name, ms, (after - before) / MiB, peak_allocated(dev) / MiB, note)
            )

    def table(self, title: str) -> str:
        w = max((len(r.name) for r in self.records), default=12)
        out = [f"\n--- {title} ---",
               f"{'phase':<{w}}  {'ms':>10}  {'retained MiB':>13}  {'peak MiB':>10}  note"]
        for r in self.records:
            flag = "  <-- RETAINS" if r.retained_mib > 50 else ""
            out.append(f"{r.name:<{w}}  {r.ms:>10.3f}  {r.retained_mib:>13.1f}  "
                       f"{r.peak_mib:>10.1f}  {r.note}{flag}")
        return "\n".join(out)

    def rows(self, **kw) -> list[Row]:
        """Expand the recorded phases into Row()s.

        Everything arrives as a keyword, including ``level`` and ``test``, so
        callers can splat the same ``base`` dict they use for standalone Row()s
        without colliding with a positional parameter.
        """
        level = kw.pop("level")
        test = kw.pop("test", "")
        for owned in ("metric", "value", "unit", "note"):
            kw.pop(owned, None)
        rows = []
        for r in self.records:
            for metric, value, unit in (
                ("ms", r.ms, "ms"),
                ("retained_mib", r.retained_mib, "MiB"),
                ("peak_mib", r.peak_mib, "MiB"),
            ):
                rows.append(Row(level=level, test=f"{test}/{r.name}", metric=metric,
                                value=value, unit=unit, note=r.note, **kw))
        return rows


class OOM(Exception):
    pass


@contextlib.contextmanager
def oom_guard(dev: torch.device, label: str):
    """Turn an OOM into data. The allocator is reset afterwards so the next
    config does not inherit a fragmented pool."""
    try:
        yield
    except torch.cuda.OutOfMemoryError as e:  # type: ignore[attr-defined]
        print(f"  [OOM] {label}: {str(e)[:110]}")
        clear(dev)
        raise OOM(label) from e
    except RuntimeError as e:  # older torch spells it differently
        if "out of memory" not in str(e).lower():
            raise
        print(f"  [OOM] {label}: {str(e)[:110]}")
        clear(dev)
        raise OOM(label) from e


def timeit(fn, dev: torch.device, warmup: int = 3, iters: int = 10) -> tuple[float, float, float]:
    """Return (mean ms, peak MiB, stdev ms). Warmup excluded; peak measured only
    over the timed region so allocator growth during warmup is not counted."""
    for _ in range(warmup):
        fn()
    sync(dev)
    reset_peak(dev)
    samples = []
    for _ in range(iters):
        sync(dev)
        t0 = time.perf_counter()
        fn()
        sync(dev)
        samples.append((time.perf_counter() - t0) * 1e3)
    peak = peak_allocated(dev) / MiB
    mean = sum(samples) / len(samples)
    var = sum((s - mean) ** 2 for s in samples) / max(len(samples) - 1, 1)
    return mean, peak, math.sqrt(var)


# ================================================================ model helpers
def pick_device(arg: str = "auto") -> torch.device:
    if arg != "auto":
        return torch.device(arg)
    if not torch.cuda.is_available():
        return torch.device("cpu")
    # On a shared multi-GPU box, take the emptiest visible card.
    best, best_free = 0, -1
    for i in range(torch.cuda.device_count()):
        try:
            free, _total = torch.cuda.mem_get_info(i)
        except Exception:
            free = 0
        if free > best_free:
            best, best_free = i, free
    print(f"[device] cuda:{best} ({best_free / 2**30:.1f} GiB free of "
          f"{torch.cuda.get_device_properties(best).total_memory / 2**30:.1f})")
    return torch.device(f"cuda:{best}")


HEAD_DIM = 64  # GPT-2 uses 64 at every size: 768/12, 1024/16, 1280/20, 1600/25.


def build_model(dev, *, n_layer=4, n_head=None, n_embd=256, T=128, V=65, tie=True,
                bias=True, dtype=torch.float32, seed=0):
    """Deterministic by default.

    Seeding here is not cosmetic: several levels rebuild the model per driver or
    per route and then compare the resulting Gramians. Without a fixed seed those
    comparisons silently diff two different random networks, and an engine that is
    working perfectly reports a disagreement.

    ``n_head=None`` derives the head count from ``n_embd`` at ``HEAD_DIM``. This
    used to be a hardcoded 4, which is right only at the default ``n_embd=256``
    (256/4 = 64) and silently wrong everywhere else: asking for GPT-2's 768 got
    4 heads of 192 rather than 12 of 64, so a run labelled "nanoGPT 124M" would
    have profiled a model with a different attention shape and a different
    parameter count. Deriving reproduces the old default exactly and scales.
    """
    if n_head is None:
        n_head = max(1, n_embd // HEAD_DIM)
    if n_embd % n_head:
        raise ValueError(f"n_embd={n_embd} not divisible by n_head={n_head}")
    torch.manual_seed(seed)
    cfg = GPTConfig(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=T,
                    vocab_size=V, dropout=0.0, bias=bias)
    model = GPT(cfg).to(device=dev, dtype=dtype)
    if not tie:
        model.transformer.wte.weight = nn.Parameter(
            model.transformer.wte.weight.detach().clone()
        )
    model.eval()  # dropout is 0.0 anyway; eval keeps the two passes identical
    modules = collect_hookable_modules(model)
    overrides = {"transformer.wpe": positional_embedding_handler}
    shared = {}
    if tie:
        shared[frozenset({"transformer.wte", "lm_head"})] = (
            lambda caps: tied_gramian(
                caps["lm_head"].A, caps["lm_head"].X,
                caps["transformer.wte"].A, caps["transformer.wte"].X,
            )
        )
    return model, modules, overrides, shared


def count_params(model: nn.Module) -> int:
    seen, total = set(), 0
    for p in model.parameters():
        if p.requires_grad and id(p) not in seen:
            seen.add(id(p))
            total += p.numel()
    return total


def synthetic_batch(m, T, V, dev, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    idx = torch.randint(0, V, (m, T), generator=g).to(dev)
    tgt = torch.randint(0, V, (m, T), generator=g).to(dev)
    return idx, tgt


def make_AX(m, T, d_out, d_in, dev, dtype, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    A = torch.randn(m, T, d_out, generator=g).to(dev, dtype)
    X = torch.randn(m, T, d_in, generator=g).to(dev, dtype)
    return A, X


def brute_force_gramian(model, idx, tgt) -> torch.Tensor:
    """Ground truth: build the [m, P] Jacobian explicitly and square it."""
    layout = param_layout(model)
    losses = per_sequence_losses(forward_logits(model, idx), tgt)
    params = [p for _, p, _ in layout]
    m = idx.shape[0]
    J = torch.empty(m, layout[-1][2].stop, dtype=torch.float64, device=idx.device)
    for i in range(m):
        grads = torch.autograd.grad(losses[i], params, retain_graph=(i < m - 1))
        J[i] = flatten_grads(grads, layout).double()
    return J @ J.T


def jdgram_gramian(model, modules, overrides, shared, idx, tgt, *, driver=None,
                   force_route=None, wdtype=None):
    return compute_gramian(
        model,
        lambda: per_sequence_losses(forward_logits(model, idx), tgt),
        modules=modules,
        handler_overrides=overrides,
        shared_handlers=shared,
        driver=driver,
        force_route=force_route,
        workspace_dtype=wdtype,
    )


def theoretical_workspace_mib(m, T, P_layer, itemsize=4) -> dict[str, float]:
    """What each route *should* cost, so measurement can be held against it.

    T-first is ``3 m T^2``, not ``2 m^2 T^2``: linear.sequence_gramian blocks over
    the objective index and holds ``K_A``, ``K_X`` and their product at ``[T, mT]``
    each, never a full ``[mT, mT]``. Using the old whole-kernel formula here made
    every measured T-first row read as *below* theory (ratios of 0.3-0.4), which
    looks like a suspiciously good result rather than a stale predictor.
    """
    return {
        "tfirst": 3 * m * T * T * itemsize / MiB,      # K_A, K_X, product at [T, mT]
        "tfirst_wholekernel": 2 * m * m * T * T * itemsize / MiB,  # pre-blocking
        "dfirst": m * P_layer * itemsize / MiB,        # one [m, P_layer] block
    }


# ======================================================== L0: identities alone
L0_SHAPES = [
    #  m,   T, d_out, d_in, label
    (4, 128, 256, 256, "small"),
    (8, 128, 256, 256, "bench-default-T128"),
    (8, 512, 256, 256, "bench-default"),
    (8, 512, 1024, 256, "mlp-c_fc"),
    (4, 512, 2048, 2048, "large-P"),
    (4, 2048, 256, 256, "long-T"),
    (8, 512, 50257, 256, "vocab-head-50k"),
    (4, 512, 32, 2048, "lora-A-r32"),
    (4, 512, 2048, 32, "lora-B-r32"),
]


def _l0_estimate_gib(m, T, d_out, d_in, itemsize) -> float:
    """Rough high-water mark for one shape: the two [mT, mT] kernels, the
    [m, d_out, d_in] block, and the A/X inputs themselves."""
    tfirst = 2 * (m * T) ** 2
    dfirst = m * d_out * d_in
    inputs = m * T * (d_out + d_in)
    return max(tfirst, dfirst) * itemsize / 2**30 + inputs * itemsize / 2**30


def level0_identities(rc, dev, log: RunLogger, viol: Violations, dtype_name: str,
                      max_alloc_gib: float = 8.0) -> None:
    """The decisive baseline: identity math with no model, no hooks, no autograd.

    If these are small while a full step is not, the arithmetic is exonerated and
    every remaining optimisation belongs to the driver.
    """
    print("\n" + "=" * 78)
    print("L0 -- identity kernels alone (no model, no hooks, no autograd)")
    print("=" * 78)
    if "jdgram" in IMPORT_ERRORS:
        print("  SKIPPED:", IMPORT_ERRORS["jdgram"])
        return
    wd = torch.float32 if dtype_name == "fp32" else torch.float64
    itemsize = 4 if dtype_name == "fp32" else 8

    print(f"{'label':<22} {'P_layer':>12} {'route':>8} "
          f"{'tfirst ms':>10} {'tfirst MiB':>11} {'dfirst ms':>10} {'dfirst MiB':>11} "
          f"{'theo tf':>9} {'theo df':>9}")
    for m, T, d_out, d_in, label in L0_SHAPES:
        P = d_out * d_in
        est = _l0_estimate_gib(m, T, d_out, d_in, itemsize)
        if est > max_alloc_gib:
            # Logged, not silently dropped: a missing row would read as "covered".
            log.log(Row(level="L0", test=label, engine="identity", dtype=dtype_name,
                        m=m, T=T, d_in=d_in, d_out=d_out, P_layer=P,
                        metric="skipped_estimate_gib", value=est, unit="GiB",
                        note=f"exceeds --max-alloc-gib={max_alloc_gib}"))
            print(f"{label:<22} {P:>12,} {'SKIP':>8}  est {est:.1f} GiB > "
                  f"--max-alloc-gib {max_alloc_gib}")
            continue
        theo = theoretical_workspace_mib(m, T, P, itemsize)
        base = dict(level="L0", test=label, engine="identity", dtype=dtype_name,
                    m=m, T=T, d_in=d_in, d_out=d_out, P_layer=P)
        if log.done(Row(**base, driver="tfirst", metric="ms")):
            continue

        res: dict[str, tuple[float, float, float]] = {}
        for rname, fn in (
            ("tfirst", lambda: linear_id.sequence_gramian(A, X, has_bias=False,
                                                          workspace_dtype=wd)),
            ("dfirst", lambda: materialized_gramian(A, X, has_bias=False,
                                                    workspace_dtype=wd)),
        ):
            clear(dev)
            try:
                A, X = make_AX(m, T, d_out, d_in, dev, wd)
                with oom_guard(dev, f"L0/{label}/{rname}"):
                    res[rname] = timeit(fn, dev)
            except OOM:
                res[rname] = (float("nan"), float("nan"), float("nan"))
                log.log(Row(**base, driver=rname, metric="ms", value=float("nan"),
                            unit="ms", oom=True, note="OOM"))
            finally:
                A = X = None
                clear(dev)

        for rname, (ms, mib, sd) in res.items():
            log.many([
                Row(**base, driver=rname, metric="ms", value=ms, unit="ms"),
                Row(**base, driver=rname, metric="ms_std", value=sd, unit="ms"),
                Row(**base, driver=rname, metric="peak_mib", value=mib, unit="MiB"),
                Row(**base, driver=rname, metric="theoretical_mib",
                    value=theo["tfirst"] if rname == "tfirst" else theo["dfirst"],
                    unit="MiB"),
            ])
        predicted = router_route(m, T, P)
        print(f"{label:<22} {P:>12,} {predicted:>8} "
              f"{res['tfirst'][0]:>10.3f} {res['tfirst'][1]:>11.1f} "
              f"{res['dfirst'][0]:>10.3f} {res['dfirst'][1]:>11.1f} "
              f"{theo['tfirst']:>9.1f} {theo['dfirst']:>9.1f}")

        # The router's job is to pick the cheaper route; check it against measurement.
        if not math.isnan(res["tfirst"][0]) and not math.isnan(res["dfirst"][0]):
            faster = "tfirst" if res["tfirst"][0] < res["dfirst"][0] else "dfirst"
            viol.check(f"L0_router_picks_faster@{label}", faster == predicted,
                       f"measured={faster} router={predicted}", severity="warn")
            leaner = "tfirst" if res["tfirst"][1] < res["dfirst"][1] else "dfirst"
            log.log(Row(**base, driver="measured", metric="faster_is_tfirst",
                        value=float(faster == "tfirst"), unit="bool"))
            log.log(Row(**base, driver="measured", metric="leaner_is_tfirst",
                        value=float(leaner == "tfirst"), unit="bool"))


# ============================================== L1: forward + hook injection
def level1_hook_overhead(rc, dev, log: RunLogger, viol: Violations, dtype_name: str,
                         m=8, T=512, V=65, n_embd=256, n_layer=4) -> None:
    """Cost of installing hooks and wrapping outputs, with no reverse pass at all."""
    print("\n" + "=" * 78)
    print(f"L1 -- forward + hook injection (m={m} T={T} d={n_embd})")
    print("=" * 78)
    if IMPORT_ERRORS:
        print("  SKIPPED:", IMPORT_ERRORS)
        return
    from jdgram.engine.edges import EdgeRegistry
    from jdgram.engine.hooks import ModuleHookManager

    model, modules, _ov, _sh = build_model(dev, n_embd=n_embd, T=T, V=V, n_layer=n_layer)
    idx, tgt = synthetic_batch(m, T, V, dev)
    base = dict(level="L1", engine="jdgram", dtype=dtype_name, m=m, T=T, V=V,
                n_embd=n_embd, n_layer=n_layer, P_total=count_params(model))

    # Each iteration drops its own graph and captures. Without that the phase's
    # "retained" column just reports N accumulated forwards, which looks like a
    # leak in the hooked case and like nothing in the bare case -- an artifact of
    # the loop, not a property of the hooks.
    tr = PhaseTracker(dev)
    reps = 5
    for _ in range(2):  # warmup, outside any phase
        per_sequence_losses(forward_logits(model, idx), tgt)
    clear(dev)
    with tr.phase("forward_bare", f"x{reps}, no hooks installed"):
        for _ in range(reps):
            losses = per_sequence_losses(forward_logits(model, idx), tgt)
            del losses
    clear(dev)
    # Timing and retained-memory want opposite things here. Freeing the hook
    # cycles needs gc.collect(), which costs ~100 ms with a live graph -- inside
    # the timed loop that is charged to the hooks and reports a 25x overhead that
    # does not exist. So: time without collecting, then measure retention
    # separately with the cleanup included.
    held: list = []
    with tr.phase("forward_hooked", f"x{reps}, hooks installed, outputs wrapped"):
        for _ in range(reps):
            edges = EdgeRegistry()
            with ModuleHookManager(edges) as mgr:
                for name, mod in modules.items():
                    mgr.hook_module(name, mod)
                losses = per_sequence_losses(forward_logits(model, idx), tgt)
            held.append((mgr, edges, losses))
            del losses, mgr, edges
    # Now the retention question, outside any timed region. The hook machinery
    # forms capture->node->graph cycles, so refcounting alone cannot free it;
    # whether gc.collect() reclaims it is the thing worth knowing.
    before_release = allocated(dev)
    for mgr, edges, losses in held:
        for cap in mgr.captures.values():
            cap.inputs.clear()
            cap.rg_outputs.clear()
            cap.clear_grads()
    held.clear()
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    reclaimed = (before_release - allocated(dev)) / MiB
    print(f"\n  reclaimed after clearing captures + gc: {reclaimed:.1f} MiB "
          f"({reps} hooked forwards were held)")
    log.log(Row(**base, test="hook_overhead", metric="reclaimed_after_gc_mib",
                value=reclaimed, unit="MiB"))

    print(tr.table("L1 hook overhead"))
    recs = {r.name: r for r in tr.records}
    if recs["forward_bare"].ms > 0:
        factor = recs["forward_hooked"].ms / recs["forward_bare"].ms
        print(f"\n  hook overhead factor on the FORWARD only: {factor:.2f}x")
        log.log(Row(**base, test="hook_overhead", metric="forward_factor",
                    value=factor, unit="x"))
        viol.check("L1_hook_overhead_small", factor < 2.0, f"{factor:.2f}x",
                   severity="warn")
    log.many(tr.rows(**base, test="hook_overhead"))
    del model, modules
    clear(dev)


# ================================================= L2: the reverse drivers
def level2_drivers(rc, dev, log: RunLogger, viol: Violations, dtype_name: str,
                   m=8, T=512, V=65, n_embd=256, n_layer=4, tie=True,
                   drivers=DRIVERS) -> None:
    """The axis that decides jdgram's memory profile.

    ``squashed`` runs one ordinary ones-seeded backward; ``batched`` vmaps the
    whole reverse over m via ``is_grads_batched``; ``loop`` runs m of them. Same
    Gramian, very different working sets -- and the previous benchmark never
    varied this, which is why route sweeps kept reporting identical peaks.
    """
    print("\n" + "=" * 78)
    print(f"L2 -- reverse drivers (m={m} T={T} V={V} d={n_embd} tie={tie})")
    print("=" * 78)
    if IMPORT_ERRORS:
        print("  SKIPPED:", IMPORT_ERRORS)
        return
    wd = torch.float32 if dtype_name == "fp32" else torch.float64
    results: dict[str, torch.Tensor] = {}
    print(f"{'driver':<10} {'ms':>10} {'peak MiB':>10} {'held MiB':>10}  note")
    for driver in drivers:
        base = dict(level="L2", test="driver", engine="jdgram", driver=driver,
                    dtype=dtype_name, m=m, T=T, V=V, n_embd=n_embd,
                    n_layer=n_layer, tie=str(tie))
        if log.done(Row(**base, metric="ms")):
            continue
        clear(dev)
        model, modules, ov, sh = build_model(dev, n_embd=n_embd, T=T, V=V,
                                             n_layer=n_layer, tie=tie)
        idx, tgt = synthetic_batch(m, T, V, dev)
        try:
            with oom_guard(dev, f"L2/{driver}"):
                # warm up allocator so first-call growth is not read as the cost
                jdgram_gramian(model, modules, ov, sh, idx, tgt, driver=driver, wdtype=wd)
                clear(dev)
                sync(dev)
                reset_peak(dev)
                t0 = time.perf_counter()
                res = jdgram_gramian(model, modules, ov, sh, idx, tgt,
                                     driver=driver, wdtype=wd)
                sync(dev)
                ms = (time.perf_counter() - t0) * 1e3
                peak = peak_allocated(dev) / MiB
                held = res.peak_held_bytes / MiB
                results[driver] = res.total.detach().clone()
            log.many([
                Row(**base, metric="ms", value=ms, unit="ms"),
                Row(**base, metric="peak_mib", value=peak, unit="MiB"),
                Row(**base, metric="held_capture_mib", value=held, unit="MiB"),
            ])
            print(f"{driver:<10} {ms:>10.2f} {peak:>10.1f} {held:>10.3f}")
        except OOM:
            log.log(Row(**base, metric="peak_mib", value=float("nan"), unit="MiB",
                        oom=True, note="OOM"))
            print(f"{driver:<10} {'OOM':>10}")
        finally:
            del model, modules
            clear(dev)

    if len(results) > 1:
        ref_name = "loop" if "loop" in results else next(iter(results))
        ref = results[ref_name]
        for name, G in results.items():
            if name == ref_name:
                continue
            d = (G - ref).abs().max().item()
            viol.check(f"L2_driver_agrees_{name}_vs_{ref_name}", d < 1e-4, f"{d:.3e}")
            log.log(Row(level="L2", test="driver_equivalence", engine="jdgram",
                        driver=name, dtype=dtype_name, m=m, T=T, V=V, n_embd=n_embd,
                        tie=str(tie), metric="max_abs_diff", value=d, unit="abs"))


# ======================================== L3: accumulation with A, X in hand
def level3_accumulate(rc, dev, log: RunLogger, viol: Violations, dtype_name: str,
                      m=8, T=512, V=65, n_embd=256, n_layer=4) -> None:
    """Identity dispatch and accumulation only: A and X are precomputed, so this
    is the pure cost of the per-layer math plus the python dispatch around it."""
    print("\n" + "=" * 78)
    print(f"L3 -- accumulation only, per route (m={m} T={T} d={n_embd})")
    print("=" * 78)
    if IMPORT_ERRORS:
        print("  SKIPPED:", IMPORT_ERRORS)
        return
    wd = torch.float32 if dtype_name == "fp32" else torch.float64
    model, modules, _ov, _sh = build_model(dev, n_embd=n_embd, T=T, V=V, n_layer=n_layer)
    linears = {n: mod for n, mod in modules.items() if isinstance(mod, nn.Linear)}

    for route in ("tfirst", "dfirst"):
        base = dict(level="L3", test="accumulate_linears", engine="jdgram",
                    route=route, dtype=dtype_name, m=m, T=T, n_embd=n_embd,
                    n_layer=n_layer)
        if log.done(Row(**base, metric="ms")):
            continue
        clear(dev)
        payload = []
        try:
            with oom_guard(dev, f"L3/{route}"):
                for name, mod in linears.items():
                    A, X = make_AX(m, T, mod.out_features, mod.in_features, dev, wd)
                    payload.append((mod, A, X))

                def run():
                    total = None
                    for mod, A, X in payload:
                        fn = (linear_id.sequence_gramian if route == "tfirst"
                              else materialized_gramian)
                        g = fn(A, X, mod.bias is not None, workspace_dtype=wd)
                        total = g if total is None else total.add_(g)
                    return total

                ms, peak, sd = timeit(run, dev, warmup=2, iters=5)
            log.many([
                Row(**base, metric="ms", value=ms, unit="ms"),
                Row(**base, metric="ms_std", value=sd, unit="ms"),
                Row(**base, metric="peak_mib", value=peak, unit="MiB"),
                Row(**base, metric="n_layers_accumulated", value=len(payload), unit="count"),
            ])
            print(f"  route={route:<8} {ms:>9.3f} ms  peak {peak:>8.1f} MiB  "
                  f"over {len(payload)} Linears")
        except OOM:
            log.log(Row(**base, metric="peak_mib", value=float("nan"), unit="MiB",
                        oom=True, note="OOM"))
        finally:
            payload.clear()
            clear(dev)
    del model, modules
    clear(dev)


# ================================================ L4: full step, phase by phase
def level4_phases(rc, dev, log: RunLogger, viol: Violations, dtype_name: str,
                  m=8, T=512, V=65, n_embd=256, n_layer=4, tie=True,
                  driver=None, force_route=None, reps=10) -> None:
    """One complete jdgram optimiser step, decomposed. A transient phase that
    retains memory after a sync is where state is being held."""
    label = f"{driver or 'default'}/{force_route or 'auto'}"
    print("\n" + "=" * 78)
    print(f"L4 -- full-step phases (m={m} T={T} V={V} d={n_embd} {label})")
    print("=" * 78)
    if IMPORT_ERRORS:
        print("  SKIPPED:", IMPORT_ERRORS)
        return
    wd = torch.float32 if dtype_name == "fp32" else torch.float64
    base = dict(level="L4", engine="jdgram", driver=driver or "default",
                route=force_route or "auto", dtype=dtype_name, m=m, T=T, V=V,
                n_embd=n_embd, n_layer=n_layer, tie=str(tie))

    clear(dev)
    tr = PhaseTracker(dev)
    with tr.phase("build_model"):
        model, modules, ov, sh = build_model(dev, n_embd=n_embd, T=T, V=V,
                                             n_layer=n_layer, tie=tie)
    idx, tgt = synthetic_batch(m, T, V, dev)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)

    def losses_fn():
        return per_sequence_losses(forward_logits(model, idx), tgt)

    with contextlib.suppress(Exception):  # allocator warmup only
        jdgram_gramian(model, modules, ov, sh, idx, tgt, driver=driver,
                       force_route=force_route, wdtype=wd)
    clear(dev)

    # Every phase is timed over `reps` iterations after a warmup pass. Measuring
    # each phase once conflates steady-state cost with one-time initialisation --
    # lazy imports, cuBLAS handle creation, kernel autotuning -- and those land
    # almost entirely on whichever phase happens to run first. A single-shot
    # `optimizer_step` reading ~22 ms for SGD on a 3.2M-parameter model is that
    # artifact, not the optimiser.
    weighting = UPGradWeighting()
    try:
        with oom_guard(dev, "L4/warmup"):
            _r = jdgram_gramian(model, modules, ov, sh, idx, tgt, driver=driver,
                                force_route=force_route, wdtype=wd)
            _w = weighting(_r.total.to(torch.float32))
            _l = losses_fn()
            _l.backward(_w.to(_l.dtype))
            opt.step()
            opt.zero_grad(set_to_none=True)
            del _r, _w, _l
        clear(dev)

        with tr.phase("forward_only", f"x{reps}, graph freed each time"):
            for _ in range(reps):
                losses = losses_fn()
                del losses  # do not let one forward's graph inflate the next phase
        with tr.phase("compute_gramian", f"x{reps}, capture + reverse + identities"):
            with oom_guard(dev, "L4/compute_gramian"):
                for _ in range(reps):
                    result = jdgram_gramian(model, modules, ov, sh, idx, tgt,
                                            driver=driver, force_route=force_route,
                                            wdtype=wd)
        G = result.total
        G32 = G.to(torch.float32)
        with tr.phase("weighting_qp", f"x{reps}, UPGrad dual-cone solve on [m,m]"):
            for _ in range(reps):
                w = weighting(G32)
        with tr.phase("final_backward", f"x{reps}, fresh forward + weighted backward"):
            for _ in range(reps):
                losses2 = losses_fn()
                losses2.backward(w.to(losses2.dtype))
                opt.zero_grad(set_to_none=True)
        with tr.phase("optimizer_step", f"x{reps}"):
            for _ in range(reps):
                opt.step()
    except OOM:
        log.log(Row(**base, test="phases", metric="peak_mib", value=float("nan"),
                    unit="MiB", oom=True, note="OOM"))
        del model, modules
        clear(dev)
        return

    for r in tr.records:
        if r.name != "build_model":
            r.ms /= reps          # report per-iteration cost, not the batch total
    print(tr.table(f"L4 phases {label} (ms = per iteration, n={reps})"))
    print(f"\n  held captures at streaming high-water mark: "
          f"{result.peak_held_bytes / MiB:.3f} MiB  (driver={result.driver})")
    log.many(tr.rows(**base, test="phases"))
    log.log(Row(**base, test="phases", metric="held_capture_mib",
                value=result.peak_held_bytes / MiB, unit="MiB"))

    cg = next((r for r in tr.records if r.name == "compute_gramian"), None)
    if cg:
        viol.check("L4_compute_gramian_is_transient", cg.retained_mib < 50,
                   f"retains {cg.retained_mib:.1f} MiB after returning an [m,m] matrix")
        biggest = max((mod.out_features * mod.in_features
                       for mod in modules.values() if isinstance(mod, nn.Linear)),
                      default=1)
        theo = theoretical_workspace_mib(m, T, biggest, 4 if dtype_name == "fp32" else 8)
        print(f"  theoretical per-layer workspace: tfirst {theo['tfirst']:.1f} MiB, "
              f"dfirst {theo['dfirst']:.1f} MiB (largest Linear P={biggest:,})")
        log.many([
            Row(**base, test="phases", metric="theoretical_tfirst_mib",
                value=theo["tfirst"], unit="MiB"),
            Row(**base, test="phases", metric="theoretical_dfirst_mib",
                value=theo["dfirst"], unit="MiB"),
        ])
    del model, modules, G
    clear(dev)


# ================================================== L5: autogram / autojac A/B
def level5_ab(rc, dev, log: RunLogger, viol: Violations, dtype_name: str,
              m=8, T=512, V=65, n_embd=256, n_layer=4) -> None:
    """Matched head-to-head. Both engines get the same model, the same batch, and
    the same batched-position forward -- autogram's ``batch_dim=0`` contract needs
    every hooked module batched on dim 0, which is exactly what jdgram's squashed
    driver needs too, so one forward now serves both."""
    print("\n" + "=" * 78)
    print(f"L5 -- jdgram vs autogram vs autojac (m={m} T={T} V={V} d={n_embd})")
    print("=" * 78)
    if IMPORT_ERRORS:
        print("  SKIPPED:", IMPORT_ERRORS)
        return
    wd = torch.float32 if dtype_name == "fp32" else torch.float64

    for tie in (False, True):
        base = dict(level="L5", dtype=dtype_name, m=m, T=T, V=V, n_embd=n_embd,
                    n_layer=n_layer, tie=str(tie))
        clear(dev)
        model, modules, ov, sh = build_model(dev, n_embd=n_embd, T=T, V=V,
                                             n_layer=n_layer, tie=tie)
        idx, tgt = synthetic_batch(m, T, V, dev)
        P_total = count_params(model)

        def losses_fn():
            return per_sequence_losses(forward_logits(model, idx), tgt)

        measured: dict[str, tuple[float, float]] = {}
        grams: dict[str, torch.Tensor] = {}

        # autojac is the arm that decides whether any of this is worth doing: it
        # materializes the whole [m, P] Jacobian, so it is the fastest option
        # until P is large enough that it cannot allocate. Leaving it out of this
        # loop meant the level printed a three-engine banner and measured two,
        # and the crossover -- the size where autojac dies and the Gramian
        # engines live -- could not be produced at all.
        for engine in ("jdgram", "autogram", "autojac"):
            row = dict(base, engine=engine, P_total=P_total)
            if log.done(Row(**row, test="ab", metric="ms")):
                continue
            clear(dev)
            try:
                with oom_guard(dev, f"L5/{engine}/tie={tie}"):
                    if engine == "jdgram":
                        def once():
                            return jdgram_gramian(model, modules, ov, sh, idx, tgt,
                                                  wdtype=wd).total
                    elif engine == "autogram":
                        eng = AutogramEngine(model, batch_dim=0)

                        def once():
                            return eng.compute_gramian(losses_fn())
                    else:
                        # No Gramian of its own: build J explicitly and square it,
                        # which is the cost being compared against.
                        aj_params = [p for p in model.parameters() if p.requires_grad]

                        def once():
                            J = torch.autograd.grad(
                                losses_fn(), aj_params,
                                grad_outputs=torch.eye(m, device=dev, dtype=wd),
                                is_grads_batched=True, retain_graph=False,
                                allow_unused=True, materialize_grads=True,
                            )
                            Jf = torch.cat([g.reshape(m, -1) for g in J], dim=1)
                            return Jf @ Jf.T

                    once()  # warm
                    clear(dev)
                    sync(dev)
                    reset_peak(dev)
                    t0 = time.perf_counter()
                    G = once()
                    sync(dev)
                    ms = (time.perf_counter() - t0) * 1e3
                    peak = peak_allocated(dev) / MiB
                    grams[engine] = G.detach().double().clone()
                    measured[engine] = (ms, peak)
                log.many([
                    Row(**row, test="ab", metric="ms", value=ms, unit="ms"),
                    Row(**row, test="ab", metric="peak_mib", value=peak, unit="MiB"),
                ])
                print(f"  tie={tie!s:<5} {engine:<9} {ms:>9.2f} ms  peak {peak:>9.1f} MiB")
            except OOM:
                log.log(Row(**row, test="ab", metric="peak_mib", value=float("nan"),
                            unit="MiB", oom=True, note="OOM"))
                print(f"  tie={tie!s:<5} {engine:<9} {'OOM':>9}")
            finally:
                clear(dev)

        if "jdgram" in measured and "autogram" in measured:
            t_ratio = measured["jdgram"][0] / max(measured["autogram"][0], 1e-9)
            p_ratio = measured["jdgram"][1] / max(measured["autogram"][1], 1e-9)
            print(f"     ratio jdgram/autogram -- time {t_ratio:.2f}x  peak {p_ratio:.2f}x")
            log.many([
                Row(**base, engine="ratio", test="ab", metric="time_ratio_jd_over_ag",
                    value=t_ratio, unit="x"),
                Row(**base, engine="ratio", test="ab", metric="peak_ratio_jd_over_ag",
                    value=p_ratio, unit="x"),
            ])

        if "jdgram" in grams and "autogram" in grams:
            d = (grams["jdgram"] - grams["autogram"]).abs().max().item()
            scale = grams["jdgram"].abs().max().item()
            rel = d / max(scale, 1e-30)
            log.log(Row(**base, engine="pair", test="ab", metric="rel_diff_jd_vs_ag",
                        value=rel, unit="rel"))
            if tie:
                # autogram sums per-module Gramians, so it drops the cross terms
                # between two modules sharing one parameter. Deviation is expected
                # and is the correctness claim -- flag its ABSENCE, not its presence.
                viol.check("L5_autogram_deviates_on_tied", rel > 1e-6,
                           f"rel={rel:.3e} (expected nonzero: dropped tied cross-terms)",
                           severity="warn")
            else:
                viol.check("L5_engines_agree_untied", rel < 1e-5, f"rel={rel:.3e}")

        # Brute force anchor, only where the Jacobian is affordable. Budget in
        # BYTES, not elements: the [m, P] anchor is float64, so the real cost is
        # 8*m*P. The old element-count guard (P*m < 4e7) allowed 40M elements
        # regardless of dtype, which at GPT-2's 124M parameters demanded m < 0.32
        # -- i.e. the anchor silently never ran at the size where the tied-weight
        # claim most needs an anchor.
        brute_bytes = 8 * m * P_total
        brute_budget = float(os.environ.get("JDGRAM_BRUTE_BUDGET_GIB", "6.0")) * 2**30
        if brute_bytes < brute_budget and "jdgram" in grams:
            try:
                with oom_guard(dev, "L5/brute"):
                    G_true = brute_force_gramian(model, idx, tgt)
                for engine, G in grams.items():
                    rel = ((G - G_true).abs().max() / G_true.abs().max()).item()
                    log.log(Row(**base, engine=engine, test="ab",
                                metric="rel_diff_vs_brute", value=rel, unit="rel"))
                    print(f"     {engine:<9} vs brute force: rel {rel:.3e}")
                    if engine == "jdgram":
                        viol.check(f"L5_jdgram_exact_tie={tie}", rel < 1e-5, f"{rel:.3e}")
            except OOM:
                pass
        del model, modules
        clear(dev)


# ==================================================== L6: scaling / complexity
def _fit_loglog(xs: list[float], ys: list[float]) -> float:
    """Least-squares slope of log y vs log x -- the measured scaling exponent."""
    pts = [(math.log(x), math.log(y)) for x, y in zip(xs, ys) if x > 0 and y > 0]
    if len(pts) < 2:
        return float("nan")
    n = len(pts)
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    num = sum((p[0] - mx) * (p[1] - my) for p in pts)
    den = sum((p[0] - mx) ** 2 for p in pts)
    return num / den if den else float("nan")


def level6_scaling(rc, dev, log: RunLogger, viol: Violations, dtype_name: str,
                   Ts=(64, 128, 256, 512, 1024), ms=(1, 2, 4, 8, 16),
                   V=65, n_embd=256, n_layer=4) -> None:
    """Measure the exponents and hold them against the intended complexity.

    Intent: the T-first route's workspace is ``m^2 T^2`` and the d-first route's
    is ``m P_layer``, both transient and per layer. If peak instead tracks the
    driver, the exponents will not match and they will not move with the route --
    which is precisely the symptom this suite exists to settle.
    """
    print("\n" + "=" * 78)
    print("L6 -- scaling laws: exponents in T and m, per driver and route")
    print("=" * 78)
    if IMPORT_ERRORS:
        print("  SKIPPED:", IMPORT_ERRORS)
        return
    wd = torch.float32 if dtype_name == "fp32" else torch.float64

    for axis, values, fixed in (("T", Ts, dict(m=8)), ("m", ms, dict(T=256))):
        for driver in DRIVERS:
            for froute in ROUTES:
                xs, peaks, times = [], [], []
                for v in values:
                    kw = dict(fixed)
                    kw[axis] = v
                    m_, T_ = kw.get("m", 8), kw.get("T", 256)
                    base = dict(level="L6", test=f"scale_{axis}", engine="jdgram",
                                driver=driver, route=froute or "auto",
                                dtype=dtype_name, m=m_, T=T_, V=V, n_embd=n_embd,
                                n_layer=n_layer)
                    clear(dev)
                    try:
                        model, modules, ov, sh = build_model(
                            dev, n_embd=n_embd, T=T_, V=V, n_layer=n_layer)
                        idx, tgt = synthetic_batch(m_, T_, V, dev)
                        with oom_guard(dev, f"L6/{axis}={v}/{driver}/{froute}"):
                            jdgram_gramian(model, modules, ov, sh, idx, tgt,
                                           driver=driver, force_route=froute, wdtype=wd)
                            clear(dev)
                            sync(dev)
                            reset_peak(dev)
                            t0 = time.perf_counter()
                            jdgram_gramian(model, modules, ov, sh, idx, tgt,
                                           driver=driver, force_route=froute, wdtype=wd)
                            sync(dev)
                            ms_ = (time.perf_counter() - t0) * 1e3
                            peak = peak_allocated(dev) / MiB
                        xs.append(float(v))
                        peaks.append(peak)
                        times.append(ms_)
                        log.many([
                            Row(**base, metric="peak_mib", value=peak, unit="MiB"),
                            Row(**base, metric="ms", value=ms_, unit="ms"),
                        ])
                    except OOM:
                        log.log(Row(**base, metric="peak_mib", value=float("nan"),
                                    unit="MiB", oom=True, note="OOM"))
                        break
                    except Exception as e:  # noqa: BLE001
                        log.log(Row(**base, metric="peak_mib", value=float("nan"),
                                    unit="MiB", note=f"ERR {type(e).__name__}"))
                        break
                    finally:
                        del model, modules
                        clear(dev)

                if len(xs) >= 3:
                    sp, st = _fit_loglog(xs, peaks), _fit_loglog(xs, times)
                    tag = f"{driver}/{froute or 'auto'}"
                    print(f"  d(log peak)/d(log {axis}) = {sp:5.2f}   "
                          f"d(log time)/d(log {axis}) = {st:5.2f}   [{tag}]")
                    log.many([
                        Row(level="L6", test=f"exponent_{axis}", engine="jdgram",
                            driver=driver, route=froute or "auto", dtype=dtype_name,
                            V=V, n_embd=n_embd, metric=f"peak_exponent_{axis}",
                            value=sp, unit="slope"),
                        Row(level="L6", test=f"exponent_{axis}", engine="jdgram",
                            driver=driver, route=froute or "auto", dtype=dtype_name,
                            V=V, n_embd=n_embd, metric=f"time_exponent_{axis}",
                            value=st, unit="slope"),
                    ])


# ================================================= L7: accuracy / convergence
def level7_accuracy(rc, dev, log: RunLogger, viol: Violations, dtype_name: str,
                    m=8, T=128, V=65, n_embd=256, n_layer=4, steps=100,
                    lr=0.01) -> None:
    """Does the engine still train? Loss curves for jdgram-UPGrad against
    autogram-UPGrad, autojac-UPGrad and plain SGD, on identical data and seeds."""
    print("\n" + "=" * 78)
    print(f"L7 -- convergence (m={m} T={T} steps={steps})")
    print("=" * 78)
    if IMPORT_ERRORS:
        print("  SKIPPED:", IMPORT_ERRORS)
        return
    wd = torch.float32 if dtype_name == "fp32" else torch.float64

    for engine in ("jdgram", "autogram", "autojac", "sgd_erm"):
        base = dict(level="L7", test="convergence", engine=engine, dtype=dtype_name,
                    m=m, T=T, V=V, n_embd=n_embd, n_layer=n_layer)
        if log.done(Row(**base, metric="final_loss")):
            continue
        torch.manual_seed(42)
        clear(dev)
        model, modules, ov, sh = build_model(dev, n_embd=n_embd, T=T, V=V,
                                             n_layer=n_layer)
        opt = torch.optim.SGD(model.parameters(), lr=lr)
        weighting = UPGradWeighting() if engine in ("jdgram", "autogram") else None
        # Engine constructed ONCE: rebuilding it per step re-hooks the model and
        # corrupts the Gramian.
        ag = AutogramEngine(model, batch_dim=0) if engine == "autogram" else None
        params = list(model.parameters()) if engine == "autojac" else []
        aggregator = UPGrad() if engine == "autojac" else None
        losses_seen: list[float] = []

        try:
            with oom_guard(dev, f"L7/{engine}"):
                for step in range(steps):
                    idx, tgt = synthetic_batch(m, T, V, dev, seed=1000 + step)

                    def losses_fn():
                        return per_sequence_losses(forward_logits(model, idx), tgt)

                    if engine == "sgd_erm":
                        loss = losses_fn().mean()
                        loss.backward()
                        losses_seen.append(loss.item())
                    elif engine == "autojac":
                        ls = losses_fn()
                        autojac_backward(ls)
                        jac_to_grad(params, aggregator)
                        losses_seen.append(ls.mean().item())
                    elif engine == "autogram":
                        # One forward, reused: see the note in level11_aggregators.
                        ls = losses_fn()
                        G = ag.compute_gramian(ls)
                        w = weighting(G.to(torch.float32))
                        ls.backward(w.to(ls.dtype))
                        losses_seen.append(ls.mean().item())
                    else:
                        G = jdgram_gramian(model, modules, ov, sh, idx, tgt,
                                           wdtype=wd).total
                        w = weighting(G.to(torch.float32))
                        ls = losses_fn()
                        ls.backward(w.to(ls.dtype))
                        losses_seen.append(ls.mean().item())
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    if step % 25 == 0:
                        print(f"  {engine:<9} step {step:>4}  loss {losses_seen[-1]:.4f}")
            log.many([
                Row(**base, metric="final_loss", value=losses_seen[-1], unit="nats"),
                Row(**base, metric="initial_loss", value=losses_seen[0], unit="nats"),
                Row(**base, metric="mean_last10",
                    value=sum(losses_seen[-10:]) / len(losses_seen[-10:]), unit="nats"),
            ])
            # `base` already carries test="convergence"; splatting it alongside an
            # explicit test= is a duplicate keyword. Third time this pattern has
            # bitten -- strip the key rather than rely on remembering.
            curve_base = {k: v for k, v in base.items() if k != "test"}
            for i, lv in enumerate(losses_seen):
                log.log(Row(**curve_base, test="loss_curve", metric=f"step_{i}",
                            value=lv, unit="nats"))
            print(f"  {engine:<9} final {losses_seen[-1]:.4f}")
        except OOM:
            log.log(Row(**base, metric="final_loss", value=float("nan"), unit="nats",
                        oom=True, note="OOM"))
        except Exception as e:  # noqa: BLE001
            log.log(Row(**base, metric="final_loss", value=float("nan"), unit="nats",
                        note=f"ERR {type(e).__name__}: {e}"[:100]))
            print(f"  {engine:<9} ERROR {type(e).__name__}: {e}")
        finally:
            del model, modules
            clear(dev)


# ================================ L11: aggregator x engine x QP-backend matrix
#: name -> (autojac Aggregator, Gramian Weighting). Only UPGrad consults a
#: dual-cone projector, so it is the only row where the jacopt axis can move.
AGGREGATORS: dict[str, tuple[str, str]] = {
    "Mean": ("Mean", "MeanWeighting"),
    "UPGrad": ("UPGrad", "UPGradWeighting"),
    "MGDA": ("MGDA", "MGDAWeighting"),
    "PCGrad": ("PCGrad", "PCGradWeighting"),
}


def _shakespeare(dev, split: str):
    """Real held-out tokens if prepared, else None (caller falls back to synthetic)."""
    import numpy as np

    path = _ROOT / "data" / "shakespeare_char" / f"{split}.bin"
    if not path.exists():
        return None
    return torch.from_numpy(np.fromfile(path, dtype=np.uint16).astype(np.int64))


def _batch_from(tokens, m, T, V, dev, seed):
    if tokens is None:
        return synthetic_batch(m, T, V, dev, seed)
    g = torch.Generator().manual_seed(seed)
    hi = len(tokens) - T - 1
    ix = torch.randint(0, hi, (m,), generator=g)
    x = torch.stack([tokens[i:i + T] for i in ix]).to(dev)
    y = torch.stack([tokens[i + 1:i + 1 + T] for i in ix]).to(dev)
    return x, y


@torch.no_grad()
def _evaluate(model, tokens, m, T, V, dev, batches=20, seed=999):
    """Held-out cross-entropy and next-token accuracy."""
    ce_sum, correct, count = 0.0, 0, 0
    for b in range(batches):
        idx, tgt = _batch_from(tokens, m, T, V, dev, seed + b)
        logits = forward_logits(model, idx)
        ce = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), reduction="sum"
        )
        ce_sum += float(ce)
        correct += int((logits.argmax(-1) == tgt).sum())
        count += tgt.numel()
    return ce_sum / max(count, 1), correct / max(count, 1)


def level11_aggregators(rc, dev, log: RunLogger, viol: Violations, dtype_name: str,
                        m=8, T=128, V=65, n_embd=256, n_layer=4, steps=200,
                        lr=0.01, eval_batches=20, use_jacopt=None) -> None:
    """Every aggregator against every engine, on identical data and seeds.

    The comparison that matters is not "is jdgram fast" but "for the aggregator you
    actually want to train with, which engine gets you there, how fast, in how much
    memory, and to what held-out loss". So each cell runs a real training loop and
    is scored on accuracy as well as cost.

    Mean/MGDA/PCGrad take no dual-cone projector, so the jacopt axis is only
    exercised on UPGrad; running it elsewhere would just duplicate rows.
    """
    print("\n" + "=" * 78)
    print(f"L11 -- aggregator x engine matrix (m={m} T={T} steps={steps})")
    print("=" * 78)
    if IMPORT_ERRORS:
        print("  SKIPPED:", IMPORT_ERRORS)
        return
    import torchjd.aggregation as tjd_agg

    wd = torch.float32 if dtype_name == "fp32" else torch.float64
    train_tokens = _shakespeare(dev, "train")
    val_tokens = _shakespeare(dev, "val")
    data_tag = "shakespeare" if train_tokens is not None else "synthetic"
    print(f"  data: {data_tag}"
          + ("" if train_tokens is not None
             else "  (run bench/prepare_shakespeare_char.py for real accuracy)"))
    log.log(Row(level="L11", test="env", metric="real_data",
                value=float(train_tokens is not None), unit="bool", note=data_tag))

    qp_backends = {"default": None}
    if use_jacopt is not False:
        try:
            from qp_backends import JacoptProjector, jacopt_available

            if jacopt_available():
                qp_backends["jacopt"] = JacoptProjector(device="keep")
            elif use_jacopt is True:
                print("  requested --jacopt but jacopt is not importable")
        except Exception as e:  # noqa: BLE001
            print("  qp_backends unavailable:", repr(e)[:100])

    cells = []
    for agg_name, (agg_cls, weight_cls) in AGGREGATORS.items():
        for engine in ("jdgram", "autogram", "autojac"):
            for qp_name in qp_backends:
                if qp_name != "default" and (agg_name != "UPGrad" or engine == "autojac"):
                    continue  # projector only reachable through a Gramian UPGrad
                cells.append((agg_name, agg_cls, weight_cls, engine, qp_name))
    cells.append(("none", "", "", "sgd_erm", "default"))
    print(f"  {len(cells)} cells\n")

    for agg_name, agg_cls, weight_cls, engine, qp_name in cells:
        base = dict(level="L11", test="train", engine=engine, driver=agg_name,
                    route=qp_name, dtype=dtype_name, m=m, T=T, V=V,
                    n_embd=n_embd, n_layer=n_layer)
        if log.done(Row(**base, metric="final_train_loss")):
            continue

        torch.manual_seed(42)
        clear(dev)
        model, modules, ov, sh = build_model(dev, n_embd=n_embd, T=T, V=V,
                                             n_layer=n_layer, seed=42)
        opt = torch.optim.SGD(model.parameters(), lr=lr)
        curve: list[float] = []
        try:
            weighting = None
            aggregator = None
            ag_engine = None
            if engine == "autojac" and agg_name != "none":
                aggregator = getattr(tjd_agg, agg_cls)()
                params = list(model.parameters())
            elif agg_name != "none":
                kw = {}
                if agg_name == "UPGrad" and qp_backends[qp_name] is not None:
                    kw["projector"] = qp_backends[qp_name]
                weighting = getattr(tjd_agg, weight_cls)(**kw)
                if engine == "autogram":
                    ag_engine = AutogramEngine(model, batch_dim=0)

            with oom_guard(dev, f"L11/{agg_name}/{engine}/{qp_name}"):
                # warm up before any timing so lazy init is not charged to step 0
                idx, tgt = _batch_from(train_tokens, m, T, V, dev, 0)
                sync(dev)
                clear(dev)
                t0 = time.perf_counter()
                for step in range(steps):
                    idx, tgt = _batch_from(train_tokens, m, T, V, dev, 1000 + step)

                    def losses_fn():
                        return per_sequence_losses(forward_logits(model, idx), tgt)

                    if engine == "sgd_erm":
                        loss = losses_fn().mean()
                        loss.backward()
                        curve.append(float(loss.detach()))
                    elif engine == "autojac":
                        ls = losses_fn()
                        autojac_backward(ls)
                        jac_to_grad(params, aggregator)
                        curve.append(float(ls.mean().detach()))
                    elif engine == "autogram":
                        # ONE forward, reused for the weighted backward. autogram's
                        # per-module remaining_counter is incremented by every
                        # forward but only decremented inside its own backward, so
                        # an extra forward desyncs it permanently: from the next
                        # step on nothing accumulates and compute_gramian returns
                        # None. It retains the graph, so reuse is also correct.
                        ls = losses_fn()
                        G = ag_engine.compute_gramian(ls)
                        w = weighting(G.to(torch.float32))
                        ls.backward(w.to(ls.dtype))
                        curve.append(float(ls.mean().detach()))
                    else:
                        # jdgram runs its own forward inside compute_gramian and
                        # frees that graph as the reverse sweeps (which is where a
                        # large part of its memory win comes from), so the weighted
                        # backward needs a fresh one. That second forward is a real
                        # cost of the design, not a harness artifact -- it is the
                        # price of not retaining activations.
                        G = jdgram_gramian(model, modules, ov, sh, idx, tgt,
                                           wdtype=wd).total
                        w = weighting(G.to(torch.float32))
                        ls = losses_fn()
                        ls.backward(w.to(ls.dtype))
                        curve.append(float(ls.mean().detach()))
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    if step == 0:  # steady state starts after the first step
                        sync(dev)
                        reset_peak(dev)
                        t0 = time.perf_counter()
                sync(dev)
                total_ms = (time.perf_counter() - t0) * 1e3
                peak = peak_allocated(dev) / MiB

            per_step = total_ms / max(steps - 1, 1)
            val_ce, val_acc = _evaluate(model, val_tokens, m, T, V, dev,
                                        batches=eval_batches)
            log.many([
                Row(**base, metric="ms_per_step", value=per_step, unit="ms"),
                Row(**base, metric="peak_mib", value=peak, unit="MiB"),
                Row(**base, metric="total_wall_ms", value=total_ms, unit="ms"),
                Row(**base, metric="final_train_loss", value=curve[-1], unit="nats"),
                Row(**base, metric="mean_last10",
                    value=sum(curve[-10:]) / len(curve[-10:]), unit="nats"),
                Row(**base, metric="val_ce", value=val_ce, unit="nats"),
                Row(**base, metric="val_next_token_acc", value=val_acc, unit="frac"),
            ])
            curve_base = {k: v for k, v in base.items() if k != "test"}
            for i, lv in enumerate(curve):
                log.log(Row(**curve_base, test="train/curve",
                            metric=f"step_{i}", value=lv, unit="nats"))
            print(f"  {agg_name:<7} {engine:<9} {qp_name:<8} "
                  f"{per_step:7.2f} ms/step  peak {peak:8.1f} MiB  "
                  f"train {curve[-1]:.4f}  val_ce {val_ce:.4f}  acc {val_acc:.4f}")
        except OOM:
            log.log(Row(**base, metric="ms_per_step", value=float("nan"),
                        unit="ms", oom=True, note="OOM"))
            print(f"  {agg_name:<7} {engine:<9} {qp_name:<8} {'OOM':>7}")
        except Exception as e:  # noqa: BLE001
            import traceback

            log.log(Row(**base, metric="ms_per_step", value=float("nan"),
                        unit="ms", note=f"ERR {type(e).__name__}: {e}"[:150]))
            print(f"  {agg_name:<7} {engine:<9} {qp_name:<8} "
                  f"ERROR {type(e).__name__}: {e}"[:110])
            # A swallowed traceback turns a code bug into a plausible-looking
            # empty cell. Keep it, and keep it with the run's artifacts.
            tb = traceback.format_exc()
            with open(rc.path("l11_errors.log"), "a") as fh:
                fh.write(f"\n=== {agg_name}/{engine}/{qp_name} ===\n{tb}")
        finally:
            del model, modules
            clear(dev)

    # jdgram and autojac implement the same mathematics; if a cell's curve does not
    # overlay its autojac reference, the engine is wrong, not merely slower.
    viol.check("L11_completed", True, f"{len(cells)} cells attempted", severity="warn")


# ============================================ L10: the QP / dual-cone projector
def level10_qp(rc, dev, log: RunLogger, viol: Violations, dtype_name: str,
               ms=(2, 4, 8, 16, 32, 64, 128), reps=20, threads=None) -> None:
    """Is the dual-cone QP worth moving off the CPU, and at what m?

    UPGrad's weights come from ``min_v v^T G v s.t. u <= v``, one solve per row of
    an ``[m, m]`` matrix. TorchJD's only projector is CPU-only (``G.cpu().numpy()``
    then ``np.apply_along_axis`` then ``quadprog``), so on a CUDA run it puts a
    device->host transfer, a Python loop and a host->device transfer on the
    critical path of every step. jacopt solves the same QP on whatever device the
    arrays live on.

    This measures both across m, on *realistic* Gramians rather than random PSD
    matrices -- conditioning and how often the unconstrained solution is already
    feasible are exactly what decide the cost, and both depend on how much the
    objectives actually conflict. Also counts forced host-device syncs, because at
    these matrix sizes transfers and syncs dominate arithmetic.
    """
    print("\n" + "=" * 78)
    print("L10 -- dual-cone QP backends (TorchJD quadprog vs jacopt)")
    print("=" * 78)
    if "torchjd" in IMPORT_ERRORS:
        print("  SKIPPED:", IMPORT_ERRORS["torchjd"])
        return
    try:
        from qp_backends import available_projectors, count_device_syncs, jacopt_error
    except Exception as e:  # noqa: BLE001
        print("  SKIPPED: cannot import bench/qp_backends.py:", repr(e))
        return

    projectors = available_projectors()
    if not projectors:
        print("  SKIPPED: no projector backends available")
        return
    print(f"  backends: {list(projectors)}")
    # Thread count is a first-class variable for this level, not an environment
    # detail. These are m x m problems; torch parallelises CPU ops above a size
    # threshold, and on a shared box the thread barrier can cost far more than
    # the arithmetic. A cliff in per-iteration cost at a fixed m with the
    # iteration count unchanged is exactly what that looks like.
    default_threads = torch.get_num_threads()
    if threads:
        torch.set_num_threads(threads)
    print(f"  torch CPU threads: {torch.get_num_threads()} "
          f"(default was {default_threads}; set with --qp-threads)")
    log.log(Row(level="L10", test="env", metric="torch_num_threads",
                value=float(torch.get_num_threads()), unit="count"))
    if jacopt_error():
        print(f"  note: jacopt unavailable -> {jacopt_error()}")
        log.log(Row(level="L10", test="env", metric="jacopt_available", value=0.0,
                    unit="bool", note=str(jacopt_error())[:180]))
    else:
        log.log(Row(level="L10", test="env", metric="jacopt_available", value=1.0,
                    unit="bool"))

    for m in ms:
        # Conflicting objectives: the regime where the constraint is ACTIVE and
        # the QP actually has to work. A random PSD Gramian is usually nearly
        # non-conflicting, the unconstrained solve is already feasible, and every
        # backend takes its fast path -- which would flatter all of them equally
        # and measure nothing.
        g = torch.Generator(device="cpu").manual_seed(m)
        J = torch.randn(m, 4 * m, generator=g, dtype=torch.float64)
        J[m // 2:] -= 1.5 * J[: m - m // 2][: m - m // 2].mean(0, keepdim=True)
        G_cpu = (J @ J.T)
        G_cpu = G_cpu / G_cpu.diagonal().mean()
        G = G_cpu.to(dev, torch.float32 if dtype_name == "fp32" else torch.float64)
        U = torch.diag(torch.full((m,), 1.0 / m, device=G.device, dtype=G.dtype))

        results: dict[str, torch.Tensor] = {}
        for label, proj in projectors.items():
            base = dict(level="L10", test="dual_cone", engine=label,
                        dtype=dtype_name, m=m)
            if log.done(Row(**base, metric="ms")):
                continue
            try:
                with oom_guard(dev, f"L10/{label}/m={m}"):
                    proj(U, G)  # warm: first call pays lazy imports and handles
                    sync(dev)
                    _, n_sync = count_device_syncs(lambda: proj(U, G), dev)
                    ms_mean, peak, sd = timeit(lambda: proj(U, G), dev,
                                               warmup=2, iters=reps)
                    results[label] = proj(U, G).detach().double().cpu()
                log.many([
                    Row(**base, metric="ms", value=ms_mean, unit="ms"),
                    Row(**base, metric="ms_std", value=sd, unit="ms"),
                    Row(**base, metric="peak_mib", value=peak, unit="MiB"),
                    # Indicative only -- torch's sync debug mode is a prototype
                    # and misses most implicit syncs, so equal counts across
                    # backends mean "undetected", not "equal".
                    Row(**base, metric="host_device_syncs_indicative",
                        value=float(n_sync), unit="count"),
                ])
                # Iterations and convergence, not just wall clock. A cliff in
                # time with no cliff in problem size is an unconverged solver
                # burning its budget, and only these two numbers can say so.
                st = dict(getattr(proj, "last_stats", {}) or {})
                if st:
                    log.many([
                        Row(**base, metric="admm_iters",
                            value=float(st.get("admm_iters", float("nan"))),
                            unit="count"),
                        Row(**base, metric="converged",
                            value=float(bool(st.get("converged", False))),
                            unit="bool"),
                    ])
                    viol.check(
                        f"L10_converged@{label}/m={m}",
                        bool(st.get("converged", False)),
                        f"iters={st.get('admm_iters')} / {st.get('max_iter')}",
                        severity="warn")
                    it = st.get("admm_iters") or 0
                    if it:
                        # ms per ADMM iteration. Constant-ish in m means the cost
                        # is arithmetic; a jump at fixed iteration count means it
                        # is dispatch, threading or contention.
                        log.log(Row(**base, metric="ms_per_admm_iter",
                                    value=ms_mean / it, unit="ms"))
                extra = (f"  iters={st.get('admm_iters')}"
                         f" converged={st.get('converged')}" if st else "")
                print(f"  m={m:<4} {label:<20} {ms_mean:8.3f} +- {sd:6.3f} ms   "
                      f"syncs~{n_sync if n_sync >= 0 else 'n/a'}{extra}")
            except OOM:
                log.log(Row(**base, metric="ms", value=float("nan"), unit="ms",
                            oom=True, note="OOM"))
            except Exception as e:  # noqa: BLE001
                log.log(Row(**base, metric="ms", value=float("nan"), unit="ms",
                            note=f"ERR {type(e).__name__}: {e}"[:150]))
                print(f"  m={m:<4} {label:<20} ERROR {type(e).__name__}: {e}"[:110])

        # Same QP, different solvers: the weights must agree or the speed
        # comparison is meaningless.
        if "torchjd_quadprog" in results:
            ref = results["torchjd_quadprog"]
            for label, W in results.items():
                if label == "torchjd_quadprog":
                    continue
                d = (W - ref).abs().max().item()
                rel = d / max(ref.abs().max().item(), 1e-30)
                log.log(Row(level="L10", test="dual_cone", engine=label,
                            dtype=dtype_name, m=m, metric="rel_diff_vs_quadprog",
                            value=rel, unit="rel"))
                viol.check(f"L10_agrees_with_quadprog@{label}/m={m}", rel < 1e-3,
                           f"rel={rel:.3e}", severity="warn")

        del results
        clear(dev)


# ================================================== L8: CUDA memory snapshot
def level8_snapshot(rc, dev, args) -> None:
    """Full allocation history. Large -- guarded, and off by default."""
    print("\n" + "=" * 78)
    print("L8 -- CUDA memory snapshot")
    print("=" * 78)
    if dev.type != "cuda":
        print("  SKIPPED (CPU)")
        return
    if IMPORT_ERRORS:
        print("  SKIPPED:", IMPORT_ERRORS)
        return
    from runtag import free_gib

    if free_gib(rc.out_dir) < 5.0:
        print(f"  SKIPPED: only {free_gib(rc.out_dir):.1f} GiB free; a snapshot can be "
              f"hundreds of MB. Re-run with --out-root on a roomier filesystem.")
        return
    clear(dev)
    torch.cuda.memory._record_memory_history(max_entries=args.snapshot_entries)
    model, modules, ov, sh = build_model(dev, n_embd=args.n_embd, T=args.T, V=args.V,
                                         n_layer=args.n_layer)
    idx, tgt = synthetic_batch(args.m, args.T, args.V, dev)
    try:
        jdgram_gramian(model, modules, ov, sh, idx, tgt, driver=args.driver)
    except Exception as e:  # noqa: BLE001
        print("  step raised:", repr(e)[:140])
    path = rc.path("memory_snapshot.pickle")
    torch.cuda.memory._dump_snapshot(str(path))
    torch.cuda.memory._record_memory_history(enabled=None)
    print(f"  wrote {path} ({path.stat().st_size / MiB:.1f} MiB)")
    print("  -> drag into https://pytorch.org/memory_viz, or run bench/profile_stats.py")
    del model, modules
    clear(dev)


# ================================================ L9: torch.profiler capture
def level9_trace(rc, dev, args) -> None:
    """Capture an op/kernel trace for bench/profile_stats.py.

    Always writes the compact per-op aggregate; the raw chrome trace is opt-in
    because it is large and the cluster filesystem is tight.
    """
    print("\n" + "=" * 78)
    print("L9 -- torch.profiler capture")
    print("=" * 78)
    if IMPORT_ERRORS:
        print("  SKIPPED:", IMPORT_ERRORS)
        return
    from torch.profiler import ProfilerActivity, profile, schedule

    wd = torch.float32 if args.dtype == "fp32" else torch.float64
    model, modules, ov, sh = build_model(dev, n_embd=args.n_embd, T=args.T, V=args.V,
                                         n_layer=args.n_layer)
    idx, tgt = synthetic_batch(args.m, args.T, args.V, dev)
    activities = [ProfilerActivity.CPU]
    if dev.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    # Wait/warmup/active so allocator and autotuner noise stays out of the trace.
    sched = schedule(wait=1, warmup=2, active=3, repeat=1)
    with profile(activities=activities, schedule=sched, record_shapes=True,
                 profile_memory=True, with_stack=args.with_stack) as prof:
        for _ in range(6):
            try:
                jdgram_gramian(model, modules, ov, sh, idx, tgt, driver=args.driver,
                               force_route=args.force_route, wdtype=wd)
            except Exception as e:  # noqa: BLE001
                print("  step raised:", repr(e)[:140])
                break
            prof.step()

    # Compact aggregate -- this is what the stats script prefers.
    rows = []
    for evt in prof.key_averages(group_by_input_shape=args.record_shapes):
        rows.append({
            "name": evt.key,
            "count": evt.count,
            "cpu_time_total_us": evt.cpu_time_total,
            "self_cpu_time_total_us": evt.self_cpu_time_total,
            "cuda_time_total_us": getattr(evt, "device_time_total", 0) or 0,
            "self_cuda_time_total_us": getattr(evt, "self_device_time_total", 0) or 0,
            "cpu_memory_usage": evt.cpu_memory_usage,
            "cuda_memory_usage": getattr(evt, "device_memory_usage", 0) or 0,
            "self_cuda_memory_usage": getattr(evt, "self_device_memory_usage", 0) or 0,
            "input_shapes": str(evt.input_shapes),
            "device_type": str(getattr(evt, "device_type", "")),
        })
    agg = rc.path("profiler_ops.csv")
    with open(agg, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["name"])
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {agg}  ({len(rows)} op groups)")

    if args.trace_raw:
        raw = rc.path("profiler_trace.json")
        prof.export_chrome_trace(str(raw))
        size = raw.stat().st_size / MiB
        print(f"  wrote {raw} ({size:.1f} MiB)")
        if size > 50:
            import gzip
            import shutil as _sh

            with open(raw, "rb") as fi, gzip.open(str(raw) + ".gz", "wb") as fo:
                _sh.copyfileobj(fi, fo)
            raw.unlink()
            print(f"  gzipped to {raw}.gz (raw trace removed to save disk)")
    del model, modules
    clear(dev)


# ============================================================ isolation runner
def run_isolated(args, levels: list[str]) -> int:
    """Re-exec one level per subprocess.

    Allocator state is process-global and sticky: a config that touched 13 GiB
    leaves a fragmented pool that inflates every later peak in the same process.
    That is the documented cause of the untrustworthy OOM-boundary numbers in the
    previous report, and a fresh process is the only clean fix.
    """
    rc_root = args.out_root
    failures = 0
    for lvl in levels:
        # Forward every shape/budget flag. Dropping one here does not fail, it
        # silently re-runs the child at the DEFAULT -- so an isolated 124M
        # campaign would quietly profile the 256-wide model and label the rows
        # with the config that was asked for.
        cmd = [sys.executable, str(Path(__file__).resolve()),
               "--version", str(args.version), "--name", f"{args.name}-{lvl}",
               "--device", args.device, "--dtype", args.dtype,
               "--levels", lvl, "--out-root", str(rc_root),
               "--m", str(args.m), "--T", str(args.T), "--V", str(args.V),
               "--n-embd", str(args.n_embd), "--n-layer", str(args.n_layer),
               "--steps", str(args.steps), "--eval-batches", str(args.eval_batches),
               "--qp-reps", str(args.qp_reps),
               "--max-alloc-gib", str(args.max_alloc_gib),
               "--notes", args.notes or f"isolated {lvl}"]
        if args.n_head is not None:
            cmd += ["--n-head", str(args.n_head)]
        if args.driver is not None:
            cmd += ["--driver", args.driver]
        if args.force_route is not None:
            cmd += ["--force-route", args.force_route]
        if args.qp_threads is not None:
            cmd += ["--qp-threads", str(args.qp_threads)]
        if args.jacopt is not None:
            cmd += ["--jacopt" if args.jacopt else "--no-jacopt"]
        if args.no_resume:
            cmd += ["--no-resume"]
        print(f"\n[isolate] {' '.join(cmd)}")
        rc = subprocess.call(cmd, cwd=str(_ROOT))
        if rc != 0:
            failures += 1
            print(f"[isolate] {lvl} exited {rc}; continuing")
    return failures


# ========================================================================= main
ALL_LEVELS = ["L0", "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9", "L10", "L11"]


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--version", type=int, required=True,
                   help="bump by hand when engine semantics change")
    p.add_argument("--name", required=True, help="short run name, e.g. squashed-baseline")
    p.add_argument("--notes", default="")
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="fp32", choices=["fp32", "fp64"])
    p.add_argument("--levels", nargs="+", default=["L0", "L2", "L4", "L5"],
                   help=f"any of {ALL_LEVELS}, or 'all'")
    p.add_argument("--out-root", default=os.environ.get("JDGRAM_RESULTS", "results"))
    p.add_argument("--isolate", action="store_true",
                   help="run each level in a fresh subprocess (avoids allocator carry-over)")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--m", type=int, default=8)
    p.add_argument("--T", type=int, default=512)
    p.add_argument("--V", type=int, default=65)
    p.add_argument("--n-embd", type=int, default=256)
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--n-head", type=int, default=None,
                   help="attention heads; default derives n_embd//64, which is "
                        "GPT-2's ratio at every size (768/12, 1024/16)")
    p.add_argument("--driver", default=None, choices=[None, *DRIVERS])
    p.add_argument("--force-route", default=None, choices=[None, "tfirst", "dfirst"])
    p.add_argument("--steps", type=int, default=100, help="L7 training steps")
    p.add_argument("--with-stack", action="store_true",
                   help="L9: record python stacks (slower, much larger trace)")
    p.add_argument("--record-shapes", action="store_true",
                   help="L9: group op aggregates by input shape")
    p.add_argument("--trace-raw", action="store_true",
                   help="L9: also export the raw chrome trace")
    p.add_argument("--snapshot-entries", type=int, default=100_000)
    p.add_argument("--eval-batches", type=int, default=20,
                   help="L11: held-out batches for val CE / accuracy")
    p.add_argument("--jacopt", dest="jacopt", action="store_true", default=None,
                   help="L11: require the jacopt projector arm")
    p.add_argument("--no-jacopt", dest="jacopt", action="store_false",
                   help="L11: skip the jacopt projector arm")
    p.add_argument("--qp-threads", type=int, default=None,
                   help="L10: pin torch CPU threads (isolates thread-contention "
                        "effects on a shared box)")
    p.add_argument("--qp-reps", type=int, default=20,
                   help="L10: timed iterations per QP backend")
    p.add_argument("--max-alloc-gib", type=float, default=8.0,
                   help="L0: skip shapes whose estimated high-water mark exceeds this")
    args = p.parse_args()

    levels = ALL_LEVELS if "all" in args.levels else list(dict.fromkeys(args.levels))
    unknown = [l for l in levels if l not in ALL_LEVELS]
    if unknown:
        p.error(f"unknown levels {unknown}; choose from {ALL_LEVELS} or 'all'")

    # Eight level functions call build_model without a head count, so an explicit
    # --n-head is applied by moving the ratio it implies rather than threading the
    # count through all of them. Set once, before any model is built.
    if args.n_head is not None:
        if args.n_embd % args.n_head:
            p.error(f"--n-embd {args.n_embd} not divisible by --n-head {args.n_head}")
        global HEAD_DIM
        HEAD_DIM = args.n_embd // args.n_head

    # L2 and L4 sweep every driver and route by default, which is what you want on
    # a 3M-parameter model and ruinous on a 124M one: `batched` needs several times
    # `squashed`'s memory and `loop` needs m backward passes, so a full 3x3 sweep at
    # GPT-2 scale spends hours OOM-ing on the two drivers we no longer ship. An
    # explicit --driver / --force-route now narrows the sweep instead of being
    # silently ignored by these two levels.
    sweep_drivers = (args.driver,) if args.driver else DRIVERS
    sweep_routes = (args.force_route,) if args.force_route else ROUTES

    if args.isolate:
        sys.exit(run_isolated(args, levels))

    dev = pick_device(args.device)
    if dev.type != "cuda":
        print("[warn] running on CPU: every memory column reads 0.0 "
              "(torch.cuda.memory_allocated is CUDA-only). Timings remain valid.")

    rc = RunContext(version=args.version, name=args.name, notes=args.notes,
                    root=Path(args.out_root))
    rc.save_manifest(vars(args) | {"import_errors": IMPORT_ERRORS,
                                   "levels_requested": levels})
    if IMPORT_ERRORS:
        print(f"\n[warn] IMPORTS FAILED -- affected levels will report SKIPPED, not zero:\n"
              f"       {json.dumps(IMPORT_ERRORS, indent=8)}")

    log = RunLogger(rc.path("rows.csv"), resume=not args.no_resume)
    viol = Violations()
    status = "ok"
    try:
        if "L0" in levels:
            level0_identities(rc, dev, log, viol, args.dtype,
                              max_alloc_gib=args.max_alloc_gib)
        if "L1" in levels:
            level1_hook_overhead(rc, dev, log, viol, args.dtype, m=args.m, T=args.T,
                                 V=args.V, n_embd=args.n_embd, n_layer=args.n_layer)
        if "L2" in levels:
            for tie in (True, False):
                level2_drivers(rc, dev, log, viol, args.dtype, m=args.m, T=args.T,
                               V=args.V, n_embd=args.n_embd, n_layer=args.n_layer,
                               tie=tie, drivers=sweep_drivers)
        if "L3" in levels:
            level3_accumulate(rc, dev, log, viol, args.dtype, m=args.m, T=args.T,
                              V=args.V, n_embd=args.n_embd, n_layer=args.n_layer)
        if "L4" in levels:
            for driver in sweep_drivers:
                for froute in sweep_routes:
                    level4_phases(rc, dev, log, viol, args.dtype, m=args.m, T=args.T,
                                  V=args.V, n_embd=args.n_embd, n_layer=args.n_layer,
                                  driver=driver, force_route=froute)
        if "L5" in levels:
            level5_ab(rc, dev, log, viol, args.dtype, m=args.m, T=args.T, V=args.V,
                      n_embd=args.n_embd, n_layer=args.n_layer)
        if "L6" in levels:
            level6_scaling(rc, dev, log, viol, args.dtype, V=args.V,
                           n_embd=args.n_embd, n_layer=args.n_layer)
        if "L7" in levels:
            level7_accuracy(rc, dev, log, viol, args.dtype, m=args.m, T=min(args.T, 128),
                            V=args.V, n_embd=args.n_embd, n_layer=args.n_layer,
                            steps=args.steps)
        if "L8" in levels:
            level8_snapshot(rc, dev, args)
        if "L9" in levels:
            level9_trace(rc, dev, args)
        if "L10" in levels:
            level10_qp(rc, dev, log, viol, args.dtype, reps=args.qp_reps,
                       threads=args.qp_threads)
        if "L11" in levels:
            level11_aggregators(rc, dev, log, viol, args.dtype, m=args.m,
                                T=min(args.T, 256), V=args.V, n_embd=args.n_embd,
                                n_layer=args.n_layer, steps=args.steps,
                                eval_batches=args.eval_batches,
                                use_jacopt=args.jacopt)
    except KeyboardInterrupt:
        status = "interrupted"
        print("\n[interrupted] rows written so far are already on disk")
    except Exception as e:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        status = f"error: {type(e).__name__}"
    finally:
        log.close()

    rc.path("violations.json").write_text(json.dumps(viol.items, indent=2))
    print("\n" + "=" * 78)
    print("VIOLATIONS:", viol.summary())
    print(f"rows: {log.n} new -> {rc.path('rows.csv')}")
    print("=" * 78)
    print(f"\nnext: python bench/profile_stats.py {rc.out_dir}")
    rc.finalize(status=status,
                summary={"violations": viol.summary(), "rows": log.n, "levels": levels})


if __name__ == "__main__":
    main()
