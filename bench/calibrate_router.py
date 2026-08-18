"""Measure both Gramian routes on this card and fit a cost model.

The router's shipped rule compares *workspace* and is blind to time. At
vocabulary scale that costs up to 13x on a single layer and up to 63% of a
whole training step, in exchange for about 1% of peak memory. Replacing it
needs coefficients that are hardware-dependent, so they are measured here
rather than derived.

    python bench/calibrate_router.py --out costmodel_a5000_fp32.json

Then, in a run:

    from jdgram import costmodel
    costmodel.load_and_use("costmodel_a5000_fp32.json")

Nothing changes until that call is made -- with no model installed the router
keeps its analytic rule exactly as before.

WHAT IS TIMED
-------------
The two identity kernels in isolation: no model, no hooks, no autograd, so
nothing but the route choice is in the frame. The default shape grid covers
GPT-2 124M's own layers (the four block linears and the vocabulary head) plus
LoRA adapter shapes, each swept over the objective counts a real ladder uses.
Both routes are exactly equivalent numerically, so a mis-fit can only cost
time, never change a result -- and ``--verify`` re-checks that equivalence on
every shape it times.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jdgram.costmodel import CostModel, Sample  # noqa: E402
from jdgram.engine.materialize import materialized_gramian  # noqa: E402
from jdgram.identities.linear import sequence_gramian  # noqa: E402

# (d_out, d_in, label). GPT-2 124M's real layers first, then adapter shapes
# that a fine-tuning use case would hit.
SHAPES = [
    (2304, 768, "attn.c_attn"),
    (768, 768, "attn.c_proj"),
    (3072, 768, "mlp.c_fc"),
    (768, 3072, "mlp.c_proj"),
    (50257, 768, "lm_head"),
    (32, 2048, "lora-A-r32"),
    (2048, 32, "lora-B-r32"),
]
MS = [1, 2, 3, 4, 6, 8, 12, 16]


def bench(fn, *a, reps=5, warmup=2, **kw) -> float:
    for _ in range(warmup):
        fn(*a, **kw)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn(*a, **kw)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3 / reps


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--T", type=int, default=512)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--max-alloc-gib", type=float, default=18.0)
    ap.add_argument("--verify", action="store_true",
                    help="assert the two routes agree numerically on every "
                         "shape timed (they must; this is the safety net)")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu"
    print(f"calibrating on {name}, T={args.T}, fp32")
    print(f"{'layer':<14}{'m':>4}{'tfirst ms':>12}{'dfirst ms':>12}"
          f"{'faster':>9}{'ratio':>8}")

    samples, skipped, worst_disagreement = [], 0, 0.0
    for d_out, d_in, label in SHAPES:
        for m in MS:
            # Two live [m,T,d] inputs plus the larger workspace, fp32.
            need = (m * args.T * (d_out + d_in)
                    + max(3.0 * m * args.T * args.T, float(m) * d_out * d_in))
            if need * 4 / 2**30 > args.max_alloc_gib:
                skipped += 1
                continue
            A = torch.randn(m, args.T, d_out, device=dev)
            X = torch.randn(m, args.T, d_in, device=dev)
            try:
                t_ms = bench(sequence_gramian, A, X, False, reps=args.reps)
                d_ms = bench(materialized_gramian, A, X, False, reps=args.reps)
                if args.verify:
                    g_t = sequence_gramian(A, X, False)
                    g_d = materialized_gramian(A, X, False)
                    rel = float((g_t - g_d).abs().max()
                                / g_t.abs().max().clamp_min(1e-30))
                    worst_disagreement = max(worst_disagreement, rel)
            except torch.cuda.OutOfMemoryError:
                skipped += 1
                del A, X
                torch.cuda.empty_cache()
                continue
            del A, X
            torch.cuda.empty_cache()
            faster = "tfirst" if t_ms < d_ms else "dfirst"
            ratio = max(t_ms, d_ms) / max(min(t_ms, d_ms), 1e-9)
            print(f"{label:<14}{m:>4}{t_ms:>12.3f}{d_ms:>12.3f}"
                  f"{faster:>9}{ratio:>8.2f}")
            samples.append(Sample(m, args.T, d_out, d_in, t_ms, d_ms))

    if not samples:
        print("no shapes fit in --max-alloc-gib; nothing to fit")
        return 1

    model = CostModel.fit(samples, device=name, dtype="fp32")
    model.save(args.out)

    print(f"\nfitted on {len(samples)} shapes ({skipped} skipped for memory)")
    print(f"  tfirst  {model.a_gemm:.3e} * gemm_flops "
          f"+ {model.a_elem:.3e} * elementwise")
    print(f"  dfirst  {model.b_build:.3e} * build_flops "
          f"+ {model.b_gram:.3e} * gram_flops")
    if args.verify:
        print(f"  routes agree numerically to {worst_disagreement:.3e} "
              f"relative (they are the same maths; this must be ~0)")

    # How often would the model have picked the loser on its own training
    # data? Not a held-out score, but it catches a fit that is outright broken.
    wrong = big = 0
    for s in samples:
        pick = model.choose(s.m, s.T, s.d_out, s.d_in)
        chosen = s.tfirst_ms if pick == "tfirst" else s.dfirst_ms
        other = s.dfirst_ms if pick == "tfirst" else s.tfirst_ms
        if chosen > other * 1.05:
            wrong += 1
            if chosen > other * 1.5:
                big += 1
    print(f"  picks the slower route on {wrong}/{len(samples)} measured "
          f"shapes ({big} of them by more than 1.5x)")

    # The same scoring for the rule being replaced, so the comparison is on
    # the record rather than asserted.
    old_wrong = old_big = 0
    for s in samples:
        pick = "tfirst" if s.m * s.T * s.T < s.d_out * s.d_in else "dfirst"
        chosen = s.tfirst_ms if pick == "tfirst" else s.dfirst_ms
        other = s.dfirst_ms if pick == "tfirst" else s.tfirst_ms
        if chosen > other * 1.05:
            old_wrong += 1
            if chosen > other * 1.5:
                old_big += 1
    print(f"  the analytic rule it replaces: {old_wrong}/{len(samples)} "
          f"({old_big} by more than 1.5x)")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
