# L4 vs L11 — why the two levels disagree about the Gramian's cost

**Author:** Arsenii
**Date:** 2026-08-11
**Status:** diagnosis complete; one harness bug fixed, the discrepancy itself is *not* a
harness bug

---

## 0. Summary for someone with two minutes

The reported finding was that L4's phase decomposition and L11's real training loop
disagree about the marginal cost of the Gramian by a factor that reproduces to three
significant figures (1.496 at m=1, 1.493 at m=2), and that a multiplier that stable
must be systematic.

**It is not systematic. It is a two-point coincidence.** The same campaign contains a
third configuration, m=3, measured in the same process the same way, and there the
same ratio-of-ratios is **0.910** — the disagreement changes *sign*. And the same
config measured twice, 19 minutes apart on the same box, moves L4's `compute_gramian`
by −28% and `final_backward` by +53% while every peak-memory reading stays
**byte-identical**. Deterministic memory with non-deterministic time is the signature
of interference from outside the process, not of a measurement methodology.

Every mechanism proposed as the cause was tested and each is worth ≤11%, not 49%:
sync serialisation 0.3%, cold allocator ≲1 ms, holding the previous result 4.7%. The
two methodologies, run against each other on a quiet box in one process, agree on the
Gramian's marginal cost to within 8%.

One genuine defect was found along the way and fixed. It is unrelated to the
discrepancy and changes the headline ratios by ~2%.

**Practical consequence:** the L4-derived floor and fusion numbers should **not** be
discounted by 1.5×. They should carry a **±30% uncertainty band** and the ~2%
optimiser correction in §6. The "architecture is capped" conclusion survives; the
three-significant-figure precision it was stated with does not.

---

## 1. The numbers, and exactly where they come from

All 124M numbers are from `results/solo/`, GPT-2 124M (n_layer 12, n_head 12, n_embd
768, V 50257), RTX A5000, fp32, T=512, `objective_mode=duplicate`,
`force_route=dfirst`, `driver=squashed`, one process per m containing L4, L5 and L11.

| run dir | m |
|---|---|
| `v10_20260811-021816_a-dup-m1-dfirst-r1_v10-20260811-solo` | 1 |
| `v10_20260811-022446_a-dup-m2-dfirst-r1_v10-20260811-solo` | 2 |
| `v10_20260811-023332_a-dup-m3-dfirst-r1_v10-20260811-solo` | 3 (run A, truncated) |
| `v10_20260811-025217_a-dup-m3-dfirst-r1_v10-20260811-solo` | 3 (run B, complete) |

L4 rows are `level=L4, metric=ms, test=phases/<phase>`; L11 rows are
`level=L11, metric=ms_per_step`.

### 1.1 The discrepancy as originally stated

L4's baseline is `final_backward + optimizer_step`; its full step is
`compute_gramian + weighting_qp + final_backward + optimizer_step`. L11's ratio is
`UPGrad/jdgram` over `none/sgd_erm`.

| run | L4 base | L4 full | L4 ratio | L11 sgd_erm | L11 jdgram | L11 ratio | L4/L11 |
|---|---|---|---|---|---|---|---|
| m=1 | 81.273 | 181.388 | 2.232 | 91.541 | 136.605 | 1.492 | **1.496** |
| m=2 | 145.843 | 417.778 | 2.865 | 155.489 | 298.318 | 1.919 | **1.493** |
| m=3 (B) | 208.695 | 461.447 | 2.211 | 236.893 | 575.882 | 2.431 | **0.910** |

The first two rows are the ones the investigation was opened on. **The third row was
available in the same campaign and was not consulted.** It disagrees by 39% and in the
opposite direction.

### 1.2 The same quantity, stated as a marginal cost

| run | L4 `compute_gramian + weighting_qp` | L11 `jdgram − sgd_erm` | L4/L11 |
|---|---|---|---|
| m=1 | 100.115 | 45.064 | 2.222 |
| m=2 | 271.935 | 142.829 | 1.904 |
| m=3 (B) | 252.753 | 338.989 | **0.746** |

At m=3 L4 *understates* the Gramian by 25%. A systematic methodological bias does not
change sign.

---

## 2. What was ruled out, and with what evidence

### 2.1 "L4 syncs around every phase and serialises CPU/GPU overlap that L11 gets free"

**Ruled out by reading, then confirmed by measurement.**

`PhaseTracker.phase()` (`bench/profile_suite.py:274`) syncs once at entry and once at
exit — and the phase body in `level4_phases` is a `for _ in range(reps)` loop with
`reps=10`. So L4 pays **two syncs per ten iterations**, and the ten iterations inside
pipeline exactly as L11's steps would. The hypothesis assumed one sync per iteration.

Measured directly on a tiny GPU config (m=1, T=128, V=1024, n_embd=128, n_layer=2,
10 reps, RTX 5070 Ti Laptop, torch 2.10.0.dev):

```
[H4] 1 sync per 10 reps (PhaseTracker) :    8.855 ms/rep
[H4] 1 sync per rep                     :    8.827 ms/rep
```

0.3%. Even the *maximal* version of this effect is not in play.

The converse also fails: L11's loop does not pipeline either. Each step calls
`float(loss.detach())` and `ls.detach().tolist()` (`profile_suite.py:1616-1617`), both
device→host reads, and `_batch_from` issues a pageable H2D copy at the top of the next
step. L11 is sync-bounded per step, L4 per ten reps — if anything L4 has *more* overlap
available, not less.

### 2.2 "L4's compute_gramian runs 10 back-to-back Gramians, so allocator state differs"

**Bounded at ≲1 ms at 124M using data already on disk.**

L4 calls `clear(dev)` — which is `gc.collect(); empty_cache(); reset_peak()` — between
its warm-up step and the first timed phase (`profile_suite.py:888`). That returns every
cached block to the driver, so the timed phases pay a one-time `cudaMalloc` bill inside
the timed region. L11 does its `clear(dev)` *before* step 0 and step 0 is excluded from
timing, so L11's allocator is fully warm for every timed step. This is a real asymmetry
and it does bias L4 upward.

How much? L5 in the *same processes* measures jdgram's Gramian with a completely
different structure: `once(); clear(dev); sync; t0; G = once(); sync` — a **single**
timed call, so it charges the *entire* post-`empty_cache` allocation bill to one
iteration, where L4 amortises it over ten. If that cost were large, L5 would read far
above L4. At m=1 it does not:

| source | method | m=1 Gramian, ms |
|---|---|---|
| L4 `compute_gramian` | 10 reps, cold allocator, cost/10 | 95.208 |
| L5 `jdgram ab` (tie=False) | 1 rep, cold allocator, full cost | 94.581 |
| L5 `jdgram ab` (tie=True) | 1 rep, cold allocator, full cost | 93.128 |

Solving `g + C = 94.58` against `g + C/10 = 95.21` gives `C ≈ −0.7 ms`, i.e. zero
within noise. **The cold-allocator penalty at 124M is under a millisecond.**

Independently, on the tiny config, running the identical phase twice — once after
`clear()`, once with the allocator already warm — inflates the reported mean by
**1.108×**, and the first rep is only 1.4× the rest against a per-rep scatter that is
itself ±40%.

*Caveat on the L5 comparison:* L5 runs `force_route=None` (router picks per layer)
while L4 pins `dfirst`. At m=1 the two land within 2% of each other, which is why the
bound holds. At m=2 they diverge (L5 189–200 ms vs L4 267 ms) exactly as expected —
`m*T² = 524288` against the 38.6M-parameter head makes the router choose `tfirst`,
which `--force-route dfirst` disables. **Do not read L5 and L4 as the same measurement
above m=1.**

### 2.3 "L4 holds the previous GramianResult alive"

**Real, worth 4.7%, not 122%.**

L4 writes `result = jdgram_gramian(...)` inside its loop, so iteration k's workspaces
are allocated while iteration k−1's `GramianResult` is still referenced; L11 writes
`jdgram_gramian(...).total` and drops everything but the `[m, m]` matrix. But
`GramianResult` carries `total`, `per_module` and `per_shared_group` — all `[m, m]`
tensors (`src/jdgram/engine/hooks.py:139`), which is bytes, not MiB. Measured on the
tiny config:

```
[H5] result=f(...)  (L4, previous result alive):    8.874 ms/rep
[H5] G=f(...).total (L11)                      :    8.479 ms/rep
```

### 2.4 "Different warm-up semantics"

**Checked; both warm up with one real step.** L4 runs a full
gramian→weight→backward→step before any phase (`profile_suite.py:877-888`); L11 runs
step 0 and resets `t0` after it (`profile_suite.py:1620-1623`). The only difference is
L4's `clear()` after its warm-up, covered in §2.2.

### 2.5 "L4 uses synthetic tokens, L11 uses the real corpus"

**Confirmed irrelevant to GPU cost, and it biases the wrong way.** `_batch_from`
(`profile_suite.py:1318`) slices a CPU tensor and copies `m*T` int64 to the device —
4 KB at m=1. It is charged to L11 only, and to *both* of L11's cells equally, so it
cancels in L11's ratio and inflates L11's absolute numbers slightly. It cannot make
L11's jdgram cell *cheaper*.

### 2.6 "The two levels call different code"

**Checked, they do not.** L4 passes `driver="squashed"`; L11 passes no `driver`, and
`_resolve_driver` defaults to `"squashed"` (`src/jdgram/engine/hooks.py:492`). Both
pass `force_route="dfirst"` and `wdtype=fp32`, both build the same tied model through
`build_model` with `n_head` derived the same way. The only differences are the model
seed (0 vs 42) and the token source, neither of which changes cost.

### 2.7 The positive control: reproduce both methodologies on a quiet box

Both methodologies were re-implemented verbatim in one process and run against each
other on a config whose vocabulary head dominates the `dfirst` materialisation, the way
it does at 124M (m=1, T=256, V=16384, n_embd=256, n_layer=2, reps=10, steps=30, three
repeats each):

```
  L4 run0: cg 9.197  qp 0.396  fb 3.118  os 0.021   as-shipped ratio 4.056   gramian+qp 9.593
  L4 run1: cg 8.632  qp 0.705  fb 3.410  os 0.018   as-shipped ratio 3.724   gramian+qp 9.337
  L4 run2: cg 8.090  qp 0.416  fb 2.507  os 0.017   as-shipped ratio 4.370   gramian+qp 8.505

  L11 run0: sgd_erm 3.956   Mean 12.994  UPGrad 13.683  PCGrad 12.684   ratio 3.459  marginal  9.727
  L11 run1: sgd_erm 3.908   Mean 12.012  UPGrad 13.777  PCGrad 13.139   ratio 3.525  marginal  9.869
  L11 run2: sgd_erm 4.238   Mean 13.481  UPGrad 14.405  PCGrad 14.202   ratio 3.399  marginal 10.167
```

L4's Gramian: 9.15 ± 0.55 ms. L11's marginal: 9.92 ± 0.23 ms. **They agree to 8%, with
L4 reading *lower*.** The ratio-of-ratios is 1.17 as shipped and 1.08 once the
optimiser is charged honestly (§3) — nowhere near 1.49.

The mechanisms the two levels do not share are therefore worth ~10% in total. The 124M
gap is something else.

---

## 3. The one real bug found (fixed; it does not explain the discrepancy)

`level4_phases` timed the optimiser on parameters whose gradients had just been
discarded:

```python
with tr.phase("final_backward", ...):
    for _ in range(reps):
        losses2 = losses_fn()
        losses2.backward(w.to(losses2.dtype))
        opt.zero_grad(set_to_none=True)     # <- leaves every p.grad = None
with tr.phase("optimizer_step", f"x{reps}"):
    for _ in range(reps):
        opt.step()                          # <- SGD skips params with grad None
```

`torch.optim.SGD` skips any parameter whose `.grad` is `None`, so the phase measured an
empty walk over the parameter list. Every campaign row confirms it: `optimizer_step`
reads **0.0405, 0.0424, 0.0405, 0.0408 ms** across the four 124M runs — flat, and
independent of m, T and the model.

Measured directly on the real GPT-2 124M parameter set (148 tensors, 473.2 MiB), no
forward or backward involved:

```
opt.step() with real grads      :   2.8505 ms
opt.step() after set_to_none    :   0.0181 ms   <-- L4's phase
missing from L4's baseline      :   2.8323 ms
traffic 3*0.462 GiB -> 522 GB/s effective
```

2.83 ms is the memory-bandwidth cost of `p -= lr*g` over 124M fp32 parameters — read
`p`, read `g`, write `p`. On the A5000 (768 GB/s nominal against the 522 GB/s measured
here) the true figure is ~2.0–2.9 ms.

**Why it matters beyond L4:** `bench/acceptance.py::floor_from_l4` builds
`base = final_backward + optimizer_step` (line 157) and derives every floor, fusion and
headroom ratio from it. A baseline missing its optimiser inflates every ratio.

**Fix applied** (`bench/profile_suite.py`, `level4_phases`): one untimed
forward+backward is run between the two phases so the optimiser has a gradient to
consume. `final_backward`'s semantics are untouched deliberately — moving the
`zero_grad` instead would have quietly changed what that phase measures, and the
campaign's existing `final_backward` rows must stay comparable. Verified at tiny scale
through the real CLI: `optimizer_step` now reads 0.091 ms against a directly measured
0.094 ms for the same optimiser, where it previously read 0.017.

**Effect on the headline numbers (using op = 2.83 ms):**

| run | ratio now | ratio fixed | floor now | floor fixed | fused floor now | fixed |
|---|---|---|---|---|---|---|
| m=1 | 2.232 | 2.191 | 1.382 | 1.370 | 1.060 | 1.058 |
| m=2 | 2.865 | 2.830 | 1.382 | 1.375 | 1.034 | 1.033 |
| m=3 (B) | 2.211 | 2.195 | 1.389 | 1.383 | 1.024 | 1.023 |

~2% on the ratio, ~1% on the floor. **The "the whole structural floor IS the redundant
forward" conclusion is unaffected.** This is a correctness fix, not an explanation.

---

## 4. What the cause actually is

**Time-varying interference on the measurement host, amplified by taking a difference
of two large, separately-measured numbers. Confidence: high (~85%) on "the
disagreement is measurement noise, not methodology"; moderate (~60%) on "external
contention" as the specific noise source, as opposed to clock/power management.**

### 4.1 The same config, measured twice, moves by half

Runs A and B of m=3 are the same command, 19 minutes apart, same box:

| phase | run A ms | run B ms | Δ | run A peak MiB | run B peak MiB | identical |
|---|---|---|---|---|---|---|
| build_model | 1901.682 | 1887.592 | −0.7% | 473.20 | 473.20 | yes |
| forward_only | 74.723 | 76.155 | +1.9% | 1952.57 | 1952.57 | yes |
| **compute_gramian** | **344.624** | **247.807** | **−28.1%** | 2686.43 | 2686.43 | yes |
| weighting_qp | 4.250 | 4.946 | +16.4% | 489.52 | 489.52 | yes |
| **final_backward** | **136.438** | **208.654** | **+52.9%** | 2247.08 | 2247.08 | yes |
| optimizer_step | 0.041 | 0.041 | +0.7% | 489.51 | 489.51 | yes |

**Every peak-memory reading is identical to the last recorded digit** while times move
by half. The process allocated exactly the same bytes in exactly the same order both
times. L4's ratio for this config is 3.556 in run A and 2.211 in run B — a 61% swing on
a quantity that was being compared to L11 at three significant figures.

The same holds in L11:

| cell | run A ms | run B ms | Δ | peak A | peak B | identical |
|---|---|---|---|---|---|---|
| UPGrad/jdgram | 565.294 | 575.882 | +1.9% | 3159.5928 | 3159.5928 | yes |
| UPGrad/autogram | 534.984 | 544.957 | +1.9% | 4015.1401 | 4015.1401 | yes |
| Mean/jdgram | 512.849 | 611.302 | +19.2% | 2686.3926 | 2686.3926 | yes |
| MGDA/autojac | 741.041 | 1143.066 | +54.3% | 4308.3633 | 4308.3633 | yes |
| **PCGrad/jdgram** | 416.034 | 612.439 | **+47.2%** | 3159.5928 | 3159.5928 | yes |
| **PCGrad/autogram** | 387.074 | 571.329 | **+47.6%** | 4015.1401 | 4015.1401 | yes |

`PCGrad/jdgram` and `PCGrad/autogram` are cells 10 and 11, run back to back, and both
moved by the same +47%. Two *adjacent in wall-clock time*, *unrelated in code* cells
degrading identically is a burst of interference occupying a time window, not a
property of either engine.

### 4.2 Cells that must cost the same, do not

Within one L11 call, `Mean/jdgram`, `UPGrad/jdgram` and `PCGrad/jdgram` run identical
GPU work — same model, same batches, same Gramian, same weighted backward — differing
only in an aggregation over an `[m, m]` matrix with m ≤ 3. Their spread:

| run | Mean | UPGrad | PCGrad | spread |
|---|---|---|---|---|
| m=1 | 186.39 | **136.60** | 168.42 | 36.4% |
| m=2 | 303.20 | **298.32** | 434.08 | 45.5% |
| m=3 (A) | 512.85 | 565.29 | 416.03 | 35.9% |
| m=3 (B) | 611.30 | **575.88** | 612.44 | 6.3% |

The controls behave the same way — `autojac` spreads 69.8% at m=2, `autogram` 43.0% at
m=3A — so this is the box, not jdgram. On the quiet local box the same three-way spread
is 8–13% (§2.7).

**Every ratio reported to the supervisor used the `UPGrad` cell, which at m=1 and m=2
is the fastest of the three.** The denominator, `sgd_erm`, is cell 13 of 13 —
reconstructed from `total_wall_ms`, it starts at t≈332 s at m=1, while `UPGrad/jdgram`
starts at t≈47 s. The numerator and the denominator of the acceptance ratio are
measured **285 seconds apart** in a run whose equivalent cells scatter by 36%.

### 4.3 Every m has a cell that matches L4's prediction

If L4 is right, an L11 jdgram cell should cost `sgd_erm + (compute_gramian +
weighting_qp)`:

| run | predicted | Mean | UPGrad | PCGrad | closest |
|---|---|---|---|---|---|
| m=1 | 191.66 | 186.39 (−2.7%) | 136.60 (−28.7%) | 168.42 (−12.1%) | Mean, 2.7% |
| m=2 | 427.42 | 303.20 (−29.1%) | 298.32 (−30.2%) | 434.08 (+1.6%) | PCGrad, 1.6% |
| m=3 (B) | 489.65 | 611.30 (+24.8%) | 575.88 (+17.6%) | 612.44 (+25.1%) | UPGrad, 17.6% |

At m=1 and m=2 an L11 cell sits within 3% of L4's prediction; at m=3 every cell is
*above* it. Choosing `UPGrad` picks the −29% end at m=1 and m=2, which is where the
"1.49×" comes from.

### 4.4 The baselines agree once the optimiser is charged

L4's `final_backward` is a fresh forward plus the weighted backward, and `forward_only`
is that forward alone, so L4 predicts sgd_erm = `final_backward` + optimiser + L11's
per-step Python/host overhead:

| run | L4 fwd | L4 bwd | L4 fb | + optimiser 2.83 | L11 sgd_erm | residual |
|---|---|---|---|---|---|---|
| m=1 | 26.18 | 55.06 | 81.23 | 84.03 | 91.54 | +7.51 |
| m=2 | 50.82 | 94.98 | 145.80 | 148.60 | 155.49 | +6.89 |
| m=3 (B) | 76.15 | 132.50 | 208.65 | 211.45 | 236.89 | +25.44 |

7 ms of residual at m=1 and m=2 is exactly what L11 adds and L4 does not: the corpus
batch fetch, `apply_objective_mode`, two device→host scalar reads and the closure. **On
the shared part of the step, the two levels agree to within 8%.** They only diverge on
the part that is obtained by subtraction.

### 4.5 Why subtraction is the wrong instrument

L11's marginal is a difference of two numbers of comparable size, each carrying the
per-cell scatter of §4.2. Propagating that scatter (using the sd/mean of the three
equivalent jdgram cells as the per-cell relative error):

| run | cell scatter | L11 marginal | ±1σ | relative | L4 says |
|---|---|---|---|---|---|
| m=1 | 15.4% | 45.06 | 25.31 | 56% | 100.12 |
| m=2 | 22.3% | 142.83 | 75.05 | 53% | 271.94 |
| m=3 (B) | 3.5% | 338.99 | 21.58 | 6% | 252.75 |

At m=1 and m=2 the L11 marginal carries **>50% relative uncertainty**. A quantity known
to ±53% cannot arbitrate a 1.49× claim about anything.

### 4.6 What I could not distinguish

External contention (another job sharing the card) and clock/power behaviour (sustained
load dropping boost clocks) both fit §4.1 and §4.2. `slurm_job_id` is empty in every
manifest and no compute-app census was recorded, so I cannot separate them from the
artifacts on disk. It does not change any conclusion below — both are properties of the
host, not of either level's methodology.

---

## 5. Physical sanity check on the two candidate answers

At m=1, L4 measures forward = 26.18 ms and forward+backward = 81.23 ms, so a plain
backward is 55.06 ms.

- **L4's answer (Gramian = 95.21 ms):** its own forward (26.18) plus a reverse over the
  same graph plus the per-layer identity arithmetic = 26.18 + 69.03. Coherent.
- **L11's answer (Gramian ≈ 40 ms):** the Gramian would have to run a forward (26.18)
  *and* a full reverse *and* the identity products in 40 ms, when the reverse alone
  costs 55 ms in the plain case. jdgram's reverse does skip writing 473 MiB of
  parameter gradients, so it is legitimately cheaper than a vanilla backward — but not
  by enough to fit a forward and a reverse and the identity work into 14 ms after the
  forward.

Using all three equivalent cells instead of the fastest one gives an L11 marginal of
72.26 ms at m=1, which is physically plausible and 28% below L4 rather than 122% below.

**L4's absolute Gramian number is the one that is physically coherent.**

---

## 6. The bound: what to trust, for what, and by how much

### Trust L4 for the *marginal cost of the Gramian* — the floor and fusion analysis

It measures that quantity directly instead of inferring it from a difference, it is
corroborated by L5 at m=1 to 0.7%, it is physically coherent (§5), and on a quiet box it
matches an L11-style loop to 8% (§2.7).

**Discount to apply: none for bias. A ±30% uncertainty band, and the §3 optimiser
correction.** The 30% comes from L4's own run-to-run reproducibility at 124M — the m=3
repeat moved `compute_gramian` by 28% and `final_backward` by 53%. Concretely, every
number in `bench/acceptance.py`'s floor/fusion table should be read as:

| quantity (m=1) | as reported | corrected | band |
|---|---|---|---|
| time ratio | 2.232 | 2.191 | 1.53 – 2.85 |
| floor ratio (free Gramian) | 1.382 | 1.370 | 1.26 – 1.48 |
| fused floor ratio | 1.060 | 1.058 | 1.04 – 1.08 |

One structural conclusion is robust to the whole band: the fused floor stays at ~1.0×
across it, so **the redundant second forward really is the entire structural floor**.
That does not need three significant figures to hold.

The unfused floor is **not** robust in the same way, and the earlier draft of this
section overstated it. The corrected m=1 floor is 1.370 with a band of 1.26 – 1.48,
which lies **entirely below** the 1.5× budget — so at m=1 the two-forward floor does
not on its own exhaust the budget, and the claim that it "stays at or above the
budget's edge" is not supported by this table. What the m=1 numbers do support is
narrower and still consequential: the floor consumes roughly **74% of the allowed
0.5× overhead** (0.370 of 0.5) before any identity kernel runs, leaving ~0.13× for
everything the Gramian actually does.

The floor does cross 1.5× at higher m in the on-disk data (m=3 run A reads 1.579), but
that specific run is the one §5 argues is corrupted by host interference; its repeat
reads 1.389. So "the floor exceeds the budget" is currently a claim resting on the
least trustworthy cell in the campaign, and should not be presented as settled until
a clean high-m measurement exists.

### Trust L11 for the *acceptance-bar ratio* — but not at one repeat

L11 measures the thing actually being claimed (a real step against a real SGD control,
including per-step host overhead). It is the right instrument for "is it 1.5×". But as
run today a single L11 cell carries ~15–22% and the ratio's numerator and denominator
are measured minutes apart, so **a single-run L11 ratio should be quoted to one
significant figure at best**: "≈1.5× at m=1, ≈1.9× at m=2, ≈2.4× at m=3", not
1.492/1.919/2.431.

### Do not compute the Gramian's cost as `jdgram − sgd_erm`

Above ±15% per-cell noise, that difference has >50% relative error (§4.5). It is the
single worst estimator available for the quantity, and it is the one that produced the
1.49×.

### Do not compare L5 and L4 above m=1

L5 runs the router on auto and L4 pins `dfirst` (§2.2).

---

## 7. The experiment that would settle it

**One change, one run: put a `torch.cuda.Event` pair around `jdgram_gramian(...)`
inside L11's own loop.**

```
L11 jdgram branch, per step:
    ev0.record(); G = jdgram_gramian(...).total; ev1.record()
    accumulate ev0.elapsed_time(ev1)
```

This measures the Gramian's cost *in the real continuously-running loop*, directly,
with no subtraction and no second cell. Report the median over steps 1..N-1 alongside
the existing `ms_per_step`, as a new `metric="gramian_ms_in_loop"` row.

It is decisive because it collapses the whole question to one comparison:

- if `gramian_ms_in_loop` ≈ L4's `compute_gramian`, L4 is vindicated and the 1.49× was
  cell scatter — which is what everything in §2–§5 predicts;
- if it lands near 40 ms, then something inside the running loop really does make the
  Gramian cheaper, and that effect is now localised to a single measured quantity
  instead of inferred from a difference of two training runs.

Run it as: one process, one config, **five alternating repeats** of
`[L4 phases] → [L11 sgd_erm] → [L11 jdgram]` so both levels sample the same interference
windows, on a card with exclusive compute mode, with `nvidia-smi --lock-gpu-clocks` set
and `nvidia-smi --query-compute-apps` logged once a second alongside. Report medians and
inter-quartile ranges, never a single value.

At 124M that is roughly 5 × (L4 ≈ 15 s + two 100-step cells ≈ 25 s) ≈ 3.5 minutes.

Two cheaper hygiene changes worth making at the same time, both of which would have
prevented this investigation:

1. **Reorder L11's cells so `sgd_erm` runs first *and* last**, and log both. The
   difference between the two readings is a direct, free measurement of the drift that
   currently sits unquantified between the numerator and the denominator of every
   acceptance ratio.
2. **Record per-step times, not just the mean**, for at least the jdgram and sgd_erm
   cells. `total_wall_ms / (steps-1)` cannot distinguish a steady 136 ms/step from
   90 ms/step with a 40-second stall in the middle, and §4.1 shows those stalls exist.

---

## 8. Reproducing everything in this document

```
# the 124M campaign data, no GPU needed
results/solo/v10_20260811-0{21816,22446,23332,25217}_a-dup-m*-dfirst-r1_*/rows.csv

# the tiny-scale bisection (§2.1, §2.2, §2.3), ~40 s on one GPU
python bench/profile_suite.py --version 10 --name l4-probe --device cuda --levels L4 \
    --m 1 --T 128 --V 1024 --n-embd 128 --n-layer 2 --n-head 2 \
    --force-route dfirst --driver squashed --objective-mode duplicate

# the optimiser measurement (§3): build the real 124M parameter set, hand it grads,
# time opt.step() with and without them. No forward, no backward.
```

Gate suite after the §3 fix: `python -m pytest gates/` → **58 passed in 1.88s**.
