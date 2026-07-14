# Results — IWRM benchmarks and TorchJD profiling (CIFAR-10)

Reproduction of the small-scale IWRM experiments from *Jacobian Descent for
Multi-Objective Optimization* (arXiv:2406.16232v3), plus engine-level profiling
of TorchJD 0.17.0. All raw data (JSON), plots, and per-step logs are in
[`results/`](results/).

**Environment:** RTX 5070 Ti Laptop GPU (12 GB), torch 2.10 nightly (cu128),
TorchJD 0.17.0, Python 3.11. See [`results/env.json`](results/env.json).

**Protocol** (matching the paper's Appendix D unless noted):

- Architecture: the CIFAR-10 CNN from Table 3, verbatim (including the grouped
  convolutions), ELU activations, PyTorch default init.
- Data: 1024-image seeded subset, batch size 32, per-channel normalization
  computed on the full training split (D.6). The whole subset is preloaded to
  GPU, so timings contain no dataloader noise.
- Optimization: plain SGD, no momentum (D.4); cross-entropy with
  `reduction='none'`, i.e. 32 objectives per step (IWRM via SSJD); 20 epochs =
  640 iterations (Table 6).
- Learning rate: selected per aggregator by area under the loss curve, the
  paper's D.1 criterion, but over a 7-point grid (3e-4 … 0.3) instead of the
  paper's 22-point coarse + 50-point refined sweep.
- Determinism: subset indices, model init `state_dict`, and per-epoch batch
  order are seeded and shared across all aggregators and both engines.
- Timing: 3 warmup epochs, then ≥10 timed epochs bracketed by
  `torch.cuda.synchronize()`; memory via `reset_peak_memory_stats` /
  `max_memory_allocated`.

Before any benchmark, `preflight` verifies that the autojac path
(`UPGrad` aggregator) and the autogram path (`UPGradWeighting`) produce the
same updates: after a multi-step trajectory from identical init, the max
parameter difference was **5.96e-08** (tolerance 5e-4).

---

## 1. Figure 2 reproduction (convergence)

![Figure 2](results/figure2_cifar10.png)

Qualitative match with the paper's Figure 2c:

- **UPGrad** (lr 0.3) converges fastest and lowest, clearly below **Mean**
  (lr 0.3) — the paper's headline result.
- **PCGrad** (lr 0.003) diverges to NaN at every lr ≥ 0.01 and only trains at
  lr ≤ 0.003. This is consistent with its definition: PCGrad **sums** the m
  projected gradients (paper Eq. 33) where UPGrad **averages** them (Eq. 4) —
  the paper notes A_PCGrad = m · A_UPGrad for m ≤ 2. At m = 32 the update is
  correspondingly larger at the same lr, which matches the ~100x smaller
  stable lr observed here.
- **MGDA** (lr 0.1) does not learn at all: the loss stays at 1.8–2.2 for all
  640 iterations, at every lr in the grid. The paper's own explanation
  (Section 5) is MGDA's sensitivity to small gradients: if any row of the
  Jacobian approaches zero, the whole aggregation approaches zero, so it can
  stall at weakly-stationary points far from optimal. A caveat on the sweep
  itself: when no lr produces learning, the AUC criterion cannot discriminate
  between lrs, so the selected lr is close to arbitrary.

## 2. Table 7 timing ratios

| Method | s/epoch (ours) | ratio, Mean = 1 (ours) | ratio (paper, L4) |
|---|---|---|---|
| SGD (ERM scalar) | 0.052 ± 0.004 | 0.12 | 0.28 |
| Mean | 0.418 ± 0.023 | 1.00 | 1.00 |
| UPGrad | 0.594 ± 0.015 | 1.42 | 1.14 |
| PCGrad | 1.154 ± 0.111 | 2.76 | 1.78 |
| MGDA | 1.996 ± 0.713 | 4.77 | 2.97 |

Ratios are compared rather than absolute times (different GPU, different
TorchJD version than the paper). Ordering matches the paper exactly
(SGD < Mean < UPGrad < PCGrad < MGDA); the magnitudes need a caveat:

- **Session-to-session variability dominates the ratios on this laptop GPU.**
  Across sessions, Mean measured between 0.39 and 0.73 s/epoch while the
  aggregators' *absolute* overheads over Mean stayed roughly constant (e.g.
  UPGrad − Mean ≈ 0.18 s/epoch in both), so a faster baseline session inflates
  every ratio. An earlier session with a slower baseline gave ratios of
  1.25 / 1.78 / 2.88, much closer to the paper's 1.14 / 1.78 / 2.97. The
  table above reports the run whose raw output is in
  [`results/table7_cifar10.md`](results/table7_cifar10.md) and
  [`results/logs/table7.log`](results/logs/table7.log). A proper fix is
  interleaved multi-session runs on a non-shared GPU (see remaining work).
- The scalar SGD baseline is proportionally cheaper here than in the paper
  (0.12 vs 0.28) — plausibly higher raw throughput on this GPU with zero
  aggregation overhead.
- MGDA's large std (± 0.713) reflects its iterative Frank–Wolfe-style solve.

## 3. Engine comparison — autojac vs autogram

![Engines](results/engines_cifar10.png)

Same UPGrad weights on both paths (verified by preflight), same data order:

| Config | s/epoch | peak MiB |
|---|---|---|
| SGD (ERM scalar) | 0.043 ± 0.002 | 54.5 |
| autojac + UPGrad | 0.573 ± 0.034 | 602.6 |
| autojac + UPGrad (`optimize_gramian_computation=True`) | 0.643 ± 0.004 | 602.6 |
| autogram + UPGradWeighting | 0.315 ± 0.007 | 83.2 |

- **1.8x faster** and **7.25x less peak memory** for autogram at this scale
  (an earlier session measured 1.95x; the memory numbers are bit-identical
  across sessions, the time ratio moves with GPU clock state).
- autogram's peak (83 MiB) is close to plain SGD's (54.5 MiB), which is what
  the paper's Appendix E.1 predicts: with the Gramian-based method only the
  m×m Gramian is stored, never the m×n Jacobian.
- `optimize_gramian_computation=True` on the autojac path changed nothing
  measurable (time or memory) at this model size, in two independent runs.

## 4. Step decomposition (where the time goes)

![Decomposition](results/decompose_cifar10.png)

One autogram step split into forward / gramian_pass / weighting_qp /
backward_step, across batch size m (= number of objectives) and model width.
QP share of total epoch time:

| width (params) | m=4 | m=8 | m=32 | m=64 |
|---|---|---|---|---|
| 1 (0.13M) | 10.6% | 11.8% | 25.5% | 65.2% |
| 4 (2.1M) | 6.9% | 8.2% | 15.8% | 40.8% |
| 8 (8.4M) | 2.9% | 2.4% | 8.1% | 19.6% |

Two trends:

- At small m (4–8), the QP share shrinks as the parameter count grows — down
  to **~2.5%** at 8.4M params — while `gramian_pass` grows to
  **9.3 s/epoch**, dominating the step outright.
- At large m the picture inverts: at m = 64, width 1, the QP is **65%** of
  the step. The QP only matters when there are many objectives.

For a regime with few objectives and many parameters, Gramian accumulation is
the bottleneck, not the QP solve (which TorchJD currently runs sequentially on
CPU; the paper notes batching it would reduce O(m^5) to O(m^4), but at m ≤ 8
that cost is already negligible).

## 5. Memory scaling with model size

![Scaling](results/scaling_cifar10.png)

UPGrad, batch 32, width-multiplied versions of the paper CNN:

| width | params | autojac peak | autogram peak | ratio |
|---|---|---|---|---|
| 1 | 0.13M | 603 MiB | 83 MiB | 7.25x |
| 2 | 0.53M | 1207 MiB | 203 MiB | 5.95x |
| 4 | 2.11M | 2516 MiB | 635 MiB | 3.96x |
| 8 | 8.42M | 5530 MiB | 2279 MiB | 2.43x |

autogram is smaller everywhere, but the ratio *shrinks* with model size.
A back-of-the-envelope check suggests why this needs a closer look: the
[32 × n] Jacobian at width 1 is only ~17 MB, far below autojac's 603 MiB peak,
so most of that peak is overhead beyond the Jacobian itself. Isolating it
(e.g. with `torch.cuda.memory_snapshot`) is listed under remaining work.

The width-16 run (~33.6M params, projected autojac peak > 12 GB) hard-crashed
this machine twice with a GPU driver watchdog error (DPC_WATCHDOG_VIOLATION,
0x133) rather than a clean CUDA OOM — autojac destabilizes the driver at this
scale before PyTorch can even raise. That point needs a larger GPU.

---

## Caveats

- **1 seed** (paper: 8 seeds with SEM bands). Curves are single runs.
- **7-point lr grid** (paper: 22 coarse + 50 refined). Mean and UPGrad
  selected lr = 0.3, the top of the grid, so their true optima may be higher;
  the paper's grid extends to 1e2.
- **CIFAR-10 only** (the SVHN architecture is implemented but not run).
- Timing on a shared laptop GPU; std is reported per row.

## Remaining experiments

- Multi-seed reruns (8 seeds, SEM bands) to match the paper's protocol.
- Extend the lr grid above 0.3 for Mean/UPGrad (both selected the endpoint).
- SVHN pass (`--dataset svhn`, architecture already implemented).
- Width ≥ 16 scaling and the autojac peak-memory breakdown
  (`torch.cuda.memory_snapshot`) — needs a GPU with ≥ 16 GB.
- Step decomposition at larger N and on a non-CNN architecture (small
  transformer) to check that the Gramian-pass-dominates conclusion holds
  beyond this model family.
- Optional: cosine-similarity-to-Mean curves (paper Figures 2b/2d).
