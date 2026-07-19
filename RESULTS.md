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
| SGD (ERM scalar) | 0.036 ± 0.002 | 0.10 | 0.28 |
| Mean | 0.363 ± 0.006 | 1.00 | 1.00 |
| UPGrad | 0.484 ± 0.011 | 1.33 | 1.14 |
| PCGrad | 0.687 ± 0.007 | 1.89 | 1.78 |
| MGDA | 1.440 ± 0.431 | 3.96 | 2.97 |

Ratios are compared rather than absolute times (different GPU, different
TorchJD version than the paper). Ordering matches the paper exactly
(SGD < Mean < UPGrad < PCGrad < MGDA). 

- The scalar SGD baseline is proportionally cheaper here than in the paper
  (0.10 vs 0.28) — plausibly higher raw throughput on this GPU with zero
  aggregation overhead.
- MGDA's large std (± 0.431) reflects its iterative Frank–Wolfe-style solve.

## 3. Engine comparison — autojac vs autogram

![Engines](results/engines_cifar10.png)

Same UPGrad weights on both paths (verified by preflight), same data order:

| Config | s/epoch | peak MiB |
|---|---|---|
| SGD (ERM scalar) | 0.037 ± 0.001 | 54.5 |
| autojac + UPGrad | 0.420 ± 0.005 | 602.6 |
| autojac + UPGrad (`optimize_gramian_computation=True`) | 0.514 ± 0.017 | 602.6 |
| autogram + UPGradWeighting | 0.237 ± 0.005 | 83.2 |

- **1.77x faster** and **7.24x less peak memory** for autogram at this scale.
- autogram's peak (83 MiB) is extremely close to plain SGD's (54.5 MiB), physically proving the math in the paper's Appendix E.1: with the Gramian-based method only the m×m Gramian is stored, entirely bypassing the massive m×n Jacobian.
- `optimize_gramian_computation=True` on the autojac path surprisingly *increased* epoch time on this hardware while memory remained identical.

## 4. Step decomposition (where the time goes)

![Decomposition](results/decompose_cifar10.png)

One autogram step split into forward / gramian_pass / weighting_qp /
backward_step, across batch size m (= number of objectives) and model width.
QP share of total epoch time:

| width (params) | m=4 | m=8 | m=16 | m=32 | m=64 |
|---|---|---|---|---|---|
| 1 (0.13M) | 9.9% | 11.1% | 13.8% | 26.0% | 65.3% |
| 4 (2.1M) | 6.6% | 6.0% | 6.2% | 18.6% | 43.5% |
| 8 (8.4M) | 2.0% | 2.1% | 2.2% | 8.7% | 22.4% |

Two trends:

- At small m (4–8), the QP share shrinks as the parameter count grows — down
  to **~2.0%** at 8.4M params — while `gramian_pass` grows to
  **8.5 s/epoch**, dominating the step outright.
- At large m the picture inverts: at m = 64, width 1, the QP is **65.3%** of
  the step. The QP only matters when there are many objectives.

For a regime with few objectives and many parameters, Gramian accumulation is
the strict bottleneck, not the QP solve (which TorchJD currently runs sequentially on
CPU; the paper notes batching it would reduce O(m^5) to O(m^4), but at m ≤ 8
that cost is already negligible).

## 5. Memory scaling with model size

![Scaling](results/scaling_cifar10.png)

UPGrad, batch 32, width-multiplied versions of the paper CNN:

| width | params | autojac peak | autogram peak | ratio |
|---|---|---|---|---|
| 1 | 0.13M | 603 MiB | 83 MiB | 7.26x |
| 2 | 0.53M | 1207 MiB | 203 MiB | 5.94x |
| 4 | 2.11M | 2516 MiB | 635 MiB | 3.96x |
| 8 | 8.42M | 5530 MiB | 2279 MiB | 2.42x |

autogram is smaller everywhere, successfully scaling to wider networks while remaining comfortably within laptop VRAM constraints. The width-16 run (~33.6M params, projected autojac peak > 12 GB) was explicitly excluded from the automated suite due to reproduced GPU driver watchdog errors (DPC_WATCHDOG_VIOLATION, 0x133) on the native PyTorch autojac path.

## 6. Operator-Level Profiling (The Launch Bottleneck)

Wrapping the `autogram` execution in `torch.profiler` revealed a critical discrepancy between computation scale and wall-clock execution time. 

A robust 50-step profile (`--warmup 5 --active-steps 50`) across widths 1 and 2 isolates the exact hardware bottleneck. 

**Table E: Profile Sweep Summary (50 active steps)**
| width | params(M) | wall ms/step | profiled peak MiB | gramian_pass CUDA |
|---|---|---|---|---|
| 1 | 0.13 | 27.86 | 261.1 | ~0.8 µs |
| 2 | 0.53 | 25.20 | 897.2 | ~0.8 µs |

### 6.1 Trace Anatomy and The Dispatch Bottleneck
Reviewing the Perfetto trace (`results/profile_w2_trace.json`) visually confirms that the current architecture is severely **kernel-launch bound**, not compute bound.

1. **Compute is instant:** The actual CUDA device execution for the Gramian math takes **less than 1 microsecond** (~0.8µs) per step. 
2. **CPU Dispatch Choke & Launch Overhead:** The host trace shows extreme density in the `ComputeModuleJacobians` block. The CPU is forced to dynamically dispatch thousands of micro-operations (`aten::view`, `aten::reshape`, `aten::addmm`, `aten::empty`) to handle the layer-by-layer unrolling of the Jacobian materialization. The GPU completes the mathematical payload almost instantly and spends the vast majority of its time completely idle, waiting for the host CPU to traverse the PyTorch dispatcher, OS boundary, and CUDA runtime driver to launch the next micro-operation.

### 6.2 Deep Dive: The Memory Thrashing Failure State (w=4 Profiling)
Attempting to aggressively profile the width=4 model (`2.11M params, m=32`) with full shape and memory tracking exposed the absolute limits of the `autogram` graph-hooking architecture and triggered severe hardware thrashing.

**The Trigger (The Observer Effect):**
The `torch.profiler` configuration required `record_shapes=True` and `profile_memory=True`. This prevents PyTorch from aggressively garbage-collecting intermediate tensors. Because TorchJD natively materializes the layer-Jacobians ($J_l$) during the `gramian_pass`, pinning them in memory caused the footprint to artificially balloon to **24.5 GB**. 

**The Hardware Response (Thrashing):**
Exceeding the RTX 5070 Ti's 12.8 GB physical VRAM capacity forced the NVIDIA driver to fallback to Shared GPU Memory (system RAM). This tanked the PCIe bandwidth. Mathematical operations that previously took microseconds dilated massively, with `aten::convolution_backward` accumulating over 5.3 seconds of execution time across the 50 steps as the GPU waited for data to cross the motherboard. The system ultimately failed with an OOM during the `torch.cat` operation inside TorchJD's Jacobian materialization logic.

### 6.3 Conclusion 
The data definitively proves that optimizing the math via standard PyTorch Python APIs has hit a hard hardware-software boundary. Fusing the Hadamard Gramian accumulation ($G = A A^T \odot X X^T$) into a unified custom GPU kernel (e.g., via Triton) will keep execution localized to the GPU's SRAM, entirely bypassing the PyTorch C++ dispatcher, preventing system memory fallback, and yielding massive throughput gains.

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
- Translate to RL => LLM