# TorchJD IWRM benchmarks — CIFAR-10

Benchmarking and profiling suite for [TorchJD](https://github.com/TorchJD/torchjd)
(v0.17.0) on the instance-wise risk minimization (IWRM) setting of
[*Jacobian Descent for Multi-Objective Optimization*](https://arxiv.org/abs/2406.16232)
(arXiv:2406.16232v3).

Goal: validate the paper's published results (Figure 2 convergence ordering,
Table 7 timing ratios) with an independent harness, then use that validated
harness to compare TorchJD's two execution engines (autojac vs autogram) and
isolate the computational bottleneck (QP solve vs Gramian accumulation).

## Key findings

Full write-up with plots, tables, caveats, and remaining work:
**[RESULTS.md](RESULTS.md)**.

- **Convergence:** UPGrad beats Mean, matching the paper. MGDA never learns
  (consistent with its known small-gradient pathology). PCGrad diverges at
  lr ≥ 0.01 and needs a ~100x smaller lr than UPGrad.
- **Engines:** autogram is ~2x faster (1.8–1.95x across sessions) and uses
  7.25x less peak memory than autojac for identical UPGrad updates (update
  equivalence asserted to ~6e-8).
- **Bottleneck:** at 4–8 objectives, the QP solve is ≤ 3% of step time on the
  larger models, while the Gramian accumulation pass dominates. The QP only
  becomes significant at large objective counts (69% at m = 64).

## Running

```bash
pip install -r requirements.txt

python run_all.py --quick     # 5-10 min sanity pass
python run_all.py             # full suite, ~60-90 min on a 12 GB GPU
python iwrm_bench.py --help   # individual subcommands
```

Outputs (plots, JSON, ratio table, per-step logs, `env.json`) land in
`results/`. Every run starts with a `preflight` step that asserts
autojac/autogram update equivalence before any benchmark is trusted.

Note: the `scaling` step is capped at width 8. Width 16 needs > 12 GB for the
autojac path and crashed the GPU driver on the 12 GB test machine; run it only
on a GPU with ≥ 16 GB.

## Environment used for the reported results

RTX 5070 Ti Laptop GPU (12 GB), torch 2.10 nightly (cu128), TorchJD 0.17.0,
Python 3.11 — see `results/env.json`.
