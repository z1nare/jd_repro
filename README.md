# TorchJD IWRM benchmarks — CIFAR-10

Independent reproduction and profiling harness for
[TorchJD](https://github.com/TorchJD/torchjd) on the IWRM setting of
[*Jacobian Descent for Multi-Objective Optimization*](https://arxiv.org/abs/2406.16232)
(arXiv:2406.16232v3), plus a custom Gramian engine (`algo3-hadamard`).

Full write-up: **[RESULTS.md](RESULTS.md)**. Raw fuji2 artifacts:
[`results_extensive_fuji2/`](results_extensive_fuji2/),
[`results_w106_112/`](results_w106_112/).

## Key findings (fuji2, RTX A5000 24 GB)

- **Convergence:** UPGrad beats Mean (AUC 381 vs 495). PCGrad diverges at
  lr ≥ 0.01. MGDA learns poorly — same qualitative story as the paper.
- **Engines at paper scale:** `algo3-hadamard` is the **fastest** path
  (0.20 s/epoch) vs `autogram` (0.33) and `autojac` (0.45), with modest
  memory vs autogram (134 vs 82 MiB).
- **Capacity:** TorchJD `autojac` / `autogram` OOM at **134M params** (w=32).
  Hadamard runs to **1.65B params** (w=112, 22.6 GB peak) — about **12×**
  TorchJD’s ceiling on the same card — then OOM at w=128.
- **Bottleneck:** at small objective counts (m=4–16), Gramian accumulation
  dominates; QP share is secondary. At m=64 the sequential QP dominates.

## Running

```bash
pip install -r requirements.txt

python run_all.py --quick     # short sanity pass
python run_all.py             # full suite (laptop-safe widths)
python iwrm_bench.py --help
```

On a 24 GB GPU, push hadamard alone:

```bash
python iwrm_bench.py scaling --dataset cifar10 \
  --widths 96 106 112 --engines algo3-hadamard --warmup 2 --timed 3
```

Every trusted suite starts with `preflight` (autojac / autogram / hadamard
equivalence).

## Environments

- **Reported here:** fuji2, 4× A5000 24 GB, torch 2.4.1+cu121, Python 3.10
- Cluster torch pins may differ from `requirements.txt` (`torch>=2.7` for
  laptop); pin CUDA wheels to match the host driver when needed.
