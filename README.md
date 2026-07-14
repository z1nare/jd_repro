# TorchJD Phase 2.5 Benchmarks — CIFAR-10

This repository contains the benchmarking and profiling suite for TorchJD (v0.17.0), evaluating performance, memory scaling, and algorithmic convergence on CIFAR-10 using an RTX 5070 Ti mobile. 

**Core Objective:** Validate paper claims (Figure 2, Table 7) and isolate the specific computational bottlenecks (QP vs. Gramian accumulation) to scope kernel-level optimizations for Phase 3.

### Key Findings
A detailed breakdown of all findings, including memory profiling and sub-routine decomposition, is available in **[RESULTS.md](./RESULTS.md)**. 
* **Convergence:** `UPGrad` matches paper headlines; `MGDA` flatlines completely.
* **Engine Efficiency:** `autogram` yields a 1.95x time speedup and a 7.25x memory reduction over `autojac`.
* **Bottleneck Isolation:** At RL-relevant objective counts ($m=4-8$), the Gramian accumulation pass dominates execution time, while the QP solver footprint shrinks to <3%.

### Reproducing the Data
All raw JSONs and logs are preserved in `results/`. To run a fast preflight sanity check of the harness locally:
```bash
pip install -r requirements.txt
python run_all.py --quick