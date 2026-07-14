"""
Phase 2.5 harness -- BUILD-IT-YOURSELF SKELETON (iwrm_bench_skeleton.py)

You are implementing this yourself; `iwrm_bench.py` in the same folder is the
answer key (fully executed, all subcommands verified). Rule of engagement:
attempt each TODO first, diff against the reference only when stuck >15 min
or after your version passes its DONE-WHEN check.

BUILD ORDER (matches the Sun-eve -> Tue schedule in PHASE25_RUNBOOK.md):
  Session 1 (Sun eve):  TODO-01..06  (registry, determinism, data, model)
                        + TODO-11..13 (preflight gates)  <- gate everything first
  Session 2 (Mon AM):   TODO-07..09  (step functions), TODO-14..15 (figure2)
  Session 3 (Mon PM):   TODO-10 (timing), TODO-16 (table7), TODO-17 (engines)
  Session 4 (Tue AM):   TODO-18 (scaling stretch), TODO-19 (CLI polish), slides

VERIFICATION CHECKPOINTS (numbers from the executed reference, CPU/synthetic):
  preflight gate 3:  max param diff ~1e-7 over a multi-step trajectory
                     (tolerance: 1e-4 CPU, 5e-4 CUDA -- cuDNN nondeterminism)
  engines (CPU!):    autojac ~6.2 s/ep vs autogram ~0.47 s/ep vs SGD-ERM ~0.24
                     -- if your autogram is NOT dramatically faster, your
                     autogram step is silently falling back to something wrong.

ENVIRONMENT (bites you before any code runs):
  pip install "torchjd[quadprog_projector]" torchvision matplotlib
  -- plain `pip install torchjd` -> UPGrad() raises ImportError (QP solver extra).
"""

from __future__ import annotations

import argparse, json, math, random, time
from pathlib import Path

import torch
from torch import nn

# =============================================================================
# TODO-01 -- Aggregator registry
# =============================================================================
# Import from torchjd.aggregation: Mean/MeanWeighting, UPGrad/UPGradWeighting,
# PCGrad/PCGradWeighting, MGDA/MGDAWeighting. Build
#   AGGREGATORS: dict[str, tuple[AggregatorCtor, WeightingCtor]]
# The two-column structure is the point: column 1 acts on J (autojac path),
# column 2 acts on the Gramian (autogram path). Same math, different engine.
#
# Also hardcode PAPER_TABLE7 ratios here (from the runbook §1) so table7 can
# print the comparison column without you retyping numbers Tuesday morning:
#   cifar10: SGD-ERM 0.50 | Mean 1.76 | UPGrad 2.01 | PCGrad 3.13 | MGDA 5.22
#   svhn:    SGD-ERM 0.79 | Mean 1.41 | UPGrad 1.80 | PCGrad 2.78 | MGDA 5.50
# And PAPER_EPOCHS = {"cifar10": 20, "svhn": 25} (Appendix D.7, Table 6).
#
# DONE WHEN: `python -c "from skeleton import AGGREGATORS"` imports clean and
# UPGrad() constructs without ImportError.


# =============================================================================
# TODO-02 -- Determinism + device utilities
# =============================================================================
# def set_seed(seed): seed random, torch, torch.cuda (manual_seed_all).
# def sync(device):   torch.cuda.synchronize(device) iff cuda -- you will call
#                     this around EVERY timed region; make it a one-liner now.
# def ema(xs, alpha=0.05): exponential smoothing for the Figure-2 plot; the
#                     alpha value goes ON the slide (paper smooths too, D-fig).
#
# DONE WHEN: two calls to set_seed(1) + torch.randn(3) give identical tensors.


# =============================================================================
# TODO-03 -- Data loading (paper D.6 preprocessing, indices pinned to disk)
# =============================================================================
# def load_data(dataset, root, subset, seed, device) -> (X [N,3,32,32] f32, Y [N] i64)
#
# (a) Support "synthetic" (torch.randn images, randint labels) so you can
#     smoke-test every subcommand on any machine with zero downloads.
# (b) CIFAR-10: torchvision.datasets.CIFAR10(train=True).data is uint8 NHWC
#     numpy -> float/255 -> permute to NCHW. SVHN: .data is already NCHW.
# (c) Normalize per channel with mean/std computed on the ENTIRE training
#     split (paper D.6 says the whole split, NOT the 1024 subset -- easy to
#     get wrong and it shifts the loss scale on slide 1).
# (d) Subset: seeded randperm -> first 1024 indices -> SAVE THE INDEX TENSOR
#     TO DISK and reload if present. Every run, both engines, all aggregators
#     must see the same 1024 images forever.
# (e) Move X, Y to device ONCE here. No DataLoader anywhere in this project --
#     preloading is what makes epoch time = pure compute in table7/engines.
#
# DONE WHEN: two consecutive invocations return bit-identical X (check
# torch.equal), and X.device is cuda on the GPU box.


# =============================================================================
# TODO-04 -- Deterministic per-epoch batch order
# =============================================================================
# def batch_order(n, batch_size, epoch, seed) -> list[LongTensor]
# Fresh Generator seeded by f(seed, epoch) -> randperm -> .split(batch_size).
# Why a function of (seed, epoch) and not global RNG state: aggregator A's run
# must NOT perturb aggregator B's data order via shared RNG consumption. This
# also covers the PCGrad projection-ordering concern -- PCGrad draws its
# internal permutation from torch's global RNG, so set_seed() before each run
# (TODO-06 pattern) pins it.
#
# DONE WHEN: batch_order(1024,32,ep=3,seed=1) is identical across processes.


# =============================================================================
# TODO-05 -- Paper-exact models (Appendix D, Tables 2-3) -- DO NOT APPROXIMATE
# =============================================================================
# def make_model(dataset, width_mult=1) -> nn.Sequential
#
# CIFAR-10 (Table 3) -- the two GROUPED convolutions are the detail everyone
# misses; without groups= your "reproduction" is a different network:
#   Conv2d(3,32,3) ELU
#   Conv2d(32,64,3,groups=32)
#   MaxPool2d(2) ELU
#   Conv2d(64,64,3,groups=64)
#   MaxPool2d(3) ELU Flatten
#   Linear(1024,128) ELU
#   Linear(128,10)
# All convs: kernel 3, stride 1, NO padding, bias=True (defaults). ELU (paper
# cites Clevert et al.), NOT ReLU. Dim check to encode as an assert:
# 32 -conv-> 30 -conv-> 28 -pool2-> 14 -conv-> 12 -pool3-> 4; 64*4*4 = 1024.
# SVHN (Table 2): 16/32/32 channels, groups 16/32, Linear(512,64), Linear(64,10).
#
# width_mult scales every channel count AND the group counts (for TODO-18).
# Keep it nn.Sequential -- autogram's Algorithm-3 lineage is plain-sequence;
# don't get creative with a custom Module (see Engine docstring constraints,
# reading item R3).
#
# DONE WHEN: model(torch.randn(2,3,32,32)).shape == (2,10) and an assert on
# the flatten width passes for width_mult in {1,2}.


# =============================================================================
# TODO-06 -- Canonical initialization pinned to disk
# =============================================================================
# def init_state(dataset, seed, out_dir, width_mult=1) -> state_dict
# set_seed -> make_model -> save state_dict to
# out/init_{dataset}_w{width}_seed{seed}.pt; reload if file exists.
# EVERY run (each aggregator, each engine, each lr in the sweep) does
# model.load_state_dict(init_state(...)) -- bit-identical starting weights is
# what makes gate 3 and Deliverable C legitimate comparisons.
#
# DONE WHEN: two fresh models loaded from it have all-equal parameters.


# =============================================================================
# TODO-07 -- autojac step  (materialize J -> aggregate -> SGD)
# =============================================================================
# loss_fn = nn.CrossEntropyLoss(reduction="none")   # <- THE IWRM line: [B]
# def make_step_autojac(model, aggregator, optimizer) -> step(x,y)->float:
#   losses = loss_fn(model(x), y)          # [32] objectives
#   torchjd.autojac.backward(losses)       # fills .jac on leaf params
#   jac_to_grad(params, aggregator)        # J -> .grad via aggregator
#   optimizer.step(); optimizer.zero_grad()
#   return float(losses.mean().detach())   # curve point for figure2
# Capture params = list(model.parameters()) ONCE outside the closure --
# parameters() is a generator, second call inside a loop is an empty iterator
# class of bug.
# Paper D.4: torch.optim.SGD, NO momentum, NO weight decay.
#
# DONE WHEN: loss on one repeated batch decreases over 20 steps.


# =============================================================================
# TODO-08 -- autogram step  (Gramian -> weights -> ONE backward; eq. 10 live)
# =============================================================================
# def make_step_autogram(model, weighting, optimizer):
#   engine = Engine(model, batch_dim=0)    # construct ONCE, hooks attach here
#   step: losses [B] -> gramian = engine.compute_gramian(losses)  # [B,B]
#         weights = weighting(gramian)     # W(G), _NonDifferentiable inside
#         losses.backward(weights)         # grad_tensor=w  ==  (w*losses).sum()
#         optimizer.step(); zero_grad()
# Notes: (i) losses.backward(weights) is the docs-canonical form of your
# Drill-2 (w.detach()*losses).sum().backward() -- same thing, cleaner;
# (ii) do NOT rebuild Engine per step (hook re-registration = fake slowdown
# that would poison Deliverable C); (iii) batch_dim=0 is what tells the engine
# instances are independent along dim 0 -- exactly the Task-1 trace.
#
# DONE WHEN: preflight gate 3 (TODO-13) passes.


# =============================================================================
# TODO-09 -- Scalar SGD-ERM baseline
# =============================================================================
# losses.mean().backward() + SGD. This is the paper's "SGD" Table-7 row and
# the grey reference bar in Deliverable C. Its role on slide 3: the distance
# autojac-vs-SGD is the Jacobian-materialization tax; autogram's job is to
# collapse that distance. Two lines of code, disproportionate slide value.


# =============================================================================
# TODO-10 -- Timing harness (the part Rui will scrutinize; get it boringly right)
# =============================================================================
# def timed_epochs(step, X, Y, warmup, timed, batch_size, seed, device)
#       -> (mean_s_per_epoch, std_s_per_epoch, peak_MiB)
# Protocol, in order:
#   1. Run `warmup` full epochs untimed (cuBLAS autotune, allocator warmup,
#      quadprog's first-call numpy import cost).
#   2. sync(device); torch.cuda.reset_peak_memory_stats(device).
#   3. Per timed epoch: sync -> t0=time.perf_counter() -> epoch -> sync -> t1.
#      Sync BOTH sides or you time kernel *launches*, not kernels.
#   4. peak = torch.cuda.max_memory_allocated()/2**20 (NaN on CPU is fine).
#   5. Return mean and sample std (n-1) over the timed epochs.
# Data is already on GPU (TODO-03e) => epoch time is pure compute by
# construction; you never have to argue about dataloader noise.
#
# DONE WHEN: two consecutive calls on the same config agree within ~5%.


# =============================================================================
# TODO-11 -- PREFLIGHT GATE 1: pref_vector audit
# =============================================================================
# Assert, at runtime: type(UPGrad().gramian_weighting) is UPGradWeighting, and
# both .pref_vector defaults are None (None -> MeanWeighting -> uniform 1/m --
# verified in torchjd 0.17 source, aggregation/_upgrad.py: UPGrad.__init__ is
# literally super().__init__(UPGradWeighting(pref_vector, projector))).
# Keep the assertion even though the answer is known: it re-verifies on every
# TorchJD upgrade, which is the actual threat model.


# =============================================================================
# TODO-12 -- PREFLIGHT GATE 2: autogram compatibility of the paper CNN
# =============================================================================
# Build the exact TODO-05 model, wrap in Engine(batch_dim=0), compute one
# Gramian on one real batch, assert shape (32,32). This is your Task-2
# incompatible-layers list executed as a check instead of consulted as a doc.
# (Reference result: passes -- grouped convs / MaxPool / ELU / Flatten /
# Linear are all fine; no BatchNorm in the paper archs, almost certainly
# deliberately, since BN couples instances across the batch dim.)
# If it ever fails: the exception names the module; substitute minimally,
# document, and that substitution is itself a slide-worthy finding.


# =============================================================================
# TODO-13 -- PREFLIGHT GATE 3: engine-equivalence trajectory assertion
# =============================================================================
# Same init (TODO-06), same batch order (TODO-04), lr=0.05:
#   model A: autojac + UPGrad;  model B: autogram + UPGradWeighting.
# Run ~3 steps x a few batches interleaved, then
#   max over params of (pA - pB).abs().max()  <  tol
# tol: 1e-4 on CPU, 5e-4 on CUDA (cuDNN kernel nondeterminism). Reference
# measured ~9e-08 on CPU. This single number is what licenses the sentence
# "same weights, two execution paths" on slide 3. If it fails on GPU only:
# try torch.backends.cuda.matmul.allow_tf32=False for the check.
#
# DONE WHEN: prints the diff and PASSES; wire it so figure2/table7/engines
# refuse to run (or loudly warn) if preflight hasn't been run this session.


# =============================================================================
# TODO-14 -- figure2: per-aggregator lr sweep (paper D.1, pragmatic version)
# =============================================================================
# Selection criterion = MINIMUM AREA UNDER THE LOSS CURVE (sum of per-iter
# mean losses). This is the paper's own criterion (D.1) -- it rewards fast AND
# stable; a diverging lr self-disqualifies via huge AUC. Grid: geometric,
# 0.003 / 0.01 / 0.03 / 0.1 / 0.3 (5 points; the paper used 22 coarse + 50
# refined = 72 runs/aggregator -- you state the simplification on the slide,
# you do not replicate it). Each candidate: fresh init from TODO-06.
# Guard: reject non-finite curves before comparing AUC.
# Budget check: 5 lrs x 4 aggs x 20 epochs at paper-L4-like ~2 s/epoch ~= 15
# min. If your GPU is slower than 2x that, drop to 3 lrs {0.01,0.03,0.1}.


# =============================================================================
# TODO-15 -- figure2: final runs + the Slide-1 plot
# =============================================================================
# For each aggregator at its best lr: fresh init, full paper horizon
# (CIFAR-10: 20 epochs = 640 iters), record per-iteration mean batch loss.
# Plot: EMA-smoothed curves, log-y, label = "AGG (lr=...)", title carries
# dataset / 1024 imgs / batch 32 / paper CNN. Dump raw curves to JSON next to
# the PNG -- Rui may ask for the unsmoothed data, and re-plotting beats
# re-running.
# Interpretation targets (paper §5 + Fig 2): UPGrad below Mean; MGDA BAD is
# CORRECT (small-gradient pathology, weak-stationarity trap) -- do not debug
# a matching result.
#
# DONE WHEN: figure2_cifar10.png exists and UPGrad's curve sits below Mean's.


# =============================================================================
# TODO-16 -- table7: ratio table
# =============================================================================
# Rows: SGD-ERM, then each aggregator via the AUTOJAC path (that is what the
# paper's Table 7 measured -- running these through autogram would be a
# category error for this deliverable). All timed with TODO-10, same warmup/
# timed counts. Emit markdown: method | yours s/ep +/- std | yours ratio
# (Mean=1) | paper ratio | comment. Paper ratio column from TODO-01 constants.
# Prepared hypothesis for deviations (have this ready BEFORE Tuesday):
# torchjd 0.17's QuadprogProjector solves the m QPs SEQUENTIALLY on CPU in
# float64 via np.apply_along_axis (see _linalg/_dual_cone.py), including a
# GPU->CPU->GPU round trip per step. If your UPGrad/MGDA ratios differ from
# the paper's, the QP path and the transfer are the first suspects -- and
# that observation IS the Phase-3 QP-branch evidence, filed with line numbers.


# =============================================================================
# TODO-17 -- engines: Deliverable C (the money slide)
# =============================================================================
# Three configs, identical init/data/order: SGD-ERM | autojac+UPGrad |
# autogram+UPGradWeighting. Time + peak memory via TODO-10. Paired bar chart
# (time panel, memory panel), speedup ratio printed into the title.
# Sanity anchor from the reference (CPU, synthetic!): 0.24 / 6.2 / 0.47 s/ep.
# The qualitative shape -- autogram lands near SGD-ERM, autojac is the
# outlier -- must survive on GPU; the magnitudes will change.
#
# DONE WHEN: engines_cifar10.png shows that shape with real GPU memory bars.


# =============================================================================
# TODO-18 -- scaling (STRETCH, time-boxed 2h, only after A-C are banked)
# =============================================================================
# width_mult in {1,2,4,8}: for both engines, time + peak mem at batch 32.
# WRAP EACH CONFIG in try/except torch.cuda.OutOfMemoryError: an autojac OOM
# is not a failure, it is the best datapoint in the deck ("autojac dies at
# X.XM params on this GPU; autogram is flat"). torch.cuda.empty_cache() after
# a catch or the next config inherits the fragmentation.
# Plot: peak MiB vs params (millions), one line per engine.


# =============================================================================
# TODO-19 -- CLI (learn from the reference's bug so you don't re-earn it)
# =============================================================================
# Subcommands: preflight | figure2 | table7 | engines | scaling.
# ARGPARSE FOOTGUN (hit during reference development): a global
# --aggs nargs="+" on the TOP-LEVEL parser swallows the subcommand token
# ("--aggs Mean UPGrad figure2" parses figure2 as an aggregator and dies).
# Fix: put shared options on a parent = argparse.ArgumentParser(add_help=False)
# and pass parents=[parent] to EVERY subparser, so all flags go AFTER the
# subcommand:  python bench.py figure2 --dataset cifar10 --aggs Mean UPGrad
#
# DONE WHEN: every command in RUNBOOK §2 parses and runs on --dataset synthetic.


# =============================================================================
# FINAL SELF-CHECK before making slides (10 min):
#   [ ] preflight passes ON THE GPU BOX (not just your laptop)
#   [ ] figure2: UPGrad < Mean; MGDA bad-as-expected; lr per curve in legend
#   [ ] table7: ratios within ~2x of paper's; deviation hypothesis written down
#   [ ] engines: autogram ~ SGD-ERM, autojac the outlier; memory bars real
#   [ ] every PNG regenerable from JSON without re-running training
#   [ ] you can say the m=32 vs m=4-8 caveat in one breath, from memory
# =============================================================================
