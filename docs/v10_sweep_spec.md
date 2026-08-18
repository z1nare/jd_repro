# v10 sweep — specification

**For:** Rui (project lead), Luo Mai
**Author:** Arsenii
**Date:** 2026-08-07
**Status:** draft for discussion

---

## 0. What this document is

Rui asked for a map of the variables worth exploring, with the constraint *"how to
minimize the number of experimentation while get to the conclusion."* This is that map:
the axes, the argument for each one that is dropped, and a run list of 8 invocations
(~9 minutes on one GPU) ordered so that stopping early still leaves a usable answer.

It assumes no knowledge of the repository. Terms are defined where first used.

---

## 1. The target, and where we are against it

Rui set two numbers:

- **Memory ≤ 1.5×** a single-objective run. *"four to six GPUs is okay, four to eight is
  a problem."*
- **Time ≤ 1.5×** at 2–3 objectives. *"we do not want the training time to be doubled or
  tripled. We want it kind of like 50% more."*

Measured against the one true single-objective control in the harness (plain SGD on the
same model, data and seed), GPT-2 124M, one RTX A5000, fp32:

| engine | time | memory |
|---|---|---|
| **jdgram** (ours) | **2.22×** | **1.00×** |
| TorchJD autogram | 3.19× | 1.66× |
| TorchJD autojac | 4.00× | 2.47× |

Source: `results/v9_20260731-154714_124m-aggregators_v9-cleanup/rows.csv`, m=4, T=128,
100 steps, UPGrad aggregator. Peak memory 1726.959 MiB for ours against 1726.960 MiB for
plain SGD.

**Memory passes. Time does not.** Both TorchJD engines fail the memory budget outright.

### 1.1 The structural obstacle

Our engine runs the model forward **twice** per training step: once to compute the
Gramian (the matrix of pairwise gradient similarities that the optimizer consumes), and
once more for the weighted backward pass.

Substituting a *free* Gramian — zero cost for all the per-layer mathematics — still
leaves **1.36–1.41×**. The budget is 1.50×. So **72–82% of the entire allowed overhead is
spent before any of the machinery we have been optimising even runs.**

This is why the sweep below leads with a single measurement rather than a grid. It also
means kernel-level optimisation cannot reach the target on its own, and something
structural has to change. That decision is §6.

---

## 2. The one axis that has never been measured

Every result we have so far grows the objective count and the batch size **together** —
in the current code an "objective" is a sequence in the batch, so `m` objectives means
`m` sequences. Doubling `m` therefore doubles the work as well as the objective count,
and the resulting 2.2×–3.4× spread cannot separate the two.

**The v10 sweep pins total tokens and varies only `m`:**

| m | T | m·T |
|---|---|---|
| 1 | 1024 | 1024 |
| 2 | 512 | 1024 |
| 4 | 256 | 1024 |
| 8 | 128 | 1024 |

This has predictive content, not just bookkeeping. One of our two contraction strategies
has a cost that is **exactly constant in `m` at fixed `m·T`**; the other carries an `m²`
term. So:

- **flat line** → Rui's rationale holds (*"most of the things are still parallelizable"*),
  and the overhead is a fixed cost independent of how many objectives you add.
- **rising line** → the cost is genuinely per-objective, and we can name the term.

`m=1` is the single-objective control and **has never been run.**

---

## 3. Axes, and the argument for each drop

| Axis | Values | Swept? | Argument |
|---|---|---|---|
| Objectives `m` at pinned tokens | 1, 2, 4, 8 | **Yes** | §2. The only axis that isolates objective count from work volume. |
| Contraction strategy | auto, seq-first, param-first | **Yes — free** | All three measured inside one process at ~0.3 s each. |
| Engine | ours, autogram, autojac, plain SGD | **Yes — free** | Three engines per invocation; SGD control adds one cell. |
| Layer shape | 5 shapes × 2 strategies | **Yes — 2 runs** | §5. Answers the hybrid question from two profiler traces. |
| Objective correlation | independent, duplicated | **Yes — at m=2** | Rui asked for the correlated case as a sanity ladder before conflicting objectives. |
| `m = 3` | — | **No** | Bracketed by m=2 and m=4, which differ by 0.25 in ratio. No value it can take flips a verdict against 1.5. |
| `m ≥ 16` | — | **No** | Already measured at 124M, and outside the stated criterion. |
| `m` and `T` independently | — | **No** | Already established that `m·T` is the scaling variable: (16,512) and (8,1024) gave 12057.9 vs 12059.3 MiB. Sweeping separately re-measures one number at 3× the cost. |
| Precision (bf16) | — | **No** | No bf16 path exists — the model builder's dtype argument is not passed by any of its ten call sites. Flagged in §7. |
| Optimizer state (Adam) | — | **No** | Plain SGD is the *adversarial* setting for a memory ratio: Adam adds state to numerator and denominator alike and dilutes every ratio toward 1.0. Our 1.00× is a lower bound. |
| Layer-group factorial (2ᵏ) | — | **No** | Per-layer contributions are summed over disjoint parameter blocks, so the cost function is separable and the best combination is a per-group minimum readable off a table. Derived, not measured. |
| QP solver | — | **No** | Measured at 0.40–0.93 ms, under 0.2% of a step. Closed on sight. |

---

## 4. The run list

Two model configurations.

**CONFIG-A** — cost measurements. 6 layers, 6 heads, width 384, vocabulary 12288 (~15.7M
parameters). Head dimension is 64, matching GPT-2 124M exactly, and the output head is
30.8% of parameters against 124M's 31.2%. **That match is the point:** the output head is
the single largest mis-routed component in the engine, and a toy 65-symbol vocabulary
would shrink it to ~4% and hide the biggest effect.

**CONFIG-B** — quality and sanity. 4 layers, 4 heads, width 256, vocabulary 65 on the
character-level Shakespeare corpus (~3.2M parameters). The only configuration where
held-out loss means anything, because it is the only vocabulary the corpus has.

```
R1  [A]  m=1  T=1024   cost + training      ~80 s
R2  [A]  m=2  T=512    cost + training      ~80 s
R3  [A]  m=8  T=128    cost                 ~20 s
R4  [A]  m=4  T=256    cost      CONDITIONAL ~20 s
R5  [A]  m=4  T=256    profiler trace, seq-first    ~15 s
R6  [A]  m=4  T=256    profiler trace, param-first  ~15 s
R7  [B]  m=2  T=512    training, independent objectives   ~90 s
R8  [B]  m=2  T=512    training, duplicated objectives    ~90 s
```

Total ~9 minutes including process startup. For contrast, the previous campaign spent
330 s on a single configuration of one of these levels.

Note on iteration speed: at these model sizes **process startup (5–10 s of imports and
CUDA context) dominates the model.** Speed is gated by invocation count, not parameter
count. This is why the list is 8 runs rather than 30 small ones.

---

## 5. What each prefix concludes

**After R1 — the floor, measured rather than inferred.** At `m=1` the Gramian is a 1×1
matrix and the aggregator returns weight 1. Everything our engine spends over plain SGD
is therefore pure fixed architectural cost — no per-layer mathematics, no aggregation, no
solver. This is the cleanest possible isolation of the two-forward overhead.

| R1 outcome | Meaning |
|---|---|
| ≥ 1.5× | The architecture is capped above budget. No amount of kernel tuning reaches the target. Go directly to §6. |
| ≈ 1.36–1.41× | Confirms the predicted floor, and gives an exact headroom figure for the kernel work. |
| < 1.3× | The floor model is wrong and the design is cheaper than predicted. Re-derive before building anything. |

**After R1+R2 — the acceptance verdict** at `m=2`, the objective count Rui named, against
a real control rather than a reconstruction.

**After R1+R2+R3 — fixed versus per-objective overhead**, both ends at identical token
budget. This is the §2 question.

**After R5+R6 — the layer-shape answer**, from two profiler traces and no new code. The
profiler groups operations by tensor shape, so per-layer-type cost is readable directly
under both strategies.

**After R7+R8 — correctness and the sanity ladder.** R7: every aggregator and engine must
land on the same held-out loss within noise; a gap is a bug, not a trade-off, and it voids
every cost number above it. R8: with two duplicated objectives the per-objective loss
curves must track each other exactly, the 2×2 Gramian must be rank-1, and UPGrad must
reduce to plain averaging. This is the ladder Rui asked for before anyone runs conflicting
objectives.

### 5.1 Stopping rules

- **Kill the `m` axis after R1** if the ratio already exceeds 1.5. The overhead then has
  nothing to do with having multiple objectives, and R2–R4 answer a question that no
  longer decides anything.
- **Skip R4** if the two ends agree within 10%. Three points cannot show a shape that two
  ends plus a closed-form cost model already determine.
- **Never read a difference under 5% from any single cell.** Every configuration is n=1;
  the one configuration we replicated showed 1.8% spread.

---

## 6. The decision this sweep is for

Our engine's second forward exists for a reason: it lets the framework free intermediate
activations as the reverse pass proceeds, which is where most of the memory advantage
comes from. TorchJD's autogram makes the opposite choice — it retains the graph, runs one
forward, and pays 1.66× memory for it.

**The second forward and the memory advantage are the same design decision.** We are at
one end of a memory-for-time dial; autogram is at the other:

| | forwards | memory | time |
|---|---|---|---|
| ours | 2, retains nothing | **1.00×** | 2.22× |
| autogram | 1, retains the graph | 1.66× | 3.19× |

Rui's criteria say memory is fine and time is not. **We are parked at the wrong end of our
own dial for the criterion we are being judged against**, with 0.26–0.50× of memory budget
unused and no time budget at all.

Moving along it projects to **1.85–2.15×**, and a further ~1.3× from the kernel work then
reaches 1.50×. That is currently the only identified path to budget.

**This is an architecture change and it needs a decision, not an assumption.** Note that
autogram is slower despite one forward, because it recomputes each module's forward
functionally — so the projection assumes we keep our captured intermediates and only stop
re-running the model. The proposed test is a half-day spike on a branch to get the number,
not a refactor.

---

## 7. Open questions

1. **The fusion decision in §6.** Architecture, not tuning. Needs a go/no-go.
2. **The acceptance criteria are stated in units we cannot test here.** Rui denominated
   the budget in H100 counts for a reinforcement-learning run — which implies bf16, Adam,
   and sharded training. We measure fp32, plain SGD, one A5000, unsharded. What ports is
   the fitted routing rule and the structural two-forward floor; the absolute time ratio
   does not. Worth agreeing explicitly what we are certifying.
3. **A correction on the layer-shape hypothesis.** Rui suggested our approach may win on
   the feed-forward layers and lose on attention, with a fallback to autogram there. In
   this codebase there is no separate attention path to fall back from: attention's only
   parameters are two ordinary linear layers, and the attention operation itself has no
   parameters and is never intercepted. Every projection in the model — attention and
   feed-forward alike — goes through the same handler and differs only in shape. So the
   question reduces to a shape-driven strategy choice, which R5/R6 answer in two runs.
   The related suggestion to stop using the Hadamard factorisation on attention is already
   true: that code path has no call sites anywhere in the transformer.
4. **The more useful version of that question.** All the dense shapes in a transformer
   cross over within a single octave of each other, so one global strategy is close to the
   best possible per-layer choice. Low-rank adapter shapes cross **~30× away**. If
   per-layer strategy is going to pay anywhere, it is on LoRA / MoE structures, not on
   attention. Worth deciding whether that is in scope.
5. **Metric.** Held-out loss at these sizes is an engine-agreement check, not a quality
   result — the only corpus available is 65-symbol character-level Shakespeare. An
   absolute quality number needs a real tokenised corpus. Is that in scope now, or after
   the cost work?

---

## Appendix — prerequisites and their status

| # | Item | Status |
|---|---|---|
| 0.1 | Sequence-length clamp silently capping the training level at 256, which would have blocked R1 and R2 | **Fixed and verified** — a run at 1024 now passes through |
| 0.2 | Corpus present so held-out loss is meaningful | **Present on the cluster** (confirmed in the v9 run record). Absent locally; local machine only runs the CPU correctness suite, which does not need it |
| 0.3 | Per-objective loss curves — previously only a mean scalar was recorded, so "do two objectives track each other" was unanswerable | **Implemented and verified** — written per run as a JSON sidecar |
| 0.4 | Perplexity — the metric Rui named appeared in no artifact; held-out loss was stored in nats and never exponentiated | **Implemented and verified** |
| 0.5 | Raw profiler traces not enabled, leaving all kernel-level analysis empty on all 20 previous runs | Outstanding — needed for R5/R6 depth, not for the acceptance numbers |
| 0.6 | Two analysis defects: a section that is written under one key and read under another, and per-kernel occupancy data that is captured and then discarded | Outstanding |
| — | Correctness suite | **44/44 passing** after the above changes |

**One finding from checking the previous campaign:** the earlier accuracy run is void.
All four engines recorded a loss flat at ~10.97 from first step to last (a total movement
of 0.005), against a floor of 10.82 for that vocabulary — the batches had no relationship
between input and target, so there was nothing to learn. The cause is fixed; the numbers
should not be cited.
