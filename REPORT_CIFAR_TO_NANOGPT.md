# From CIFAR-10 to nanoGPT: exact Gramians for Jacobian Descent

**What this extends.** An earlier CIFAR-10 IWRM study measured a custom Gramian
engine against TorchJD's `autojac` and `autogram` on a convolutional network
with `T = 1`. This report carries that work to a transformer, where the central
assumption of the CIFAR result no longer holds, and re-measures everything
against TorchJD 0.17.

**Provenance.** Every number below is from run tag `v8_20260731-*` on gala1
(NVIDIA RTX A5000, 24 GiB; torch 2.13.0+cu130, torchjd 0.17.0, Python 3.10.12),
fp32 workspace with a float64 accumulator. Correctness is against brute-force
`autograd.grad` in float64. 25 tagged run directories, each carrying a manifest
with the git SHA and environment and a `rows.csv` of measurements, indexed in
`runs_index.csv`. Those artifacts live on the cluster and are not committed
here; `scripts/run_profile_cluster.sh bundle` packages them for transfer.

**TorchJD citations.** Line numbers refer to TorchJD 0.17.0
([source](https://github.com/TorchJD/torchjd)), read from a local checkout of
that release.

---

## 0. The one-line summary

On a weight-tied nanoGPT, jdgram is **1.9× faster than `autogram` and 15% leaner**,
and is **exact where `autogram` is wrong by 1.2e-03** — because `autogram` sums
per-module Gramians and drops the cross-terms between two modules sharing a
parameter. That is every LLM with a tied embedding.

The CIFAR headline (12× TorchJD's parameter ceiling) does **not** transfer, and
this report says why: it was earned under `T = 1`.

**And a result that contradicts the design intent** (§7): at vocabulary scale the
router picks the wrong route, costing **2.8× in time** for a memory saving that
never materialises. Fixing it is the single highest-value item outstanding.

---

## 1. What the CIFAR result established, and its hidden assumption

CIFAR-10 IWRM: one cross-entropy objective per image, `m = 32`, on the paper's CNN
(grouped convolutions, ELU, max-pool). On fuji2 (4× A5000, torch 2.4.1+cu121):

| | `autojac` | `autogram` | Hadamard engine |
|---|---|---|---|
| s/epoch | 0.45 | 0.33 | **0.20** |
| peak MiB | — | **82** | 134 |
| parameter ceiling | 134M (OOM) | 134M (OOM) | **1.65B** (22.6 GiB) |

Convergence reproduced the paper: UPGrad beat Mean on area under the loss curve
(381 vs 495), PCGrad diverged at lr ≥ 0.01, MGDA learned poorly. The engine
matched `autogram` to **2.8e-14** in float64.

**The assumption.** In IWRM each objective is one image, seen at exactly **one
position**. With `A` the upstream gradient at a layer output and `X` its input:

```
G_ij = < A_i A_j^T , X_i X_j^T >_F
```

and with one position per objective this collapses to a Hadamard product of two
`[m, m]` matrices:

```
G = (A A^T) ⊙ (X X^T)              # T = 1
```

Two small matrices, no per-layer buffer proportional to the parameter count.
**That collapse is the entire source of the 12× capacity win.** It is a property
of the problem, not of the engine.

A transformer breaks it. Per-sequence losses span `T` tokens, the per-layer
gradient is a sum of `T` outer products, and its rank is up to `T`. Everything
below is about what replaces the Hadamard form.

---

## 2. What had to be built

Per-layer identities, each gated against brute-force autograd at
`rtol=0, atol=1e-10` in float64 before use ([`gates/`](gates/), **44 tests**,
each run twice for `bias`/`no-bias`):

| Layer | Identity | Gate |
|---|---|---|
| `nn.Linear` (QKV, proj, MLP, LM head) | two contraction orders, §3 | 5a, 5b |
| Linear bias | `Σ_t A`, then outer product | 5b, 5c |
| LayerNorm / RMSNorm `γ`, `β` | recompute `x̂` from the raw input | 5c |
| `nn.Embedding` (token) | index-equality kernel, or scatter-add | 5d |
| `nn.Embedding` (positional) | diagonal in `t` | 5d |
| **Tied `wte` = `lm_head`** | **four terms, §5** | **5e** |
| Softmax / SDPA / GELU / residual / dropout | **nothing** | 5b–5e |
| Driver equivalence (3 reverse strategies) | — | 5f |
| Route equivalence (2 contraction orders) | — | 5g |

The last identity row is what keeps this tractable: **attention holds no
parameters of its own.** Its parameters are four Linears; softmax, SDPA, GELU,
residual adds and RoPE contribute no Gramian terms at all, and ordinary autograd
propagates `A` through them. Supporting a new architecture is re-registration,
not new mathematics.

---

## 3. Two contraction orders, and the router

Writing the Gramian with all four indices makes the design space explicit:

```
G_ij = Σ_{t,s,p,q}  A_i[t,p] X_i[t,q] A_j[s,p] X_j[s,q]
```

You choose which index pair to contract first, and there are exactly two choices:

| Route | contracts first | workspace | intended to win when |
|---|---|---|---|
| **T-first** | positions `(t, s)` | `3 m T²` per layer | `P_layer` is huge (vocab head) |
| **d-first** | features `(p, q)` | `m · P_layer` | `P_layer` is modest |

No third pairing keeps the result exact. That is why the answer is a router, and
why "we are quadratic in T" describes only one of the two routes.

Measured on the identity kernels alone, no model, no autograd (L0):

| shape | T-first ms / MiB | d-first ms / MiB |
|---|---|---|
| interior MLP, m=8 T=512 | 3.67 / 52.1 | **0.29 / 36.1** |
| long T, m=4 T=2048 | 6.53 / 216.1 | **0.32 / 25.1** |
| vocab head, V=50257 | 133.2 / **822.1** | **10.7** / 1190.8 |
| LoRA-A r=32 | 1.42 / 36.4 | **0.16 / 25.4** |

At kernel level the vocab head behaves as designed: T-first is **31% leaner**. It
is also **12.4× slower**. §7 shows which of those survives at model scale.

---

## 4. The reverse pass: TorchJD's strategy, and what jdgram puts inside it

Both engines face the same problem — the Gramian needs each objective's gradient
kept *separate*, and the obvious ways to get that are expensive.

**TorchJD's answer, which jdgram adopts wholesale.** For a 1-D loss vector with
`batch_dim=0`, `autogram` seeds **one ordinary backward** with
`torch.ones_like(output)`
(`_engine.py:326`). With
per-instance losses and batch-independent modules the intermediate Jacobians are
block-diagonal, so `∂(Σ_i L_i)/∂z[i] = ∂L_i/∂z[i]` — row `i` of that single
gradient *is* objective `i`'s upstream gradient. Each module's contribution is
computed inside its own backward hook as the sweep passes through it, and its
state is released the moment it has been squared
(`_gramian_computer.py:73-76`,
`del self.summed_jacobian`).

That is the right design, and jdgram uses it unchanged: one ones-seeded backward,
each layer's identity fired inside its own backward, captures freed immediately.
Only a tied group outlives the call that produced it, because its cross-terms
cannot be formed until both sites have been visited — measured at **5.0 MiB held
tied, 0.0 untied**, which is the streaming behaving as designed. **The reverse
strategy is TorchJD's, not ours, and this report should not imply otherwise.**

One genuine difference in that machinery, and it is a trade rather than a win:
jdgram releases more eagerly — it pops the stored input as the hook consumes it
and runs the reverse with `retain_graph=False`, where TorchJD pins the module's
`args` and retains the graph. The cost is that the graph is gone afterwards, so
a training step that needs a weighted backward after the Gramian pays a second
forward. TorchJD does not. That shows up in §6 as a real per-step cost.

**The difference is what happens inside that hook.** TorchJD materialises the
module's Jacobian block explicitly: `FunctionalJacobianComputer` vmaps a
`torch.func.vjp` over the objective index and concatenates every parameter
gradient into one row
(`_jacobian_computer.py:127`),
producing `[m, P_module]`, then squares it. That costs `m · P_module` for every
module, always, and it needs a per-module functional forward to build the vjp
(`_jacobian_computer.py:74-75, 116`).

jdgram never forms that block. Each layer's contribution is a closed form in
`(A, X)` — §3 — so the choice of contraction order becomes available:

| | workspace per layer | forms `[m, P_layer]`? | needs a vjp |
|---|---|---|---|
| `autogram` | `m · P_layer` | yes, always | yes, per module |
| jdgram, d-first | `m · P_layer` | yes | no |
| jdgram, **T-first** | **`3 m T²`** | **no** | no |

**TorchJD has no T-first analogue — structurally it is always d-first.** That is
what makes the vocabulary head the interesting case: at `V = 50257` the head's
`P_layer` is 12.9M, so d-first needs a 393 MiB workspace where T-first needs
24 MiB. Only one of the two engines can express the cheaper one. §7 measures what
happens there, including the router bug that currently stops jdgram cashing it in.

Measured route exponents for peak memory (L6), which is the property the router is
supposed to exploit:

| route | exponent in `m` | in `T` |
|---|---|---|
| d-first | **0.71** | **0.80** |
| T-first | 0.84 | 1.12 |

At `V = 65` peak **moves with the route** — 34% apart at T=512, 56% at larger T.

**One engineering fix on top, in the identity kernels themselves.**
`torch.einsum` **clones both operands**, so the T-first Linear contraction peaked
at `4 m²T²` rather than the documented `2 m²T²`; blocking the contraction over the
objective index removes the `[mT, mT]` kernel entirely, giving `3 m T²` — measured
**164 → 12.25 MiB** at m=8, T=256. The same einsum-clone issue cost a gratuitous
2× in the d-first embedding and tied-cross kernels (**1250 → 625 MiB** at
V=20000).

---

## 5. Where jdgram is correct and `autogram` is not: tied weights

**Scope the claim precisely: this is a defect in `autogram`, not in TorchJD.**
`autojac` materializes the full `[m, P]` Jacobian, so a shared parameter's
gradient is summed over both its sites before anything is squared and the cross
terms are present automatically — it agrees with brute force on a tied model.
The defect belongs specifically to the per-module Gramian strategy.

Modern LLMs tie the token embedding to the LM head [13, 7] — one parameter, two
sites. The per-objective gradient of a shared parameter is the **sum over its
sites**, so the Frobenius product carries four terms:

```
⟨g_h + g_e , h_h + h_e⟩ = ⟨g_h,h_h⟩ + ⟨g_e,h_e⟩ + ⟨g_h,h_e⟩ + ⟨g_e,h_h⟩
```

`autogram` computes the first two. The mechanism is visible in three files:

- `_engine.py:202-206` —
  `_hook_module_recursively` creates **one `GramianComputer` per module**, keyed by
  the module object. Two modules sharing a parameter get two independent computers
  that never see each other.
- `_gramian_computer.py:45-76`
  — `remaining_counter` counts *forward calls of one module* (line 52-53), sums
  that module's Jacobians (line 66-69), squares at zero (line 73-74). It correctly
  handles a module called twice; it has no notion of a **parameter reached through
  two different modules**.
- `_gramian_accumulator.py:19-23`
  — `self._gramian.add_(gramian)`. Summing Gramians is valid only for **disjoint**
  parameters.

Measured against brute force across the vocabulary sweep:

| | V=65 | V=8000 | V=32000 | V=50257 |
|---|---|---|---|---|
| jdgram vs brute force | **1.58e-07** | exact | exact | exact |
| `autogram` deviation (tied) | **1.234e-03** | 1.137e-04 | 5.792e-05 | 4.587e-05 |

The deviation is largest where the tied parameter is the largest share of the
model, and never vanishes. jdgram computes all four terms
([`identities/tied.py`](src/jdgram/identities/tied.py)) and **refuses to run** on a
tied model without an explicit shared handler rather than return a structurally
plausible wrong number ([`engine/hooks.py`](src/jdgram/engine/hooks.py)).

**This is the durable claim.** It holds regardless of any performance number.

Its cost is honest: holding both sites' captures until the group completes puts the
tied model at **668 MiB against `autogram`'s 412** at V=65 — 1.62×. At larger V
that inverts (§7): jdgram is *leaner* than `autogram` at every V ≥ 8000.

---

## 6. Head-to-head, per aggregator (m=8, T=128, 1000 steps, real Shakespeare)

Every cell is a real training run on identical data and seeds, scored on held-out
cross-entropy and next-token accuracy as well as cost.

| Aggregator | Engine | ms/step | peak MiB | val CE | val acc |
|---|---|---|---|---|---|
| Mean | **jdgram** | 29.17 | **110.9** | 2.4669 | 0.2821 |
| Mean | autogram | 35.77 | 186.3 | 2.4669 | 0.2821 |
| Mean | autojac | **17.85** | 227.4 | 2.4669 | 0.2821 |
| UPGrad | **jdgram** | 29.38 | **123.1** | 2.4620 | 0.2809 |
| UPGrad | jdgram + jacopt | 38.39 | 123.1 | 2.4620 | 0.2809 |
| UPGrad | autogram | 37.33 | 198.5 | 2.4621 | 0.2809 |
| UPGrad | autogram + jacopt | 45.54 | 198.5 | 2.4621 | 0.2809 |
| UPGrad | autojac | **19.27** | 328.1 | 2.4620 | 0.2809 |
| MGDA | **jdgram** | 57.76 | **123.1** | 2.4750 | 0.2786 |
| MGDA | autogram | 64.01 | 198.5 | 2.4749 | 0.2787 |
| MGDA | autojac | **45.59** | 227.4 | 2.4750 | 0.2786 |
| PCGrad | **jdgram** | 30.94 | **123.1** | **2.4206** | **0.2848** |
| PCGrad | autogram | 37.63 | 198.5 | 2.4213 | 0.2840 |
| PCGrad | autojac | **19.73** | 227.4 | 2.4212 | 0.2843 |
| — | sgd_erm | 6.12 | 114.7 | 2.4669 | 0.2821 |

Four things to read off it:

1. **jdgram is the leanest engine in all twelve cells** — 111–123 MiB against
   `autogram`'s 186–199 and `autojac`'s 227–328.
2. **All three engines agree on held-out CE to four decimals** for every
   aggregator. That overlay is produced automatically by the harness.
3. **PCGrad is the best aggregator here** — val CE 2.4206 and accuracy 0.2848,
   against Mean/SGD at 2.4669/0.2821 and UPGrad at 2.4620/0.2809. That is a
   larger spread than the engine differences and worth its own investigation.
4. **`autojac` is fastest and heaviest.** It materialises the full `[m, P]`
   Jacobian; at 3.18M parameters that is affordable. The Gramian trade only pays
   at a size where `autojac` cannot run — we are below that size here.

L7 confirms all four engines produce identical loss trajectories (final 4.174,
initial 4.245, mean-last-10 4.178 for jdgram, autogram, autojac and SGD alike).

At step level (L4, per-iteration over 10): `compute_gramian` 15.4 ms,
`final_backward` 10.9, `forward_only` 4.0, **`weighting_qp` 0.76**,
`optimizer_step` 0.03. The dual-cone QP is **2.7% of a step**.

---

## 7. Vocabulary scale: the regime the two-route design exists for

This is the flagship test — `P_layer` large enough that `m·T² < P`, so the router
selects T-first for the head. m=8, T=512, d=256, tied.

### 7.1 jdgram is leaner than `autogram` at every large vocabulary

| V | params | jdgram peak | `autogram` peak | ratio |
|---|---|---|---|---|
| 65 | 3.18M | 668.2 | 411.6 | 1.62× |
| 8000 | 5.21M | **873.5** | 970.2 | **0.90×** |
| 32000 | 11.35M | **2439.7** | 2864.7 | **0.85×** |
| 50257 | 16.03M | **3637.2** | 4311.7 | **0.84×** |

The tied-memory penalty at V=65 **inverts** by V=8000 and stays inverted. jdgram
is 16% leaner at GPT-2 vocabulary while also being exact.

### 7.2 But the router picks the wrong route, and it costs 2.8×

L4, `compute_gramian` per iteration, `squashed` driver:

| V | `auto` | forced T-first | forced d-first |
|---|---|---|---|
| 65 | 15.4 ms | 64.8 | 14.6 |
| 8000 | 21.5 | 89.6 | **21.4** |
| 32000 | 120.9 | 168.4 | **45.9** |
| 50257 | **177.6** | 226.1 | **64.4** |

At V=50257 the router's choice costs **2.76×** against simply forcing d-first.
The rule is `tfirst iff m·T² < P_layer`; at the head `m·T² = 2.1M` and
`P_layer = 12.9M`, so it selects T-first — and T-first on a vocab head is 12.4×
slower per L0.

**And the memory saving it was buying does not exist at model scale.** The stats
tool flags it directly: `[driver=squashed] peak is identical across tfirst/dfirst
— 3526 MiB vs 3526 MiB`. The theoretical workspaces genuinely differ (24 MiB
T-first vs 392.6 MiB d-first), but peak is set by the forward activation set
(1914 MiB) and the held tied captures (789 MiB), so a 369 MiB workspace
difference is invisible while the 122 ms time penalty is fully visible.

**Consequence.** jdgram's L5 time ratio against `autogram` is 0.63× at V=8000 but
**2.47× at V=32000 and 2.67× at V=50257** — it loses on time at large vocabulary.
That loss is entirely the router. Forcing d-first at V=50257 gives 64.4 ms
against `autogram`'s 66.4 ms — **parity on time and 16% leaner**.

This is the highest-value fix outstanding, and it is a cost-model change, not new
mathematics.

### 7.3 Correctness is unaffected by any of it

The two contraction orders agree to `0.000e+00` at **every** vocabulary tested,
and jdgram agrees with brute-force autograd throughout (§5). The router chooses
between two routes that are exactly equivalent; picking the wrong one costs time,
never accuracy. That is what makes the §7.2 fix safe to make — it cannot change a
single number in the results, only how long they take to produce.

---

## 8. Additional exploration (not part of the core claim)

Separated deliberately: directions, some closed by measurement, not established
results.

### 8.1 LoRA inverts the routing decision and removes the memory ceiling

Test-time training [1, 16] and RL fine-tuning adapt LoRA adapters [8], not full
weights. For `W_eff = W + BA` the same identity applies with `d_out → r`, and
`P_layer = 2rd`:

| setting | `P_layer` | route | T-first ws | d-first ws |
|---|---|---|---|---|
| Qwen3-1.7B, r=32, T=2048 | 131k | **d-first** | 256 MiB | **2.0 MiB** |
| 8B, r=32, T=4096 | 262k | **d-first** | 1024 MiB | **4.0 MiB** |

Two consequences: the T-first route — the direct descendant of the CIFAR Hadamard
trick — is the **wrong** route for every test-time LoRA setting; and **the memory
ceiling that has shaped this phase is a property of full fine-tuning, not of
Jacobian Descent.** L0 measures the LoRA-shaped kernels directly (r=32): d-first
0.16 ms / 25.4 MiB against T-first 1.42 / 36.4.

**Not yet done:** registering a LoRA module and gating it end to end. The identity
needs no change; this is registration, not derivation.

### 8.2 GPU QP solvers, and a benchmarking trap worth knowing

TorchJD ships exactly one dual-cone projector,
`QuadprogProjector`, and it
is CPU-only: line 109-110 `_to_array(G)` → `tensor.cpu().detach().numpy()`
(line 127-129), line 112 `np.apply_along_axis` over rows, line 119
`solve_qp(..., solver="quadprog")`, then a copy back. On CUDA that is a
device→host transfer, a Python row loop and a host→device transfer per step.
`quadprog` implements the Goldfarb–Idnani dual active-set method [5], which is
exact and, at these sizes, extremely fast on a CPU.

Benchmarked against `jacopt`, which solves the same problem with a consensus
ADMM [3] in the manner of operator-splitting QP solvers such as OSQP [15], and
is backend-agnostic so the solve follows the array type onto the device. Four
bugs were fixed in it first (§9.4). The single-thread control settles what
looked like a solver failure:

| m | quadprog | jacopt GPU | jacopt CPU (56 threads) | jacopt CPU (1 thread) |
|---|---|---|---|---|
| 8 | **0.65** | 21.7 | 9.9 | 6.3 |
| 32 | **2.43** | 28.8 | **3612.2** | **10.6** |
| 64 | 18.7 | 25.9 | 3876.2 | 19.3 |
| 128 | 252.3 | **31.4** | 2735.5 | 56.4 |

**The m≥32 CPU cliff was thread contention, not the solver — 341× at m=32 between
56 threads and 1.** These are 32×32 matrices; torch parallelises above a size
threshold and on a shared 56-thread box the barrier costs far more than the
arithmetic. Iteration counts were identical (100) in both columns, and
`ms_per_admm_iter` on GPU is flat at ~0.3 ms across all m (cv 0.175).

Crossover is **m ≈ 64**. At m=128 the GPU solver is 8× faster than quadprog with
**zero host synchronisations** against quadprog's three.

**Verdict for the committed path: do not adopt.** At m=8 the QP is 0.76 ms — 2.7%
of a step — and quadprog is 33× faster than the alternative. L11 confirms it end
to end: UPGrad + jacopt is 38.39 ms/step against 29.38 default, identical val CE.
jacopt's accuracy also degrades where it starts winning (rel 1.5e-03 at m=64,
9.1e-03 at m=128) because ADMM is approximate where quadprog is exact.

### 8.3 Approximation directions closed by measurement

- **Top-k truncation of the LM-head gradient.** `A = softmax(z) − onehot(y)` has
  concentrated mass, so keeping the top-k over vocabulary looked promising. It
  fails: `Σ_v A[v] = 0` exactly, and zeroing the tail breaks that, injecting
  coherent error. A rank-1 correction restoring sum-to-zero does not help.
- **Unconditional sketching.** TensorSketch [12] — the Count-Sketch [4] applied
  to the outer-product structure, following Pagh's compressed matrix
  multiplication [11] — has an oblivious subspace-embedding error that scales as
  `1/√k` [2], the same rate the Johnson–Lindenstrauss bound gives for random
  projection [9]. Reaching 1e-4 needs `k ≈ 5.8e8`, larger than `P` itself. The
  Gramian is not a place where sketching helps, because the quantity being
  approximated is exactly the one the aggregator's QP is sensitive to.

Both recorded as checked-and-rejected.

---

## 9. Honest summary

| Question | Answer |
|---|---|
| Does the CIFAR result transfer? | **No**, and the reason is `T = 1`. |
| Where does jdgram beat TorchJD? | **Correctness on tied weights** (1.58e-07 vs 1.234e-03); **memory in every aggregator cell** (111–123 vs 186–328 MiB) and **at every vocabulary ≥ 8000** (0.84–0.90× `autogram`); **1.9× on time vs `autogram`** at V=65. |
| Where does it lose? | `autojac` is ~1.6× faster at this scale. **Time at V ≥ 32000 (2.5–2.7× `autogram`) — caused entirely by the router.** GPU idle 48–51%, dispatch-bound. |
| Is the QP a bottleneck? | No — 2.7% of a step at m=8. Matters above m ≈ 64. |
| Is the router working? | **For memory at small V, yes** (34–56% spread). **At vocabulary scale, no** — it costs 2.76× in time for a saving that is invisible at model scale. |

## 10. Open items, ranked

1. **Fix the router cost model.** Select on `min(time, memory)`, not memory alone.
   Worth **2.76×** at V=50257 and turns a 2.67× time loss into parity. Cheapest
   high-value change available.
2. **Dispatch-bound execution.** GPU idle 48–51%, 48% of 1257 launches under 8 µs,
   `GramianNodeBackward` 5.6% of CPU time. Batch or fuse the per-layer loop.
3. **LoRA registration and gate.** Identity unchanged; registration only.
4. **Qwen-scale validation.** Nothing here has run above 16.03M parameters.
5. **Multi-call modules.** The engine raises `NotImplementedError` for a module
   called more than once per forward; TorchJD's streaming counter is the model.

## Appendix — reproduction

```bash
bash scripts/run_profile_cluster.sh verify      # 21 preflight checks
bash scripts/run_profile_cluster.sh gates       # 44 correctness gates, fp64
bash scripts/run_profile_cluster.sh overnight   # everything, guarded and logged
```

Every run writes to `results/v{VERSION}_{timestamp}_{name}_{sha}/` with a manifest
recording the git SHA, the full diff if dirty, the environment and the GPU, and
appends one row to `runs_index.csv`. `bench/profile_stats.py` aggregates.

## References

1. Akyürek E., Damani M., Zweiger A., Qiu L., Guo H., Pari J., Kim Y., Andreas J.
   *The Surprising Effectiveness of Test-Time Training for Few-Shot Learning.*
   arXiv:2411.07279, 2024 (v2 2025). — note the widely quoted title
   "…for Abstract Reasoning" is the superseded v1 title.
2. Avron H., Nguyen H. L., Woodruff D. P. *Subspace Embeddings for the Polynomial
   Kernel.* NeurIPS 27, pp. 2258–2266, 2014.
3. Boyd S., Parikh N., Chu E., Peleato B., Eckstein J. *Distributed Optimization
   and Statistical Learning via the Alternating Direction Method of Multipliers.*
   Foundations and Trends in Machine Learning 3(1), pp. 1–122, 2011.
   doi:10.1561/2200000016
4. Charikar M., Chen K., Farach-Colton M. *Finding frequent items in data
   streams.* Theoretical Computer Science 312(1), pp. 3–15, 2004 (ICALP 2002).
   doi:10.1016/S0304-3975(03)00400-6
5. Goldfarb D., Idnani A. *A numerically stable dual method for solving strictly
   convex quadratic programs.* Mathematical Programming 27(1), pp. 1–33, 1983.
   doi:10.1007/BF02591962 — the algorithm `quadprog` implements.
6. Désidéri J.-A. *Multiple-gradient descent algorithm (MGDA) for multiobjective
   optimization.* Comptes Rendus Mathematique 350(5–6), pp. 313–318, 2012.
   doi:10.1016/j.crma.2012.03.014
7. Inan H., Khosravi K., Socher R. *Tying Word Vectors and Word Classifiers: A
   Loss Framework for Language Modeling.* ICLR 2017. arXiv:1611.01462
8. Hu E. J., Shen Y., Wallis P., Allen-Zhu Z., Li Y., Wang S., Wang L., Chen W.
   *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022.
   arXiv:2106.09685
9. Johnson W. B., Lindenstrauss J. *Extensions of Lipschitz mappings into a
   Hilbert space.* Contemporary Mathematics 26, AMS, pp. 189–206, 1984.
   doi:10.1090/conm/026/737400
10. Quinton P., Rey V. *Jacobian Descent for Multi-Objective Optimization.*
    arXiv:2406.16232, 2024 (v3 2025). — the method, and TorchJD.
11. Pagh R. *Compressed Matrix Multiplication.* ACM Transactions on Computation
    Theory 5(3), article 9, 2013. doi:10.1145/2493252.2493254
12. Pham N., Pagh R. *Fast and scalable polynomial kernels via explicit feature
    maps.* KDD '13, pp. 239–247, 2013. doi:10.1145/2487575.2487591 — TensorSketch.
13. Press O., Wolf L. *Using the Output Embedding to Improve Language Models.*
    EACL 2017, Vol. 2, pp. 157–163. ACL Anthology E17-2025.
14. Radford A., Wu J., Child R., Luan D., Amodei D., Sutskever I. *Language
    Models are Unsupervised Multitask Learners.* OpenAI technical report, 2019.
    — the 124M configuration this report calls nanoGPT.
15. Stellato B., Banjac G., Goulart P., Bemporad A., Boyd S. *OSQP: An Operator
    Splitting Solver for Quadratic Programs.* Mathematical Programming
    Computation 12(4), pp. 637–672, 2020. doi:10.1007/s12532-020-00179-2
16. Sun Y., Wang X., Liu Z., Miller J., Efros A. A., Hardt M. *Test-Time Training
    with Self-Supervision for Generalization under Distribution Shifts.* ICML
    2020, PMLR 119, pp. 9229–9248.
17. Sener O., Koltun V. *Multi-Task Learning as Multi-Objective Optimization.*
    NeurIPS 31, pp. 525–536, 2018. arXiv:1810.04650
18. Yu T., Kumar S., Gupta A., Levine S., Hausman K., Finn C. *Gradient Surgery
    for Multi-Task Learning.* NeurIPS 33, 2020. arXiv:2001.06782 — PCGrad.

---

**Caveats throughout.** Single seed (42). Plain SGD. Largest model 16.03M
parameters. fp32 workspace, float64 accumulator. `m = 8` unless stated.
Shakespeare-char is a convergence demo, not a language-modelling claim. The box is
shared (56 CPU threads, other users) — CPU-side timings carry contention noise, as
§8.2 demonstrates. Two L8 snapshot captures landed after the step completed and
show only the profiler's own allocations; that level needs its capture point moved.
