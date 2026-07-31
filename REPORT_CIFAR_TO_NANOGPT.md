# From CIFAR-10 to nanoGPT: exact Gramians for Jacobian Descent

**What this extends.** An earlier CIFAR-10 IWRM study measured a custom Gramian
engine against TorchJD's `autojac` and `autogram` on a convolutional network
with `T = 1`. This report carries that work to a transformer, where the central
assumption of the CIFAR result no longer holds, and re-measures everything
against TorchJD 0.17.

**Provenance.** §0–§7 and §9–§11 are from run tag `v8_20260731-*` on gala1
(25 runs); §8 is from `v9_20260731-*` on fuji2 (20 runs, campaign complete). The
two boxes carry different torch builds, so ratios hold within a version and
absolute timings do not transfer across them.

gala1 is an NVIDIA RTX A5000, 24 GiB, torch 2.13.0+cu130, torchjd 0.17.0,
Python 3.10.12; fuji2 is 4x RTX A5000 24 GiB on torch 2.4.1+cu121. Both run an
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

**Now measured at full GPT-2 124M** (§8): `autojac` stops fitting on a 24 GiB
card **between m=7 and m=8** at T=512 and dies at m=8/T=1024, while both Gramian
engines keep running — the first direct evidence that there is a reachable size
where materialising `[m, P]` is impossible and this approach is not. jdgram is
exact there (**4.0e-08** against brute force), holds **0.72–0.75×** `autogram`'s
peak untied, and — reversing the small-model result — is **the fastest and
leanest engine in all twelve aggregator cells**, 1.45× faster than `autogram`
and 1.5–2.0× faster than `autojac`.

Two honest corrections come with it: the tied-memory advantage of §7.1
**reverses** at this scale, and the router is *right* below `m·T² ≈ 500k` and
wrong above it — the penalty reaching **1.79×**.

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

## 8. GPT-2 124M: the scale test

Run tag `v9_20260731-*` on **fuji2** (4x RTX A5000 24 GiB, torch 2.4.1+cu121) --
a different box from sections 0-7, so compare ratios within a version, not
absolute timings across. Full GPT-2 124M: 12 layers, 12 heads, 768 wide,
V=50257, 124,439,808 trainable parameters with tying on. 20 tagged runs,
campaign complete. **One caveat first:** the gate suite did not execute on this
box (`pytest` absent from that env), so correctness here rests on the in-run
brute-force anchor rather than the 44 gates.

### 8.1 `autojac` stops fitting between m=7 and m=8 -- the crossover, measured

The claim the whole approach rests on, and until now untested at a size where it
could fail. Peak MiB, T=512, untied:

| m | jdgram | `autogram` | `autojac` |
|---|---|---|---|
| 4 | **3529** | 4707 | 8775 |
| 6 | **4975** | 6742 | 16378 |
| 7 | **5698** | 7759 | 21063 |
| 8 | **6421** | 8777 | **OOM** |
| 9 | **7144** | 9794 | **OOM** |
| 12 | **9313** | 12847 | **OOM** |

At m=8, T=1024 -- the full GPT-2 context -- `autojac` asks for 12.27 GiB beyond
the card and dies; jdgram runs in 12206 MiB and `autogram` in 14562.

**There is a real, reachable size at which materialising `[m, P]` is impossible
and both Gramian engines keep running**, and jdgram holds **0.72-0.75x**
`autogram`'s peak across the whole untied sweep.

### 8.2 At 124M jdgram is the fastest engine, not the slowest

The section 6 matrix had `autojac` fastest by ~1.6x, with the note that the trade
only pays where it cannot run. At 124M that reverses completely (m=4, T=128,
200 steps, real Shakespeare):

| Aggregator | Engine | ms/step | peak MiB | val CE | val acc |
|---|---|---|---|---|---|
| Mean | **jdgram** | **84.4** | **1255** | 3.1739 | 0.2083 |
| Mean | autogram | 122.5 | 2389 | 3.1739 | 0.2083 |
| Mean | autojac | 125.4 | 4265 | 3.2426 | 0.1939 |
| UPGrad | **jdgram** | **85.4** | **1727** | 2.8452 | 0.2589 |
| UPGrad | autogram | 122.8 | 2861 | 2.8436 | 0.2598 |
| UPGrad | autojac | 154.1 | 4265 | 2.8450 | 0.2590 |
| MGDA | **jdgram** | **100.8** | **1727** | 2.8315 | 0.2579 |
| MGDA | autogram | 139.0 | 2861 | 2.8225 | 0.2597 |
| MGDA | autojac | 169.6 | 4265 | 2.8843 | 0.2316 |
| PCGrad | **jdgram** | **84.4** | **1727** | 5.1042 | 0.0848 |
| PCGrad | autogram | 122.3 | 2861 | 4.0904 | 0.0527 |
| PCGrad | autojac | 154.1 | 4265 | 3.6217 | 0.0527 |
| -- | sgd_erm | 38.5 | 1727 | 3.1739 | 0.2083 |

**jdgram is now fastest and leanest in all twelve cells** -- 1.45x faster than
`autogram` and 1.5-2.0x faster than `autojac`, at 0.53x and 0.29-0.40x their
peaks. The Gramian trade begins paying at exactly the scale predicted, and this
is the first table in this project where it does.

Mean, UPGrad and MGDA agree on held-out CE across all three engines to three
decimals. **PCGrad does not**, and it diverges: val CE 5.10 / 4.09 / 3.62 for
jdgram / `autogram` / `autojac` against UPGrad's 2.845. Three engines computing
the same Gramian should not produce three different PCGrad outcomes, so this is
either PCGrad's own instability at this scale or a projection-order sensitivity;
either way it is **unexplained and should not be quoted as a result**.

### 8.3 The router: right where it matters, wrong where it does not

`compute_gramian` ms per iteration, `squashed`:

| config | m*T^2 | `auto` | T-first | d-first | verdict |
|---|---|---|---|---|---|
| m=2, T=256 | 131k | **43.4** | 43.4 | 86.3 | auto right, 2.0x |
| m=4, T=256 | 262k | **96.5** | 96.6 | 123.7 | auto right, 1.28x |
| m=8, T=256 | 524k | 245.1 | 245.7 | **199.3** | auto wrong, 1.23x |
| m=4, T=512 | 1.0M | 234.3 | 244.1 | **199.0** | auto wrong, 1.18x |
| m=8, T=512 | 2.1M | 614.2 | 745.5 | **343.5** | auto wrong, **1.79x** |
| m=16, T=512 | 4.2M | 1071.6 | 2557.3 | **637.6** | auto wrong, **1.68x** |
| m=8, T=1024 | 8.4M | 1104.9 | 2599.1 | **669.0** | auto wrong, **1.65x** |

And the peak spread between the two routes, from the stats tool:

| config | route peak spread | verdict |
|---|---|---|
| m=2, T=256 | 1256 vs 1516 MiB -- **17%** | router *is* controlling memory |
| m=4, T=256 | 1935 vs 2542 -- **24%** | controlling memory |
| m=8, T=256 | 3381 vs 4596 -- **26%** | controlling memory |
| m=4, T=512 | 3382 vs 3419 -- **1%** | flagged: workspace is not the peak |
| m=8, T=512 | 6274 vs 6348 -- **1%** | flagged |
| m=16, T=512 | 12058 vs 12207 -- **1%** | flagged |
| m=8, T=1024 | 12059 vs 12059 -- **0%** | flagged |

Read together these are the whole diagnosis, and it is cleaner than the earlier
single data point suggested:

- Below `m*T^2` of roughly 500k the workspace genuinely **is** a meaningful share
  of peak (17-26% spread) and the rule picks the faster route. **The router is
  correct there, and switching it off would cost 2.0x at the smallest rung.**
- Above that, the workspace difference collapses into the noise (0-1%) because
  peak is set by forward activations and held captures, while the *time*
  difference grows to **1.79x**.

So the rule optimises a quantity that stops existing exactly as the penalty for
optimising it starts to matter. The fix is a genuine cost model weighing both
axes, not a blanket switch to d-first. Note also that `auto` at m=8/T=512 sits
between the two forced routes (614 against 745 and 344) -- routing is per layer,
so it is already choosing d-first for some layers and T-first for others.

### 8.4 The tied-memory inversion does *not* hold at 124M

Section 7.1 reported jdgram becoming leaner than `autogram` on tied models at
large vocabulary. At full GPT-2 that reverses:

| | untied | tied |
|---|---|---|
| m=4, T=512 | 0.75x | 1.54x |
| m=8, T=512 | 0.73x | 1.03x |
| m=12, T=512 | 0.72x | 1.03x |
| m=8, T=1024 | 0.84x | 1.19x |

Held captures scale exactly linearly in m -- 99.7 MiB at m=2, 199.3 at m=4,
398.6 at m=8, 1594.5 at m=16 -- which is `m * V * d * 4` bytes, the cost of
keeping both tied sites alive until the cross terms can be formed. At 124M that
group is 38.6M parameters and the bill exceeds what `autogram` saves by omitting
the terms it omits.

**Leaner untied, heavier tied, exact in both.** The memory win and the
correctness win do not stack at this scale, and section 7.1 should be read as a
property of that smaller model rather than a general result.

### 8.5 Correctness holds at 124M

The brute-force anchor runs at this size now (it was gated off by an
element-count guard that demanded `m < 0.32` at 124M):

| | vs brute force |
|---|---|
| jdgram | **4.04e-08** (m=4), **5.80e-08** (m=6) |
| `autogram`, tied | 3.88e-06 to 4.21e-06 |
| `autojac` | 8.48e-05 / 5.57e-05 |

jdgram is exact at full scale, not only at gate scale. `autojac` is the **least**
accurate of the three -- three orders of magnitude looser than jdgram -- because
its vmapped fp32 reverse accumulates differently. Not a defect, but it retires
"autojac is the trustworthy reference" as an argument.

### 8.6 Execution profile, and two things that did not work

At 124M the step is **GEMM-dominated rather than dispatch-bound**, the opposite
of the small-model finding: 677 ms self-CUDA against 549 ms self-CPU over 48164
op calls, with `ampere_sgemm_128x64_tn` the top kernel at 13% and `aten::mm`
variants filling most of the rest. `GramianNodeBackward` is 5.2% of CPU time
(8.6% across all three of its entries) and `cudaLaunchKernel` 9.5% over 6375
launches. The per-layer Python loop is no longer the first thing to fix here.

Two negative results worth recording:

- **L7 learned nothing.** All four engines produced identical trajectories, but
  flat ones -- 10.972 initial, 10.967 final, no movement in 200 steps. Training a
  50257-wide head on a 65-symbol corpus at the default learning rate does not
  learn, so L7 in this configuration confirms engine *agreement* and nothing
  about convergence. L11 did learn (11.08 to 3.17) because it trains longer.
- **No QP arm.** `jacopt` is not installed on fuji2, so the GPU-QP comparison has
  not been repeated at this scale. The dual-cone solve costs 0.46-0.92 ms here,
  under 1% of a step, so it is not currently a bottleneck.

---

## 9. Additional exploration (not part of the core claim)

Separated deliberately: directions, some closed by measurement, not established
results.

### 9.1 LoRA inverts the routing decision and removes the memory ceiling

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

### 9.2 GPU QP solvers, and a benchmarking trap worth knowing

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
bugs were fixed in it first. The single-thread control settles what
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

### 9.3 Approximation directions closed by measurement

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

## 10. Honest summary

| Question | Answer |
|---|---|
| Does the CIFAR result transfer? | **No**, and the reason is `T = 1`. |
| Is there a size where `autojac` cannot run and this can? | **Yes, and it is now measured** — between m=7 and m=8 at GPT-2 124M, T=512 (§8.1). This is the claim the approach rests on and it was untested until v9. |
| Where does jdgram beat TorchJD? | **Correctness on tied weights** at every scale (4.0e-08 vs 3.9e-06 at 124M). **Memory untied**, 0.72–0.75× `autogram` throughout. **Speed and memory in all twelve aggregator cells at 124M** — 1.45× faster than `autogram`, 1.5–2.0× faster than `autojac`, at 0.53× and 0.29–0.40× their peaks. |
| Where does it lose? | **Tied memory at 124M** (1.03–1.54× `autogram`) — the held-capture bill exceeds what `autogram` saves by being wrong. **Time at large `m·T²`**, entirely the router. At small scale `autojac` is still ~1.6× faster. |
| Is the QP a bottleneck? | No — under 1% of a step at 124M (0.46–0.92 ms). Matters above m ≈ 64. |
| Is the router working? | **Below `m·T² ≈ 500k`, yes** — it controls 17–26% of peak *and* picks the faster route. **Above it, no** — the memory difference vanishes (0–1%) while the time penalty grows to 1.79×. |

## 11. Open items, ranked

1. **Fix the router cost model.** Worth **1.79×** at 124M, m=8/T=512, and the
   penalty is now bounded on both sides: the rule must keep choosing T-first
   below `m·T² ≈ 500k`, where it is worth 2.0×, and stop choosing it above.
   A rule that weighs measured per-route time against workspace *as a share of
   peak* is the shape; the coefficients are hardware-dependent and have to be
   measured on the target card. Both routes are numerically identical, so this
   can only change how long a run takes.
2. **The tied-capture bill.** Held captures are `m·V·d·4` bytes — 1.6 GiB at
   m=16 — and are what makes jdgram heavier than `autogram` on tied models at
   124M. Chunking the tied cross-term over the vocabulary would trade time for
   this, and would turn the one remaining memory loss into a win.
3. **PCGrad disagrees across engines at 124M** (val CE 5.10 / 4.09 / 3.62 for
   jdgram / `autogram` / `autojac`) while Mean, UPGrad and MGDA agree to three
   decimals. Unexplained. Either PCGrad instability at this scale or an
   ordering sensitivity; it needs isolating before any PCGrad number is quoted.
4. **Qwen-scale validation.** 124M is measured; Qwen-1.7B is 14× beyond it and
   remains arithmetic.
5. **LoRA registration and gate.** Identity unchanged; registration only.
6. **Multi-call modules.** The engine raises `NotImplementedError` for a module
   called more than once per forward; TorchJD's streaming counter is the model.

Dispatch-boundedness has dropped off this list: at 124M the step is
GEMM-dominated (§8.6), so the per-layer Python loop is no longer the first thing
to fix. It remains the right target at small model sizes.

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
§9.2 demonstrates. Two L8 snapshot captures landed after the step completed and
show only the profiler's own allocations; that level needs its capture point moved.
