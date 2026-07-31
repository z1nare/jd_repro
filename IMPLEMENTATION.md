# jdgram — implementation specification

What the engine does, why each decision was made, and what it replaced. Written
as of run tag `v9_20260731-*` (GPT-2 124M campaign, fuji2).

Measurements referenced here live in `REPORT_CIFAR_TO_NANOGPT.md`; this document
is the design record, not the results.

---

## 0. What the library computes

Jacobian Descent aggregates `m` per-objective gradients — UPGrad, MGDA, PCGrad
and Mean all take the same input, the Gramian

```
G = J Jᵀ        J = [m, P]        G = [m, m]
```

`G` is `[m, m]`. `J` is the model size times the objective count. **The entire
point of this library is that `G` can be computed without `J` ever existing.**

The decomposition that makes that possible:

```
G_ij = Σ_layers ⟨ ∂L_i/∂W_ℓ , ∂L_j/∂W_ℓ ⟩_F
```

Each layer's Frobenius product has a closed form in two quantities the backward
pass already produces — the upstream gradient `A` at the layer's output and the
layer's input `X` — so the per-layer `[m, P_layer]` block need not exist either.

---

## 1. Architecture

```
src/jdgram/
  identities/       one module per layer family; each returns an [m,m] block
    linear.py         T-first and d-first contractions for nn.Linear
    embedding.py      token and positional embedding
    norm.py           LayerNorm / RMSNorm gamma and beta
    tied.py           the four-term shared-parameter Gramian
    propagation.py    parameter-free ops: the executable statement that they
                      contribute nothing
    precision.py      workspace vs accumulator dtype policy
  engine/
    hooks.py          compute_gramian(): the sole entry point
    registry.py       type -> MRO -> predicate dispatch to a handler
    router.py         per-layer contraction-order choice
    materialize.py    the d-first route
    accumulate.py     streaming accumulator + tied-group barrier
    node.py, edges.py autograd plumbing, ported from TorchJD (MIT)
  costmodel.py      where a measured router rule would live (NOT implemented)
gates/              44 tests against brute-force autograd, float64
bench/              profiling harness, levels L0-L11
```

**Entry point.** `compute_gramian(model, compute_losses, modules=...,
handler_overrides=..., shared_handlers=..., driver=..., force_route=...,
workspace_dtype=...)`. It takes a *callable* that produces the losses, because it
owns the forward pass (see §4.3).

---

## 2. The identity layer

### 2.1 Specification

Every identity takes `(module, A, X)` where `A` is `[m, T, d_out]` and `X` is
`[m, T, d_in]`, and returns `[m, m]` in float64.

| Layer | Form | Notes |
|---|---|---|
| `nn.Linear` weight | `Σ_{t,s} (A_i[t]·A_j[s]) (X_i[t]·X_j[s])` | two contraction orders, §3 |
| `nn.Linear` bias | `(Σ_t A_i)·(Σ_t A_j)` | rank-1 in `t` |
| `nn.Embedding` token | index-equality kernel, or scatter-add | rows only touched where indices match |
| `nn.Embedding` positional | diagonal in `t` | each position hits one row |
| LayerNorm / RMSNorm | recompute `x̂` from the raw input, then `[m, d]` | `d` is small; d-first always |
| Tied `wte` = `lm_head` | `G_hh + G_ee + G_he + G_ehᵀ` | §5 |
| Softmax, SDPA, GELU, residual, dropout, RoPE | **nothing** | no parameters, no terms |

### 2.2 Why the parameter-free row matters more than it looks

Attention needed no new mathematics. Its parameters are four `nn.Linear`s;
softmax, SDPA, GELU, the residual adds and RoPE hold no parameters and therefore
contribute no Gramian terms at all — ordinary autograd propagates `A` through
them. **Supporting a new architecture is re-registration, not derivation.** This
is the single most load-bearing structural fact in the design and it is why the
CIFAR-to-transformer port was tractable at all.

### 2.3 Precision policy

Heavy intermediates run in `workspace_dtype` (default float32); the `[m, m]`
accumulator is **always** float64. The Gramian is tiny and feeds a QP regularised
at ~1e-4, so fp64 there is nearly free, while fp32 on the workspace is ~30× on
GA102. Gates pin float64 workspace so brute-force diffs at `atol=1e-10` stay
meaningful.

---

## 3. Two contraction orders and the router

### 3.1 Why there are exactly two

Write the Gramian with all four indices:

```
G_ij = Σ_{t,s,p,q}  A_i[t,p] X_i[t,q] A_j[s,p] X_j[s,q]
```

You may contract positions `(t,s)` first or features `(p,q)` first. **No third
pairing keeps the result exact.** That is why the answer is a router rather than
a single kernel, and why "quadratic in T" describes only one of the two routes.

| Route | contracts first | workspace | forms `[m, P_layer]` |
|---|---|---|---|
| **T-first** | positions | `3 m T²` | no |
| **d-first** | features | `m · P_layer` | yes |

### 3.2 Current rule, and its known defect

`engine/router.py`:

```python
tfirst if m * T * T < P_layer else dfirst
```

**This is a memory rule, and it is measurably wrong above `m·T² ≈ 500k`.** The
v9 campaign bounded it on both sides:

- Below that threshold the workspace is genuinely 17–26% of peak *and* the rule
  picks the faster route. Forcing d-first there costs **2.0×**.
- Above it the two routes differ by 0–1% in peak, because peak is set by forward
  activations and held captures — while the time difference grows to **1.79×**
  against the rule's choice.

So the rule optimises a quantity that stops existing exactly as the penalty for
optimising it begins to matter. `costmodel.py` documents this and raises rather
than pretending to have a measured model.

**The fix is not "always use d-first".** That would make the small case twice as
slow. It requires weighing measured per-route time against workspace *as a share
of peak*, with hardware-dependent coefficients. Both routes are numerically
identical (agreement 0.000e+00 at every configuration tested), so this can only
change how long a run takes, never what it produces.

Routing is **per layer**, which is visible in the data: at m=8/T=512 the `auto`
timing sits between both forced routes, so it is already choosing differently for
different layers.

---

## 4. The reverse pass

### 4.1 The problem

The Gramian needs each objective's gradient kept *separate*. The obvious routes
are expensive: `m` separate backward passes, or one `is_grads_batched` pass that
vmaps the whole reverse sweep and gives every intermediate a leading `m`
dimension.

### 4.2 What is implemented: `squashed`, adopted from TorchJD

For a 1-D loss vector with batch dim 0, seed **one ordinary backward** with
`torch.ones_like(losses)`. With per-instance losses and batch-independent
modules the intermediate Jacobians are block-diagonal, so

```
∂(Σ_i L_i)/∂z[i] = ∂L_i/∂z[i]
```

Row `i` of that single gradient *is* objective `i`'s upstream gradient.

**This strategy is TorchJD's, not ours.** `autogram/_engine.py:326` does exactly
this, fires each module's contribution inside its own backward hook, and releases
that module's state as soon as it is squared
(`_gramian_computer.py:73-76`, `del self.summed_jacobian`). The library uses that
design unchanged and the source says so.

Three drivers remain selectable (`squashed`, `batched`, `loop`) purely as an
equivalence check — gate 5f asserts all three agree. `squashed` is the default
and the only one that should ever ship.

### 4.3 Precondition, and the cost it carries

Every hooked module must be batched on dim 0, which is why `forward_logits`
expands the position vector from `[T]` to `[B, T]` — `wpe` must produce
`[B, T, d]`, not `[T, d]`, or autograd's broadcast-backward sums over the batch
before the hook sees the gradient.

The one genuine divergence from TorchJD, and it is a **trade, not a win**: this
engine releases more eagerly — it pops the stored input as the hook consumes it
and runs with `retain_graph=False`, where TorchJD pins the module's `args` and
retains the graph. The cost is that the graph is gone afterwards, so a training
step needing a weighted backward after the Gramian pays a **second forward**.
That is why `compute_gramian` takes a callable rather than a tensor, and it shows
up in the measured step breakdown as a real cost.

---

## 5. Shared parameters — the correctness claim

### 5.1 The mathematics

A parameter reached through two modules has a per-objective gradient that is the
**sum over its sites**, so the Frobenius product expands to four terms:

```
⟨g_h + g_e , h_h + h_e⟩ = ⟨g_h,h_h⟩ + ⟨g_e,h_e⟩ + ⟨g_h,h_e⟩ + ⟨g_e,h_h⟩
```

Summing per-module Gramians computes the first two. This is not an approximation
choice — it is a missing term.

### 5.2 What is implemented

`accumulate.py` holds a `SharedGroup` barrier: each member's capture is submitted
as the reverse sweep reaches it, and the group Gramian is formed only once every
member has been visited. `identities/tied.py` computes all four terms.

`hooks.py` **raises** if any parameter is owned by two modules and no
`shared_handlers` entry covers them. Refusing is deliberate: summing per-module
Gramians returns a structurally plausible wrong answer, and a wrong `[m, m]`
matrix propagates silently through the aggregator into the weights.

### 5.3 Scope the claim correctly

This is a defect in `autogram`, **not in TorchJD**. `autojac` materialises the
full `[m, P]` Jacobian, so a shared parameter's gradient is summed over both
sites before anything is squared and the cross terms are present automatically —
it agrees with brute force on a tied model. The defect belongs specifically to
the per-module Gramian strategy, which is the strategy this library also uses.

### 5.4 The cost, which is real

Holding both sites alive costs `m · V · d · 4` bytes — measured at exactly
99.7 / 199.3 / 398.6 / 1594.5 MiB for m = 2 / 4 / 8 / 16 at GPT-2 124M. At that
scale the bill **exceeds** what `autogram` saves by omitting the terms, so the
library is leaner than `autogram` untied and heavier tied. Chunking the tied
cross-term over the vocabulary would trade time for this and is the natural next
optimisation.

---

## 6. Dispatch

`registry.py` resolves a module to a handler by exact type, then MRO, then
predicate. `handler_overrides` pins a specific module path (used for `wpe`, which
is an `nn.Embedding` structurally but positional semantically).

`check_module_supported` rejects BatchNorm and anything with
`track_running_stats`: those couple batch elements, so per-instance objectives
are not independent and the block-diagonal assumption of §4.2 fails.

A module called more than once per forward raises `NotImplementedError`. TorchJD
handles this with a `remaining_counter` that sums the module's Jacobians across
calls before squaring; the same approach applies here and is not yet built.

---

## 7. The benchmark harness

`bench/profile_suite.py`, levels L0–L11: identity kernels alone (L0), hook
overhead (L1), reverse drivers (L2), accumulation (L3), full-step phase
decomposition (L4), engine A/B with a brute-force anchor (L5), scaling exponents
(L6), convergence (L7), memory snapshot (L8), profiler trace (L9), QP backends
(L10), aggregator × engine matrix (L11).

Three design rules, each of which exists because its absence produced a wrong
conclusion at least once:

1. **Every run is tagged.** `v{VERSION}_{timestamp}_{name}_{sha}` with a manifest
   recording the git SHA, the environment, and the full diff if the tree is
   dirty. Untagged results written to shared paths cannot be attributed to code.
2. **Resume is keyed on every distinguishing field.** A field left out of
   `RunLogger.KEY` silently collapses two configurations into one and skips the
   second as "already done" — which reads as a completed sweep with a missing
   row, not as a bug.
3. **`bench/preflight.py` proves the remote checkout is the code you think it
   is.** An rsync that transferred nothing looks exactly like one that
   transferred everything, and a stale engine produces numbers — just the wrong
   ones.

---

## 8. How this differs from the previous design

| Area | Previous | Now | Why |
|---|---|---|---|
| **Problem setting** | CIFAR-10 IWRM, one objective per image | transformer, per-sequence losses over `T` tokens | The Hadamard collapse `G = (AAᵀ)⊙(XXᵀ)` needs one position per objective. It is a property of the problem, not the engine, and it does not survive `T > 1`. |
| **Engine shape** | manual reverse walk over a flat `nn.Sequential`, caching every layer input | hook-driven, autograd propagates `A` | A manual walk needs an explicit rule per op. Hooks need rules only where parameters live, which is why attention required nothing. |
| **Reverse driver** | `is_grads_batched` with `eye(m)`, vmapping the whole pass | one ones-seeded ordinary backward | vmap gives every intermediate a leading `m` dimension. Adopting TorchJD's strategy cut peak 5.7× and time 3.6×. |
| **Accumulation** | all captures held until the reverse pass completed | fired inside each module's backward, freed immediately | Holding everything made peak scale with depth. Only tied groups now outlive the call that produced them. |
| **Contraction** | one route | two routes plus a per-layer router | The vocabulary head and an interior MLP want opposite orders. Neither is universally right. |
| **Linear T-first kernel** | whole `[mT, mT]` kernel via `einsum` | blocked over the objective index | `torch.einsum` clones both operands, so the kernel cost `4 m²T²` where the docstring claimed `2 m²T²`. Blocking removes the `[mT, mT]` intermediate entirely: `3 m T²`, measured 164 → 12.25 MiB. |
| **Embedding / tied kernels** | `einsum` | `reshape` + `matmul` | Same clone; 1250 → 625 MiB at V=20000. |
| **Precision** | float64 throughout | fp32 workspace, fp64 accumulator | ~30× on the heavy intermediates; the `[m,m]` result is too small for its dtype to matter. |
| **Shared parameters** | not handled | four terms, or refuse | Summing per-module Gramians silently drops two terms. |
| **Attention** | expected to need a new identity | needed nothing | Its parameters are four Linears. |
| **Result provenance** | shared output paths | tagged run directories + manifest + index | Several generations of results had become unattributable to code. |

### 8.1 Claims that were retracted or narrowed

Recorded because they were stated before they were tested:

- **"12× TorchJD's parameter ceiling."** True on CIFAR, and it was a `T = 1`
  result. Does not transfer.
- **"Firing identities inside backward and freeing captures immediately is our
  improvement."** It is not — TorchJD already does both. The real difference is
  that TorchJD materialises a per-module `[m, P_module]` Jacobian block and
  squares it, whereas this engine uses a closed form and can therefore choose the
  T-first order, which `autogram` has no analogue for.
- **"`autogram`'s counter desync is permanent."** It is not; `compute_gramian`
  resets every computer in a `finally`. The defect is an undocumented,
  unchecked contract with a cryptic failure, which is still worth reporting.
- **"jdgram becomes leaner than `autogram` on tied models at large vocabulary."**
  True on a small trunk with a big head; **reverses at GPT-2 124M**, where the
  held-capture bill dominates.
- **"The router is wrong at vocabulary scale."** Too broad. It is *right* below
  `m·T² ≈ 500k` — worth 2.0× there — and wrong above it.
- **"jacopt does not converge at m ≥ 32."** Diagnosed twice, wrong both times.
  It was CPU thread contention on a shared box: 341× between 56 threads and one,
  at identical iteration counts.

---

## 9. Invariants

Anything that breaks one of these is a bug, not a tuning question.

1. Both contraction routes return the same Gramian. Measured agreement is
   `0.000e+00` at every configuration tested.
2. All three reverse drivers return the same Gramian (gate 5f).
3. `compute_gramian` returns an `[m, m]` matrix and retains **nothing** — the
   harness asserts retained memory is ~0 after it returns.
4. A tied model either gets all four cross terms or raises.
5. The accumulator is float64 regardless of workspace dtype.
6. No identity ships without a gate that checks it against brute-force
   `autograd.grad` in float64 at `rtol=0, atol=1e-10`.

---

## 10. Deliberately not implemented

- **A measured router cost model** (§3.2) — the highest-value outstanding item.
- **Chunked tied cross-terms** — would convert the one remaining memory loss
  into a win.
- **Multi-call modules** — raises rather than guessing.
- **LoRA registration.** The identity needs no change: `W_eff = W + BA` is the
  same form with `d_out → r`. This is registration, not derivation.
- **GRPO head seed** — the derivation is not finished, so there is no gate, so
  it does not ship.
