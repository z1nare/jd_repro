# The attention/FFN hybrid question, answered from the source

**For:** Rui (project lead)
**Author:** Arsenii
**Date:** 2026-08-11
**Status:** finding — derived from source and existing measurements, no new runs

---

## Executive summary

Rui asked whether jdgram should use its own identity on the feed-forward layers and
fall back to autogram on the attention layers. The premise behind the question — that
the FFN is where the parameters are — is exactly right and is worth 2× attention in
this model (56.6M vs 28.3M weights at GPT-2 124M). But the fallback has nothing to fall
back from: **there is no attention identity in this codebase and never was.** In
nanoGPT, attention's only parameters are two ordinary `nn.Linear` modules; the attention
operation itself is a parameter-free function that the engine never intercepts. All 49
Linears in GPT-2 124M — QKV, attention output, both FFN projections, and the LM head —
resolve to the same handler, and the only thing that differs between them is one
integer, `P_layer = d_out · d_in`. So the question is not "which engine on which layer"
but "which contraction order at which shape", and that is derivable rather than
searchable: every dense shape in a GPT-style block crosses over within **exactly one
octave** of every other, independently of model width, so a single global route is
already close to the best possible per-layer hybrid. The good news is the scope
collapse — a research programme becomes the two profiler traces already scheduled as
R5/R6 in the v10 sweep. The two suggestions inside the proposal that *are* actionable
are also already resolved: dropping the Hadamard factorisation on attention is a no-op
(that code path has zero call sites), and the autogram-style contraction is already
available as the d-first route and is in fact what produced the headline numbers.

---

## 1. What is true in the premise

**FFN is where the parameters are.** Per block, with width `d`:

| | modules | weights |
|---|---|---|
| attention | `c_attn` (`d → 3d`), `attn.c_proj` (`d → d`) | `4d²` |
| feed-forward | `c_fc` (`d → 4d`), `mlp.c_proj` (`4d → d`) | `8d²` |

`models/nanogpt/model.py:35,37,82,84`. FFN is **66.7%** of each block's weights. At
GPT-2 124M (`d=768`, 12 blocks) that is 56.6M against 28.3M — the reason MoE is applied
there and not to attention is the same reason any per-layer optimisation pays off there
first. That part of the reasoning transfers to this engine unchanged.

**Both routes are exactly interchangeable, so a hybrid is always legal.** T-first and
d-first produce bit-comparable Gramians (`gates/test_5g_routes.py`, "Route equivalence
— 5g — pass", `docs/operator_table.md:25`; L5 measures them agreeing to `0.000e+00` at
every vocabulary tested, `src/jdgram/costmodel.py:23`). Nothing about mixing
strategies per layer can produce a wrong number. The only question a hybrid can ever
answer is a cost question.

---

## 2. Why the fallback has nothing to fall back from

### 2.1 Attention's parameters are two Linears; the attention math has none

`CausalSelfAttention` (`models/nanogpt/model.py:29`) declares exactly two parameterised
submodules:

```
models/nanogpt/model.py:35   self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, ...)
models/nanogpt/model.py:37   self.c_proj = nn.Linear(config.n_embd, config.n_embd, ...)
```

The attention operation itself is `torch.nn.functional.scaled_dot_product_attention`
(`models/nanogpt/model.py:64`) — a free function with no parameters, no module, and
therefore no Gramian contribution. The engine states this as a design invariant:

> "attention's only parameters are its four Linears"
> — `src/jdgram/identities/propagation.py:6`

and again in the module table: `Softmax / SDPA / FlashAttention | no | none | — |
autograd only` (`docs/operator_table.md:20`). Parameter-free ops contribute no Gramian
terms whatsoever; they only shape how the upstream gradient `A` propagates, which
ordinary autograd already does (`src/jdgram/identities/propagation.py:8-13`).

Verified empirically: `dict(attn.named_parameters(recurse=False))` is `{}` on a real
GPT-2 124M instance (§6).

### 2.2 The collector never even sees the attention module

`collect_hookable_modules` recurses only into modules that own no direct trainable
parameters (`src/jdgram/engine/registry.py:199`), which is precisely the condition that
skips container modules:

> "For nanoGPT this yields `wte`, `wpe`, each block's norms and Linears, `ln_f` and
> `lm_head` — and skips `Block` / `MLP` / `CausalSelfAttention`, whose only parameters
> live in children."
> — `src/jdgram/engine/registry.py:189-191`

Verified on GPT-2 124M: 76 hooked modules — 2 `Embedding`, 25 `LayerNorm`, 49 `Linear` —
and the set of hooked `Block` / `MLP` / `CausalSelfAttention` modules is empty (§6).
There is no attention node in the hook graph on which a per-layer engine choice could be
made.

### 2.3 All five projection shapes are one code path

`linear_handler` is registered against `nn.Linear`
(`src/jdgram/engine/registry.py:114`) and dispatch resolves by exact type then MRO
(`src/jdgram/engine/registry.py:85-98`). `c_attn`, `attn.c_proj`, `c_fc`, `mlp.c_proj`
and `lm_head` are all plain `nn.Linear`, so all 49 of them resolve to the identical
function object — confirmed by enumeration (§6). The handler body is nine lines:

```
src/jdgram/engine/registry.py:119   m, T, _ = A.shape
src/jdgram/engine/registry.py:120   P_layer = module.out_features * module.in_features
src/jdgram/engine/registry.py:121   if route(m, T, P_layer) == "dfirst":
```

`P_layer` is the *only* per-module quantity that enters. "FFN vs attention" is therefore
not a difference in kind, not a separate code path, and not a registration question —
it is a difference in one integer, feeding one inequality.

### 2.4 "Stop using the Hadamard factorisation on attention" is already true

`rank1_gramian` — the `G = (AAᵀ) ⊙ (XXᵀ)` form — lives at
`src/jdgram/identities/linear.py:25`. A repository-wide grep finds **exactly one
occurrence of the name: its own `def` line.** No call sites, in the engine, the bench,
or the gates.

This is not an oversight. The Hadamard collapse requires one position per objective,
which is the CIFAR/IWRM setting and not the transformer one:

> "The Hadamard collapse `G = (AAᵀ)⊙(XXᵀ)` needs one position per objective. It is a
> property of the problem, not the engine, and it does not survive `T > 1`."
> — `IMPLEMENTATION.md:291`

It is kept as the executable statement of the derivation
(`docs/design/hadamard_cifar_derivation.md`) and as the CIFAR result's provenance. So
this half of the proposal is a no-op that is already in effect.

### 2.5 "Go back to autogram" is already available — it is the d-first route

The two routes are not "ours" and "theirs". The d-first route *is* the autogram
algorithm, per-module:

> ":mod:`~jdgram.engine.materialize` is the d-first route, which forms `[m, P_layer]`
> and squares it — structurally what autogram does for every module. The T-first route
> in :mod:`jdgram.identities.linear` has no autogram analogue: it contracts positions
> first and never forms that block."
> — `src/jdgram/engine/__init__.py:11-14`

Two consequences:

1. Selecting "autogram-style contraction on attention" needs no integration work: it is
   `route() == "dfirst"` for that module, which is a shape comparison the engine already
   makes per layer, every step.
2. **The headline v10 numbers were produced with the autogram-style contraction on
   every layer, attention included.** All three solo runs pass `--force-route dfirst`
   (`results/solo/v10_20260811-021816_a-dup-m1-dfirst-r1_v10-20260811-solo/manifest.json`,
   `argv`). The proposed hybrid is therefore a strict *subset* of what has already been
   measured — it would turn the autogram-style route off on the FFN, not on on attention.

The other reading — literally calling into TorchJD's engine for the attention modules —
would buy nothing, because the per-module mathematics on the other side of that call is
the same materialise-and-square, and it would import autogram's graph retention (1.66×
memory, `docs/v10_sweep_spec.md:36`) into a design whose only passing criterion is
memory.

---

## 3. What the question becomes: a shape crossover, in closed form

Both routes compute the same `G`, so the choice is cost only
(`src/jdgram/engine/router.py:3-4`). The workspace model the bench holds measurement
against (`bench/profile_suite.py:487-500`, `itemsize` bytes per element):

```
W_tfirst = 3 · m · T²          # K_A, K_X and their product, each [T, mT]
W_dfirst = m · P_layer         # one [m, d_out, d_in] block
```

and the implemented rule (`src/jdgram/engine/router.py:43`):

```
tfirst  iff  m · T² < P_layer = d_out · d_in
```

Solving for the crossover:

```
T*(m) = sqrt(d_out · d_in / m)        m*(T) = d_out · d_in / T²
```

### 3.1 The width cancels, and the dense band is exactly one octave

Every dense weight in a GPT-style block is `P_layer = k · d²` with the shape multiplier
`k` fixed by the architecture, not by the size:

| module | shape | `k` | `P_layer` at `d=768` |
|---|---|---|---|
| `attn.c_proj` | `d → d` | 1 | 589,824 |
| `c_attn` (fused QKV) | `d → 3d` | 3 | 1,769,472 |
| `mlp.c_fc` | `d → 4d` | 4 | 2,359,296 |
| `mlp.c_proj` | `4d → d` | 4 | 2,359,296 |

Substituting, `T*(m) = d · sqrt(k / m)`, so **`T*/d` depends only on `k`**. `k` ranges
over `{1, 3, 4, 4}` for any transformer with fused QKV and a 4× FFN, so the crossovers
span a factor of exactly `sqrt(4/1) = 2` — **one octave in `T`, independent of `d`, of
`m`, and of model size.** Widening the model moves every crossover by the same factor.

Note also that `c_fc` and `mlp.c_proj` are *indistinguishable* to the router: both have
`P = 4d²`, only transposed. The two FFN projections can never be routed differently
from each other.

At `d = 768`:

| module | `P_layer` | `T*` at m=2 | `T*` at m=3 | `m*` at T=512 |
|---|---|---|---|---|
| `attn.c_proj` | 589,824 | 543 | 443 | **2.25** |
| `c_attn` | 1,769,472 | 941 | 768 | 6.75 |
| `mlp.c_fc` | 2,359,296 | 1086 | 887 | 9.00 |
| `mlp.c_proj` | 2,359,296 | 1086 | 887 | 9.00 |
| `lm_head` (V=50257) | 38,597,376 | 4393 | 3587 | 147.24 |

At the measured operating point (`T=512`, `m=1..3`) the engine's actual per-layer
decisions are:

```
attn.c_proj   m=1 tfirst   m=2 tfirst   m=3 dfirst
c_attn        m=1 tfirst   m=2 tfirst   m=3 tfirst   (flips at m=6.75)
mlp.c_fc      m=1 tfirst   m=2 tfirst   m=3 tfirst   (flips at m=9)
mlp.c_proj    m=1 tfirst   m=2 tfirst   m=3 tfirst   (flips at m=9)
lm_head       tfirst through m=147
```

So across the whole `m=1..3` range that Rui's acceptance bar covers, an attention/FFN
split differs from a single global route on **exactly one** of the four in-block
Linears — `attn.c_proj`, and only at `m=3`. That module is 8.3% of a block's weights.

(One refinement on `lm_head`: nanoGPT ties it to `wte`
(`models/nanogpt/model.py:138`), so at runtime its contribution is computed by
`tied.tied_gramian` rather than by `linear_handler`. That changes nothing here — the
tied path calls the *same* router on the *same* `P = V·d`,
`src/jdgram/identities/tied.py:127` — so the crossover above applies unchanged.)

`lm_head` is the one genuine outlier at 65× the smallest dense shape, and it is already
the documented weak point (`docs/operator_table.md:46-51`, `src/jdgram/costmodel.py`) —
an argument for fixing the head's route, not for splitting attention from FFN.

### 3.2 A hybrid cannot improve the workspace high-water mark at all

The engine holds one layer's `(A, X)` at a time and fires each identity inside that
module's own backward hook (`src/jdgram/engine/hooks.py:13-15`), so per-layer workspaces
do not coexist: the high-water mark is a `max` over layers, not a sum.

`W_tfirst = 3mT²` has **no `P_layer` term** — it is the same for every layer. Therefore
any mixed assignment with at least one T-first layer has high-water mark

```
max(3mT²,  m · max{P_layer : layers routed d-first})  ≥  3mT²
```

which is the all-T-first cost. **A hybrid can never beat all-T-first on peak workspace;
it can only match it or lose.** The best any per-layer split can achieve is therefore
exactly what `force_route("tfirst")` already achieves globally, with no split at all. At
`m=3, T=512, fp32` that is 9.00 MiB, against a measured `compute_gramian` peak of
**2686.4 MiB**
(`results/solo/v10_20260811-025217_a-dup-m3-dfirst-r1_v10-20260811-solo/rows.csv`,
`L4|phases/compute_gramian|peak_mib`). So the memory headroom a hybrid could unlock over
the best *global* route is exactly zero, and that best global route's whole contraction
workspace is 0.33% of the phase peak it sits inside.

That is consistent with what was already measured directly: the v9 route sweep found
`tfirst=3382 MiB` against `dfirst=3419 MiB` on the same model
(`results/stats_124m.txt`, flagged there as "peak is identical across tfirst/dfirst —
the contraction workspace is NOT the peak for this driver"), and `costmodel.py:13`
records 3526 MiB either way.

**The memory side of the hybrid question is closed: the answer is zero.** What is left
is time, where the two routes genuinely differ — T-first measures 12.4× slower per
kernel on the vocabulary head (`src/jdgram/costmodel.py:10`) — and time is exactly what
R5/R6 measure.

### 3.3 Where per-layer routing does pay: low-rank shapes

`L0_SHAPES` (`bench/profile_suite.py:504-515`) already contains the contrast, in one
table, at one `(m, T)`:

| entry | `d_out × d_in` | `P_layer` | route at m=4, T=512 | `T*` |
|---|---|---|---|---|
| `large-P` | 2048 × 2048 | 4,194,304 | **tfirst** | 1024 |
| `lora-A-r32` | 32 × 2048 | 65,536 | **dfirst** | 128 |
| `lora-B-r32` | 2048 × 32 | 65,536 | **dfirst** | 128 |

Replacing a `d × d` weight with rank-`r` adapters divides `P_layer` by `d/r` and `T*` by
`sqrt(d/r)` — here 64× and exactly 8×. Both adapter matrices land on the same `P = r·d`,
so like the FFN pair they can never be routed differently from each other; the split
is between the *base weight* and the *adapters*, and at the harness's own default
`(m=4, T=512)` the two sides land on **opposite routes in the same table**.

For comparison, the LoRA crossover sits 9× (vs `attn.c_proj`) to 36× (vs the FFN
projections) below the GPT-2 dense band in `P`, i.e. 3–6× in `T`. This refines the
"~30×" estimate in `docs/v10_sweep_spec.md:233` to an exact range. Either way the
conclusion there stands and is now derived rather than estimated: **if per-layer routing
pays anywhere, it pays on LoRA / MoE structures, not on attention.** An MoE expert is
an FFN of reduced width receiving a fraction of the tokens, which moves both `P_layer`
and effective `T` — it lands in this same analysis. (Derived, not measured; no MoE model
exists in-repo.)

### 3.4 A separate, smaller correction to the rule itself

While deriving the above: the rule `m·T² < P_layer` is documented as deciding "on
workspace alone" (`docs/operator_table.md:46-47`), but setting the two stated workspaces
equal cancels `m`:

```
3 · m · T²  =  m · P_layer     ⟹     3T² = P_layer     ⟹     T* = d · sqrt(k/3)
```

The workspace-minimising rule is `m`-free, and carries a factor 3 the implemented rule
does not. The two coincide **exactly at m=3** and diverge on either side. Checked
against the engine at `T=512`:

```
m=1  attn.c_proj  rule=tfirst  tf= 3.00 MiB  df= 2.25 MiB  -> cheaper is dfirst   MISMATCH
m=2  attn.c_proj  rule=tfirst  tf= 6.00 MiB  df= 4.50 MiB  -> cheaper is dfirst   MISMATCH
m=3  attn.c_proj  rule=dfirst  tf= 9.00 MiB  df= 6.75 MiB  -> cheaper is dfirst   ok
m=3  c_attn / c_fc / lm_head                               -> rule agrees         ok
```

The likely cause is a stale constant rather than a design choice: the `3mT²` figure
dates from the blocking rewrite of the T-first kernel, which replaced a `4m²T²`
whole-kernel form (`IMPLEMENTATION.md:296`, `src/jdgram/identities/linear.py:71-82`) and
post-dates the router. **This changes no result** — both routes are exact, §3.2 shows
the workspace is not the peak, and it is a strictly smaller issue than the
time-blindness `costmodel.py` already documents. Recorded so it is not rediscovered.

---

## 4. What this means for the next step

1. **Nothing needs to be built.** No attention identity to write, no fallback branch, no
   second engine to integrate, no registration. Every one of the five projection shapes
   already goes through one handler and one shape-driven inequality.

2. **The question is already on the v10 run list, at two runs.** R5 and R6
   (`docs/v10_sweep_spec.md:127-128`) are profiler traces at forced T-first and forced
   d-first. The profiler groups operations by tensor shape, so per-shape cost is readable
   directly from those two traces. And because per-layer Gramian contributions are summed
   over **disjoint** parameter blocks, the cost function is separable: the best possible
   per-layer hybrid is the per-shape minimum read off the two pure runs. No `2^k` grid,
   no search. This was the argument for dropping the layer-group factorial axis
   (`docs/v10_sweep_spec.md:103`); it applies verbatim here.

3. **Expect the hybrid to be worth approximately nothing at the current operating
   point, and know that before running it.** §3.2 proves it cannot help peak workspace at
   all; §3.1 shows the dense shapes sit inside one octave, and that at `m=1..3, T=512` a
   split touches one Linear holding 8.3% of a block's weights. R5/R6 should be read as a
   *time* measurement confirming a bound, not as an open search.

4. **The two places the same reasoning says effort should go instead**, both already
   identified: the LM head's route, which is 65× outside the dense band and measures
   12.4× slower per kernel on the wrong side (`src/jdgram/costmodel.py`); and the
   two-forward structural floor, which consumes 72–82% of the entire time budget before
   any of this machinery runs (`docs/v10_sweep_spec.md:52`). Route selection cannot
   reach the 1.5× bar; §6 of the sweep spec is still the decision that matters.

5. **One thing worth agreeing explicitly.** Rui's shape intuition is right, just aimed
   one architecture over: the layers where a per-layer strategy genuinely diverges are
   low-rank and expert-sharded ones, not attention. Whether LoRA / MoE registration is in
   scope for this phase is a real open question (`docs/v10_sweep_spec.md:235`,
   `REPORT_CIFAR_TO_NANOGPT.md:556-575`) — the identity needs no change, only registration
   and a gate.

---

## 5. Summary table

| Rui's proposal | Status | Evidence |
|---|---|---|
| Apply jdgram's identity on the FFN | Already done — same handler as everything else | `registry.py:114-123` |
| Fall back to autogram on attention | Nothing to fall back from: no attention identity exists | `model.py:29-76`, `registry.py:189-191`, `propagation.py:6` |
| Attention needs a different treatment in kind | No — one integer differs, `P_layer` | `registry.py:120` |
| Stop using the Hadamard factorisation on attention | Already true — zero call sites | grep; `linear.py:25`; `IMPLEMENTATION.md:291` |
| Use autogram's contraction on some layers | Already available as the d-first route, and already used on **all** layers in the headline runs | `engine/__init__.py:11-14`; solo `manifest.json` |
| Hybrid saves memory | Provably not: hybrid ≥ all-T-first on peak workspace, and workspace is not the peak | §3.2; `stats_124m.txt`; `costmodel.py:13` |
| Hybrid saves time | Open, bounded, and measured by two already-scheduled runs | R5/R6, `v10_sweep_spec.md:127-128` |

---

## 6. Verification

Environment: `C:/Users/garba/miniconda3/envs/torch-cuda/python.exe`,
`PYTHONPATH=<repo>;<repo>/src;<repo>/bench`. GPT-2 124M built on `torch.device("meta")`
(`n_layer=12, n_head=12, n_embd=768, block_size=512, vocab_size=50257, bias=True`) — no
GPU, no weights allocated.

```
hooked modules: 76
  Embedding    x2    e.g. transformer.wte
  LayerNorm    x25   e.g. transformer.h.0.ln_1
  Linear       x49   e.g. transformer.h.0.attn.c_attn

block 0 hooked names:
    transformer.h.0.ln_1        -> layernorm_handler
    transformer.h.0.attn.c_attn -> linear_handler
    transformer.h.0.attn.c_proj -> linear_handler
    transformer.h.0.ln_2        -> layernorm_handler
    transformer.h.0.mlp.c_fc    -> linear_handler
    transformer.h.0.mlp.c_proj  -> linear_handler
    lm_head                     -> linear_handler

any Block / MLP / CausalSelfAttention hooked? []
distinct handlers over all Linears: {'linear_handler'} (49 Linears)
CausalSelfAttention direct params: []
MLP direct params: []
```

Grep for the Hadamard form across the repository returns one line, its own definition:

```
src/jdgram/identities/linear.py:25:def rank1_gramian(
```

Gate suite unchanged by this document (Markdown only), re-run to confirm the baseline:

```
python -m pytest gates/ -q   ->  58 passed
```

Crossovers and route decisions in §3 come from calling `jdgram.engine.router.route` and
`bench.profile_suite.theoretical_workspace_mib` directly, not from hand arithmetic.
Measured peaks come from `results/solo/*/rows.csv` and `results/stats_124m.txt`.

**Not verified here:** anything about *time*. Every timing statement in this document is
quoted from an existing measurement (`costmodel.py`, `stats_124m.txt`,
`v10_sweep_spec.md`), not re-measured — no GPU jobs were run, per the standing
constraint that a cluster campaign is in flight.
