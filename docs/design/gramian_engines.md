# Gramian Engines for Jacobian Descent at LLM Scale — Reevaluation, Identities, and Plan

**Status:** working design doc · follows the group meeting (CIFAR retired, transformers now, MuSiQue/Qwen approved) · all derivations in Part II are **pending gate verification** unless marked proven.

---

## Part I — Reevaluating the contribution at the real target

### I.1 What is proven vs. what the target changes

**Proven (CIFAR/IWRM regime):** the Hadamard-factorized engine computes the exact Gramian
(preflight gate 4, $10^{-13}$ vs `autogram`), beats both TorchJD engines on speed at every
shared width, and fits **~12×** more parameters before OOM (1.65 B on 24 GB vs TorchJD's
134 M ceiling). Luo named this "likely our contribution."

**But the benchmark regime and the target regime differ on the two variables that decide
which engine wins:**

| | IWRM benchmark | LLM/RL target |
|---|---|---|
| $m$ (objectives) | 32 (= batch; per-instance losses) | **2–8** (per-objective GRPO losses) |
| per-objective gradient at a layer | rank-1 outer product $a_i x_i^\top$ | rank-up-to-$BT$ (sum over rollouts × tokens) |
| largest layer | $\propto w^2$ Linear | MLP linears (~13.8 M) and the **vocab head (~233 M)** |

### I.2 The honest arithmetic at the target (Qwen-1.5B-ish, $B{=}8$, $T{=}1024$, fp32)

**Per-layer materialization** (autogram-style: form $J_\ell \in \mathbb{R}^{m\times P_\ell}$,
add $J_\ell J_\ell^\top$, discard):

| $m$ | biggest MLP linear block | LM-head block |
|---|---|---|
| 4 | 0.21 GiB | **3.5 GiB** |
| 8 | 0.41 GiB | **7.0 GiB** |
| 32 | 1.6 GiB | 27.8 GiB |
| 64 | 3.3 GiB | 55.6 GiB |

**$T{\times}T$ contraction** (Hadamard route, Part II.1): workspace is $O((BT)^2)$ per pair
(0.25 GiB dense, MBs chunked) — but its FLOP cost at an interior $d{\times}d$ linear is
**≈ 7× ($m{=}4$) to 12× ($m{=}8$)** the weight-gradient part of the backward it accompanies.

**Consequences, stated plainly:**

1. **For interior linears at small $m$, per-layer materialization wins.** A 0.2–0.4 GiB
   transient block is not a memory wall, and it costs no extra FLOPs beyond the $m$-seed
   backward. Using the $T{\times}T$ contraction there would spend ~an order of magnitude
   more compute to save memory that didn't need saving. The 12× CIFAR result does **not**
   transfer to these layers at $m{\le}8$ — because it was earned in the $m{=}32$, rank-1
   regime, and honesty about that is what keeps the rest of the claims credible.
2. **The vocab head (and tied embedding) is where the trick is decisive at the target.**
   A 3.5–7 GiB transient block for one layer, on a 24 GB card already holding 1.5 B params
   + optimizer state, is the wall. The closed form removes it entirely — and the GRPO seed
   structure (I.4) makes it nearly free across objective pairs.
3. **At large $m$ the original regime returns.** Per-rollout objectives ($m$ = group size,
   16–64) or the $m{\to}100{+}$ embedding-space stretch goal put every layer back where
   CIFAR was: materialization blocks in the GiB–tens-of-GiB range, Hadamard scaling as
   $O(m^2)$ smallness. The trick is the *enabler* of any future large-$m$ direction.

### I.3 The reframed contribution: a cost-model-driven hybrid engine

The deliverable is not "Hadamard everywhere." It is **one exact Gramian engine that selects
the cheapest correct identity per layer**:

$$\text{route}(\ell) = \begin{cases} \text{closed form (Part II)} & \text{if } m\,P_\ell \gg (BT)^2\text{-workspace, i.e. head/embedding, or } m \text{ large}\\ \text{materialize } J_\ell J_\ell^\top & \text{if } m P_\ell \text{ small (interior linears, } m \le 8) \end{cases}$$

with the crossover chosen by an explicit, measured cost model rather than taste. Both
routes are mathematically identical (same $G$), so the choice is pure engineering — and the
selection rule itself, with its measured crossover, is a presentable systems result.

**What this project can now credibly deliver, ranked:**

1. **Runnable JD at 1.5–2 B scale on 24–48 GB** — the success criterion Rui set verbatim.
   The head/embedding closed forms are the specific thing that makes this fit.
2. **$G$ in one $m$-seed backward with no $O(mP)$ gradient storage** — vs. the naive
   m-backwards-and-store baseline that motivated the project.
3. **The GRPO-seed head shortcut (I.4)** — all $m^2$ head-layer Gramian entries from one
   shared kernel. Derivation below; potentially the single best FLOP result in the project.
4. **Transformer coverage with exact-equivalence gates** (Phase B) — the item Luo's
   MuSiQue approval is gated on.
5. **GPU-resident batched QP** (jacopt port) and **fused contraction kernels** (Triton) —
   bounded engineering, already scoped, now targeted at the layers the cost model routes
   through closed forms.
6. **Stochastic $G$ estimation** (research-flavored): $G$ is $m{\times}m$ and only feeds a
   weighting — subsampling positions/rollouts (SSJD-style, the paper's own §2.1 machinery)
   can cut contraction FLOPs by $(BT/s)^2$ at bounded weight error. Unvalidated; listed as
   an opportunity, not a promise.

### I.4 The GRPO seed structure — why the head layer collapses (derivation, ungated)

Per-objective GRPO losses share every token's log-probability gradient direction and differ
only by per-token scalars (advantage × clip mask):
$$A_i^{\text{logits}}[b,t,:] \;=\; c_i(b,t)\cdot s(b,t,:),\qquad s = \nabla_z \log\pi \text{ shared across } i.$$
Because the head is applied per-token (no mixing), this diagonal structure is exact there:
$$K_{A,ij} = \mathrm{diag}(c_i)\, S S^\top \mathrm{diag}(c_j) \;\Rightarrow\; G^{\text{head}}_{ij} = c_i^\top \big( S S^\top \odot X X^\top \big)\, c_j .$$
**One** $(BT){\times}(BT)$ kernel, computed once, then $m^2$ cheap quadratic forms. Below
the head, attention mixes tokens and the diagonal structure breaks — the general
contraction (II.1) applies there. Gate this derivation like everything else before using it.

### I.5 Honest costs and open risks

- Interior-layer contraction FLOPs (7–12× backward) if the cost model is ignored — the
  reason the hybrid framing exists.
- Head-seed caching: $A^{\text{head}}$ is $[BT, V]$ (~4.7 GiB fp32 at $V{=}152$k) if held
  naively across the backward; the I.4 factorization stores $c_i$ ($[m,BT]$) + shared
  structure instead. Must be engineered, not assumed.
- BatchNorm remains out of scope (cross-instance coupling); irrelevant for the target
  (LayerNorm/RMSNorm), stated for completeness.
- nanochat trains with Muon+AdamW, not plain Adam — Stage-B ("Adam after aggregation")
  interaction to revisit at integration, not now.
- All Part II identities are exact algebra, but **nothing ships until its gate passes**
  against brute-force autograd — same discipline as gates 1–4.

---

## Part II — The identities (corrected; all pending gates unless marked proven)

Setup: $m$ objectives; objective $i$'s loss is a mean over rollouts $b$ and tokens $t$
(write the combined position index as $u = (b,t)$, $U = BT$ positions). $A_i \in
\mathbb{R}^{U \times d_\text{out}}$ is objective $i$'s upstream gradient at a layer;
$X \in \mathbb{R}^{U \times d_\text{in}}$ is the layer input, **shared across objectives**
(same forward pass). Every identity below computes exact entries of $G = JJ^\top$.

### II.1 Linear layers — the general sequence contraction

Per-objective weight gradient is a sum of outer products, rank up to $U$:
$$\frac{\partial L_i}{\partial W} = \sum_u A_i[u]\, X[u]^\top
\quad\Rightarrow\quad
G_{ij} \mathrel{+}= \sum_{u,v}\; \big(A_i[u]\!\cdot\!A_j[v]\big)\,\big(X[u]\!\cdot\!X[v]\big)
= \big\langle A_i A_j^\top,\; XX^\top \big\rangle_F .$$

- $K_X = XX^\top \in \mathbb{R}^{U\times U}$ is **shared across all $(i,j)$ pairs** —
  compute once per layer.
- Rank-1 special case ($U{=}1$): recovers the proven CIFAR identity
  $G = (AA^\top)\odot(XX^\top)$ exactly. One formula, not two tiers of math; the LM head
  is *not* mathematically special (the earlier per-token rank-1 claim was wrong for
  per-sequence losses) — it is special only for **weight tying** (II.4) and the **seed
  structure** (I.4).
- Bias: $\partial L_i/\partial b = \sum_u A_i[u]$, so
  $G_{ij} \mathrel{+}= \big(\sum_u A_i[u]\big)\cdot\big(\sum_v A_j[v]\big)$ —
  materialize $[m, d_\text{out}]$ and multiply.
- **Naive einsum builds $[m,m,U,U]$** — fine at gate sizes, ~17 GiB at $m{=}32$,
  $U{=}2048$. Ship two implementations: naive (gates) and chunked-per-pair
  ($[U,U]$ or row-blocks only), gated against each other.

### II.2 Token embedding — indicator kernel (X is indices, not activations)

The embedding gradient scatters rows; the pairwise inner product only picks up positions
holding the **same token**:
$$G_{ij} \mathrel{+}= \sum_{u,v:\; \text{tok}[u] = \text{tok}[v]} A_i[u]\cdot A_j[v]
= \big\langle A_i A_j^\top,\; \mathbb{1}[\text{tok}[u]{=}\text{tok}[v]] \big\rangle_F .$$
Same contraction shape as II.1 with $K_X$ replaced by a boolean equality kernel
(`tok.unsqueeze(0) == tok.unsqueeze(1)` per pair of sequences).

### II.3 Positional embedding (nanoGPT `wpe`) — diagonal shortcut

Every sequence uses positions $0..T{-}1$, so the equality kernel is the identity across
matching $t$ within each $(b,b')$ block:
$$G_{ij} \mathrel{+}= \sum_{b,b',t} A_i[b,t]\cdot A_j[b',t].$$

### II.4 Tied weights (wte = lm_head) — four-permutation cross terms

With one shared $W$ used at two sites, the true per-objective gradient is the **sum** of
site gradients, so
$$G_{ij} = G^{hh}_{ij} + G^{ee}_{ij} + G^{he}_{ij} + G^{eh}_{ij},$$
head×head via II.1, emb×emb via II.2, and the cross terms via an index-gather contraction:
$$G^{he}_{ij} = \sum_{u,v} A^{\text{head}}_i\big[u,\ \text{tok}[v]\big]\;\big(X^{\text{head}}[u]\cdot B_j[v]\big),$$
where $B_j$ is the embedding-site upstream gradient. **This is not a plain Hadamard of two
Grams** — the earlier one-routine-for-all-four sketch was wrong on this point.
Mechanically: reuse TorchJD's `remaining_counter` cache verbatim — collect $(A, X)$ at the
head, wait for the embedding's backward, then compute all four terms and flush.

### II.5 LayerNorm / RMSNorm — small-vector materialization

Per-objective parameter gradients are $d$-vectors:
$\partial L_i/\partial\gamma = \sum_u A_i[u]\odot\hat x[u]$ (and analogously for $\beta$).
Materialize $[m, d]$, then $G \mathrel{+}= MM^\top$. No trick needed — $d$ is small.

### II.6 Parameter-free ops — no identity required (proven principle)

Softmax/SDPA, SiLU/GELU, residual adds, RoPE contribute **no** Gramian terms; they only
shape how $A$ propagates, and ordinary autograd does that propagation (gate 4 already
demonstrated exactness through ELU/MaxPool). This is why the hybrid hook engine never
needs to "port attention" — attention's parameters are its four Linears.

---

## Part III — Execution plan (nanoGPT sandbox → gates → nanochat)

**Standing rules:** every gate is a committed script (they become the transformer
preflight, as gates 1–4 were for CIFAR); no step starts before the previous gate passes;
on failure, shrink the hooked-module set before touching math; `dropout = 0.0` everywhere
or brute-force and hooked passes see different networks; **no `torch.compile` until all
gates pass eager** — it is a Phase-E knob, not a correctness tool; fp64 for gates, dtype
configurable for runs (A5000 fp64 throughput is rate-limited).

- **Step 0 — sandbox (½ d).** `nanogpt_engine/` inside `jd_repro`; copy `model.py` only
  from `karpathy/nanoGPT` at a pinned commit (record the hash). Gate config:
  `n_layer=2, n_head=2, n_embd=64, block_size=16, vocab=65, dropout=0.0`, $m{=}4$
  per-sequence mean-CE objectives, fp64, run gates with `bias=True` **and** `False`.
  Deliverable: `brute_force.py` — per-objective `torch.autograd.grad`, fixed param
  flattening order, $G_\text{true} = JJ^\top$. Ground truth for everything; written first.
- **Step 1 — one Linear, tying disabled (1 d).** Delete the tying line. Hook plumbing
  (TorchJD forward-hook + `AutogramNode` skeleton, borrowed) on `lm_head` only, II.1
  naive einsum. **Gate 5a:** vs brute-force Jacobian of `lm_head.weight` alone,
  atol ≈ 1e-10.
- **Step 2 — all Linears (½–1 d).** Add `c_attn` (fused QKV = just a Linear with
  out $=3d$), `c_proj`, `c_fc`, `mlp.c_proj`. Nothing for attention itself (II.6).
  **Gate 5b:** per-layer comparison printed, so failures name their layer.
- **Step 3 — Norms + biases (½ d).** II.5. **Gate 5c:** everything except embeddings.
- **Step 4 — embeddings (1 d).** II.2 + II.3. **Gate 5d:** full model, tying disabled —
  the complete-coverage gate.
- **Step 5 — re-enable tying, cross terms (1 d, hardest).** II.4 with the
  `remaining_counter` cache. **Gate 5e:** full tied model. On failure, diff against the
  gate-5d engine on the untied model — isolates cross-term logic by construction.
- **Step 6 — chunked variant + first real-size run (½ d).** Chunked II.1 gated against
  naive; one fp32 run at real nanoGPT-small shapes on the cluster → seeds the B3 overlay
  and the cost model's measured crossover (I.3).
- **Then:** B3 convergence overlay (Hadamard vs autogram-style vs scalar, same seed,
  curves coincide within fp noise — and the *explanation* is the point), B4 operator
  table (op | parameterized? | identity | rank | route | status | cost), head-seed
  shortcut gate (I.4), nanochat port (RMSNorm/GQA/Flash-Attn are covered by II.5/II.1/II.6
  respectively — the port is re-registration, not new math).

**Gate 5e passing = "layer ports finished" = the condition Luo's MuSiQue approval was
gated on.** Say exactly that when reporting it.
