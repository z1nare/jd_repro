# Gramian via Hadamard Factorization — derivation and provenance

## Starting point: JD paper, Section 6

The [JD paper](https://arxiv.org/pdf/2406.16232) states that once aggregator weights are a
function of the Gramian alone — not the full Jacobian $J$ — the efficient
path is to compute

$$
G = J J^\top \in \mathbb{R}^{m \times m}
$$

directly from pairwise gradient inner products $G_{ij} = \langle g_i, g_j
\rangle$, rather than materializing $J \in \mathbb{R}^{m \times P}$. That's
their own stated motivation for an efficient implementation, and the
starting question for this method: **is there a closed form for $G_{ij}$
that never touches $P$ at all, for the layer types we actually use?**

`autogram` (TorchJD's own engine) already answers this question one way: it
never stores the *global* Jacobian, and per layer it materialises $J_\ell \in
\mathbb{R}^{m \times P_\ell}$ once, then computes $G_\ell = J_\ell J_\ell^\top$
as a single matrix multiplication — already an efficient BLAS call, not a
loop. Its real cost is **materialising $J_\ell$ in the first place**, an
$[m, P_\ell]$ tensor, before that matmul can run at all.

*(A genuinely column-wise, per-group Python-loop recursion — summing $G_\ell
= \sum_k j_k j_k^\top$ one column/group at a time — was my own first
prototype of this, before vectorising it. That loop is what became
dispatch-bound on GPU; it isn't what `autogram` does.)*

## The seed: cross-entropy gradient w.r.t. logits

The recursion needs a starting upstream gradient $A$ at the network's
output. For cross-entropy loss on logits $z$ with integer label $y$,

$$
L = -\log\big(\mathrm{softmax}(z)_y\big)
\quad\Rightarrow\quad
\frac{\partial L}{\partial z} = \mathrm{softmax}(z) - \mathrm{onehot}(y),
$$

the standard softmax-cross-entropy gradient identity. Every layer-wise term
below is only correct if this seed is correct — it's stated explicitly here
because getting it wrong (e.g. assuming an MSE-shaped seed) was the first of
four real bugs found while building this.

## The layer-wise identity (Linear layers)

For a Linear layer, the per-instance weight gradient is an **outer product**:

$$
g_i = a_i x_i^\top, \qquad a_i \in \mathbb{R}^{d_\text{out}},\ \ x_i \in
\mathbb{R}^{d_\text{in}}.
$$

The pairwise inner product of two such gradients (as flattened vectors)
factors cleanly:

$$
\langle g_i, g_j \rangle
= \langle a_i x_i^\top,\ a_j x_j^\top \rangle_F
= \langle a_i, a_j \rangle \cdot \langle x_i, x_j \rangle.
$$

*(Frobenius inner product of two rank-1 matrices is the product of the two
vector inner products — a standard identity for outer products.)*

Stacking this over the whole batch turns the *global* Gramian contribution
of the layer into a **Hadamard (elementwise) product of two small $m \times
m$ matrices**:

$$
G_W = (A A^\top) \odot (X X^\top), \qquad
A = \begin{bmatrix} a_1^\top \\ \vdots \\ a_m^\top \end{bmatrix}, \quad
X = \begin{bmatrix} x_1^\top \\ \vdots \\ x_m^\top \end{bmatrix}.
$$

The bias term is $G_b = A A^\top$ (the per-instance bias gradient is $a_i$
directly, no outer product). Upstream propagation for the next layer back
is $A \leftarrow A W$ (matches PyTorch's $[d_\text{out}, d_\text{in}]$
weight-shape convention, so shapes work out to $[m, d_\text{in}]$).

**Why this is cheaper, precisely:** $J_\ell$ (shape $[m, d_\text{out}
\cdot d_\text{in}]$) is never formed. Only $A A^\top$ and $X X^\top$
(each $[m,m]$) are computed, then multiplied elementwise. This is what makes
it cheaper than `autogram`, specifically — not a faster way to run a sum
that was already a single matmul, but a different closed form that skips
materialising $J_\ell$ at all.

## Grouped / depthwise Conv2d

The same outer-product structure holds per-group after an `unfold`, so each
group's contribution factors the same way; summing over groups (vectorised
as one batched `einsum` rather than a Python loop) gives the full layer's
Gramian contribution without forming the per-group Jacobian either.

## Provenance of the identity

The outer-product fact itself — $g_i = a_i x_i^\top$, and pairwise
quantities of such gradients decompose into two small Gram-like objects —
is not new. It is the same identity behind:

> Goodfellow, [*"Efficient Per-Example Gradient Computations"*](https://arxiv.org/pdf/1510.01799)

**Note on the source:** this is a 2-page arXiv technical report, not a
peer-reviewed conference paper — that's why it's short. It's the
commonly-cited origin of this specific technique in the per-example-gradient
/ DP-SGD literature (e.g. cited directly by Rochette, Manoel & Tramel's CNN
extension, arXiv:1912.06015). Goodfellow's version computes only the
**diagonal** — per-example gradient norms, $s_j = \|a_j\|^2 \|x_j\|^2$ in
his notation — for exactly this reason: importance-sampling by gradient
norm. The Gramian's **off-diagonal** terms are the direct extension of the
same identity to the full pairwise case, which is what JD's aggregator
weights actually need.

## Correctness

Verified numerically, at matching precision, against `autogram`'s own
Gramian on the paper-exact architecture (preflight gate 4):

```
max abs diff = 2.8e-14   (both engines in float64, batch=8)
```

Comparing at mismatched precision (this implementation's float64
accumulation against `autogram` run on a native float32 model) still passes
but is a much weaker check — diff ≈ 4.6e-5 against a correspondingly loose
tolerance, mostly reflecting float32 rounding noise on the reference side
rather than the true achievable agreement. The number above is the one that
means something.