"""Linear-layer Gramian identities.

Two regimes, one formula.  With ``A_i`` the upstream gradient of objective ``i``
and ``X`` the (objective-shared) layer input:

    dL_i/dW = sum_u A_i[u] X[u]^T
    G_ij   += <A_i A_j^T, X X^T>_F

* **rank-1** (``U = 1``, one position per objective -- the CIFAR/IWRM case):
  collapses to the Hadamard form ``G = (A A^T) . (X X^T)``.  Proven, gate 4.
* **sequence** (``U = BT`` positions -- the transformer case): needs the full
  Frobenius contraction against the shared kernel ``K_X = X X^T``.  Design doc
  II.1.  Not yet gated.
"""

from __future__ import annotations

import torch


def rank1_gramian(A: torch.Tensor, x_in: torch.Tensor, has_bias: bool) -> torch.Tensor:
    """``(A A^T) . (X X^T)`` plus the bias term ``A A^T``.

    ``A`` is ``[m, d_out]``, ``x_in`` is ``[m, d_in]``; one position per
    objective.  Accumulated in float64 -- the Gramian is only ``m x m``, so the
    precision is nearly free and it keeps the gate tolerances meaningful.
    """
    AAt = (A @ A.T).double()
    g = AAt * (x_in @ x_in.T).double()
    if has_bias:
        g = g + AAt
    return g


def backprop(A: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Propagate the upstream gradient through a Linear: ``A <- A W``.

    PyTorch stores ``weight`` as ``[d_out, d_in]``, so this yields ``[m, d_in]``.
    """
    return A @ weight


def sequence_gramian(*args, **kwargs):
    """PLACEHOLDER -- design doc II.1, the general sequence contraction.

    To implement: ``G_ij += <A_i A_j^T, K_X>_F`` where ``K_X = X X^T`` is
    ``[U, U]`` and **shared across all (i, j) pairs** -- compute it once per
    layer, not once per pair.  Bias term: ``G_ij += (sum_u A_i[u]) . (sum_v A_j[v])``,
    which is a plain ``[m, d_out]`` materialization.

    Ship two implementations, gated against each other:
      * naive einsum -- builds ``[m, m, U, U]``, fine at gate sizes, ~17 GiB at
        m=32 / U=2048, so gates only;
      * chunked per-pair -- holds ``[U, U]`` or row-blocks only, the shippable one.

    Gates: 5a (head only), 5b (all linears), 5f (chunked vs naive).
    """
    raise NotImplementedError("II.1 sequence contraction: not yet implemented")
