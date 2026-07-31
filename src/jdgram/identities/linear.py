"""Linear-layer Gramian identities.

Two regimes, one formula.  With ``A_i`` the upstream gradient of objective ``i``
and ``X`` the (objective-shared) layer input:

    dL_i/dW = sum_u A_i[u] X[u]^T
    G_ij   += <A_i A_j^T, X X^T>_F

* **rank-1** (``U = 1``, one position per objective -- the CIFAR/IWRM case):
  collapses to the Hadamard form ``G = (A A^T) . (X X^T)``.  Proven, gate 4.
* **sequence** (``U = BT`` positions -- the transformer case): T-first
  contraction here; d-first lives in :func:`jdgram.engine.materialize.materialized_gramian`.

Heavy workspace runs in ``workspace_dtype`` (default float32); the final
``[m, m]`` is always float64. See :mod:`jdgram.identities.precision`.
"""

from __future__ import annotations

import torch

from jdgram.identities.precision import resolve_workspace_dtype


def rank1_gramian(
    A: torch.Tensor,
    x_in: torch.Tensor,
    has_bias: bool,
    *,
    workspace_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """``(A A^T) . (X X^T)`` plus the bias term ``A A^T``.

    ``A`` is ``[m, d_out]``, ``x_in`` is ``[m, d_in]``; one position per
    objective.  Workspace in ``workspace_dtype``; return ``[m, m]`` float64.
    """
    wd = resolve_workspace_dtype(workspace_dtype)
    Aw = A.to(wd)
    Xw = x_in.to(wd)
    AAt = Aw @ Aw.T
    g = (AAt * (Xw @ Xw.T)).double()
    if has_bias:
        g = g + AAt.double()
    return g


def backprop(A: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Propagate the upstream gradient through a Linear: ``A <- A W``.

    PyTorch stores ``weight`` as ``[d_out, d_in]``, so this yields ``[m, d_in]``.
    """
    return A @ weight


def sequence_gramian(
    A: torch.Tensor,
    X: torch.Tensor,
    has_bias: bool,
    *,
    workspace_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """T-first Linear Gramian, one objective row at a time.

    A: [m, T, d_out], X: [m, T, d_in] → G: [m, m] float64.

    ``G[i,j] = sum_{t,s} <A_i[t], A_j[s]> * <X_i[t], X_j[s]>``.  Written that way
    the ``i`` index is free: row ``i`` only needs ``A_i`` against *all* of ``A``,
    which is ``[T, mT]`` -- not the full ``[mT, mT]``.  So the loop below never
    materializes an ``[mT, mT]`` kernel at all.

    Peak workspace is ``3 m T^2`` (``K_A``, ``K_X``, their product) instead of the
    ``4 m^2 T^2`` the whole-kernel form actually costs -- an ``m``-fold reduction,
    at the price of ``m`` GEMMs instead of one.  Each is still ``[T, d] x [d, mT]``,
    so this stays GEMM-shaped rather than becoming launch-bound.

    The ``4x`` is not a typo and is why this was rewritten: the previous
    implementation built both ``[mT, mT]`` kernels and contracted them with
    ``einsum("itjs,itjs->ij", ...)`` in the belief that einsum fuses the product.
    It does not -- it materializes a *clone of each operand* first (verified by
    dispatch trace), so the peak was two live kernels plus two clones.  Even the
    naive ``(K_A * K_X).sum(...)`` it was avoiding would have been cheaper at
    ``3 m^2 T^2``.
    """
    if A.ndim != 3 or X.ndim != 3:
        raise ValueError("A and X must both be rank-3 tensors")
    if A.shape[:2] != X.shape[:2]:
        raise ValueError(
            f"A and X must share [m, T], got {A.shape[:2]} and {X.shape[:2]}"
        )
    m, t, _ = A.shape
    wd = resolve_workspace_dtype(workspace_dtype)
    Aw = A.to(wd)
    Xw = X.to(wd)

    A_flat = Aw.reshape(m * t, -1)
    X_flat = Xw.reshape(m * t, -1)
    G = torch.zeros(m, m, dtype=torch.float64, device=A.device)
    for i in range(m):
        k_a = Aw[i] @ A_flat.T                        # [t, m*t]
        k_x = Xw[i] @ X_flat.T                        # [t, m*t]
        G[i] = (k_a * k_x).view(t, m, t).sum(dim=(0, 2)).double()
    if has_bias:
        b = Aw.sum(dim=1)
        G = G + (b @ b.T).double()
    return G
