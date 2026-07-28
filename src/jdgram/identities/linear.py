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


def sequence_gramian(A:torch.Tensor, X:torch.Tensor, has_bias:bool) -> torch.Tensor:
    """Exact Linear weight/bias Gramian for per-sequence objectives.
    A: [m, T, d_out], where A[i] = d(loss_i) / d(linear_output[i]).
    X: [m, T, d_in], where X[i] is the Linear input for sequence i.
    Returns:
        G: [m, m] float64.
    """
    if A.ndim != 3 or X.ndim != 3:
        raise ValueError("A and X must both be rank-3 tensors")
    if A.shape[:2] != X.shape[:2]:
        raise ValueError(
            f"A and X must share [m, T], got {A.shape[:2]} and {X.shape[:2]}"
        )
    m, t, _ = A.shape
    A = A.double()
    X = X.double()

    G = torch.zeros((m,m), dtype=torch.float64, device = X.device)

    #Flatten
    A_flat = A.reshape(m*t, -1) # [m*T, d_out]
    X_flat = X.reshape(m*t, -1) # [m*T, d_in]

    m_a = A_flat @ A_flat.T # [m*T, m*T]
    m_x = X_flat @X_flat.T # [m*T, m*T]
    # Fixing shapes
    K_A = m_a.reshape(m, t, m, t).permute(0, 2, 1, 3)   # [m, m, T, T]
    K_X = m_x.reshape(m, t, m, t).permute(0, 2, 1, 3)   # [m, m, T, T]

    G = (K_A * K_X).sum(dim=(-2, -1))       # [m, m]
    if has_bias:
        bias_grad = A.sum(dim=1)             # [m, d_out]
        G = G + bias_grad @ bias_grad.T
    return G
