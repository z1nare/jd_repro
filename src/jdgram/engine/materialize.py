"""d-first / materialization route: form each objective's weight gradient, Gram it, discard.

Not a second-class citizen: for interior linears at modest m this is often
cheaper than the T-first ``[mT,mT]`` contraction (design doc I.3). Memory is a
transient ``[m, P_layer]`` block — not autojac's full-model Jacobian.

Both routes produce the identical ``G``; :mod:`jdgram.engine.router` picks.
"""

from __future__ import annotations

import torch

from jdgram.identities.precision import resolve_workspace_dtype


def materialized_gramian(
    A: torch.Tensor,
    X: torch.Tensor,
    has_bias: bool,
    *,
    workspace_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """d-first Linear Gramian: ``B[i] = Σ_t A[i,t]⊗X[i,t]``, then ``G = B:B``.

    A: [m, T, d_out], X: [m, T, d_in] → G: [m, m] float64.
    """
    if A.ndim != 3 or X.ndim != 3:
        raise ValueError("A and X must both be rank-3 tensors")
    if A.shape[:2] != X.shape[:2]:
        raise ValueError(
            f"A and X must share [m, T], got {A.shape[:2]} and {X.shape[:2]}"
        )
    wd = resolve_workspace_dtype(workspace_dtype)
    Aw = A.to(wd)
    Xw = X.to(wd)
    B = torch.einsum("itp,itq->ipq", Aw, Xw)  # [m, d_out, d_in]
    # Flatten-then-matmul: contracting all trailing axes against the same axes of
    # the other operand is exactly a matmul on the flattened view, and reshape on
    # a contiguous tensor is a view. Measurably leaner than einsum here even
    # though this particular label order happens not to clone.
    B_flat = B.reshape(B.shape[0], -1)
    G = (B_flat @ B_flat.T).double()
    if has_bias:
        b = Aw.sum(dim=1)
        G = G + (b @ b.T).double()
    return G
