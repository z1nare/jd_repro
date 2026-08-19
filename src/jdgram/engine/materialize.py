"""d-first / materialization route: form each objective's weight gradient, Gram it, discard.

Not a second-class citizen: for interior linears at modest m this is often
cheaper than the T-first ``[mT,mT]`` contraction (design doc I.3). Memory is a
transient ``[m, P_layer]`` block — not autojac's full-model Jacobian.

Both routes produce the identical ``G``; :mod:`jdgram.engine.router` picks.
"""

from __future__ import annotations

import torch

from jdgram.identities.precision import resolve_workspace_dtype

# Cap on the live [m, block, d_in] slice, in elements. Sized so the vocab head at
# m=16 holds a few hundred MiB rather than the ~2 GiB a whole [m, V, d] costs,
# while staying large enough that each bmm is still GEMM-shaped rather than a
# stream of tiny launches.
_BLOCK_ELEMS = 32 * 1024 * 1024


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
    m, _, d_out = Aw.shape
    d_in = Xw.shape[2]

    # Build and Gram B in slices along d_out instead of all at once. The Gramian
    # is a plain sum over parameters, so slicing the parameter axis and
    # accumulating is exact -- and it wins on three axes at once:
    #
    #   memory   peak is [m, block, d_in], not the full [m, d_out, d_in]. On
    #            GPT-2's vocab head at m=8: 998 MiB against 1984 MiB.
    #   accuracy each slice's [m,m] lands in float64 before being summed, so the
    #            long d_out reduction accumulates in fp64 rather than fp32.
    #            Measured against an fp64 reference at V=50257, m=8: 1.4e-07
    #            chunked against 2.3e-06 whole.
    #   time     unchanged to within noise -- same FLOPs, same bmm kernels.
    #
    # einsum("itp,itq->ipq", ...) was measured to dispatch to exactly this bmm,
    # so writing the bmm out costs nothing and makes the slicing obvious.
    block = max(1, min(d_out, _BLOCK_ELEMS // max(1, m * d_in)))
    G = torch.zeros(m, m, dtype=torch.float64, device=A.device)
    for start in range(0, d_out, block):
        B = torch.bmm(Aw[:, :, start:start + block].transpose(1, 2), Xw)
        G += (B.reshape(m, -1) @ B.reshape(m, -1).T).double()
    if has_bias:
        b = Aw.sum(dim=1)
        G = G + (b @ b.T).double()
    return G
