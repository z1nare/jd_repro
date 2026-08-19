from __future__ import annotations

import torch

from jdgram.identities.precision import resolve_workspace_dtype


def norm_gramian(
    A: torch.Tensor,
    X: torch.Tensor,
    has_bias: bool = True,
    eps: float = 1e-5,
    center: bool = True,
    *,
    workspace_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """
    Gramian for LayerNorm (center=True) or RMSNorm (center=False).

    Args:
        A:      [m, ..., d] upstream grads at the norm *output* (same device as model).
        X:      [m, ..., d] *raw* hook input (pre-normalization), not xhat.
        has_bias: LayerNorm beta present; usually False for RMSNorm.
        eps:    match module.eps (nanoGPT LayerNorm uses 1e-5).
        center: True => subtract mean (LayerNorm); False => RMSNorm.
    Returns: G: [m, m] float64

    Any rank >= 2 is accepted, not just ``[m, T, d]``. Normalisation is over the
    last axis whatever the rank, and gamma's gradient sums every axis between the
    objective and the feature. Qwen3.5's per-head ``q_norm``/``k_norm`` are called
    on ``[m, T, n_heads, head_dim]``: with a hardcoded ``sum(dim=1)`` that left a
    rank-3 ``g_gamma``, whose ``.T`` reverses all three axes, and the Gramian then
    failed on a shape mismatch -- loudly here, but ``.T`` on a >2-D tensor is
    deprecated rather than an error, so it is worth not relying on that.
    """
    if A.ndim < 2 or X.ndim < 2:
        raise ValueError(
            f"A and X need rank >= 2 ([m, ..., d]); got {tuple(A.shape)} and {tuple(X.shape)}"
        )
    if A.shape != X.shape:
        raise ValueError(
            f"A and X must have the same shape; got {tuple(A.shape)} and {tuple(X.shape)}"
        )
    wd = resolve_workspace_dtype(workspace_dtype)
    Aw = A.to(wd)
    Xw = X.to(wd)

    # xhat on-device. biased var matches F.layer_norm (unbiased=False).
    if center:
        mu = Xw.mean(dim=-1, keepdim=True)
        var = Xw.var(dim=-1, keepdim=True, unbiased=False)
        xhat = (Xw - mu) * torch.rsqrt(var + eps)
    else:
        ms = Xw.pow(2).mean(dim=-1, keepdim=True)
        xhat = Xw * torch.rsqrt(ms + eps)

    # Every axis between the objective and the feature. Empty for a bare [m, d],
    # where there is nothing to reduce -- and `sum(dim=())` reduces *everything*
    # in torch, so that case has to skip the call rather than pass an empty tuple.
    mid = tuple(range(1, Aw.ndim - 1))

    prod = Aw * xhat
    g_gamma = prod.sum(dim=mid) if mid else prod
    G = (g_gamma @ g_gamma.T).double()

    if has_bias:
        g_beta = Aw.sum(dim=mid) if mid else Aw
        G = G + (g_beta @ g_beta.T).double()

    return G
