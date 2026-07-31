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
        A:      [m, T, d] upstream grads at the norm *output* (same device as model).
        X:      [m, T, d] *raw* hook input (pre-normalization), not xhat.
        has_bias: LayerNorm beta present; usually False for RMSNorm.
        eps:    match module.eps (nanoGPT LayerNorm uses 1e-5).
        center: True => subtract mean (LayerNorm); False => RMSNorm.
    Returns: G: [m, m] float64
    """
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

    g_gamma = (Aw * xhat).sum(dim=1)
    G = (g_gamma @ g_gamma.T).double()

    if has_bias:
        g_beta = Aw.sum(dim=1)
        G = G + (g_beta @ g_beta.T).double()

    return G
