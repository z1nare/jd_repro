from __future__ import annotations

import torch
def norm_gramian(
    A: torch.Tensor,
    X: torch.Tensor,
    has_bias: bool = True,
    eps: float = 1e-5,
    center: bool = True,
) -> torch.Tensor:
    """
    Gramian for LayerNorm (center=True) or RMSNorm (center=False).

    Args:
        A:      [m, T, d] upstream grads at the norm *output* (same device as model).
        X:      [m, T, d] *raw* hook input (pre-normalization), not xhat.
        has_bias: LayerNorm beta present; usually False for RMSNorm.
        eps:    match module.eps (nanoGPT LayerNorm uses 1e-5).
        center: True => subtract mean (LayerNorm); False => RMSNorm.
    Returns: G: [m, m] 
    """
    # Keep compute on whatever device A/X already sit on (GPU for cluster).
    # Do not .cpu() / .item() / host transfers here.
    dtype = torch.float64  # gates; for throughput runs use A.dtype instead
    A = A.to(dtype=dtype)
    X = X.to(dtype=dtype)

    # xhat on-device. biased var matches F.layer_norm (unbiased=False).
    if center:
        mu = X.mean(dim=-1, keepdim=True)
        var = X.var(dim=-1, keepdim=True, unbiased=False)
        xhat = (X - mu) * torch.rsqrt(var + eps)
    else:
        # RMSNorm: no mean subtract
        ms = X.pow(2).mean(dim=-1, keepdim=True)
        xhat = X * torch.rsqrt(ms + eps)

    # [m, d] then [m, m] — all device-local
    g_gamma = (A * xhat).sum(dim=1)
    G = g_gamma @ g_gamma.T

    if has_bias:
        g_beta = A.sum(dim=1)
        G = G + g_beta @ g_beta.T

    return G