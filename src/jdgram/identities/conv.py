"""Grouped/depthwise Conv2d Gramian identity.

Derivation of the vectorized form (why this is the same math):
  a per-group loop computed, for each g:
      J_g = A_g_reshaped @ unfolded_g^T          [m, Cout_g, Cin_g*kH*kW]
      gramian += flatten(J_g) @ flatten(J_g)^T
  Summing ``J_g J_g^T`` over g is exactly the Frobenius inner product of the
  full per-example weight-Jacobian rows, i.e. what one gets by stacking the
  per-group ``J_g`` blocks into ``J = [m, G*Cout_g*Cin_g*kH*kW]`` and doing a
  single ``J @ J^T``.  The einsum does all G small matmuls as one batched
  kernel launch instead of G launches from Python.

Memory is identical to the loop version -- the ``[m, P_layer]`` block ``J`` is
materialized either way (grouped-conv ``P_layer`` is small: ``Cout * Cin_g * kH*kW``).
The win is kernel-launch count and Python overhead, i.e. exactly the
dispatch-bound regime the Perfetto traces showed.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def gramian(A: torch.Tensor, x_in: torch.Tensor, layer: nn.Conv2d) -> torch.Tensor:
    """Weight (and bias) Gramian contribution of one Conv2d, as ``[m, m]`` float64."""
    m = x_in.shape[0]
    G = layer.groups
    Cout_g = layer.out_channels // G

    unf = F.unfold(x_in,
                   kernel_size=layer.kernel_size,
                   padding=layer.padding,
                   stride=layer.stride)                  # [m, Cin*kk, L]
    L = unf.shape[-1]
    unf = unf.view(m, G, -1, L)                          # [m, G, Cin_g*kk, L]
    A_r = A.reshape(m, G, Cout_g, L)                     # [m, G, Cout_g, L]
    J = (A_r @ unf.transpose(-1, -2)).reshape(m, -1)
    g = (J @ J.T).double()

    if layer.bias is not None:
        temp = A.sum(dim=(2, 3))
        g = g + (temp @ temp.T).double()
    return g


def backprop(A: torch.Tensor, x_in: torch.Tensor, layer: nn.Conv2d) -> torch.Tensor:
    """Propagate the upstream gradient to the conv's input."""
    return torch.nn.grad.conv2d_input(
        x_in.shape, layer.weight, A,
        stride=layer.stride, padding=layer.padding,
        dilation=layer.dilation, groups=layer.groups,
    )
