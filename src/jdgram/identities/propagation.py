"""Parameter-free ops: propagation rules only, no Gramian terms (design doc II.6).

Softmax/SDPA, ELU/SiLU/GELU, residual adds, RoPE, pooling and reshapes hold no
parameters, so they contribute nothing to ``G``.  They only shape how the
upstream gradient ``A`` propagates.  This is why the engine never needs to
"port attention": attention's parameters are its four Linears.

The ELU/MaxPool2d/Flatten rules below came from ``hadamard.py`` and are already
covered by the legacy CIFAR gate.  Transformer-side rules (RoPE, SDPA,
residual) are not needed as explicit entries -- ordinary autograd propagates
``A`` through them once the engine is hook-driven rather than a manual reverse
walk.  They are listed in ``docs/operator_table.md`` for completeness.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def elu_backprop(A: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """ELU derivative applied to the upstream gradient; ``z`` is the ELU *input*."""
    return A * torch.where(z > 0, torch.ones_like(z), torch.exp(z))


def maxpool2d_backprop(A: torch.Tensor, x_in: torch.Tensor, layer: nn.MaxPool2d) -> torch.Tensor:
    """Route the upstream gradient back through the argmax positions."""
    _, indices = F.max_pool2d(
        x_in, kernel_size=layer.kernel_size, stride=layer.stride,
        padding=layer.padding, return_indices=True,
    )
    return F.max_unpool2d(
        A, indices, kernel_size=layer.kernel_size, stride=layer.stride,
        padding=layer.padding, output_size=x_in.shape[-2:],
    )


def flatten_backprop(A: torch.Tensor, in_shape: torch.Size) -> torch.Tensor:
    return A.reshape(in_shape)
