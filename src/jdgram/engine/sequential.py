"""Manual reverse-walk Gramian engine for flat ``nn.Sequential`` models.

This is the CIFAR-era engine, extracted from the top-level ``hadamard.py``: it
caches every layer input on the forward pass, seeds the upstream gradient at
the logits, then walks the layers in reverse, accumulating each layer's Gramian
contribution from :mod:`jdgram.identities`.

It is kept because it is *proven* (old preflight gate 4, re-run by
``gates/test_legacy_cifar.py``) and because it is the reference the hook-driven
engine must agree with on the CIFAR net.

It does **not** generalise to transformers, and it is not meant to: it assumes
a flat ``nn.Sequential``, no weight sharing, and one position per objective.
The replacement is :mod:`jdgram.engine.hooks` + :mod:`jdgram.engine.registry`.
"""

from __future__ import annotations

import torch
from torch import nn

from jdgram import seeds
from jdgram.identities import conv as conv_id
from jdgram.identities import linear as linear_id
from jdgram.identities import propagation as prop


def sequential_gramian(model: nn.Sequential, x: torch.Tensor, y: torch.Tensor,
                       num_classes: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(G, logits)`` with ``G`` the exact ``[m, m]`` Gramian in float64.

    One objective per row of ``x`` (per-instance cross-entropy, the IWRM setting).
    """
    m = x.shape[0]
    layers = list(model.children())

    cache: dict[nn.Module, torch.Tensor] = {}
    out = x
    for layer in layers:
        cache[layer] = out
        out = layer(out)

    A = seeds.softmax_cross_entropy(out, y, num_classes)
    gramian = torch.zeros(m, m, dtype=torch.float64, device=x.device)

    for layer in reversed(layers):
        if isinstance(layer, nn.Linear):
            x_in = cache[layer]
            gramian += linear_id.rank1_gramian(A, x_in, layer.bias is not None)
            A = linear_id.backprop(A, layer.weight)

        elif isinstance(layer, nn.ELU):
            A = prop.elu_backprop(A, cache[layer])

        elif isinstance(layer, nn.MaxPool2d):
            A = prop.maxpool2d_backprop(A, cache[layer], layer)

        elif isinstance(layer, nn.Conv2d):
            x_in = cache[layer]
            gramian += conv_id.gramian(A, x_in, layer)
            A = conv_id.backprop(A, x_in, layer)

        elif isinstance(layer, nn.Flatten):
            A = prop.flatten_backprop(A, cache[layer].shape)

    return gramian, out


# Name used by the frozen legacy harness (legacy/cifar/iwrm_bench.py).
algorithm3 = sequential_gramian
