"""Upstream-gradient seeds: the ``A`` matrix that starts the reverse recursion.

Every layer identity in :mod:`jdgram.identities` consumes an upstream gradient
``A[i]`` = d(objective i) / d(that layer's output).  The seed is ``A`` at the
network output.  Getting the seed wrong invalidates every downstream identity,
so the seeds live here explicitly rather than being inlined in an engine.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def softmax_cross_entropy(logits: torch.Tensor, targets: torch.Tensor,
                          num_classes: int | None = None) -> torch.Tensor:
    """Seed for per-instance cross-entropy objectives (the CIFAR/IWRM setting).

    For ``L = -log softmax(z)[y]`` the standard identity is
    ``dL/dz = softmax(z) - onehot(y)``.  One objective per row of ``logits``.
    """
    C = num_classes or logits.shape[-1]
    return torch.softmax(logits, dim=-1) - F.one_hot(targets, num_classes=C).to(logits.dtype)


def grpo(*args, **kwargs):
    """PLACEHOLDER -- seed for per-objective GRPO losses (design doc I.4).

    To implement: return the shared per-token score direction
    ``s = d log pi / dz`` together with the per-objective scalars
    ``c_i(b, t) = advantage x clip-mask``, so the head-layer Gramian collapses
    to ``G_ij = c_i^T (S S^T . X X^T) c_j`` -- one (BT x BT) kernel reused
    across all m^2 pairs, instead of an [BT, V] block per objective.

    Returning ``(c, s)`` rather than a dense ``A`` is the whole point: dense
    ``A`` at the vocab head is ~4.7 GiB fp32 at V=152k.

    Not gated yet: the gate has to come with the derivation.
    """
    raise NotImplementedError("GRPO seed: design doc I.4, not yet derived-and-gated")
