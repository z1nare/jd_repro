"""Autograd node that captures a layer's upstream gradient during backward.

Ported from TorchJD 0.17.0 ``torchjd/autogram/_module_hook_manager.py``
(``AutogramNode``), MIT licence, (c) Valerian Rey, Pierre Quinton.

Identity on forward, side effect on backward.  The node exists so that a
layer's upstream gradient ``A`` is observable at the exact point in the reverse
pass where it is live, without holding the whole Jacobian anywhere.

Divergence from upstream: TorchJD fires the Gramian computation *inside*
``backward`` and accumulates immediately.  Here ``backward`` only hands the
gradients to a capture object; the identities fire once per module after the
driver's m-objective loop.  That is a deliberate gate-scale simplification --
see :mod:`jdgram.engine.hooks` for the orchestration and why.
"""

from __future__ import annotations

from typing import Any, Protocol

import torch
from torch import Tensor


class GradientSink(Protocol):
    """Receives one backward visit's gradients for a single module."""

    def record_backward(self, grad_outputs: tuple[Tensor, ...]) -> None: ...


class GramianNode(torch.autograd.Function):
    generate_vmap_rule = True

    @staticmethod
    def forward(_sink: GradientSink, *rg_tensors: Tensor) -> tuple[Tensor, ...]:
        # detach, not `return rg_tensors`: handing back the inputs verbatim makes
        # autograd treat this as a no-op and the node never enters the graph.
        return tuple(t.detach() for t in rg_tensors)

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple, _output: Any) -> None:
        ctx.sink = inputs[0]

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Tensor) -> tuple:
        ctx.sink.record_backward(tuple(g.detach() for g in grad_outputs))
        # Pass gradients through untouched: this node must not perturb the
        # reverse pass it is observing.
        return None, *grad_outputs
