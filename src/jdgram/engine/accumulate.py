"""Streaming Gramian accumulation: one ``[m, m]`` buffer, contributions added in place.

The point is *lifetime*, not arithmetic.  A layer's ``(A, X)`` are the only large
tensors its identity needs, and they stop being needed the instant the ``[m, m]``
contribution exists.  Accumulating here -- from inside the layer's backward, as
the reverse pass sweeps -- means the engine holds one layer's captures at a time
instead of every layer's at once.

Shared parameters are the one exception.  Per-module Gramians are additive only
for disjoint parameters (see :mod:`jdgram.engine.hooks`), so a tied group's
members must be held until the last one fires and the four-term identity can run.
:class:`SharedGroup` counts them down; everything else frees immediately.

Ported in spirit from TorchJD 0.17.0 ``autogram/_gramian_accumulator.py`` and
``_gramian_computer.py`` (MIT, (c) Valerian Rey, Pierre Quinton).  The divergence
that matters: TorchJD's ``remaining_counter`` is per *module call*, so it never
sees two distinct modules sharing one parameter and silently drops the cross
terms.  :class:`SharedGroup` counts *group members*, which is what keeps the tied
``wte``/``lm_head`` Gramian exact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

import torch
from torch import Tensor, nn


class LayerCaptureLike(Protocol):
    name: str
    module: nn.Module
    A: Tensor
    X: object


@dataclass
class GramianAccumulator:
    """Running ``[m, m]`` total, plus the per-module breakdown gates diff against.

    ``add`` clones the first contribution rather than aliasing it, so a handler
    that returns a cached or view-backed tensor cannot be mutated underneath the
    caller by the subsequent ``add_``.
    """

    total: Tensor | None = None
    per_module: dict[str, Tensor] = field(default_factory=dict)
    per_shared_group: dict[frozenset[str], Tensor] = field(default_factory=dict)

    def add(self, contribution: Tensor) -> None:
        if self.total is None:
            self.total = contribution.clone()
        else:
            self.total.add_(contribution)

    def add_module(self, name: str, contribution: Tensor) -> None:
        self.per_module[name] = contribution
        self.add(contribution)

    def add_group(self, group: frozenset[str], contribution: Tensor) -> None:
        self.per_shared_group[group] = contribution
        self.add(contribution)

    def reset(self) -> None:
        self.total = None
        self.per_module.clear()
        self.per_shared_group.clear()


class SharedGroup:
    """Holds a tied group's captures until every member has been visited.

    ``remaining`` counts *members not yet seen this reverse pass*.  A group is
    fired exactly once, when it hits zero; the captures are dropped immediately
    afterwards so the tied pair is not carried through the rest of the pass.
    """

    def __init__(
        self,
        names: frozenset[str],
        handler: Callable[[dict[str, LayerCaptureLike]], Tensor],
    ) -> None:
        self.names = names
        self.handler = handler
        self.remaining = len(names)
        self.captures: dict[str, LayerCaptureLike] = {}

    def reset(self) -> None:
        self.remaining = len(self.names)
        self.captures.clear()

    def held_bytes(self) -> int:
        """Bytes currently pinned by this group -- reported by the profiler."""
        total = 0
        for capture in self.captures.values():
            for tensor in (capture.A, capture.X):
                if torch.is_tensor(tensor):
                    total += tensor.numel() * tensor.element_size()
        return total

    def submit(self, capture: LayerCaptureLike) -> Tensor | None:
        """Record one member; return the group Gramian once the last one lands."""
        if capture.name in self.captures:
            raise RuntimeError(
                f"{capture.name} was visited twice in one reverse pass while part of "
                f"shared group {sorted(self.names)}. Multi-call modules inside a tied "
                f"group need a summed-Jacobian formulation, not a summed Gramian."
            )
        self.captures[capture.name] = capture
        self.remaining -= 1
        if self.remaining > 0:
            return None
        try:
            return self.handler(dict(self.captures))
        finally:
            self.captures.clear()
