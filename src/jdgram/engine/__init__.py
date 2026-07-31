"""Engines: the machinery that drives the identities over a whole model.

:mod:`~jdgram.engine.hooks` is the entry point, with
:mod:`~jdgram.engine.node`, :mod:`~jdgram.engine.edges`,
:mod:`~jdgram.engine.accumulate` and :mod:`~jdgram.engine.registry` as its
parts; the hook injection, phase flag and gradient-edge bookkeeping are ported
from TorchJD's autogram engine (MIT), as is the reverse strategy itself -- one
ones-seeded backward, each module's contribution fired inside its own backward
hook and its captures released immediately.

:mod:`~jdgram.engine.materialize` is the d-first route, which forms
``[m, P_layer]`` and squares it -- structurally what autogram does for every
module. The T-first route in :mod:`jdgram.identities.linear` has no autogram
analogue: it contracts positions first and never forms that block.
:mod:`~jdgram.engine.router` picks between them per layer (``m*T^2`` vs
``P_layer``); :mod:`~jdgram.costmodel` is where a *measured* rule would live.
"""

from jdgram.engine.accumulate import GramianAccumulator, SharedGroup
from jdgram.engine.hooks import (
    GramianResult,
    LayerCapture,
    ModuleHookManager,
    compute_gramian,
)
from jdgram.engine.materialize import materialized_gramian
from jdgram.engine.registry import (
    collect_hookable_modules,
    dispatch,
    find_shared_parameters,
    positional_embedding_handler,
    register,
    register_predicate,
)
from jdgram.engine.router import force_route, route

__all__ = [
    "GramianAccumulator",
    "GramianResult",
    "LayerCapture",
    "ModuleHookManager",
    "SharedGroup",
    "compute_gramian",
    "collect_hookable_modules",
    "dispatch",
    "find_shared_parameters",
    "force_route",
    "materialized_gramian",
    "positional_embedding_handler",
    "register",
    "register_predicate",
    "route",
]
