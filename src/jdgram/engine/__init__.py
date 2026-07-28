"""Engines: the machinery that drives the identities over a whole model.

:mod:`~jdgram.engine.sequential` is the proven CIFAR-era manual reverse walk.
:mod:`~jdgram.engine.hooks` is the hook-driven replacement for transformers,
with :mod:`~jdgram.engine.node`, :mod:`~jdgram.engine.edges` and
:mod:`~jdgram.engine.registry` as its parts; the hook injection, phase flag and
gradient-edge bookkeeping are ported from TorchJD's autogram engine (MIT).

:mod:`~jdgram.engine.materialize` and :mod:`~jdgram.engine.router` stay
placeholders until ``bench/crossover.py`` produces measured data to route on.
"""

from jdgram.engine.hooks import (
    GramianResult,
    LayerCapture,
    ModuleHookManager,
    compute_gramian,
)
from jdgram.engine.registry import (
    collect_hookable_modules,
    dispatch,
    find_shared_parameters,
    positional_embedding_handler,
    register,
    register_predicate,
)

__all__ = [
    "GramianResult",
    "LayerCapture",
    "ModuleHookManager",
    "compute_gramian",
    "collect_hookable_modules",
    "dispatch",
    "find_shared_parameters",
    "positional_embedding_handler",
    "register",
    "register_predicate",
]
