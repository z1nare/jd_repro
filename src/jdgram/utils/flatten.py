"""PLACEHOLDER -- the canonical parameter ordering.  Write this before any gate.

Both ``gates/brute_force.py`` and the engine must agree on how parameters are
ordered and sliced, or a gate failure says "the Gramian is wrong" without
saying *which layer* is wrong -- and gate 5b explicitly requires per-layer
attribution so failures name their layer.

To implement: return ``(name, param, slice)`` triples in a deterministic order
(module traversal order, parameters in ``named_parameters()`` order, tied
parameters appearing **once**), plus helpers to flatten a per-objective
gradient dict into a single ``[m, P]`` row block and to slice a ``[m, P]``
block back into per-layer sub-blocks keyed by name.

Tied parameters are the subtle case: they must appear once in the flattening,
because the true per-objective gradient of a shared ``W`` is the *sum* over its
sites (design doc II.4).  Counting it twice silently changes the ground truth.
"""

from __future__ import annotations


def param_layout(*args, **kwargs):
    raise NotImplementedError("canonical flattening: step 0 of the execution plan")
