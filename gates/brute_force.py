"""PLACEHOLDER -- ground truth for every transformer gate.  Write this first.

Nothing else in ``gates/`` is meaningful until this is trusted, so it is step 0
of the execution plan and it gets written before any engine code.

To implement:
  * build the m objectives (``gate_tiny``: 4 per-sequence mean-CE losses);
  * for each objective, ``torch.autograd.grad(L_i, params, retain_graph=True)``;
  * flatten each objective's gradients using the **canonical ordering** from
    :mod:`jdgram.utils.flatten` -- not ``model.parameters()`` order chosen
    ad hoc here, or gate 5b cannot attribute a failure to a layer;
  * stack into ``J`` of shape ``[m, P]`` and return ``G_true = J @ J.T``;
  * also return the per-layer sub-block Gramians, so gates can print a per-layer
    delta table rather than one scalar.

fp64 throughout.  Dropout must be 0.0 or the brute-force and hooked passes see
different networks.

Sanity check before trusting it: ``diag(G_true)[i]`` must equal
``sum of grad_i.pow(2).sum()`` over parameters, to machine precision.  Tied
parameters must be counted **once** -- the true gradient of a shared ``W`` is
the sum over its sites (design doc II.4), not two separate entries.
"""

from __future__ import annotations


def true_gramian(*args, **kwargs):
    raise NotImplementedError("brute-force ground truth: step 0 of the execution plan")
