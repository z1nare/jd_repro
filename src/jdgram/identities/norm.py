"""PLACEHOLDER -- LayerNorm / RMSNorm (design doc II.5).

No trick needed here.  The per-objective parameter gradients are ``d``-vectors::

    dL_i/d(gamma) = sum_u A_i[u] . xhat[u]        (and analogously for beta)

so materializing ``[m, d]`` and taking ``M M^T`` is already cheap -- ``d`` is
small.  This is the *materialization route* of the I.3 cost model, chosen on
purpose rather than by omission.

BatchNorm is deliberately out of scope (cross-instance coupling breaks the
per-objective decomposition).  Irrelevant for the transformer target, which
uses LayerNorm/RMSNorm.

Gate: gates/test_5c_norms_bias.py.
"""

from __future__ import annotations


def norm_gramian(*args, **kwargs):
    raise NotImplementedError("II.5 norm materialization: not yet implemented")
