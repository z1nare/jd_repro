"""PLACEHOLDER -- the materialization route: form J_l, add J_l J_l^T, discard.

The autogram-style fallback, and *not* a second-class citizen: per the I.3 cost
model this is the **correct** route for interior linears at small m.  At
m <= 8 a transient [m, P_l] block for a Qwen-1.5B MLP linear is 0.2-0.4 GiB,
which is not a memory wall, and it costs no extra FLOPs beyond the m-seed
backward -- whereas the II.1 contraction there would spend roughly 7x (m=4) to
12x (m=8) the weight-gradient backward cost to save memory that did not need
saving.

Both routes produce the identical ``G``, so every gate must pass with routing
on and with routing forced to either side.  That equivalence is the regression
net for :mod:`jdgram.engine.router`.
"""

from __future__ import annotations


def materialized_gramian(*args, **kwargs):
    raise NotImplementedError("materialization route: step 6 of the execution plan")
