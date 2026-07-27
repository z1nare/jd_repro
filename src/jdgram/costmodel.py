"""PLACEHOLDER -- measured cost model backing :mod:`jdgram.engine.router`.

Holds the crossover between the closed-form contraction and the materialization
route, as a function of ``(m, P_layer, U = B*T, dtype)``.

The numbers must be **loaded from measurement**, not hardcoded from arithmetic:
``bench/crossover.py`` sweeps m x layer size on the target card and writes
``results/transformer/crossover.json``; this module reads it and exposes a
predicate the router can call per layer.

Analytic starting estimates from the design doc (to be confirmed or refuted):
  * per-layer materialization at B=8, T=1024, fp32: ~0.21 GiB (m=4) to ~0.41 GiB
    (m=8) for the biggest MLP linear; ~3.5 GiB (m=4) to ~7.0 GiB (m=8) for the
    LM head;
  * the (BT)x(BT) contraction workspace is ~0.25 GiB dense, MBs when chunked,
    but costs ~7-12x the weight-gradient backward FLOPs at an interior linear.
"""

from __future__ import annotations


def crossover(*args, **kwargs):
    raise NotImplementedError("cost model: populated by bench/crossover.py")
