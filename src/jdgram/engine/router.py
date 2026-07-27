"""PLACEHOLDER -- per-layer route selection (design doc I.3).

The reframed contribution is not "closed form everywhere".  It is one exact
engine that picks the cheapest correct identity per layer::

    route(l) = closed form        if m*P_l >> (BT)^2 workspace
                                  (vocab head, tied embedding, or large m)
               materialize J J^T  if m*P_l is small
                                  (interior linears at m <= 8)

Both routes are mathematically identical, so the choice is pure engineering --
which is exactly why the crossover must come from ``jdgram.costmodel`` (i.e.
from ``bench/crossover.py`` measurements) rather than from taste.  The
selection rule *with* its measured crossover is the presentable systems result;
the rule without measurements is an opinion.
"""

from __future__ import annotations


def route(*args, **kwargs):
    raise NotImplementedError("cost-model routing: step 7 of the execution plan")
