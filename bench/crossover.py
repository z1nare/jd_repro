"""PLACEHOLDER -- measure where the closed form beats materialization.

Produces the numbers behind :mod:`jdgram.costmodel`, and therefore behind the
whole hybrid-engine framing: without a measured crossover, ``router.route()``
is taste with extra steps.

Sweep ``m`` in {2, 4, 8, 16, 32} against representative layer sizes (interior
MLP linear, attention projections, vocab head) at target ``B``/``T``, timing
both routes and recording peak memory for each.  Report the crossover in terms
the router can evaluate cheaply at registration time, and write
``results/transformer/crossover.json``.

Two results are worth stating plainly whichever way they come out: the expected
~7-12x contraction overhead at interior linears for m <= 8, and the head-layer
case where materialization needs 3.5-7 GiB transient and the closed form needs
megabytes.  If the measurements contradict the analysis, the measurements win
and the design doc gets corrected.
"""

from __future__ import annotations


def main():
    raise NotImplementedError("crossover measurement: step 7 of the execution plan")
