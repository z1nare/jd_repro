"""PLACEHOLDER -- timing, peak memory and capacity at real transformer shapes.

The transformer counterpart of the frozen ``legacy/cifar/iwrm_bench.py``.  It
should reuse that harness's *protocol* and not its code: warmup then timed
iterations bracketed by ``cuda.synchronize()``, peak memory via
``max_memory_allocated``, OOM caught per configuration and recorded as a result
rather than crashing the sweep, and an explicit ``del`` + ``gc.collect()`` +
``empty_cache()`` between configurations so one engine's OOM cannot contaminate
the next one's measurement.

That last point is not hypothetical -- it was a real bug in the CIFAR sweep and
it silently produced wrong OOM boundaries until it was fixed.

Sweep axes that matter here: ``m`` (2-8 for GRPO, 16-64 for per-rollout
objectives), sequence length ``T``, batch ``B``, and route (closed form /
materialize / router's choice).  Writes to ``results/transformer/``.

Success criterion to aim at: runnable JD at 1.5-2B parameters on 24-48 GB.
"""

from __future__ import annotations


def main():
    raise NotImplementedError("transformer benchmark: step 6 of the execution plan")
