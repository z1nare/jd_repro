"""PLACEHOLDER -- shared gate fixtures.

To implement:
  * a ``gate_model`` fixture built from ``models.configs.gate_tiny``,
    **parametrized over ``bias in (True, False)``** -- bias terms are separate
    summands in every identity, so a bias-free gate can pass while the biased
    path is wrong;
  * fp64 default dtype for gates;
  * ``dropout = 0.0`` asserted, not merely configured -- if the brute-force and
    hooked passes see different networks the comparison is vacuous;
  * a fixed seed, and ``torch.use_deterministic_algorithms`` where it does not
    conflict with the ops under test.

Standing rules for everything in this directory (design doc, Part III):
  * every gate is a committed script -- these become the transformer preflight,
    exactly as gates 1-4 became the CIFAR preflight;
  * no step starts before the previous gate passes;
  * on failure, shrink the hooked-module set before touching the math;
  * no ``torch.compile`` until all gates pass eager -- it is a performance knob,
    not a correctness tool.
"""

from __future__ import annotations
