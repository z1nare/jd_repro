"""PLACEHOLDER -- autograd-graph node that fires a layer's Gramian identity.

Skeleton borrowed from TorchJD's ``AutogramNode``.  Two responsibilities:

1. Sit in the backward graph so the identity runs with ``A`` in hand and can
   release it immediately afterwards.
2. Hold the ``remaining_counter`` cache used for **weight sharing**: a tied
   parameter is reached by the backward pass more than once, and its Gramian
   contribution is only complete after the *last* site has been visited.
   Counting down the expected visits, accumulating ``(A, X)`` per site, and
   flushing on zero is what makes the II.4 cross terms computable at all.

First needed at gate 5a (single site); the counter logic first matters at
gate 5e (tied ``wte``/``lm_head``).
"""

from __future__ import annotations


class GramianNode:
    """PLACEHOLDER."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("AutogramNode skeleton: step 1 of the execution plan")
