"""PLACEHOLDER -- hook plumbing that captures (A, X) per layer.

Replaces the manual reverse walk in :mod:`jdgram.engine.sequential`, which
cannot survive contact with a transformer (nested blocks, shared weights,
non-Sequential control flow).

To implement:
  * a **forward hook** per registered module storing the layer input ``X``
    (shared across all m objectives -- one forward pass);
  * a **backward** path that exposes the m-seeded upstream gradient ``A`` at
    that module, so the layer's identity can fire once and the captured
    tensors be freed immediately;
  * borrow TorchJD's forward-hook + ``AutogramNode`` structure rather than
    inventing one (see :mod:`jdgram.engine.node`).

The whole point is that ``G`` comes out of **one m-seed backward** with no
``O(m P)`` gradient storage -- if a hook holds ``A`` and ``X`` alive past the
layer's own identity call, that property is lost.

First use: gate 5a, registered on ``lm_head`` only.
"""

from __future__ import annotations


class GramianHooks:
    """PLACEHOLDER -- context manager installing/removing the capture hooks."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("hook plumbing: step 1 of the execution plan")
