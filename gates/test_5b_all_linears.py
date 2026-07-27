"""PLACEHOLDER -- Gate 5b: every Linear in the block.

Adds ``c_attn`` (fused QKV -- just a Linear with ``out = 3d``), ``attn.c_proj``,
``mlp.c_fc``, ``mlp.c_proj``.  Nothing is registered for attention itself: it
holds no parameters of its own (design doc II.6).

Assert: full-model Gramian restricted to linear weights matches brute force,
and **print a per-layer delta table** so a failure names its layer.  That
attribution depends on both sides using the canonical ordering from
:mod:`jdgram.utils.flatten`.
"""

import pytest

pytest.skip("gate 5b not yet implemented", allow_module_level=True)
