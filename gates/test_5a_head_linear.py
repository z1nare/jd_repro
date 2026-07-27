"""PLACEHOLDER -- Gate 5a: one Linear (``lm_head``), weight tying disabled.

Scope: hook plumbing (:mod:`jdgram.engine.hooks` + :mod:`jdgram.engine.node`)
registered on ``lm_head`` **only**, using the naive einsum form of II.1.

Assert: the engine's contribution for ``lm_head.weight`` matches the
brute-force Jacobian of that parameter alone, atol ~ 1e-10 at fp64.

Delete the tying line in the model for this gate -- tying is gate 5e.
"""

import pytest

pytest.skip("gate 5a not yet implemented", allow_module_level=True)
