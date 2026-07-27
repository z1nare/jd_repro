"""PLACEHOLDER -- Gate 6: the GRPO head-seed shortcut (design doc I.4).

Claim under test: per-objective GRPO losses share every token's log-probability
gradient direction and differ only by per-token scalars, so at the (per-token,
non-mixing) head::

    A_i[b,t,:] = c_i(b,t) * s(b,t,:)
    G^head_ij  = c_i^T (S S^T . X X^T) c_j

One ``(BT)x(BT)`` kernel computed once, then m^2 cheap quadratic forms.

Assert: against brute-force autograd on a GRPO-shaped loss -- not against the
II.1 engine, which would only prove the two implementations agree, not that the
diagonal-structure assumption holds.

The derivation is ungated as written.  It is potentially the best FLOP result
in the project, which is exactly why it does not get to skip its gate.
"""

import pytest

pytest.skip("gate 6 not yet implemented", allow_module_level=True)
