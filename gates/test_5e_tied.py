"""PLACEHOLDER -- Gate 5e: weight tying re-enabled.  The hardest step.

Scope: II.4's four terms (``G^hh + G^ee + G^he + G^eh``) via the
``remaining_counter`` cache in :mod:`jdgram.engine.node` -- collect ``(A, X)``
at the head, wait for the embedding's backward, compute all four, flush.

Assert: the full **tied** model matches brute force.  Note that brute force
must count the shared parameter once, with its gradient summed over both sites.

On failure: diff against the gate-5d engine on the untied model.  The untied
result is known good, so any discrepancy is in the cross terms by construction.

**Passing this gate is what "layer ports finished" means** -- the condition
the MuSiQue approval was gated on.  Report it in those words.
"""

import pytest

pytest.skip("gate 5e not yet implemented", allow_module_level=True)
