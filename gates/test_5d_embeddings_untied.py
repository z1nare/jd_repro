"""PLACEHOLDER -- Gate 5d: embeddings, tying still disabled.  Complete coverage.

Scope: II.2 (token embedding indicator kernel) and II.3 (positional embedding
diagonal shortcut).

Assert: the **full untied model** Gramian matches brute force.  This is the
complete-coverage gate -- after it passes, every parameter in the model is
accounted for by some identity, and the only thing left is weight sharing.

Keep this engine reachable after 5e lands: it is the diff target that isolates
cross-term logic when the tied gate fails.
"""

import pytest

pytest.skip("gate 5d not yet implemented", allow_module_level=True)
