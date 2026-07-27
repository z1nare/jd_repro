"""PLACEHOLDER -- Gate 5c: LayerNorm/RMSNorm parameters and all bias terms.

Scope: II.5 materialization for gamma/beta, plus the bias summand of II.1.

Assert: everything except the embeddings now matches brute force.  Run with
``bias=True`` and ``bias=False`` -- the false case must not silently skip a
term that the true case gets wrong.
"""

import pytest

pytest.skip("gate 5c not yet implemented", allow_module_level=True)
