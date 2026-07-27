"""PLACEHOLDER -- Gate 5f: chunked II.1 against the naive einsum.

The naive form builds ``[m, m, U, U]``: acceptable at gate sizes, ~17 GiB at
m=32 / U=2048.  The chunked form holds only ``[U, U]`` or row-blocks and is the
one that actually ships, so it needs its own equivalence gate rather than
inheriting trust from 5a/5b.

Assert: chunked == naive to fp64 machine precision, across at least one
chunk-size that does not divide ``U`` evenly.  Off-by-one on the tail block is
the obvious failure mode and an even split would hide it.
"""

import pytest

pytest.skip("gate 5f not yet implemented", allow_module_level=True)
