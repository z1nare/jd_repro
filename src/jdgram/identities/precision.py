"""Workspace vs accumulator precision for Gramian identities.

Heavy intermediates (``[mT, mT]``, unfolded Jacobians, etc.) run in
``workspace_dtype`` (default float32 — ~30× on GA102 vs fp64). The final
``[m, m]`` Gramian is always accumulated / returned in float64; that matrix is
tiny and feeds the aggregator's QP (reg ~1e-4), so fp64 there is nearly free.

Gates pin float64 workspace via :func:`workspace_dtype` so brute-force diffs
at ``atol=1e-10`` stay meaningful. Flip the default only after a convergence
overlay (fp32 vs fp64 workspace, same seed) coincides.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch

# Default for bench / training. Gates override to float64.
_workspace_dtype: torch.dtype = torch.float32


def get_workspace_dtype() -> torch.dtype:
    return _workspace_dtype


@contextmanager
def workspace_dtype(dtype: torch.dtype) -> Iterator[None]:
    """Temporarily set the identity workspace dtype (e.g. float64 in gates)."""
    global _workspace_dtype
    prev = _workspace_dtype
    _workspace_dtype = dtype
    try:
        yield
    finally:
        _workspace_dtype = prev


def resolve_workspace_dtype(explicit: torch.dtype | None) -> torch.dtype:
    return _workspace_dtype if explicit is None else explicit
