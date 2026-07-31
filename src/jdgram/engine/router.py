"""Per-layer route selection between T-first contraction and d-first materialize.

Both routes return the same Gramian; the choice is cost only. Analytic rule
(until :mod:`jdgram.costmodel` loads measured crossovers)::

    tfirst  if  m · T²  <  P_layer
    dfirst  otherwise

``P_layer`` is the layer's parameter count (e.g. ``d_out * d_in`` for Linear,
``V * d`` for Embedding). At large vocab, tfirst wins; at interior linears
with long T, dfirst is linear in T and usually cheaper.

Force a side for gates via :func:`force_route` (``None`` restores the rule).
"""

from __future__ import annotations

from typing import Literal

Route = Literal["tfirst", "dfirst"]

_force: Route | None = None


def force_route(route: Route | None) -> None:
    """Pin every :func:`route` decision, or ``None`` to clear."""
    global _force
    if route is not None and route not in ("tfirst", "dfirst"):
        raise ValueError(f"unknown route {route!r}; expected 'tfirst', 'dfirst', or None")
    _force = route


def get_force_route() -> Route | None:
    return _force


def route(m: int, T: int, P_layer: int) -> Route:
    """Pick contraction order for one layer."""
    if _force is not None:
        return _force
    if P_layer <= 0:
        raise ValueError(f"P_layer must be positive, got {P_layer}")
    return "tfirst" if m * T * T < P_layer else "dfirst"
