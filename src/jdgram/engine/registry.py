"""PLACEHOLDER -- module type -> Gramian identity dispatch.

The table that makes the nanochat port "re-registration, not new math":
RMSNorm maps to II.5, GQA/Flash-Attention projections map to II.1, and the
attention math itself maps to nothing (II.6).

To implement: a mapping from ``type(module)`` (plus a predicate escape hatch
for tied/shared parameters) to an identity callable with a uniform signature,
so :mod:`jdgram.engine.hooks` never special-cases a layer type inline.

Keep it in sync with ``docs/operator_table.md`` -- that table is the
human-readable view of this dict, including each entry's gate status.
"""

from __future__ import annotations

REGISTRY: dict = {}


def register(*args, **kwargs):
    raise NotImplementedError("identity registry: step 2 of the execution plan")
