"""jdgram -- exact Gramian engines for Jacobian Descent.

``G = J J^T`` computed from one m-seed backward, without ever materializing the
``[m, P]`` Jacobian.  The engine selects the cheapest *correct* identity per
layer (see :mod:`jdgram.engine.router`); every route yields the same ``G``.

Status: the transformer identities are built and gated -- Linear, bias,
LayerNorm/RMSNorm, token and positional embeddings, and the tied embedding/head,
which is the case TorchJD's ``autogram`` gets wrong.  See
``docs/operator_table.md`` for per-operator status and ``gates/`` for the
float64 brute-force checks.  Nothing ships until its gate passes.
"""

__all__ = ["seeds", "identities", "engine", "costmodel", "utils"]
