"""jdgram -- exact Gramian engines for Jacobian Descent.

``G = J J^T`` computed from one m-seed backward, without ever materializing the
``[m, P]`` Jacobian.  The engine selects the cheapest *correct* identity per
layer (see :mod:`jdgram.engine.router`); every route yields the same ``G``.

Status: the CIFAR/IWRM path (:mod:`jdgram.engine.sequential`) is proven against
TorchJD's ``autogram``.  The transformer identities are being built and gated
one layer at a time -- see ``docs/design/gramian_engines.md`` and
``docs/operator_table.md``.  Nothing ships until its gate passes.
"""

__all__ = ["seeds", "identities", "engine", "costmodel", "utils"]
