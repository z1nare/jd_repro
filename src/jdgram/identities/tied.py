"""PLACEHOLDER -- tied weights, wte == lm_head (design doc II.4).

With one shared ``W`` used at two sites, the true per-objective gradient is the
**sum** of the two site gradients, so the Gramian gets four terms::

    G_ij = G^hh_ij + G^ee_ij + G^he_ij + G^eh_ij

``G^hh`` via II.1 (linear.sequence_gramian), ``G^ee`` via II.2
(embedding.token_embedding_gramian), and the cross terms via an index-gather
contraction::

    G^he_ij = sum_{u,v} A^head_i[u, tok[v]] . (X^head[u] . B_j[v])

where ``B_j`` is the embedding-site upstream gradient.

This is **not** a plain Hadamard of two Grams -- an earlier sketch that treated
all four terms with one routine was wrong on exactly this point, and that is
why the cross terms get their own file.

Mechanics: reuse TorchJD's ``remaining_counter`` cache verbatim -- collect
``(A, X)`` at the head, wait for the embedding's backward, then compute all
four terms and flush.

Gate: gates/test_5e_tied.py.  This is the hardest step; on failure, diff
against the gate-5d engine on the *untied* model, which isolates the cross-term
logic by construction.  Passing 5e is what "layer ports finished" means.
"""

from __future__ import annotations


def tied_gramian(*args, **kwargs):
    raise NotImplementedError("II.4 four-permutation cross terms: not yet implemented")
