"""PLACEHOLDER -- embedding Gramian identities (design doc II.2, II.3).

**II.2 token embedding (``wte``).** The input is indices, not activations, so
the gradient scatters rows and a pair of positions only contributes when they
hold the *same* token::

    G_ij += <A_i A_j^T, 1[tok[u] == tok[v]]>_F

Same contraction shape as II.1 with ``K_X`` replaced by a boolean equality
kernel (``tok.unsqueeze(0) == tok.unsqueeze(1)``).  Implement it by reusing the
II.1 contraction with a swapped kernel, not as a separate code path.

**II.3 positional embedding (``wpe``).** Every sequence uses positions
``0..T-1``, so the equality kernel is the identity across matching ``t`` within
each ``(b, b')`` block::

    G_ij += sum_{b, b', t} A_i[b, t] . A_j[b', t]

Gate: gates/test_5d_embeddings_untied.py (tying disabled -- the cross terms
between the embedding site and the head are II.4's problem, not this file's).
"""

from __future__ import annotations


def token_embedding_gramian(*args, **kwargs):
    raise NotImplementedError("II.2 indicator kernel: not yet implemented")


def positional_embedding_gramian(*args, **kwargs):
    raise NotImplementedError("II.3 diagonal shortcut: not yet implemented")
