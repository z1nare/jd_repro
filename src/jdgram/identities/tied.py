"""One shared W at two sites => per-objective grad is the SUM of site grads.
Frobenius product of two sums => four terms, not two:

    G = G_hh + G_ee + G_he + G_eh

G_hh  via linear.sequence_gramian
G_ee  via embedding.sequence_gramian
G_he  via the index-gather cross term below
G_eh  = G_he.T   (write + G_he.T explicitly; do not replace with 2*G_he)"""

from __future__ import annotations

import torch

from jdgram.identities import embedding as embedding_id
from jdgram.identities import linear as linear_id
import torch.nn.functional as F

def head_embedding_cross(
    A_head: torch.Tensor,
    X_head: torch.Tensor,
    A_emb: torch.Tensor,
    tokens: torch.Tensor,
) -> torch.Tensor:
    """
    Cross term G_he only.
    Args:
        A_head: [m, T, V]  -- d(loss_i) / d(logits[i])
        X_head: [m, T, d]  -- lm_head input (final hidden states)
        A_emb:  [m, T, d]  -- d(loss_i) / d(embedding_output[i])
        tokens: [m, T]     -- input token ids (long)
    Returns:
        G_he: [m, m] float64    """

    V = A_head.shape[-1]
    
    # 1. Summarize Head side over sequence length T -> [m, V, d]
    H = torch.einsum("itv, itd -> ivd", A_head.double(), X_head.double())

    # 2. Scatter Embedding side over sequence length T into vocab bins -> [m, V, d]
    O = F.one_hot(tokens, num_classes=V).double()
    E = torch.einsum("jsv, jsd -> jvd", O, A_emb.double())

    # 3. Final cross-term Gramian matrix -> [m, m]
    G_he = torch.einsum("ivd, jvd -> ij", H, E)

    return G_he


def tied_gramian(
    A_head: torch.Tensor,
    X_head: torch.Tensor,
    A_emb: torch.Tensor,
    tokens: torch.Tensor,
) -> torch.Tensor:
    """
    All four permutation terms for the tied wte / lm_head weight.
    Returns:
        G: [m, m] float64  (= G_hh + G_ee + G_he + G_eh)
    """
    G_hh = linear_id.sequence_gramian(A_head, X_head, has_bias=False)
    G_ee = embedding_id.sequence_gramian(A_emb, tokens)
    G_he = head_embedding_cross(A_head, X_head, A_emb, tokens)
    G_eh = G_he.T
    return G_hh+G_ee+G_he+G_eh