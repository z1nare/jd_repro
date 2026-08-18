"""One shared W at two sites => per-objective grad is the SUM of site grads.
Frobenius product of two sums => four terms, not two:

    G = G_hh + G_ee + G_he + G_eh

G_hh  via linear.sequence_gramian
G_ee  via embedding.sequence_gramian
G_he  via the index-gather cross term below (d-first or T-first)
G_eh  = G_he.T   (write + G_he.T explicitly; do not replace with 2*G_he)

Cross-term contraction orders (same G; router picks by cost):

* **d-first** — form ``[m, V, d]`` summaries H, E then ``H:E``. Cheap when
  ``m·V·d`` beats ``m²T²`` (short T, modest V).
* **T-first** — gather logits at token ids and contract over ``(t,s)``.
  Workspace ``2·m²T²``; wins at large V (Qwen-scale vocab).
"""

from __future__ import annotations

from typing import Literal

import torch

from jdgram.identities import embedding as embedding_id
from jdgram.identities import linear as linear_id
from jdgram.identities.precision import resolve_workspace_dtype

CrossRoute = Literal["dfirst", "tfirst"]


def head_embedding_cross_dfirst(
    A_head: torch.Tensor,
    X_head: torch.Tensor,
    A_emb: torch.Tensor,
    tokens: torch.Tensor,
    *,
    workspace_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """G_he via ``[m, V, d]`` summaries (d-first). Scatter-add, never one-hot."""
    wd = resolve_workspace_dtype(workspace_dtype)
    V = A_head.shape[-1]
    m, _, d = X_head.shape
    Aw = A_head.to(wd)
    Xw = X_head.to(wd)
    Aew = A_emb.to(wd)

    H = torch.einsum("itv, itd -> ivd", Aw, Xw)
    E = torch.zeros(m, V, d, dtype=wd, device=A_emb.device)
    E.scatter_add_(1, tokens.unsqueeze(-1).expand(-1, -1, d), Aew)

    # Same reasoning as embedding.sequence_gramian_dfirst: the trailing (v, d)
    # axes are contracted against the same axes of the other operand, so this is
    # a matmul on the flattened views. einsum would clone both [m, V, d] blocks.
    return (H.reshape(m, -1) @ E.reshape(m, -1).T).double()


def head_embedding_cross_tfirst(
    A_head: torch.Tensor,
    X_head: torch.Tensor,
    A_emb: torch.Tensor,
    tokens: torch.Tensor,
    *,
    workspace_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """G_he via T×T contraction (T-first).

    ``G_he[i,j] = Σ_{t,s} A_head[i,t,tokens[j,s]] · ⟨X_head[i,t], A_emb[j,s]⟩``

    Workspace ``O(m²T²)`` instead of ``O(m·V·d)`` — preferred at large V.
    """
    wd = resolve_workspace_dtype(workspace_dtype)
    Aw = A_head.to(wd)
    Xw = X_head.to(wd)
    Aew = A_emb.to(wd)

    # gathered[i,t,j,s] = A_head[i, t, tokens[j,s]]
    gathered = Aw[:, :, tokens]
    inner = torch.einsum("itd,jsd->itjs", Xw, Aew)
    return (gathered * inner).sum(dim=(1, 3)).double()


def head_embedding_cross(
    A_head: torch.Tensor,
    X_head: torch.Tensor,
    A_emb: torch.Tensor,
    tokens: torch.Tensor,
    *,
    route: CrossRoute = "dfirst",
    workspace_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """G_he only; ``route`` selects d-first or T-first (same value)."""
    if route == "tfirst":
        return head_embedding_cross_tfirst(
            A_head, X_head, A_emb, tokens, workspace_dtype=workspace_dtype
        )
    if route == "dfirst":
        return head_embedding_cross_dfirst(
            A_head, X_head, A_emb, tokens, workspace_dtype=workspace_dtype
        )
    raise ValueError(f"unknown cross route {route!r}; expected 'dfirst' or 'tfirst'")


def tied_gramian(
    A_head: torch.Tensor,
    X_head: torch.Tensor,
    A_emb: torch.Tensor,
    tokens: torch.Tensor,
    *,
    cross_route: CrossRoute | None = None,
    workspace_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """
    All four permutation terms for the tied wte / lm_head weight.
    Returns:
        G: [m, m] float64  (= G_hh + G_ee + G_he + G_eh)

    Diagonal head/emb terms and the cross term all follow
    :func:`jdgram.engine.router.route` on ``P = V * d`` unless
    ``cross_route`` is set explicitly (cross only).
    """
    from jdgram.engine.materialize import materialized_gramian
    from jdgram.engine.router import route as pick_route

    m, T, d = X_head.shape
    V = A_head.shape[-1]
    head_route = pick_route(m, T, V * d, V, d)
    if cross_route is None:
        cross_route = head_route

    if head_route == "dfirst":
        G_hh = materialized_gramian(
            A_head, X_head, has_bias=False, workspace_dtype=workspace_dtype
        )
    else:
        G_hh = linear_id.sequence_gramian(
            A_head, X_head, has_bias=False, workspace_dtype=workspace_dtype
        )
    G_ee = embedding_id.sequence_gramian(
        A_emb,
        tokens,
        num_embeddings=V,
        route=head_route,
        workspace_dtype=workspace_dtype,
    )
    G_he = head_embedding_cross(
        A_head,
        X_head,
        A_emb,
        tokens,
        route=cross_route,
        workspace_dtype=workspace_dtype,
    )
    return G_hh + G_ee + G_he + G_he.T
