"""Embedding Gramian identities (token and positional).

Token embedding has the same two contraction orders as Linear:

* **T-first** — equality mask on ``[mT, mT]`` times ``A A^T`` (wins at huge V).
* **d-first** — scatter grads into ``[m, V, d]``, then Gram (wins at modest V).

Positional embedding is diagonal in t (II.3): flatten ``(T, d)`` then ``AA^T``.
"""

from __future__ import annotations

import torch

from jdgram.identities.precision import resolve_workspace_dtype


def sequence_gramian_tfirst(
    A: torch.Tensor,
    idx: torch.Tensor,
    *,
    workspace_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """T-first embedding Gramian. Bool mask selects; no fp64 mask buffer."""
    wd = resolve_workspace_dtype(workspace_dtype)
    Aw = A.to(wd)
    m, t, _ = Aw.shape

    A_flat = Aw.reshape(m * t, -1)
    m_a = (A_flat @ A_flat.T).reshape(m, t, m, t)

    idx_flat = idx.reshape(-1)
    # bool [m,t,m,t] — multiply in workspace dtype via einsum (no .double() mask)
    mask = (idx_flat.unsqueeze(0) == idx_flat.unsqueeze(1)).reshape(m, t, m, t)
    return torch.einsum("itjs,itjs->ij", m_a, mask.to(wd)).double()


def sequence_gramian_dfirst(
    A: torch.Tensor,
    idx: torch.Tensor,
    num_embeddings: int,
    *,
    workspace_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """d-first embedding Gramian: scatter into ``[m, V, d]``, then ``E:E``."""
    if num_embeddings <= 0:
        raise ValueError(f"num_embeddings must be positive, got {num_embeddings}")
    wd = resolve_workspace_dtype(workspace_dtype)
    Aw = A.to(wd)
    m, _, d = Aw.shape
    E = torch.zeros(m, num_embeddings, d, dtype=wd, device=A.device)
    E.scatter_add_(1, idx.unsqueeze(-1).expand(-1, -1, d), Aw)
    # Flatten-then-matmul rather than einsum("ivd,jvd->ij", E, E). Contracting
    # every trailing axis against the same axes of the other operand *is* a
    # matmul on the flattened view, and `reshape` on a contiguous tensor is free.
    # einsum is not: it clones both operands first (label order decides -- "ivd"
    # clones where the identical "ipq" does not), which at Qwen vocab is two
    # spurious [m, V, d] copies.
    E_flat = E.reshape(m, -1)
    return (E_flat @ E_flat.T).double()


def sequence_gramian(
    A: torch.Tensor,
    idx: torch.Tensor,
    *,
    num_embeddings: int | None = None,
    route: str | None = None,
    workspace_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Embedding weight Gramian; optional ``route`` / router when ``num_embeddings`` set.

    If ``route`` is omitted and ``num_embeddings`` is given, uses
    :func:`jdgram.engine.router.route`. If both omitted, defaults to T-first
    (backward-compatible).
    """
    if route is None and num_embeddings is not None:
        from jdgram.engine.router import route as pick_route

        m, T, d = A.shape
        route = pick_route(m, T, num_embeddings * d)
    if route == "dfirst":
        if num_embeddings is None:
            raise ValueError("dfirst embedding route requires num_embeddings")
        return sequence_gramian_dfirst(
            A, idx, num_embeddings, workspace_dtype=workspace_dtype
        )
    return sequence_gramian_tfirst(A, idx, workspace_dtype=workspace_dtype)


def positional_embedding_gramian(
    A: torch.Tensor,
    *,
    workspace_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    wd = resolve_workspace_dtype(workspace_dtype)
    m, t, d = A.shape
    Aw = A.to(wd)
    A_flat = Aw.reshape(m, t * d)
    return (A_flat @ A_flat.T).double()
