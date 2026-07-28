from __future__ import annotations
import torch

def sequence_gramian(A: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Embedding weight-Gramian contribution for per-sequence objectives.
 
    A:   [m, T, d] -- A[i, t] = d(objective_i) / d(embedding_output[i, t]).
    idx: [m, T]    -- the token ids that produced that output (int64).
    Returns [m, m] float64. No bias term -- nn.Embedding has none.
    """
    A=A.double()
    m,t,_ = A.shape

    A_flat=A.reshape(m*t, -1)
    m_a = A_flat@A_flat.T
    K_A = m_a.reshape(m,t,m,t).permute(0, 2, 1, 3) # m, m, t, t

    idx_flat = idx.reshape(-1)
    mask = (idx_flat.unsqueeze(0) == idx_flat.unsqueeze(1)) # [m*t,m*t]
    mask = mask.reshape(m,t,m,t).permute(0,2,1,3) # m,m,t,t
    return (mask.double()*K_A).sum(dim=(-2, -1))


def positional_embedding_gramian(A: torch.Tensor) -> torch.Tensor:
    m, t, d = A.shape
    A = A.double()
    A_flat = A.reshape(m, t*d)
    return A_flat @ A_flat.T
    