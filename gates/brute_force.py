from __future__ import annotations

import torch

from jdgram.utils.flatten import flatten_grads, param_layout, validate_layout
from models.configs import forward_logits, per_sequence_losses


def assert_gramian_sanity(G: torch.Tensor, J: torch.Tensor, *, atol: float = 1e-10) -> None:
    row_norms = (J * J).sum(dim=1)
    torch.testing.assert_close(G.diagonal(), row_norms, rtol=0, atol=atol)
    torch.testing.assert_close(G, G.T, rtol=0, atol=atol)


def true_gramian(
    model,
    idx: torch.Tensor,
    targets: torch.Tensor,
    *,
    layout=None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """
    Returns:
      G_true: [m, m] float64
      J:      [m, P] float64
      G_by_layer: {param_name: [m,m]} for per-layer gate 5b checks
    """
    if layout is None:
        layout = param_layout(model)
    validate_layout(layout)

    model = model.double()
    idx = idx.to(dtype=torch.long, device=next(model.parameters()).device)
    targets = targets.to(dtype=torch.long, device=idx.device)

    logits = forward_logits(model, idx)
    losses = per_sequence_losses(logits, targets)

    params = [p for _, p, _ in layout]
    m = idx.shape[0]
    p = layout[-1][2].stop
    j = torch.empty(m, p, dtype=torch.float64, device=idx.device)

    for i in range(m):
        grads = torch.autograd.grad(
            losses[i],
            params,
            create_graph=False,
            retain_graph=(i < m - 1),
            allow_unused=False,
        )
        j[i] = flatten_grads(grads, layout)

    g_true = j @ j.T

    g_by_layer: dict[str, torch.Tensor] = {}
    for name, _, sl in layout:
        j_layer = j[:, sl]
        g_by_layer[name] = j_layer @ j_layer.T

    assert_gramian_sanity(g_true, j)
    return g_true, j, g_by_layer
