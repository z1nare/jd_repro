"""Exact Gramian block for parameters that have no closed-form identity.

The engine's value is avoiding the ``[m, P]`` Jacobian for the parameters that
dominate ``P``. It does not follow that *every* parameter has to be handled that
way. A model can carry a handful of parameters whose gradient is not determined
by ``(A, X)`` at a module boundary -- Qwen3.5's ``A_log``/``dt_bias`` sit inside
the delta-rule recurrence gate, and its ``RMSNormGated`` weight is multiplied by
a second input the forward hook never sees.

For those, materialising is the honest answer and costs nothing at the sizes
involved: Qwen3.5-0.8B has 445.2K such parameters out of 752.4M (0.059%), so the
fp64 block is 3.4 MiB per objective -- 57 MiB at m=16, against the ~2 GiB the
vocabulary head alone would need.

This keeps the Gramian **exact**. The alternative on offer is excluding those
parameters, which silently returns a Gramian of a different function.

Cost is ``m`` backward passes. Under ``driver="loop"`` the engine is already
paying that, so pass ``retain_graph=True`` and reuse one graph for both.
"""

from __future__ import annotations

import torch
from torch import nn


def residual_params(
    model: nn.Module,
    handled: set[str],
) -> list[tuple[str, torch.nn.Parameter]]:
    """Trainable parameters not owned by any handled module, in a stable order.

    ``handled`` holds qualified *module* names whose direct parameters the engine
    accounts for. Deduplicates on tensor identity, so a tied weight reachable
    from two modules is returned at most once.
    """
    seen: set[int] = set()
    out: list[tuple[str, torch.nn.Parameter]] = []
    for mod_name, mod in model.named_modules():
        if mod_name in handled:
            for p in mod.parameters(recurse=False):
                seen.add(id(p))
    for mod_name, mod in model.named_modules():
        if mod_name in handled:
            continue
        for p_name, p in mod.named_parameters(recurse=False):
            if not p.requires_grad or id(p) in seen:
                continue
            seen.add(id(p))
            out.append((f"{mod_name}.{p_name}" if mod_name else p_name, p))
    return out


def residual_gramian(
    losses: torch.Tensor,
    params: list[torch.nn.Parameter],
    *,
    retain_graph: bool = True,
    allow_unused: bool = True,
) -> torch.Tensor:
    """``[m, m]`` float64 Gramian of the given parameters, by explicit Jacobian.

    :param losses: the ``[m]`` per-objective loss vector, still attached to the graph.
    :param params: parameters to account for. Keep this small -- the whole point
        is that it is the tail of the model, not its body.

    A parameter that does not participate in an objective's graph yields ``None``
    from ``autograd.grad``; that is a true zero row-block, not an error, so
    ``allow_unused`` defaults to True and the gradient is taken as zero.
    """
    if not params:
        m = losses.shape[0]
        return torch.zeros(m, m, dtype=torch.float64, device=losses.device)

    m = losses.shape[0]
    rows = []
    for i in range(m):
        grads = torch.autograd.grad(
            losses[i], params,
            retain_graph=retain_graph or (i < m - 1),
            allow_unused=allow_unused,
        )
        rows.append(torch.cat([
            (torch.zeros_like(p) if g is None else g).reshape(-1).double()
            for g, p in zip(grads, params)
        ]))
    J = torch.stack(rows)
    return J @ J.T


def numel(params: list[torch.nn.Parameter]) -> int:
    return sum(p.numel() for p in params)
