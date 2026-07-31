"""Canonical parameter ordering for Jacobian / Gramian ground truth.

Both gates/brute_force.py and the engine must use THIS module for flattening.
If brute force uses model.parameters() ad hoc and the engine uses param_layout(),
gate failures will be meaningless.
"""

from __future__ import annotations

from typing import TypeAlias

import torch
from torch import nn

LayoutEntry: TypeAlias = tuple[str, nn.Parameter, slice]


def param_layout(model: nn.Module) -> list[LayoutEntry]:
    layout: list[LayoutEntry] = []
    offset = 0
    seen: set[int] = set()

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        pid = id(param)
        if pid in seen:
            continue
        n = param.numel()
        layout.append((name, param, slice(offset, offset + n)))
        offset += n
        seen.add(pid)
    return layout


def num_params(layout: list[LayoutEntry]) -> int:
    """Total flat dimension P."""
    if not layout:
        return 0
    return layout[-1][2].stop


def validate_layout(layout: list[LayoutEntry]) -> None:
    """Self-test invariants. Call once per model build."""
    expected = 0
    seen_ids: set[int] = set()
    for name, param, sl in layout:
        assert sl.start == expected, f"gap before {name}: expected start {expected}, got {sl.start}"
        assert sl.stop - sl.start == param.numel(), f"slice size mismatch for {name}"
        assert id(param) not in seen_ids, f"duplicate parameter id for {name}"
        seen_ids.add(id(param))
        expected = sl.stop
    assert expected == sum(p.numel() for _, p, _ in layout)


def flatten_grads(grads: list[torch.Tensor], layout: list[LayoutEntry]) -> torch.Tensor:
    if len(grads) != len(layout):
        raise ValueError(f"flatten_grads: {len(grads)} grads vs {len(layout)} layout entries")

    P = num_params(layout)
    vec = torch.empty(P, dtype=grads[0].dtype, device=grads[0].device)

    for i, (name, param, sl) in enumerate(layout):
        g = grads[i]
        if g is None:
            raise ValueError(f"flatten_grads: None grad for {name}")
        if g.numel() != param.numel():
            raise ValueError(f"flatten_grads: shape mismatch for {name}")
        vec[sl] = g.reshape(-1)

    return vec


def flatten_grad_dict(
    grad_dict: dict[str, torch.Tensor],
    layout: list[LayoutEntry],
) -> torch.Tensor:
    grads = [grad_dict[name] for name, _, _ in layout]
    return flatten_grads(grads, layout)


def unflatten(vec: torch.Tensor, layout: list[LayoutEntry]) -> dict[str, torch.Tensor]:
    if vec.numel() != num_params(layout):
        raise ValueError(f"unflatten: vec length {vec.numel()} != P {num_params(layout)}")
    out: dict[str, torch.Tensor] = {}
    for name, param, sl in layout:
        out[name] = vec[sl].view_as(param)
    return out


def layer_slices(layout: list[LayoutEntry]) -> dict[str, slice]:
    return {name: sl for name, _, sl in layout}
