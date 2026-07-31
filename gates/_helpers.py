"""Shared helpers for transformer Gramian gates."""

from __future__ import annotations

from typing import Callable

import torch
from torch import nn

from jdgram.engine.hooks import compute_gramian
from jdgram.engine.registry import IdentityHandler, collect_hookable_modules
from models.configs import forward_logits, per_sequence_losses


def make_compute_losses(
    model: nn.Module,
    idx: torch.Tensor,
    targets: torch.Tensor,
) -> Callable[[], torch.Tensor]:
    def compute_losses() -> torch.Tensor:
        return per_sequence_losses(forward_logits(model, idx), targets)

    return compute_losses


def expected_module_gramian(
    g_by_layer: dict[str, torch.Tensor],
    module_name: str,
) -> torch.Tensor:
    """Sum brute-force per-parameter blocks owned by ``module_name``."""
    blocks = [
        g
        for pname, g in g_by_layer.items()
        if pname == module_name or pname.startswith(module_name + ".")
    ]
    if not blocks:
        raise KeyError(f"no brute-force blocks for module {module_name!r}")
    out = blocks[0]
    for block in blocks[1:]:
        out = out + block
    return out


def collect_linears(model: nn.Module) -> dict[str, nn.Module]:
    return {
        name: module
        for name, module in collect_hookable_modules(model).items()
        if isinstance(module, nn.Linear)
    }


def collect_linears_and_norms(model: nn.Module) -> dict[str, nn.Module]:
    return {
        name: module
        for name, module in collect_hookable_modules(model).items()
        if isinstance(module, nn.Linear) or type(module).__name__.endswith(("LayerNorm", "RMSNorm"))
    }


def run_engine(
    model: nn.Module,
    idx: torch.Tensor,
    targets: torch.Tensor,
    modules: dict[str, nn.Module],
    *,
    handler_overrides: dict[str, IdentityHandler] | None = None,
    shared_handlers=None,
    use_leaf_edges: bool = True,
    driver: str | None = None,
    batched_backward: bool | None = None,
):
    """Drive the engine. ``driver=None`` means the engine default (``"squashed"``),
    so gates 5a-5e guard the path that actually ships; 5f pins all three together."""
    return compute_gramian(
        model,
        make_compute_losses(model, idx, targets),
        modules=modules,
        handler_overrides=handler_overrides,
        shared_handlers=shared_handlers,
        use_leaf_edges=use_leaf_edges,
        driver=driver,
        batched_backward=batched_backward,
    )
