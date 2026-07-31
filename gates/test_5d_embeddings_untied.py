"""Gate 5d -- full untied model. The complete-coverage gate.

Compares against full G_true. Passes ``positional_embedding_handler`` for
``transformer.wpe`` because both embeddings are ``nn.Embedding`` and type
dispatch cannot separate them.
"""

from __future__ import annotations

import torch

from gates._helpers import expected_module_gramian, run_engine
from gates.brute_force import true_gramian
from jdgram.engine.registry import (
    collect_hookable_modules,
    positional_embedding_handler,
)


def test_5d_full_untied_model(gate_model, gate_data, gate_layout):
    idx, targets = gate_data
    g_true, _, _ = true_gramian(gate_model, idx, targets, layout=gate_layout)

    modules = collect_hookable_modules(gate_model)
    result = run_engine(
        gate_model,
        idx,
        targets,
        modules,
        handler_overrides={"transformer.wpe": positional_embedding_handler},
    )

    diff = (result.total - g_true).abs().max().item()
    print(f"max|G_engine - G_true| = {diff:.3e}")
    torch.testing.assert_close(result.total, g_true, rtol=0, atol=1e-10)


def test_5d_covers_every_parameter(gate_model, gate_layout):
    """Every layout parameter must belong to a hooked module."""
    hooked = set(collect_hookable_modules(gate_model))
    for name, _, _ in gate_layout:
        owner = name.rsplit(".", 1)[0]
        assert owner in hooked, f"{name} belongs to unhooked module {owner}"


def test_5d_per_module_matches_brute_force(gate_model, gate_data, gate_layout):
    idx, targets = gate_data
    _, _, g_by_layer = true_gramian(gate_model, idx, targets, layout=gate_layout)

    modules = collect_hookable_modules(gate_model)
    result = run_engine(
        gate_model,
        idx,
        targets,
        modules,
        handler_overrides={"transformer.wpe": positional_embedding_handler},
    )

    failures: list[tuple[str, float]] = []
    for name in modules:
        expected = expected_module_gramian(g_by_layer, name)
        diff = (result.per_module[name] - expected).abs().max().item()
        print(f"{name:40s} max|diff| = {diff:.3e}")
        if diff > 1e-10:
            failures.append((name, diff))
    assert not failures, f"per-layer mismatches: {failures}"
