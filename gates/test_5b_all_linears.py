"""Gate 5b -- every Linear correct, with per-layer failure attribution."""

from __future__ import annotations

import torch

from gates._helpers import collect_linears, expected_module_gramian, run_engine
from gates.brute_force import true_gramian


def test_5b_all_linears(gate_model, gate_data, gate_layout):
    idx, targets = gate_data
    _, _, g_by_layer = true_gramian(gate_model, idx, targets, layout=gate_layout)

    modules = collect_linears(gate_model)
    result = run_engine(gate_model, idx, targets, modules)

    failures: list[tuple[str, float]] = []
    for name in modules:
        expected = expected_module_gramian(g_by_layer, name)
        diff = (result.per_module[name] - expected).abs().max().item()
        print(f"{name:40s} max|diff| = {diff:.3e}")
        if diff > 1e-10:
            failures.append((name, diff))
    assert not failures, f"per-layer mismatches: {failures}"

    names = list(modules)
    expected_total = expected_module_gramian(g_by_layer, names[0])
    for name in names[1:]:
        expected_total = expected_total + expected_module_gramian(g_by_layer, name)
    torch.testing.assert_close(result.total, expected_total, rtol=0, atol=1e-10)
