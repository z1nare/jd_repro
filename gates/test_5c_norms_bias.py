"""Gate 5c -- norms and biases. Everything except embeddings is covered.

Requires :func:`jdgram.identities.norm.norm_gramian` (II.5).
"""

from __future__ import annotations

import torch

from gates._helpers import (
    collect_linears_and_norms,
    expected_module_gramian,
    run_engine,
)
from gates.brute_force import true_gramian


def test_5c_norms_and_bias(gate_model, gate_data, gate_layout):
    idx, targets = gate_data
    _, _, g_by_layer = true_gramian(gate_model, idx, targets, layout=gate_layout)

    modules = collect_linears_and_norms(gate_model)
    result = run_engine(gate_model, idx, targets, modules)

    failures: list[tuple[str, float]] = []
    for name in modules:
        expected = expected_module_gramian(g_by_layer, name)
        diff = (result.per_module[name] - expected).abs().max().item()
        print(f"{name:40s} max|diff| = {diff:.3e}")
        if diff > 1e-10:
            failures.append((name, diff))
    assert not failures, f"per-layer mismatches: {failures}"
