"""Gate 5a -- hook mechanism correct on one untied Linear (lm_head).

Compares against the per-layer brute-force block, not full G_true: G_true
includes every other parameter's contribution.

lm_head is constructed bias=False in nanoGPT regardless of config.bias, so the
bias path is not exercised here -- gate 5b is where that first happens.
"""

from __future__ import annotations

import torch

from gates._helpers import run_engine
from gates.brute_force import true_gramian


def test_5a_head_linear(gate_model, gate_data, gate_layout):
    idx, targets = gate_data
    _, _, g_by_layer = true_gramian(gate_model, idx, targets, layout=gate_layout)

    modules = {"lm_head": gate_model.lm_head}
    result = run_engine(gate_model, idx, targets, modules)

    expected = g_by_layer["lm_head.weight"]
    torch.testing.assert_close(
        result.per_module["lm_head"], expected, rtol=0, atol=1e-10
    )
    torch.testing.assert_close(result.total, expected, rtol=0, atol=1e-10)


def test_5a_head_has_no_bias(gate_model):
    assert gate_model.lm_head.bias is None
