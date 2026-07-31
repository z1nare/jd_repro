"""Step 0 gate: brute-force Gramian ground truth is internally consistent."""

from __future__ import annotations

import torch

from gates.brute_force import assert_gramian_sanity, true_gramian
from jdgram.utils.flatten import num_params


def test_0_brute_force_sanity(gate_model, gate_data, gate_layout):
    idx, targets = gate_data
    g_true, j, g_by_layer = true_gramian(
        gate_model, idx, targets, layout=gate_layout
    )

    m = idx.shape[0]
    p = num_params(gate_layout)
    assert g_true.shape == (m, m)
    assert j.shape == (m, p)
    assert g_true.dtype == torch.float64
    assert j.dtype == torch.float64
    assert_gramian_sanity(g_true, j)

    for name, sl in ((n, s) for n, _, s in gate_layout):
        g_layer = g_by_layer[name]
        assert g_layer.shape == (m, m)
        torch.testing.assert_close(
            g_layer,
            j[:, sl] @ j[:, sl].T,
            rtol=0,
            atol=1e-10,
            msg=f"per-layer Gramian mismatch for {name}",
        )


def test_0_untied_weights(gate_model):
    wte = gate_model.transformer.wte.weight
    head = gate_model.lm_head.weight
    # Untying clones values, so tensors may still compare equal; what matters
    # is that they are distinct Parameter objects (independent grads).
    assert wte is not head
    assert id(wte) != id(head)
