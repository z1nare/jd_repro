"""Shared fixtures for transformer Gramian gates (design doc Part III, step 0).

Every gate in this directory should use these fixtures so bias, dtype, dropout,
and seed are consistent.  Nothing downstream is trustworthy until
``test_0_brute_force_sanity`` passes for both bias variants.
"""

from __future__ import annotations

import pytest
import torch

from jdgram.utils.flatten import param_layout, validate_layout
from models.configs import GateTinyConfig, build_gate_model, gate_batch

# Fixed seeds — change only when intentionally refreshing reference values.
GATE_SEED = 0
DATA_SEED = 1


@pytest.fixture(autouse=True, scope="session")
def _ieee_fp32_matmul():
    """Pin full-precision fp32 matmul for the whole session.

    TF32 silently drops fp32 mantissa from 24 bits to 11, which is far coarser
    than the atol=1e-10 these gates assert. PyTorch has defaulted it off, but the
    knob has already been renamed once (``allow_tf32`` -> ``fp32_precision`` in
    2.9) and a future default flip would turn every gate here into a
    rubber stamp rather than a failure. Set it explicitly instead of inheriting.
    """
    saved: list[tuple[object, str, object]] = []
    for backend in (torch.backends.cuda.matmul, torch.backends.cudnn):
        if hasattr(backend, "fp32_precision"):
            saved.append((backend, "fp32_precision", backend.fp32_precision))
            backend.fp32_precision = "ieee"
        elif hasattr(backend, "allow_tf32"):
            saved.append((backend, "allow_tf32", backend.allow_tf32))
            backend.allow_tf32 = False
    try:
        yield
    finally:
        for backend, attr, value in saved:
            setattr(backend, attr, value)


@pytest.fixture(autouse=True)
def _fp64_identity_workspace():
    """Gates diff against float64 brute force at atol=1e-10; pin workspace."""
    from jdgram.identities.precision import workspace_dtype

    with workspace_dtype(torch.float64):
        yield


@pytest.fixture(params=[True, False], ids=["bias", "no-bias"])
def bias(request: pytest.FixtureRequest) -> bool:
    return request.param


@pytest.fixture
def gate_config(bias: bool) -> GateTinyConfig:
    cfg = GateTinyConfig(bias=bias, dropout=0.0, tie_weights=False)
    assert cfg.dropout == 0.0, "dropout must be 0.0 or BF and hooked passes diverge"
    assert not cfg.tie_weights, "gates 5a-5d require an untied model"
    return cfg


@pytest.fixture
def gate_model(gate_config: GateTinyConfig) -> torch.nn.Module:
    torch.manual_seed(GATE_SEED)
    model = build_gate_model(
        bias=gate_config.bias,
        tie_weights=False,
        device="cpu",
        dtype=torch.float64,
    )
    model.eval()
    return model


@pytest.fixture
def gate_data(gate_config: GateTinyConfig) -> tuple[torch.Tensor, torch.Tensor]:
    idx, targets = gate_batch(gate_config, seed=DATA_SEED, device="cpu")
    assert idx.shape[0] == gate_config.m
    assert idx.shape == targets.shape
    assert idx.dtype == torch.long
    assert targets.dtype == torch.long
    return idx, targets


@pytest.fixture
def gate_layout(gate_model: torch.nn.Module):
    layout = param_layout(gate_model)
    validate_layout(layout)
    assert layout[-1][2].stop > 0
    return layout


@pytest.fixture
def tied_model(bias: bool) -> torch.nn.Module:
    """Gate 5e only: wte.weight IS lm_head.weight, as upstream nanoGPT ships it."""
    torch.manual_seed(GATE_SEED)
    model = build_gate_model(
        bias=bias,
        tie_weights=True,
        device="cpu",
        dtype=torch.float64,
    )
    model.eval()
    return model


@pytest.fixture
def tied_layout(tied_model: torch.nn.Module):
    layout = param_layout(tied_model)
    validate_layout(layout)
    assert layout[-1][2].stop > 0
    return layout
