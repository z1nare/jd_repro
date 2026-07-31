"""Route equivalence: auto / forced tfirst / forced dfirst must agree.

Both contraction orders are exact; this is the regression net for
:mod:`jdgram.engine.router` and :mod:`jdgram.engine.materialize`.
"""

from __future__ import annotations

import torch
import pytest

from gates._helpers import collect_linears, run_engine
from jdgram.engine.materialize import materialized_gramian
from jdgram.engine.registry import collect_hookable_modules, positional_embedding_handler
from jdgram.engine.router import force_route, route
from jdgram.identities import embedding as embedding_id
from jdgram.identities import linear as linear_id
from jdgram.identities.tied import (
    head_embedding_cross_dfirst,
    head_embedding_cross_tfirst,
)


@pytest.fixture(autouse=True)
def _clear_force_route():
    force_route(None)
    yield
    force_route(None)


def test_router_rule():
    assert route(4, 2048, 12_600_000) == "dfirst"  # interior MLP @ long T
    assert route(4, 2048, 311_000_000) == "tfirst"  # vocab-sized
    assert route(4, 16, 65 * 32) == "tfirst"  # gate-scale: mT²=1024, P=2080


def test_linear_tfirst_matches_dfirst():
    torch.manual_seed(0)
    m, T, d_out, d_in = 4, 16, 32, 24
    A = torch.randn(m, T, d_out, dtype=torch.float64)
    X = torch.randn(m, T, d_in, dtype=torch.float64)
    Gt = linear_id.sequence_gramian(A, X, has_bias=True)
    Gd = materialized_gramian(A, X, has_bias=True)
    assert (Gt - Gd).abs().max().item() < 1e-10


def test_embedding_tfirst_matches_dfirst():
    torch.manual_seed(0)
    m, T, d, V = 4, 16, 32, 65
    A = torch.randn(m, T, d, dtype=torch.float64)
    idx = torch.randint(0, V, (m, T))
    Gt = embedding_id.sequence_gramian_tfirst(A, idx)
    Gd = embedding_id.sequence_gramian_dfirst(A, idx, V)
    assert (Gt - Gd).abs().max().item() < 1e-10


def test_tied_cross_tfirst_matches_dfirst():
    torch.manual_seed(0)
    m, T, d, V = 4, 16, 32, 65
    A_head = torch.randn(m, T, V, dtype=torch.float64)
    X_head = torch.randn(m, T, d, dtype=torch.float64)
    A_emb = torch.randn(m, T, d, dtype=torch.float64)
    tokens = torch.randint(0, V, (m, T))
    Gd = head_embedding_cross_dfirst(A_head, X_head, A_emb, tokens)
    Gt = head_embedding_cross_tfirst(A_head, X_head, A_emb, tokens)
    assert (Gt - Gd).abs().max().item() < 1e-10


@pytest.mark.parametrize("forced", [None, "tfirst", "dfirst"])
def test_engine_routes_agree_with_brute_scale(gate_model, gate_data, forced):
    """Full hooked model: routing on / forced tfirst / forced dfirst."""
    idx, targets = gate_data
    modules = collect_linears(gate_model)
    force_route(forced)
    a = run_engine(gate_model, idx, targets, modules)
    force_route("tfirst")
    t = run_engine(gate_model, idx, targets, modules)
    force_route("dfirst")
    d = run_engine(gate_model, idx, targets, modules)
    force_route(None)

    assert (a.total - t.total).abs().max().item() < 1e-8
    assert (t.total - d.total).abs().max().item() < 1e-8
