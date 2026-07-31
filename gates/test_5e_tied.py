"""Gate 5e -- full tied model. Passing this is "layer ports finished".

The engine refuses to sum per-module Gramians across a shared parameter (that
would drop the II.4 cross terms). Supply ``shared_handlers`` that unpacks the
two site captures and calls :func:`jdgram.identities.tied.tied_gramian`.
"""

from __future__ import annotations

import torch

from gates._helpers import run_engine
from gates.brute_force import true_gramian
from jdgram.engine.hooks import LayerCapture
from jdgram.engine.registry import (
    collect_hookable_modules,
    positional_embedding_handler,
)
from jdgram.identities.tied import tied_gramian

TIED_GROUP = frozenset({"transformer.wte", "lm_head"})


def _tied_shared_handler(captures: dict[str, LayerCapture]) -> torch.Tensor:
    head = captures["lm_head"]
    emb = captures["transformer.wte"]
    return tied_gramian(head.A, head.X, emb.A, emb.X)


def test_5e_weights_are_tied(tied_model):
    assert tied_model.transformer.wte.weight is tied_model.lm_head.weight


def test_5e_layout_dedups_shared_param(tied_model, tied_layout):
    """param_layout dedups by id(), so the tied W appears exactly once."""
    names = [n for n, _, _ in tied_layout]
    assert not ("lm_head.weight" in names and "transformer.wte.weight" in names)


def test_5e_tied_full_model(tied_model, gate_data, tied_layout):
    idx, targets = gate_data
    g_true, _, _ = true_gramian(tied_model, idx, targets, layout=tied_layout)

    modules = collect_hookable_modules(tied_model)
    result = run_engine(
        tied_model,
        idx,
        targets,
        modules,
        handler_overrides={"transformer.wpe": positional_embedding_handler},
        shared_handlers={TIED_GROUP: _tied_shared_handler},
    )

    diff = (result.total - g_true).abs().max().item()
    print(f"max|G_engine - G_true| = {diff:.3e}")
    torch.testing.assert_close(result.total, g_true, rtol=0, atol=1e-10)


def test_5e_refuses_without_shared_handler(tied_model, gate_data):
    idx, targets = gate_data
    modules = collect_hookable_modules(tied_model)
    try:
        run_engine(
            tied_model,
            idx,
            targets,
            modules,
            handler_overrides={"transformer.wpe": positional_embedding_handler},
        )
    except ValueError as exc:
        assert "shared" in str(exc).lower() or "cross" in str(exc).lower()
    else:
        raise AssertionError("expected ValueError when shared_handlers is omitted")
