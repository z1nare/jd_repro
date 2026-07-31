"""Gate 5f -- driver A/B/C: squashed vs m-loop vs is_grads_batched must agree.

The three drivers obtain the same ``A`` by different routes and differ only in
cost, so any disagreement is a driver bug rather than an identity bug. This is
the regression net for :func:`jdgram.engine.hooks.compute_gramian`'s reverse
paths, and in particular for the squashed driver's block-diagonal assumption --
the one that turns m reverse passes (or one vmapped pass costing m times the
memory) into a single ordinary backward.

It also pins the two structural properties that make squashed worth having:
per-module streaming frees captures on the spot, and the driver refuses to run
when the assumption it rests on does not hold.
"""

from __future__ import annotations

import pytest
import torch

from gates._helpers import make_compute_losses, run_engine
from jdgram.engine.hooks import compute_gramian
from jdgram.engine.registry import collect_hookable_modules, positional_embedding_handler
from models.configs import forward_logits, per_sequence_losses

DRIVERS = ("squashed", "loop", "batched")


@pytest.fixture
def hooked(gate_model):
    return collect_hookable_modules(gate_model), {
        "transformer.wpe": positional_embedding_handler
    }


def test_all_drivers_agree(gate_model, gate_data, hooked):
    """Total and every per-module block must match across all three drivers."""
    idx, targets = gate_data
    modules, overrides = hooked

    results = {
        d: run_engine(gate_model, idx, targets, modules,
                      handler_overrides=overrides, driver=d)
        for d in DRIVERS
    }

    ref = results["loop"]
    for name in DRIVERS:
        if name == "loop":
            continue
        d_total = (results[name].total - ref.total).abs().max().item()
        print(f"[driver A/B] max|G_{name} - G_loop| = {d_total:.3e}")
        assert d_total < 1e-8, f"{name} disagrees with loop on the total: {d_total}"
        for mod_name in ref.per_module:
            d_mod = (
                results[name].per_module[mod_name] - ref.per_module[mod_name]
            ).abs().max().item()
            assert d_mod < 1e-8, f"{name} vs loop at {mod_name}: {d_mod}"


def test_squashed_streams_captures(gate_model, gate_data, hooked):
    """Untied model: the streaming driver holds nothing between layers.

    ``peak_held_bytes`` is the high-water mark of captures pinned across layer
    boundaries. With no shared parameters every layer's ``(A, X)`` is consumed
    inside its own backward, so the mark is exactly zero -- whereas the
    non-streaming drivers hold every layer's captures at once.
    """
    idx, targets = gate_data
    modules, overrides = hooked

    squashed = run_engine(gate_model, idx, targets, modules,
                          handler_overrides=overrides, driver="squashed")
    loop = run_engine(gate_model, idx, targets, modules,
                      handler_overrides=overrides, driver="loop")

    print(f"held bytes: squashed={squashed.peak_held_bytes} loop={loop.peak_held_bytes}")
    assert squashed.peak_held_bytes == 0
    assert loop.peak_held_bytes > 0, "loop driver should hold every layer's captures"


def test_squashed_refuses_unbatched_module(gate_model, gate_data):
    """Bare ``[T]`` positions make wpe unbatched, and a ones-seeded backward then
    sums over objectives there. The driver must say so rather than return a
    plausible wrong Gramian."""
    idx, targets = gate_data
    modules = collect_hookable_modules(gate_model)

    def losses_unbatched():
        return per_sequence_losses(
            forward_logits(gate_model, idx, batched_positions=False), targets
        )

    with pytest.raises(ValueError, match="batched on dim 0"):
        compute_gramian(
            gate_model,
            losses_unbatched,
            modules=modules,
            handler_overrides={"transformer.wpe": positional_embedding_handler},
            driver="squashed",
        )


def test_unbatched_positions_still_work_on_loop(gate_model, gate_data):
    """The unbatched forward is not broken, just restricted to the other drivers,
    and it must produce the same Gramian as the batched forward."""
    idx, targets = gate_data
    modules = collect_hookable_modules(gate_model)
    overrides = {"transformer.wpe": positional_embedding_handler}

    batched = run_engine(gate_model, idx, targets, modules,
                         handler_overrides=overrides, driver="squashed")
    unbatched = compute_gramian(
        gate_model,
        lambda: per_sequence_losses(
            forward_logits(gate_model, idx, batched_positions=False), targets
        ),
        modules=modules,
        handler_overrides=overrides,
        driver="loop",
    )
    d = (batched.total - unbatched.total).abs().max().item()
    print(f"batched-pos vs bare-pos positions: {d:.3e}")
    assert d < 1e-10
