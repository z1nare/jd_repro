"""Gate 5h: folding the residual tail into the reverse pass is exact and free.

Parameters with no closed-form identity used to cost a second forward plus ``m``
more backwards (``residual_gramian`` on a fresh graph). ``compute_gramian``'s
``residual_params=`` harvests them from the passes it already runs instead.

The point of the gate: that reordering must not move ``G`` by a single ulp, and
must still agree with a brute-force Jacobian over *every* parameter -- hooked and
residual alike.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from jdgram.engine.hooks import compute_gramian
from jdgram.engine.residual import residual_gramian

M, T, D, V = 3, 5, 8, 16


class _TailModel(nn.Module):
    """``scale`` is owned by the root, so it is never a hooked module."""

    def __init__(self) -> None:
        super().__init__()
        self.inp = nn.Linear(D, D, bias=False)
        self.head = nn.Linear(D, V, bias=False)
        self.scale = nn.Parameter(torch.randn(V, dtype=torch.float64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.inp(x)) * self.scale


@pytest.fixture
def setup():
    torch.manual_seed(0)
    model = _TailModel().double()
    x = torch.randn(M, T, D, dtype=torch.float64)
    calls = {"n": 0}

    def losses() -> torch.Tensor:
        calls["n"] += 1
        return model(x).pow(2).mean(dim=(1, 2))

    modules = {"inp": model.inp, "head": model.head}
    tail = [model.scale]
    return model, losses, modules, tail, calls


def _brute_force(losses_fn, params) -> torch.Tensor:
    losses = losses_fn()
    rows = []
    for i in range(M):
        g = torch.autograd.grad(
            losses[i], params, retain_graph=(i < M - 1), allow_unused=True
        )
        rows.append(torch.cat([
            (torch.zeros_like(p) if gi is None else gi).reshape(-1).double()
            for gi, p in zip(g, params)
        ]))
    J = torch.stack(rows)
    return J @ J.T


@pytest.mark.parametrize("driver", ["loop", "batched"])
def test_folded_total_matches_brute_force(setup, driver):
    model, losses, modules, tail, _ = setup
    ref = _brute_force(losses, [model.inp.weight, model.head.weight, model.scale])
    res = compute_gramian(
        model, losses, modules=modules, driver=driver,
        workspace_dtype=torch.float64, residual_params=tail,
    )
    torch.testing.assert_close(res.total, ref, atol=1e-10, rtol=0)


@pytest.mark.parametrize("driver", ["loop", "batched"])
def test_folded_block_identical_to_standalone(setup, driver):
    """The fold must reproduce residual_gramian exactly, not merely closely."""
    model, losses, modules, tail, _ = setup
    res = compute_gramian(
        model, losses, modules=modules, driver=driver,
        workspace_dtype=torch.float64, residual_params=tail,
    )
    standalone = residual_gramian(losses(), tail)
    assert torch.equal(res.residual, standalone)


def test_fold_saves_a_forward(setup):
    """One forward with the fold; two without it."""
    model, losses, modules, tail, calls = setup

    calls["n"] = 0
    compute_gramian(model, losses, modules=modules, driver="loop",
                    workspace_dtype=torch.float64, residual_params=tail)
    folded = calls["n"]

    calls["n"] = 0
    compute_gramian(model, losses, modules=modules, driver="loop",
                    workspace_dtype=torch.float64)
    residual_gramian(losses(), tail)
    separate = calls["n"]

    assert folded == 1
    assert separate == 2


def test_no_residual_params_leaves_result_unchanged(setup):
    """Omitting residual_params must not perturb the hooked-module total."""
    model, losses, modules, tail, _ = setup
    with_tail = compute_gramian(
        model, losses, modules=modules, driver="loop",
        workspace_dtype=torch.float64, residual_params=tail,
    )
    without = compute_gramian(
        model, losses, modules=modules, driver="loop",
        workspace_dtype=torch.float64,
    )
    assert without.residual is None
    torch.testing.assert_close(
        with_tail.total - with_tail.residual, without.total, atol=1e-12, rtol=0
    )


def test_squashed_refuses_residual_params(setup):
    """A ones-seeded backward cannot yield per-objective rows; say so loudly."""
    model, losses, modules, tail, _ = setup
    with pytest.raises(NotImplementedError, match="per-objective residual rows"):
        compute_gramian(model, losses, modules=modules, driver="squashed",
                        residual_params=tail)
