"""Gate 6 -- degenerate objective families: G checked against structure, not data.

Gates 0-5g pin the engine against a brute-force ``[m, P]`` Jacobian on
*independent* objectives. That is the strongest numerical check available, but it
is also the weakest *diagnostic* one: brute force and the engine share the same
forward, so when they agree the answer is "the identities reproduce autograd",
and when they disagree the failure names a layer and nothing else.

This gate takes the other route. Each test arranges the m objectives so that the
exact shape of ``G = J Jt`` is known in closed form *before* anything runs, from
the relationship between the objectives alone. Writing ``g_i`` for objective i's
gradient, ``G[i][j] = g_i . g_j``, so:

  S1  m = 1                 G is [1, 1] and G[0][0] = ||g||^2
  S2  L2 = L1               both rows of J are g: G = ||g||^2 * ones(2, 2)
  S3  L2 = 1.1 * L1         G = ||g||^2 * [[1, 1.1], [1.1, 1.21]], det G = 0
  S4  L2 = L1 + c           constants have no gradient: G identical to S2's
  S5  S2 on a TIED model    rank-1 must survive the four-term II.4 sum
  S6  L2 = -L1              g_2 = -g_1, so G[0][1] is strictly NEGATIVE

S2 is the load-bearing one, and it is worth being exact about why, because the
obvious reading of it is too generous. Duplication forces a rank-1 Gramian, and
rank 1 is a measure-zero object -- but any engine that treats the two objective
rows symmetrically lands on ``a * ones(2, 2)`` for *some* ``a``, so the pattern
alone only convicts bugs that break that symmetry: a mis-indexed objective row,
an objective axis read transposed, a driver that recovers one objective's
upstream gradient and reuses it. The other half of the claim is the value. ``a``
has to equal ``||g||^2`` taken off the brute-force Jacobian, and that is what
convicts a dropped cross-term or a double-counted layer. Every test below asserts
both halves against brute force, and S3's unequal coefficients then remove the
symmetry degeneracy outright by demanding three different entries.

S5 points that same pair of assertions at
:func:`jdgram.identities.tied.tied_gramian`, whose ``G_hh + G_ee + G_he + G_he.T``
is exactly where a sign error or a missing transpose would hide. Measured rather
than assumed: deleting ``G_he.T`` on the bias=True gate model leaves the Gramian
perfectly rank 1 and only moves ``a``, from 25.341 to 25.408 -- so on this family
of objectives it is the *value* assertion that catches the tied-weight bug, not
the rank. S5 is the gate protecting that work; it earns it through the anchor,
and the rank-1 shape is what makes the anchor cheap to read.

The objective relationships mirror ``apply_objective_mode`` in
``bench/profile_suite.py`` (its ``duplicate`` / ``scaled`` / ``conflicting``
modes) so that what the benchmark campaign measures and what this gate proves are
the same constructions, down to the coefficient values.
"""

from __future__ import annotations

import torch

from gates._helpers import make_compute_losses, run_engine
from gates.brute_force import true_gramian
from jdgram.engine.hooks import LayerCapture, compute_gramian
from jdgram.engine.registry import (
    collect_hookable_modules,
    positional_embedding_handler,
)
from jdgram.identities.tied import tied_gramian
from models.configs import forward_logits, per_sequence_losses

# Every comparison here is against a closed-form value in float64, and conftest's
# autouse ``_fp64_identity_workspace`` pins the identity workspace to match, so
# the tolerance is the same absolute one the rest of the suite uses. rtol=0 is
# deliberate: a relative tolerance would silently scale with ||g||^2 and stop
# testing anything on the entries that are supposed to be zero.
ATOL = 1e-10

# Same values as bench/profile_suite.py's objective modes, so the campaign's
# "scaled" and "conflicting" rows and these gates describe one construction.
SCALED_COEFFS = (1.0, 1.1)
CONFLICTING_COEFFS = (1.0, -1.0)
CONSTANT_OFFSET = 3.7

# gate 5e's group and handler, restated rather than imported: S5's whole claim is
# about what happens *inside* the four-term sum, so the call into it belongs in
# the file that makes the claim. Keep in sync with gates/test_5e_tied.py.
TIED_GROUP = frozenset({"transformer.wte", "lm_head"})


def _tied_shared_handler(captures: dict[str, LayerCapture]) -> torch.Tensor:
    head = captures["lm_head"]
    emb = captures["transformer.wte"]
    return tied_gramian(head.A, head.X, emb.A, emb.X)


def _duplicate_rows(
    idx: torch.Tensor,
    targets: torch.Tensor,
    m: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batch row 0, repeated m times -- ``apply_objective_mode``'s ``duplicate``.

    Slice ``0:1`` and repeat, never index ``0`` and stack: indexing drops the
    batch dimension, and every hooked module must stay batched on dim 0 or the
    squashed driver refuses the run outright (see ``_is_batched`` in
    :mod:`jdgram.engine.hooks`). The same slice-don't-index rule is why S1 builds
    its single-objective batch with ``idx[0:1]``.
    """
    reps = (m, *([1] * (idx.ndim - 1)))
    return idx[0:1].repeat(*reps), targets[0:1].repeat(*reps)


def _coefficient_losses(model, idx, targets, coeffs: torch.Tensor):
    """``compute_losses`` for ``compute_gramian``, with per-objective scaling.

    :func:`gates._helpers.run_engine` builds its loss callable internally from
    ``(idx, targets)`` and has nowhere to say "scale objective 1 by 1.1", so S3
    and S6 call :func:`jdgram.engine.hooks.compute_gramian` directly with this.
    Scaling the loss *vector* rather than the data is what makes the expected
    Gramian exact: ``L_i -> c_i L_i`` implies ``g_i -> c_i g_i`` implies
    ``G -> (c c^T) * G``, with no assumption about the model at all.
    """

    def compute_losses() -> torch.Tensor:
        losses = per_sequence_losses(forward_logits(model, idx), targets)
        return losses * coeffs.to(losses.dtype)

    return compute_losses


def _gramian(model, compute_losses, *, shared_handlers=None):
    """Engine run over the whole model, with the one override no run may omit.

    ``transformer.wpe`` and ``transformer.wte`` are both ``nn.Embedding``, so type
    dispatch cannot tell them apart, and without the override the position
    embedding is handed the token-embedding identity.

    Measured, because the honest version is not the scary one: on this model the
    override is numerically a no-op. ``pos = arange(T)`` makes every index
    distinct, so the token identity's ``idx == idx`` mask collapses to ``t == s``
    and it reduces to exactly II.3's ``A A^T``. Omitting it changes no gate's
    answer -- not this one's, not 5d's, not 5e's. That is the reason to funnel
    every call through one helper rather than repeat the override six times: the
    habit has to survive reaching a model where the two identities stop agreeing
    (repeated or reset position ids, packed sequences, a wpe indexed by anything
    other than a bijection), and no test in this suite will remind anyone.
    """
    return compute_gramian(
        model,
        compute_losses,
        modules=collect_hookable_modules(model),
        handler_overrides={"transformer.wpe": positional_embedding_handler},
        shared_handlers=shared_handlers,
    )


def _outer_scaled_brute(g_brute: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
    """``(c c^T) * G_brute`` -- the brute-force anchor, corrected for coefficients.

    :func:`gates.brute_force.true_gramian` builds its own unweighted losses and
    cannot see the coefficients, so on a duplicated batch it returns the unscaled
    ``||g||^2 * ones(m, m)``. Applying the outer product here is not a fudge: it
    is the scaling law ``G -> (c c^T) * G`` written out, so the cross-check tests
    that law explicitly instead of assuming it.
    """
    return torch.outer(coeffs, coeffs) * g_brute


def _squared_norm(J: torch.Tensor, row: int = 0) -> torch.Tensor:
    """``||g_row||^2`` straight off the brute-force Jacobian, as a 0-d tensor."""
    return (J[row] * J[row]).sum()


# --------------------------------------------------------------------------- S1
def test_s1_single_objective_gramian_is_the_squared_norm(
    gate_model, gate_data, gate_layout
):
    """m = 1. G is [1, 1] and its only entry is ||g||^2.

    The degenerate end of the ladder, and the one that catches an engine that has
    quietly grown an m >= 2 assumption -- an ``eye(m)`` seed, a cross-term loop
    that never executes, a shape check written against a square block.
    """
    idx, targets = gate_data
    # 0:1, not 0. Indexing would hand the model a [T] batch and the squashed
    # driver would (correctly) reject every hooked module as unbatched.
    idx, targets = idx[0:1], targets[0:1]

    g_true, J, _ = true_gramian(gate_model, idx, targets, layout=gate_layout)
    assert g_true.shape == (1, 1)
    assert J.shape[0] == 1

    result = _gramian(gate_model, make_compute_losses(gate_model, idx, targets))
    assert result.total.shape == (1, 1)

    diff = (result.total - g_true).abs().max().item()
    print(f"S1 m=1: ||g||^2 = {g_true[0, 0].item():.6f}  max|G - G_true| = {diff:.3e}")
    torch.testing.assert_close(result.total, g_true, rtol=0, atol=ATOL)
    torch.testing.assert_close(result.total[0, 0], _squared_norm(J), rtol=0, atol=ATOL)


# --------------------------------------------------------------------------- S2
def test_s2_duplicate_objectives_give_a_rank_one_gramian(
    gate_model, gate_data, gate_layout
):
    """L2 = L1 exactly, so every row of J is the same g and G = ||g||^2 * ones.

    For a symmetric 2x2, "all four entries equal" *is* rank 1. Both halves of that
    matter and they catch different bugs (see the module docstring): the shared
    *pattern* convicts an engine that handles the two objective rows differently,
    and the shared *value* -- anchored to ``||g||^2`` off the brute-force Jacobian
    rather than to the engine's own diagonal -- convicts one that gets the
    magnitude wrong while staying symmetric.
    """
    idx, targets = _duplicate_rows(*gate_data, m=2)

    g_true, J, _ = true_gramian(gate_model, idx, targets, layout=gate_layout)
    # Premise check, on brute force rather than on the engine: if the two rows of
    # J were not identical the rest of this test would be asserting the wrong
    # thing, and it would be the batch construction at fault, not the engine.
    torch.testing.assert_close(J[0], J[1], rtol=0, atol=ATOL)

    result = _gramian(gate_model, make_compute_losses(gate_model, idx, targets))
    norm_sq = _squared_norm(J)
    expected = norm_sq * torch.ones(2, 2, dtype=torch.float64)

    diff = (result.total - expected).abs().max().item()
    print(f"S2 duplicate: ||g||^2 = {norm_sq.item():.6f}  max|G - rank1| = {diff:.3e}")
    torch.testing.assert_close(result.total, g_true, rtol=0, atol=ATOL)
    torch.testing.assert_close(result.total, expected, rtol=0, atol=ATOL)


# --------------------------------------------------------------------------- S3
def test_s3_scaled_duplicate_scales_the_gramian_quadratically(
    gate_model, gate_data, gate_layout
):
    """L2 = 1.1 * L1, so G = ||g||^2 * [[1, 1.1], [1.1, 1.21]] -- still rank 1.

    S2 cannot distinguish "the engine tracks each objective" from "the engine
    computes one gradient and broadcasts it", because under S2 those give the same
    answer. Unequal coefficients separate them: the entries are now three
    different numbers, and only a G that carries the objective axis correctly gets
    all three right while keeping the determinant at zero.
    """
    idx, targets = _duplicate_rows(*gate_data, m=2)
    coeffs = torch.tensor(SCALED_COEFFS, dtype=torch.float64)

    g_brute, J, _ = true_gramian(gate_model, idx, targets, layout=gate_layout)
    expected = _outer_scaled_brute(g_brute, coeffs)

    result = _gramian(
        gate_model, _coefficient_losses(gate_model, idx, targets, coeffs)
    )
    G = result.total

    diff = (G - expected).abs().max().item()
    print(
        f"S3 scaled {SCALED_COEFFS}: G = {G.flatten().tolist()}  "
        f"max|G - (c c^T)*G_brute| = {diff:.3e}"
    )
    torch.testing.assert_close(G, expected, rtol=0, atol=ATOL)
    torch.testing.assert_close(G[0, 1], G[1, 0], rtol=0, atol=ATOL)
    torch.testing.assert_close(
        G[1, 1], _squared_norm(J) * coeffs[1] * coeffs[1], rtol=0, atol=ATOL
    )

    # Rank 1 restated as the vanishing 2x2 determinant. Normalised by the product
    # of the diagonal because det carries ||g||^4: an absolute atol on the raw
    # determinant is a tolerance that tightens or loosens with the model's
    # gradient scale, which is not a property this gate should depend on.
    det = G[0, 0] * G[1, 1] - G[0, 1] * G[1, 0]
    normalised = det / (G[0, 0] * G[1, 1])
    print(f"S3 det G = {det.item():.3e}  normalised = {normalised.item():.3e}")
    torch.testing.assert_close(
        normalised, torch.zeros((), dtype=torch.float64), rtol=0, atol=ATOL
    )


# --------------------------------------------------------------------------- S4
def test_s4_constant_offset_leaves_the_gramian_untouched(
    gate_model, gate_data, gate_layout
):
    """L2 = L1 + c for a python-float c. d(c)/dtheta = 0, so G is S2's G exactly.

    This is the gate against an engine that has reached for the loss *values*
    anywhere -- a normalisation by the loss, a weighting derived from it, a
    reduction that is not the one the identity assumes. None of that is visible
    while the objectives are unshifted, because the values happen to be the ones
    the arithmetic wants; adding a constant separates value from gradient and any
    such dependence turns into a diff.
    """
    idx, targets = _duplicate_rows(*gate_data, m=2)
    g_true, J, _ = true_gramian(gate_model, idx, targets, layout=gate_layout)

    offsets = torch.tensor([0.0, CONSTANT_OFFSET], dtype=torch.float64)

    def shifted_losses() -> torch.Tensor:
        return per_sequence_losses(forward_logits(gate_model, idx), targets) + offsets

    # Guard against a vacuous pass: if the offset never reached the loss vector
    # this test would be S2 run twice and would agree with itself perfectly.
    values = shifted_losses()
    torch.testing.assert_close(
        values[1] - values[0],
        torch.tensor(CONSTANT_OFFSET, dtype=values.dtype),
        rtol=0,
        atol=ATOL,
    )

    plain = _gramian(gate_model, make_compute_losses(gate_model, idx, targets))
    shifted = _gramian(gate_model, shifted_losses)
    expected = _squared_norm(J) * torch.ones(2, 2, dtype=torch.float64)

    diff = (shifted.total - plain.total).abs().max().item()
    print(
        f"S4 offset c={CONSTANT_OFFSET}: losses {values.tolist()}  "
        f"max|G_shifted - G_plain| = {diff:.3e}"
    )
    torch.testing.assert_close(shifted.total, plain.total, rtol=0, atol=ATOL)
    torch.testing.assert_close(shifted.total, g_true, rtol=0, atol=ATOL)
    torch.testing.assert_close(shifted.total, expected, rtol=0, atol=ATOL)


# --------------------------------------------------------------------------- S5
def test_s5_tied_model_duplicate_objectives_stay_rank_one(
    tied_model, gate_data, tied_layout
):
    """S2 on a tied model, which routes the shared W through the four-term II.4 sum.

    ``wte.weight is lm_head.weight``, so the per-objective gradient of that one
    matrix is the *sum* of its two site gradients, and the Frobenius product of
    two sums is ``G_hh + G_ee + G_he + G_he.T`` -- four terms, not two. Gate 5e
    already diffs that against brute force on independent objectives; this runs
    the same sum on objectives whose answer is known in closed form, so a failure
    reads as "the tied identity is off by this much" instead of "these two
    computations of the same thing disagree".

    Do not expect the rank to be what fails. Dropping ``G_he.T``, flipping its
    sign or transposing it the wrong way all keep the matrix symmetric, positive
    on the diagonal and -- on a duplicated batch -- still rank 1; what moves is
    the shared entry. So the assertion that does the work here is the one against
    ``||g||^2``, and against ``g_true``.
    """
    idx, targets = _duplicate_rows(*gate_data, m=2)

    g_true, J, _ = true_gramian(tied_model, idx, targets, layout=tied_layout)
    torch.testing.assert_close(J[0], J[1], rtol=0, atol=ATOL)

    result = _gramian(
        tied_model,
        make_compute_losses(tied_model, idx, targets),
        shared_handlers={TIED_GROUP: _tied_shared_handler},
    )
    # The tied group has to have actually contributed; if the shared handler never
    # fired, the rank-1 assertion below could still pass on a model whose head and
    # embedding were simply left out of the sum.
    assert TIED_GROUP in result.per_shared_group

    norm_sq = _squared_norm(J)
    expected = norm_sq * torch.ones(2, 2, dtype=torch.float64)

    diff = (result.total - expected).abs().max().item()
    print(f"S5 tied duplicate: ||g||^2 = {norm_sq.item():.6f}  max|G - rank1| = {diff:.3e}")
    torch.testing.assert_close(result.total, g_true, rtol=0, atol=ATOL)
    torch.testing.assert_close(result.total, expected, rtol=0, atol=ATOL)


# --------------------------------------------------------------------------- S6
def test_s6_negated_objective_gives_a_strictly_negative_offdiagonal(
    gate_model, gate_data, gate_layout
):
    """L2 = -L1, so g_2 = -g_1 and G = ||g||^2 * [[1, -1], [-1, 1]].

    Conflict has to be *constructed*, not sampled. Two cross-entropy objectives on
    independent corpus windows are usually positively correlated, so a conflict
    test built from random data passes for the wrong reason and fails
    intermittently. Negating a coefficient makes the off-diagonal exactly
    ``-||g||^2``, and that is the only arrangement in which "strictly negative" is
    a statement about the engine rather than about the draw.

    It looks wrong at first that this works at all. The default ``squashed`` driver
    runs ONE backward seeded with ``ones(m)``, which differentiates the SUM of the
    objectives -- and here that sum is ``L - L = 0``. The driver never uses the
    summed value. It relies on block-diagonality: with per-instance losses and
    batch-independent modules, ``d(sum_i L_i)/dz[i] == dL_i/dz[i]``, so row i of
    the gradient arriving at a layer output is objective i's upstream gradient,
    read separately from every other row. S6 therefore doubles as a test of that
    assumption -- if the driver were really differentiating a scalar sum, the
    Gramian of a batch whose losses cancel would come back zero.
    """
    idx, targets = _duplicate_rows(*gate_data, m=2)
    coeffs = torch.tensor(CONFLICTING_COEFFS, dtype=torch.float64)

    g_brute, J, _ = true_gramian(gate_model, idx, targets, layout=gate_layout)
    expected = _outer_scaled_brute(g_brute, coeffs)

    compute_losses = _coefficient_losses(gate_model, idx, targets, coeffs)

    # The cancellation is the point, so state it rather than leave it implied.
    summed = compute_losses().sum()
    torch.testing.assert_close(
        summed, torch.zeros((), dtype=torch.float64), rtol=0, atol=ATOL
    )

    result = _gramian(gate_model, compute_losses)
    G = result.total

    diff = (G - expected).abs().max().item()
    print(
        f"S6 conflicting: sum(L) = {summed.item():.3e}  G = {G.flatten().tolist()}  "
        f"max|G - (c c^T)*G_brute| = {diff:.3e}"
    )
    torch.testing.assert_close(G, expected, rtol=0, atol=ATOL)
    # Strictly negative, and non-degenerately so: a G that came back all zeros
    # would satisfy assert_close against nothing useful, so anchor the sign to a
    # diagonal that is genuinely positive.
    assert G[0, 0].item() > 0.0, "degenerate: the squared gradient norm vanished"
    assert G[0, 1].item() < 0.0, f"expected a conflicting pair, got G[0][1]={G[0, 1]}"
    torch.testing.assert_close(G[0, 1], -G[0, 0], rtol=0, atol=ATOL)
    torch.testing.assert_close(G[1, 1], G[0, 0], rtol=0, atol=ATOL)


# ---------------------------------------------------- driver-independence anchor
def test_s2_rank_one_holds_on_every_driver(gate_model, gate_data, gate_layout):
    """S2 must not be a property of the default driver.

    ``squashed`` is the one that ships and the one whose block-diagonal shortcut
    S6 leans on, but ``||g||^2 * ones(2, 2)`` is a statement about the Gramian,
    not about how ``A`` was obtained. Gate 5f already A/Bs the three drivers, and
    that is the check this one is not: 5f compares them to *each other*, so three
    drivers that drifted together would agree with each other and pass. Here each
    one is held against a closed-form answer independently, which is the only
    version that survives a common-mode change to the reverse path.
    """
    idx, targets = _duplicate_rows(*gate_data, m=2)
    _, J, _ = true_gramian(gate_model, idx, targets, layout=gate_layout)
    expected = _squared_norm(J) * torch.ones(2, 2, dtype=torch.float64)

    modules = collect_hookable_modules(gate_model)
    overrides = {"transformer.wpe": positional_embedding_handler}

    for driver in ("squashed", "loop", "batched"):
        result = run_engine(
            gate_model,
            idx,
            targets,
            modules,
            handler_overrides=overrides,
            driver=driver,
        )
        G = result.total
        print(f"S2/{driver}: max|G - rank1| = {(G - expected).abs().max().item():.3e}")
        torch.testing.assert_close(G, expected, rtol=0, atol=ATOL)
