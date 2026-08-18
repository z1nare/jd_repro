"""Gates for the measured cost model and its wiring into the router.

The router's route choice can only ever change *cost*, never results -- both
routes compute the same Gramian, and gate 5 already pins that. So these gates
guard the two things that could actually go wrong when a cost model is
installed:

1. it changes a decision it should not (regressions in the fallback path), and
2. it fails to change the decision it exists to change (the vocabulary head).
"""

from __future__ import annotations

import math

import pytest
import torch

from jdgram import costmodel
from jdgram.costmodel import CostModel, Sample
from jdgram.engine.materialize import materialized_gramian
from jdgram.engine.router import force_route, route
from jdgram.identities.linear import sequence_gramian


@pytest.fixture(autouse=True)
def _clean_router():
    """No test may leak a forced route or an installed model into another."""
    force_route(None)
    costmodel.use(None)
    yield
    force_route(None)
    costmodel.use(None)


# --------------------------------------------------------------- fallback
def test_without_a_model_the_analytic_rule_is_unchanged():
    # The exact inequality the router shipped with, spot-checked either side.
    assert route(1, 8, 10_000) == "tfirst"      # m*T^2 = 64 < 10000
    assert route(8, 512, 65_536) == "dfirst"    # 2.1M >= 65536
    # Supplying the factored dims must not change anything while no model is
    # installed -- otherwise merely threading them through is a behaviour
    # change, and every existing measurement silently becomes incomparable.
    assert route(1, 8, 10_000) == route(1, 8, 10_000, 100, 100)
    assert route(8, 512, 65_536) == route(8, 512, 65_536, 256, 256)


def test_force_route_still_wins_over_an_installed_model():
    costmodel.use(CostModel(a_gemm=1e-9, a_elem=0.0, b_build=1e9, b_gram=0.0))
    for forced in ("tfirst", "dfirst"):
        force_route(forced)
        assert route(8, 512, 50257 * 768, 50257, 768) == forced


def test_a_caller_without_factored_dims_gets_the_old_rule():
    # A model is installed but the call cannot say how P factors, so the
    # cost model must not be consulted on a guess.
    costmodel.use(CostModel(a_gemm=0.0, a_elem=0.0, b_build=1e9, b_gram=1e9))
    assert route(1, 8, 10_000) == "tfirst"


# ----------------------------------------------------------------- fitting
def _synthetic(a_gemm, a_elem, b_build, b_gram, shapes):
    from jdgram.costmodel import dfirst_terms, tfirst_terms
    out = []
    for m, T, d_out, d_in in shapes:
        f1, f2 = tfirst_terms(m, T, d_out, d_in)
        g1, g2 = dfirst_terms(m, T, d_out, d_in)
        out.append(Sample(m, T, d_out, d_in,
                          a_gemm * f1 + a_elem * f2,
                          b_build * g1 + b_gram * g2))
    return out


def test_fit_recovers_known_coefficients():
    shapes = [(m, 512, d_out, d_in)
              for m in (1, 2, 4, 8, 16)
              for d_out, d_in in ((768, 768), (3072, 768), (50257, 768))]
    truth = (2.0e-10, 5.0e-10, 3.0e-10, 7.0e-10)
    model = CostModel.fit(_synthetic(*truth, shapes))
    for got, want in zip(
            (model.a_gemm, model.a_elem, model.b_build, model.b_gram), truth):
        assert got == pytest.approx(want, rel=1e-3)


def test_fit_never_returns_a_negative_coefficient():
    # Noisy, nearly-collinear data is exactly where an unconstrained least
    # squares hands back a negative rate and the model starts predicting
    # negative time. The constrained solve must not.
    shapes = [(m, 512, 768, 768) for m in (1, 2, 3, 4, 6, 8)]
    samples = _synthetic(1e-10, 0.0, 1e-10, 0.0, shapes)
    for i, s in enumerate(samples):          # deterministic perturbation
        s.tfirst_ms *= 1.0 + 0.3 * (-1) ** i
        s.dfirst_ms *= 1.0 - 0.3 * (-1) ** i
    model = CostModel.fit(samples)
    for c in (model.a_gemm, model.a_elem, model.b_build, model.b_gram):
        assert c >= 0.0
    t, d = model.predict(4, 512, 768, 768)
    assert t >= 0.0 and d >= 0.0


def test_measured_shapes_beat_the_fit():
    """An exactly-measured shape must be reported as measured, not smoothed."""
    s = Sample(8, 512, 50257, 768, tfirst_ms=141.3, dfirst_ms=10.8)
    model = CostModel.fit([s])
    t, d = model.predict(8, 512, 50257, 768)
    assert t == pytest.approx(141.3)
    assert d == pytest.approx(10.8)
    assert model.choose(8, 512, 50257, 768) == "dfirst"


# ------------------------------------------------- the decision that matters
def test_vocabulary_head_is_routed_to_the_faster_side():
    """The whole reason this module exists.

    The analytic rule needs m > 147 before it stops sending GPT-2's 50k-vocab
    head down the 13x-slower route. A model fitted to the measured numbers has
    to fix that at a realistic objective count.
    """
    model = CostModel.fit([Sample(8, 512, 50257, 768, 141.3, 10.8)])
    costmodel.use(model)
    assert route(8, 512, 50257 * 768) == "tfirst", (
        "sanity: without factored dims this is still the old rule")
    assert route(8, 512, 50257 * 768, 50257, 768) == "dfirst"


def test_near_ties_go_to_the_leaner_route():
    """Memory is the engine's strongest result; do not spend it on noise."""
    m, T, d_out, d_in = 4, 512, 256, 256
    # Same predicted time for both routes -> whichever needs less workspace.
    model = CostModel.fit([Sample(m, T, d_out, d_in, 10.0, 10.0)])
    from jdgram.costmodel import dfirst_workspace, tfirst_workspace
    leaner = ("tfirst" if tfirst_workspace(m, T, d_out, d_in)
              <= dfirst_workspace(m, T, d_out, d_in) else "dfirst")
    assert model.choose(m, T, d_out, d_in) == leaner


def test_workspace_cap_overrides_the_time_preference():
    from jdgram.costmodel import dfirst_workspace, tfirst_workspace
    # 256x256 at T=512: d-first's [m, P] block is far leaner than t-first's
    # 3mT^2 kernels, so a cap can sit strictly between the two.
    m, T, d_out, d_in = 4, 512, 256, 256
    w_t = tfirst_workspace(m, T, d_out, d_in)
    w_d = dfirst_workspace(m, T, d_out, d_in)
    assert w_d < w_t, "test shape must have a leaner d-first workspace"

    model = CostModel.fit([Sample(m, T, d_out, d_in, 1.0, 100.0)])
    assert model.choose(m, T, d_out, d_in) == "tfirst"   # 100x faster on time

    cap = (w_d + w_t) / 2                                # only tfirst violates
    assert model.choose(m, T, d_out, d_in, max_workspace_elems=cap) == "dfirst"


def test_an_unsatisfiable_cap_falls_back_to_time():
    """If neither route fits the cap, the cap cannot decide -- pick the fast
    one rather than silently preferring whichever is listed first."""
    model = CostModel.fit([Sample(4, 512, 256, 256, 1.0, 100.0)])
    assert model.choose(4, 512, 256, 256, max_workspace_elems=1.0) == "tfirst"


def test_roundtrip_through_json(tmp_path):
    model = CostModel.fit([Sample(8, 512, 50257, 768, 141.3, 10.8)],
                          device="test", dtype="fp32")
    back = CostModel.load(model.save(tmp_path / "cm.json"))
    assert back.device == "test"
    assert back.choose(8, 512, 50257, 768) == "dfirst"
    assert back.predict(8, 512, 50257, 768) == pytest.approx((141.3, 10.8))


# ------------------------------------------------------------- correctness
@pytest.mark.parametrize("has_bias", [False, True])
def test_both_routes_still_agree_whatever_the_model_says(has_bias):
    """The safety net: a routing change can only ever cost time."""
    torch.manual_seed(0)
    m, T, d_out, d_in = 3, 7, 5, 4
    A = torch.randn(m, T, d_out, dtype=torch.float64)
    X = torch.randn(m, T, d_in, dtype=torch.float64)
    g_t = sequence_gramian(A, X, has_bias, workspace_dtype=torch.float64)
    g_d = materialized_gramian(A, X, has_bias, workspace_dtype=torch.float64)
    assert torch.allclose(g_t, g_d, rtol=1e-10, atol=1e-10)
