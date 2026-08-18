"""A *measured* cost model for the router's route choice.

:mod:`jdgram.engine.router` shipped with one analytic inequality,
``tfirst iff m*T^2 < P_layer``, which compares the two routes' **workspace**.
Profiling at vocabulary scale showed that rule making the wrong call, because
it optimises a quantity that has stopped mattering:

  * At V=50257 the LM head has ``m*T^2 = 2.1M`` against ``P_layer = 12.9M``, so
    the rule selects T-first.
  * T-first on a vocabulary head measures 13x slower per kernel than d-first.
  * The workspace it saves is invisible at model scale, where peak is set by
    the forward activations. Measured at 124M, forcing d-first everywhere cost
    **1.0-1.2% more peak memory and saved up to 63% of step time**.

So the rule faithfully minimises bytes that are not the peak, and is blind to
time -- the only axis on which the two routes meaningfully differ.

WHAT THIS MODULE FITS
---------------------
Not a black box: the two terms per route come from the kernels themselves.

*T-first* (:func:`jdgram.identities.linear.sequence_gramian`) loops over the m
objectives; iteration i does two ``[T, d] x [d, mT]`` GEMMs and one elementwise
product over ``[T, mT]``::

    flops  ~  m^2 T^2 (d_out + d_in)          elementwise  ~  m^2 T^2

*d-first* (:func:`jdgram.engine.materialize.materialized_gramian`) forms
``B[i] = sum_t A[i,t] (x) X[i,t]`` then Grams it::

    flops  ~  m P T   (building B)            gram  ~  m^2 P

Each route is therefore a two-term nonnegative least squares against measured
times. Coefficients are hardware-dependent -- they encode achieved GEMM
efficiency at a given shape -- so they are *measured on the target card*, never
derived. :mod:`bench.calibrate_router` produces the table.

WHY NOT JUST "ALWAYS D-FIRST"
-----------------------------
Because it is false at small m. On GPT-2 124M at T=512 the measured whole-model
crossover sits at m ~ 3.5: T-first is genuinely faster at m=1,2,3 and d-first
from m=4 up. A rule that ignored that would trade one wrong answer for another.

SAFETY
------
Both routes are exactly equivalent numerically -- level L5 of
``bench/profile_suite.py`` measures them agreeing to 0.000e+00 at every
vocabulary tested -- so a wrong choice here can only cost time or memory, never
change what a run produces. With no table loaded the analytic rule is used
unchanged, so importing this module changes nothing until it is calibrated.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal, Sequence

Route = Literal["tfirst", "dfirst"]

# A choice inside this relative margin counts as a tie, and ties go to the
# route with the smaller workspace. The memory advantage is the strongest
# result the engine currently has; it should not be spent for noise.
TIE_MARGIN = 0.10


def tfirst_terms(m: int, T: int, d_out: int, d_in: int) -> tuple[float, float]:
    """Work terms for the T-first route: (GEMM flops, elementwise elements)."""
    mt2 = float(m) * float(m) * float(T) * float(T)
    return mt2 * float(d_out + d_in), mt2


def dfirst_terms(m: int, T: int, d_out: int, d_in: int) -> tuple[float, float]:
    """Work terms for the d-first route: (building B, Gramming B)."""
    P = float(d_out) * float(d_in)
    return float(m) * P * float(T), float(m) * float(m) * P


def tfirst_workspace(m: int, T: int, d_out: int, d_in: int) -> float:
    """Peak workspace elements, T-first: k_a, k_x and their product."""
    return 3.0 * m * T * T


def dfirst_workspace(m: int, T: int, d_out: int, d_in: int) -> float:
    """Peak workspace elements, d-first: the [m, d_out, d_in] block."""
    return float(m) * float(d_out) * float(d_in)


def _nnls2(rows: Sequence[tuple[float, float]], y: Sequence[float]
           ) -> tuple[float, float]:
    """Two-parameter nonnegative least squares, solved exactly.

    Small enough to do by hand and so avoid a scipy dependency inside the
    package. Solve the 2x2 normal equations; if either coefficient comes out
    negative the true constrained optimum lies on that axis, so refit the
    remaining single term (which is a one-line ratio) and return the better of
    the two candidates.
    """
    a = sum(r[0] * r[0] for r in rows)
    b = sum(r[0] * r[1] for r in rows)
    c = sum(r[1] * r[1] for r in rows)
    p = sum(r[0] * v for r, v in zip(rows, y))
    q = sum(r[1] * v for r, v in zip(rows, y))
    det = a * c - b * b
    if abs(det) > 1e-30:
        x0 = (c * p - b * q) / det
        x1 = (a * q - b * p) / det
        if x0 >= 0 and x1 >= 0:
            return x0, x1

    def sse(u: float, v: float) -> float:
        return sum((u * r[0] + v * r[1] - t) ** 2 for r, t in zip(rows, y))

    cands = []
    if a > 1e-30:
        cands.append((max(p / a, 0.0), 0.0))
    if c > 1e-30:
        cands.append((0.0, max(q / c, 0.0)))
    if not cands:
        return 0.0, 0.0
    return min(cands, key=lambda uv: sse(*uv))


@dataclass
class Sample:
    """One measured shape: milliseconds for each route."""
    m: int
    T: int
    d_out: int
    d_in: int
    tfirst_ms: float
    dfirst_ms: float


@dataclass
class CostModel:
    """Fitted coefficients plus the measurements they came from."""
    a_gemm: float = 0.0          # T-first, per GEMM flop
    a_elem: float = 0.0          # T-first, per elementwise element
    b_build: float = 0.0         # d-first, per flop building B
    b_gram: float = 0.0          # d-first, per flop Gramming B
    device: str = "?"
    dtype: str = "?"
    n_samples: int = 0
    # Exact measurements win over the fit wherever a shape was actually timed.
    measured: dict = field(default_factory=dict)

    # ------------------------------------------------------------- fitting
    @staticmethod
    def fit(samples: Sequence[Sample], device: str = "?", dtype: str = "?"
            ) -> "CostModel":
        if not samples:
            raise ValueError("cannot fit a cost model with no samples")
        tf_rows = [tfirst_terms(s.m, s.T, s.d_out, s.d_in) for s in samples]
        df_rows = [dfirst_terms(s.m, s.T, s.d_out, s.d_in) for s in samples]
        a0, a1 = _nnls2(tf_rows, [s.tfirst_ms for s in samples])
        b0, b1 = _nnls2(df_rows, [s.dfirst_ms for s in samples])
        measured = {_key(s.m, s.T, s.d_out, s.d_in):
                    [s.tfirst_ms, s.dfirst_ms] for s in samples}
        return CostModel(a_gemm=a0, a_elem=a1, b_build=b0, b_gram=b1,
                         device=device, dtype=dtype, n_samples=len(samples),
                         measured=measured)

    # ---------------------------------------------------------- prediction
    def predict(self, m: int, T: int, d_out: int, d_in: int
                ) -> tuple[float, float]:
        """(tfirst_ms, dfirst_ms), measured where known, fitted otherwise."""
        hit = self.measured.get(_key(m, T, d_out, d_in))
        if hit:
            return float(hit[0]), float(hit[1])
        f1, f2 = tfirst_terms(m, T, d_out, d_in)
        g1, g2 = dfirst_terms(m, T, d_out, d_in)
        return (self.a_gemm * f1 + self.a_elem * f2,
                self.b_build * g1 + self.b_gram * g2)

    def choose(self, m: int, T: int, d_out: int, d_in: int,
               max_workspace_elems: float | None = None) -> Route:
        """Pick the faster route, breaking near-ties toward less memory."""
        t_t, t_d = self.predict(m, T, d_out, d_in)
        w_t = tfirst_workspace(m, T, d_out, d_in)
        w_d = dfirst_workspace(m, T, d_out, d_in)

        if max_workspace_elems is not None:
            over_t = w_t > max_workspace_elems
            over_d = w_d > max_workspace_elems
            if over_t and not over_d:
                return "dfirst"
            if over_d and not over_t:
                return "tfirst"

        lo = min(t_t, t_d)
        if lo <= 0 or not math.isfinite(lo):
            return "tfirst" if w_t <= w_d else "dfirst"
        if abs(t_t - t_d) / lo < TIE_MARGIN:
            return "tfirst" if w_t <= w_d else "dfirst"
        return "tfirst" if t_t < t_d else "dfirst"

    # ------------------------------------------------------------ storage
    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json(), encoding="utf-8")
        return p

    @staticmethod
    def load(path: str | Path) -> "CostModel":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return CostModel(**d)


def _key(m: int, T: int, d_out: int, d_in: int) -> str:
    return f"{m}x{T}x{d_out}x{d_in}"


# ------------------------------------------------------ the active model
_active: CostModel | None = None


def use(model: CostModel | None) -> None:
    """Install a cost model, or ``None`` to fall back to the analytic rule."""
    global _active
    _active = model


def active() -> CostModel | None:
    return _active


def load_and_use(path: str | Path) -> CostModel:
    model = CostModel.load(path)
    use(model)
    return model


def crossover(m: int, T: int, d_out: int, d_in: int) -> Route:
    """Route for one layer under the active model.

    Raises if no model is installed -- callers that want a graceful fallback
    should check :func:`active` first, as :mod:`jdgram.engine.router` does.
    """
    if _active is None:
        raise NotImplementedError(
            "no measured cost model installed. Run "
            "`python bench/calibrate_router.py --out <file>` on the target "
            "card and load it with jdgram.costmodel.load_and_use(<file>), or "
            "leave it unset to keep the analytic rule."
        )
    return _active.choose(m, T, d_out, d_in)
