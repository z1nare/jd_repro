"""Dual-cone projector backends, so the QP stage can be compared like for like.

UPGrad's weights come from projecting each row of ``U`` onto the dual cone of the
Jacobian rows, which by Proposition 1 of the JD paper is the QP

    min_v  v^T G v      subject to   u <= v

TorchJD exposes this behind :class:`torchjd.linalg.DualConeProjector` and ships
exactly one implementation, :class:`QuadprogProjector`, which is CPU-only: it
calls ``G.cpu().numpy().astype(float64)``, loops the rows with
``np.apply_along_axis``, solves each with ``quadprog``, and copies the result
back. On a CUDA run that is a device->host transfer, a Python-level loop, and a
host->device transfer on the critical path of every optimiser step.

jacopt solves the same QP with a backend-agnostic consensus ADMM
that follows the array type, so handing it CUDA tensors keeps the solve on the
device. :class:`JacoptProjector` below adapts it to TorchJD's interface so the
two can be swapped under an otherwise identical training step:

    UPGradWeighting(projector=JacoptProjector())

Whether that is *faster* is an empirical question and the answer depends on m --
see level L10 of bench/profile_suite.py. At small m the QP is an m x m problem
where CPU quadprog is microseconds and GPU kernels are launch-latency, so the
device solve can easily lose; the transfer and the Python loop are what actually
cost. This module exists so that question is settled by measurement.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import torch
from torch import Tensor

_JACOPT_ERROR: str | None = None
try:
    from jacopt import dual_cone_project as _jacopt_dual_cone_project
except Exception as e:  # noqa: BLE001
    _jacopt_dual_cone_project = None  # type: ignore[assignment]
    _JACOPT_ERROR = repr(e)

_TORCHJD_ERROR: str | None = None
try:
    from torchjd.linalg import QuadprogProjector
except Exception as e:  # noqa: BLE001
    QuadprogProjector = None  # type: ignore[assignment,misc]
    _TORCHJD_ERROR = repr(e)


def jacopt_available() -> bool:
    return _jacopt_dual_cone_project is not None


def jacopt_error() -> str | None:
    return _JACOPT_ERROR


@dataclass
class JacoptProjector:
    """TorchJD ``DualConeProjector`` backed by ``jacopt.dual_cone_project``.

    Deliberately does NOT subclass TorchJD's ABC: the ABC is only a typing
    contract (``__call__(U, G) -> Tensor``), and keeping this free-standing means
    the module imports even when TorchJD is absent.

    :param device: ``"keep"`` runs the solve wherever ``G`` already lives (the
        point of the exercise). ``"cpu"`` moves to host first, which is the honest
        control -- it isolates "jacopt vs quadprog" from "GPU vs CPU".
    :param normalize: jacopt objective conditioning mode; ``None`` uses its legacy
        ``norm_eps`` behaviour. TorchJD's QuadprogProjector normalises by trace
        with ``norm_eps=1e-4``, so ``"trace"`` is the closest match.
    :param reg_eps: diagonal regularisation, matching TorchJD's default so the two
        solve the same regularised problem rather than two different ones.
    """

    device: str = "keep"
    normalize: str | None = "trace"
    norm_eps: float = 1e-4
    reg_eps: float = 1e-4
    #: Bound the ADMM. Measured on conflicting Gramians it does not converge and
    #: runs its whole budget (seconds per projection at m>=32), so an uncapped
    #: benchmark measures the ceiling rather than the solver.
    max_iter: int = 2000

    def __post_init__(self) -> None:
        #: Populated by the last call: admm_iters, converged, max_iter, fastpath.
        self.last_stats: dict = {}

    def __call__(self, U: Tensor, G: Tensor) -> Tensor:
        if _jacopt_dual_cone_project is None:
            raise RuntimeError(f"jacopt is not importable: {_JACOPT_ERROR}")
        src_device, src_dtype = G.device, G.dtype
        g = G.detach()
        u = U.detach()
        if self.device == "cpu":
            g, u = g.cpu(), u.cpu()
        # jacopt's ADMM promotes to float64 when either input is float64; give it
        # float64 explicitly so the comparison against quadprog (which casts to
        # float64 unconditionally) is precision-matched rather than precision-lucky.
        g = g.to(torch.float64)
        u = u.to(torch.float64)
        stats: dict = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            W = _jacopt_dual_cone_project(
                g, u, normalize=self.normalize, norm_eps=self.norm_eps,
                reg_eps=self.reg_eps, max_iter=self.max_iter, stats=stats,
            )
        self.last_stats = stats
        return torch.as_tensor(W, device=src_device, dtype=src_dtype)

    def __repr__(self) -> str:  # shows up in run manifests
        return (f"JacoptProjector(device={self.device!r}, "
                f"normalize={self.normalize!r}, reg_eps={self.reg_eps}, "
                f"max_iter={self.max_iter})")


def available_projectors(include_jacopt: bool = True) -> dict[str, object]:
    """Every projector we can benchmark in this environment, by label."""
    out: dict[str, object] = {}
    if QuadprogProjector is not None:
        out["torchjd_quadprog"] = QuadprogProjector(norm_eps=1e-4, reg_eps=1e-4)
    if include_jacopt and jacopt_available():
        out["jacopt_keep_device"] = JacoptProjector(device="keep")
        out["jacopt_force_cpu"] = JacoptProjector(device="cpu")
    return out


def count_device_syncs(fn, device: torch.device) -> tuple[object, int]:
    """Run ``fn`` and report how many host-device synchronisations it forced.

    Uses ``torch.cuda.set_sync_debug_mode("warn")``, which makes every implicit
    sync emit a warning. That is the cheapest way to answer "does this stage drag
    the GPU back to the host, and how often" without a full profiler trace.
    Returns ``(result, n_syncs)``; ``n_syncs`` is -1 when counting is unavailable.
    """
    if device.type != "cuda":
        return fn(), -1
    try:
        torch.cuda.synchronize(device)
        torch.cuda.set_sync_debug_mode("warn")
    except Exception:  # noqa: BLE001
        return fn(), -1
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = fn()
        n = sum(1 for w in caught
                if "synchron" in str(w.message).lower()
                or "cudaStreamSynchronize" in str(w.message))
        return result, n
    finally:
        try:
            torch.cuda.set_sync_debug_mode("default")
        except Exception:  # noqa: BLE001
            pass
