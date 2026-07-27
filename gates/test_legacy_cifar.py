"""Legacy gate 4: the CIFAR Gramian identity still matches TorchJD's ``autogram``.

This is the old preflight gate 4 from ``legacy/cifar/iwrm_bench.py``, lifted out
so it runs standalone against :mod:`jdgram.engine.sequential` -- the code that
used to live in the top-level ``hadamard.py``.  Its job is to prove the Phase-0
split was lossless, and to keep proving it while the transformer identities are
built on top.

The model is defined here rather than imported from the frozen legacy harness:
the harness pulls in torchvision, matplotlib and a data loader, none of which
this comparison needs.  It is the paper's Appendix D Table 3 architecture,
which is the only thing gate 4 ever depended on.

Reference number from the fuji2 run: max abs diff 2.8e-14 at float64, batch 8.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
autogram = pytest.importorskip("torchjd.autogram")

from torch import nn  # noqa: E402

from jdgram.engine.sequential import sequential_gramian  # noqa: E402

TOL = 1e-6  # the tolerance the original gate asserted; observed diff is ~1e-14


def paper_cifar_cnn(width_mult: int = 1) -> nn.Sequential:
    """Appendix D, Table 3.  3x3 convs, stride 1, no padding, bias, ELU."""
    w = width_mult
    return nn.Sequential(
        nn.Conv2d(3, 32 * w, 3), nn.ELU(),
        nn.Conv2d(32 * w, 64 * w, 3, groups=32 * w),
        nn.MaxPool2d(2), nn.ELU(),
        nn.Conv2d(64 * w, 64 * w, 3, groups=64 * w),
        nn.MaxPool2d(3), nn.ELU(), nn.Flatten(),
        nn.Linear(1024 * w, 128 * w), nn.ELU(),
        nn.Linear(128 * w, 10),
    )


def test_sequential_gramian_matches_autogram():
    torch.manual_seed(1)
    m = 8
    model = paper_cifar_cnn().double()
    x = torch.randn(m, 3, 32, 32, dtype=torch.float64)
    y = torch.randint(0, 10, (m,))

    loss_fn = nn.CrossEntropyLoss(reduction="none")
    engine = autogram.Engine(model, batch_dim=0)
    g_reference = engine.compute_gramian(loss_fn(model(x), y))

    g_ours, _ = sequential_gramian(model, x, y)

    max_diff = (g_ours - g_reference).abs().max().item()
    assert max_diff < TOL, f"Gramian mismatch vs autogram: {max_diff:.2e}"
