"""
Benchmark and profiling harness for TorchJD on the IWRM setting of
"Jacobian Descent for Multi-Objective Optimization" (arXiv:2406.16232v3).

Reproduces the paper's small-scale experiments and adds engine-level profiling:
  - Figure 2 loss curves (UPGrad / Mean / PCGrad / MGDA) on CIFAR-10 or SVHN,
    1024-image subset, batch 32, IWRM via SSJD, Appendix D architectures.
  - Table 7 per-aggregator epoch times (ratios relative to Mean).
  - autojac vs autogram engine comparison (time and peak memory).
  - Per-step timing decomposition and model-width scaling.

Subcommands:
  preflight  -- pref_vector audit, engine-equivalence assertion, autogram compat check
  figure2    -- aggregator training runs -> loss-curve plot (optional per-aggregator lr sweep)
  table7     -- per-aggregator epoch timing -> ratio table vs paper's Table 7
  engines    -- autojac vs autogram (UPGrad): time + peak memory bar chart
  scaling    -- width-multiplier sweep of both engines
  decompose  -- per-segment timing of one autogram step across (m, width)

Examples:
  python iwrm_bench.py preflight --dataset cifar10
  python iwrm_bench.py figure2  --dataset cifar10 --sweep
  python iwrm_bench.py table7   --dataset cifar10 --warmup 3 --timed 10
  python iwrm_bench.py engines  --dataset cifar10 --warmup 3 --timed 10
  python iwrm_bench.py scaling  --dataset cifar10 --widths 1 2 4 8

Requires: torch, torchvision, matplotlib, and `pip install "torchjd[quadprog_projector]"`.
`--dataset synthetic` runs on random data with the same shapes (no downloads).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from torch import nn

# ----------------------------------------------------------------------------- torchjd
from torchjd.aggregation import (
    MGDA,
    Mean,
    MeanWeighting,
    MGDAWeighting,
    PCGrad,
    PCGradWeighting,
    UPGrad,
    UPGradWeighting,
)
from torchjd.autogram import Engine
from torchjd.autojac import backward, jac_to_grad
from torch.profiler import profile, ProfilerActivity, record_function
AGGREGATORS = {
    # name -> (Aggregator ctor [autojac path], Weighting ctor [autogram path])
    "Mean": (Mean, MeanWeighting),
    "UPGrad": (UPGrad, UPGradWeighting),
    "PCGrad": (PCGrad, PCGradWeighting),
    "MGDA": (MGDA, MGDAWeighting),
}

# Paper Table 7 (NVIDIA L4, batch 32, autojac-era implementation), seconds/epoch.
# Used ONLY to compare ratios relative to Mean. "SGD-ERM" is the scalar baseline.
PAPER_TABLE7 = {
    "cifar10": {"SGD-ERM": 0.50, "Mean": 1.76, "UPGrad": 2.01, "PCGrad": 3.13, "MGDA": 5.22},
    "svhn":    {"SGD-ERM": 0.79, "Mean": 1.41, "UPGrad": 1.80, "PCGrad": 2.78, "MGDA": 5.50},
}

# Paper training horizons (Appendix D.7): epochs at 1024 images / batch 32.
PAPER_EPOCHS = {"cifar10": 20, "svhn": 25, "synthetic": 5}


# ----------------------------------------------------------------------------- utils
def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def device_from_arg(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


def ema(xs: list[float], alpha: float = 0.05) -> list[float]:
    out, m = [], None
    for x in xs:
        m = x if m is None else (1 - alpha) * m + alpha * x
        out.append(m)
    return out


# ----------------------------------------------------------------------------- data
def load_data(dataset: str, root: str, subset: int, seed: int, device: torch.device):
    """Return (X, Y) preloaded on `device`. X: [N,3,32,32] float32, Y: [N] int64.

    Preprocessing matches the paper (Appendix D.6): per-channel normalization with
    mean/std computed on the ENTIRE training split. Subset indices are seeded and
    saved next to the data so every run uses the same 1024 images.
    """
    if dataset == "synthetic":
        g = torch.Generator().manual_seed(seed)
        X = torch.randn(subset, 3, 32, 32, generator=g)
        Y = torch.randint(0, 10, (subset,), generator=g)
        return X.to(device), Y.to(device)

    import torchvision  # imported lazily so synthetic mode works without it

    if dataset == "cifar10":
        ds = torchvision.datasets.CIFAR10(root, train=True, download=True)
        data = torch.from_numpy(ds.data).float().div_(255)          # [50000,32,32,3]
        data = data.permute(0, 3, 1, 2).contiguous()                # [N,3,32,32]
        labels = torch.tensor(ds.targets, dtype=torch.long)
    elif dataset == "svhn":
        ds = torchvision.datasets.SVHN(root, split="train", download=True)
        data = torch.from_numpy(ds.data).float().div_(255)          # [N,3,32,32]
        labels = torch.from_numpy(ds.labels).long()
    else:
        raise ValueError(f"unknown dataset {dataset}")

    # Per-channel stats over the FULL training split (paper D.6).
    mean = data.mean(dim=(0, 2, 3), keepdim=True)
    std = data.std(dim=(0, 2, 3), keepdim=True)
    data = (data - mean) / std

    idx_file = Path(root) / f"{dataset}_subset{subset}_seed{seed}.pt"
    if idx_file.exists():
        idx = torch.load(idx_file)
    else:
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(len(data), generator=g)[:subset]
        idx_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(idx, idx_file)

    return data[idx].to(device), labels[idx].to(device)


def batch_order(n: int, batch_size: int, epoch: int, seed: int) -> list[torch.Tensor]:
    """Deterministic per-epoch shuffle, identical across aggregators/engines."""
    g = torch.Generator().manual_seed(seed * 100_003 + epoch)
    perm = torch.randperm(n, generator=g)
    return list(perm.split(batch_size))


# ----------------------------------------------------------------------------- models
def make_model(dataset: str, width_mult: int = 1) -> nn.Sequential:
    """Paper-exact CNNs (Appendix D, Tables 2-3). All convs: 3x3, stride 1, no
    padding, bias. Activation: ELU. Grouped convolutions as specified.

    width_mult scales channel counts (and the grouped-conv group counts with them)
    for the `scaling` subcommand; width_mult=1 is the paper's architecture.
    """
    w = width_mult
    if dataset in ("cifar10", "synthetic"):
        # Table 3. Dims: 32->30->28 -pool2-> 14->12 -pool3-> 4; 64*4*4 = 1024.
        return nn.Sequential(
            nn.Conv2d(3, 32 * w, 3), nn.ELU(),
            nn.Conv2d(32 * w, 64 * w, 3, groups=32 * w),
            nn.MaxPool2d(2), nn.ELU(),
            nn.Conv2d(64 * w, 64 * w, 3, groups=64 * w),
            nn.MaxPool2d(3), nn.ELU(), nn.Flatten(),
            nn.Linear(1024 * w, 128 * w), nn.ELU(),
            nn.Linear(128 * w, 10),
        )
    if dataset == "svhn":
        # Table 2. 32*4*4 = 512.
        return nn.Sequential(
            nn.Conv2d(3, 16 * w, 3), nn.ELU(),
            nn.Conv2d(16 * w, 32 * w, 3, groups=16 * w),
            nn.MaxPool2d(2), nn.ELU(),
            nn.Conv2d(32 * w, 32 * w, 3, groups=32 * w),
            nn.MaxPool2d(3), nn.ELU(), nn.Flatten(),
            nn.Linear(512 * w, 64 * w), nn.ELU(),
            nn.Linear(64 * w, 10),
        )
    raise ValueError(dataset)


def init_state(dataset: str, seed: int, out_dir: Path, width_mult: int = 1) -> dict:
    """One canonical initialization per (dataset, seed, width): saved to disk so every
    aggregator/engine run starts from bit-identical weights."""
    f = out_dir / f"init_{dataset}_w{width_mult}_seed{seed}.pt"
    if f.exists():
        return torch.load(f)
    set_seed(seed)
    sd = make_model(dataset, width_mult).state_dict()
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(sd, f)
    return sd


# ----------------------------------------------------------------------------- steps
loss_fn = nn.CrossEntropyLoss(reduction="none")  # IWRM: one CE loss per instance


def make_step_autojac(model, aggregator, optimizer, optimize_gramian: bool = False):
    params = list(model.parameters())

    def step(x, y) -> float:
        losses = loss_fn(model(x), y)                    # [B] objectives
        backward(losses)
        if optimize_gramian:
            jac_to_grad(params, aggregator, optimize_gramian_computation=True)
        else:
            jac_to_grad(params, aggregator)
        optimizer.step()
        optimizer.zero_grad()
        return float(losses.mean().detach())

    return step


def make_step_autogram(model, weighting, optimizer, batch_dim: int = 0):
    engine = Engine(model, batch_dim=batch_dim)

    def step(x, y) -> float:
        losses = loss_fn(model(x), y)                    # [B] objectives
        gramian = engine.compute_gramian(losses)         # [B,B], J never stored
        weights = weighting(gramian)                     # eq. 10: W(G)
        losses.backward(weights)                         # one scalar-equiv backward
        optimizer.step()
        optimizer.zero_grad()
        return float(losses.mean().detach())

    return step


def make_step_sgd_erm(model, optimizer):
    """Scalar baseline: ERM with plain SGD (paper's 'SGD' row in Table 7)."""

    def step(x, y) -> float:
        loss = loss_fn(model(x), y).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return float(loss.detach())

    return step


def run_training(step, X, Y, epochs: int, batch_size: int, seed: int,
                 device: torch.device) -> list[float]:
    """Run `epochs` epochs; return per-iteration mean batch loss."""
    curve = []
    for ep in range(epochs):
        for idx in batch_order(len(X), batch_size, ep, seed):
            curve.append(step(X[idx], Y[idx]))
    sync(device)
    return curve


def timed_epochs(step, X, Y, warmup: int, timed: int, batch_size: int, seed: int,
                 device: torch.device) -> tuple[float, float, float]:
    """Return (mean s/epoch, std s/epoch, peak GPU memory in MiB during timed part)."""
    for ep in range(warmup):
        for idx in batch_order(len(X), batch_size, ep, seed):
            step(X[idx], Y[idx])
    sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    times = []
    for ep in range(warmup, warmup + timed):
        sync(device)
        t0 = time.perf_counter()
        for idx in batch_order(len(X), batch_size, ep, seed):
            step(X[idx], Y[idx])
        sync(device)
        times.append(time.perf_counter() - t0)
    peak_mib = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else float("nan")
    mean = sum(times) / len(times)
    std = math.sqrt(sum((t - mean) ** 2 for t in times) / max(len(times) - 1, 1))
    return mean, std, peak_mib


# ----------------------------------------------------------------------------- runners
def fresh(dataset, agg_name, lr, args, engine: str):
    """Build (model, step_fn) from the canonical init for a given aggregator+engine."""
    device = device_from_arg(args.device)
    model = make_model(dataset, getattr(args, "width", 1))
    model.load_state_dict(init_state(dataset, args.seed, Path(args.out), getattr(args, "width", 1)))
    model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr)  # paper D.4: plain SGD, no momentum
    if engine == "sgd-erm":
        return model, make_step_sgd_erm(model, opt)
    agg_ctor, weighting_ctor = AGGREGATORS[agg_name]
    if engine == "autojac":
        return model, make_step_autojac(model, agg_ctor(), opt)
    if engine == "autojac-ogc":  # optimize_gramian_computation=True (memory-saving path)
        return model, make_step_autojac(model, agg_ctor(), opt, optimize_gramian=True)
    if engine == "autogram":
        return model, make_step_autogram(model, weighting_ctor(), opt)
    raise ValueError(engine)


def cmd_preflight(args):
    device = device_from_arg(args.device)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    X, Y = load_data(args.dataset, args.data_root, args.subset, args.seed, device)

    # --- Gate 1: pref_vector audit (structural). UPGrad must wrap UPGradWeighting
    # with identical defaults; both default to uniform 1/m via MeanWeighting.
    u, w = UPGrad(), UPGradWeighting()
    assert type(u.gramian_weighting) is type(w), "UPGrad does not wrap UPGradWeighting!"
    assert u.pref_vector is None and w.pref_vector is None, \
        f"pref_vector default mismatch: {u.pref_vector} vs {w.pref_vector}"
    print("[gate 1] pref_vector defaults identical (None -> uniform 1/m). OK")

    # --- Gate 2: autogram compatibility of the paper architecture (Task-2 check).
    set_seed(args.seed)
    model = make_model(args.dataset).to(device)
    try:
        engine = Engine(model, batch_dim=0)
        losses = loss_fn(model(X[: args.batch_size]), Y[: args.batch_size])
        G = engine.compute_gramian(losses)
        assert G.shape == (args.batch_size, args.batch_size)
        print(f"[gate 2] autogram accepts the paper CNN; Gramian {tuple(G.shape)}. OK")
    except Exception as e:  # noqa: BLE001
        print(f"[gate 2] FAILED - autogram incompatible layer in paper CNN: {e!r}")
        raise

    # --- Gate 3: single-batch update equivalence: autojac+UPGrad vs autogram+
    # UPGradWeighting must produce the same aggregated update from the same state.
    lr = 0.05
    torch.use_deterministic_algorithms(False)
    mA, stepA = fresh(args.dataset, "UPGrad", lr, args, "autojac")
    mB, stepB = fresh(args.dataset, "UPGrad", lr, args, "autogram")
    for k in range(args.k_steps):
        for idx in batch_order(len(X), args.batch_size, k, args.seed)[:4]:
            stepA(X[idx], Y[idx]); stepB(X[idx], Y[idx])
    max_diff = max(
        (pa - pb).abs().max().item()
        for pa, pb in zip(mA.parameters(), mB.parameters())
    )
    tol = 1e-4 if device.type == "cpu" else 5e-4
    print(f"[gate 3] {args.k_steps}-step trajectory max param diff = {max_diff:.2e} "
          f"(tolerance {tol:.0e})")
    assert max_diff < tol, "autojac and autogram trajectories diverged"
    print("preflight passed (3/3 gates)")


def lr_sweep(dataset, agg_name, grid, args, X, Y, epochs) -> tuple[float, float]:
    """Paper D.1 (pragmatic version): pick lr minimizing area under the loss curve."""
    best_lr, best_auc = None, float("inf")
    n_finite = 0
    for lr in grid:
        _, step = fresh(dataset, agg_name, lr, args, "autojac")
        curve = run_training(step, X, Y, epochs, args.batch_size, args.seed,
                             device_from_arg(args.device))
        auc = sum(curve)
        finite = all(map(math.isfinite, curve))
        n_finite += int(finite)
        marker = ""
        if finite and auc < best_auc:
            best_lr, best_auc, marker = lr, auc, "  <- best so far"
        else:
            marker = "  (diverged)" if not finite else ""
        print(f"    lr={lr:<8g} AUC={auc:10.2f}{marker}")

    if best_lr is None:
        raise RuntimeError(
            f"[{agg_name}] every lr in the grid diverged ({grid}); extend the grid downward.")
    if n_finite == 1:
        print(f"    warning [{agg_name}]: only one finite lr ({best_lr}); selection is "
              "not a real optimum, extend the grid downward.")
    elif best_lr in (grid[0], grid[-1]):
        print(f"    warning [{agg_name}]: selected lr={best_lr} is a grid endpoint; the "
              "optimum may lie outside the tested range.")
    return best_lr, best_auc


def cmd_figure2(args):
    device = device_from_arg(args.device)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    X, Y = load_data(args.dataset, args.data_root, args.subset, args.seed, device)
    epochs = args.epochs or PAPER_EPOCHS.get(args.dataset, 20)
    grid = [float(g) for g in args.lr_grid.split(",")]

    results = {}
    for agg in args.aggs:
        if args.sweep:
            print(f"[{agg}] lr sweep (criterion: area under loss curve, paper D.1):")
            lr, _ = lr_sweep(args.dataset, agg, grid, args, X, Y, epochs)
        else:
            lr = args.lr
        print(f"[{agg}] final run @ lr={lr}")
        _, step = fresh(args.dataset, agg, lr, args, "autojac")
        curve = run_training(step, X, Y, epochs, args.batch_size, args.seed, device)
        results[agg] = {"lr": lr, "curve": curve}

    (out / f"figure2_{args.dataset}.json").write_text(json.dumps(results))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 4.5))
    for agg, r in results.items():
        plt.plot(ema(r["curve"], args.smooth), label=f"{agg} (lr={r['lr']:g})", linewidth=1.8)
    plt.yscale("log"); plt.xlabel("Iteration"); plt.ylabel("Mean per-instance CE (EMA)")
    plt.title(f"IWRM on {args.dataset} - 1024 imgs, batch 32, paper CNN (Fig. 2 repro)")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    fig_path = out / f"figure2_{args.dataset}.png"
    plt.savefig(fig_path, dpi=160)
    print(f"Saved {fig_path}")


def cmd_table7(args):
    device = device_from_arg(args.device)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    X, Y = load_data(args.dataset, args.data_root, args.subset, args.seed, device)
    paper = PAPER_TABLE7.get(args.dataset, {})

    rows = []
    for name, engine in [("SGD-ERM", "sgd-erm")] + [(a, "autojac") for a in args.aggs]:
        _, step = fresh(args.dataset, name if engine != "sgd-erm" else "Mean", args.lr, args, engine)
        mean_s, std_s, _ = timed_epochs(step, X, Y, args.warmup, args.timed,
                                        args.batch_size, args.seed, device)
        rows.append((name, mean_s, std_s))
        print(f"[{name:8s}] {mean_s:7.3f} +/- {std_s:.3f} s/epoch")

    ours_mean = {n: m for n, m, _ in rows}["Mean"]
    paper_mean = paper.get("Mean")
    lines = ["| Method | yours s/epoch | yours ratio (Mean=1) | paper ratio (Mean=1) |",
             "|---|---|---|---|"]
    for n, m, s in rows:
        pr = f"{paper[n] / paper_mean:.2f}" if paper_mean and n in paper else "-"
        lines.append(f"| {n} | {m:.3f} +/- {s:.3f} | {m / ours_mean:.2f} | {pr} |")
    table = "\n".join(lines)
    print("\n" + table)
    (out / f"table7_{args.dataset}.md").write_text(table + "\n")
    print(f"\nSaved {out / f'table7_{args.dataset}.md'}")
    print("Compare RATIOS only (different GPU / TorchJD version than the paper).")


def cmd_engines(args):
    device = device_from_arg(args.device)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    X, Y = load_data(args.dataset, args.data_root, args.subset, args.seed, device)

    configs = [("SGD-ERM (scalar)", "Mean", "sgd-erm"),
               ("autojac + UPGrad", "UPGrad", "autojac"),
               ("autojac + UPGrad\n(optimize_gramian)", "UPGrad", "autojac-ogc"),
               ("autogram + UPGradWeighting", "UPGrad", "autogram")]
    rows = []
    for label, agg, engine in configs:
        _, step = fresh(args.dataset, agg, args.lr, args, engine)
        mean_s, std_s, peak = timed_epochs(step, X, Y, args.warmup, args.timed,
                                           args.batch_size, args.seed, device)
        rows.append((label, mean_s, std_s, peak))
        print(f"[{label:28s}] {mean_s:7.3f} +/- {std_s:.3f} s/epoch | peak {peak:9.1f} MiB")

    (out / f"engines_{args.dataset}.json").write_text(json.dumps(
        [{"label": l, "s_per_epoch": m, "std": s, "peak_mib": p} for l, m, s, p in rows]))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [r[0] for r in rows]
    colors = ["#999", "#c44", "#e58", "#2a7"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].bar(labels, [r[1] for r in rows], yerr=[r[2] for r in rows], color=colors)
    axes[0].set_ylabel("s / epoch"); axes[0].set_title("Time per epoch")
    axes[1].bar(labels, [r[3] for r in rows], color=colors)
    axes[1].set_ylabel("peak MiB"); axes[1].set_title("Peak GPU memory")
    for ax in axes:
        ax.tick_params(axis="x", rotation=12, labelsize=8); ax.grid(alpha=0.3, axis="y")
    by_label = {r[0]: r for r in rows}
    speedup = by_label["autojac + UPGrad"][1] / by_label["autogram + UPGradWeighting"][1]
    fig.suptitle(f"{args.dataset}: autogram is {speedup:.2f}x faster than autojac (same UPGrad weights)")
    fig.tight_layout()
    fig_path = out / f"engines_{args.dataset}.png"
    fig.savefig(fig_path, dpi=160)
    print(f"Saved {fig_path}")


def cmd_scaling(args):
    device = device_from_arg(args.device)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    X, Y = load_data(args.dataset, args.data_root, args.subset, args.seed, device)

    results = {"autojac": [], "autogram": []}
    for w in args.widths:
        args.width = w
        n_params = sum(p.numel() for p in make_model(args.dataset, w).parameters())
        for engine in ("autojac", "autogram"):
            try:
                _, step = fresh(args.dataset, "UPGrad", args.lr, args, engine)
                mean_s, _, peak = timed_epochs(step, X, Y, args.warmup, args.timed,
                                               args.batch_size, args.seed, device)
                results[engine].append({"width": w, "params": n_params,
                                        "s_per_epoch": mean_s, "peak_mib": peak})
                print(f"[w={w} | {n_params/1e6:.2f}M params | {engine:8s}] "
                      f"{mean_s:.3f} s/epoch, peak {peak:.0f} MiB")
            except torch.cuda.OutOfMemoryError:
                print(f"[w={w} | {engine}] OOM (recorded)")
                results[engine].append({"width": w, "params": n_params, "oom": True})
                torch.cuda.empty_cache()
    args.width = 1

    (out / f"scaling_{args.dataset}.json").write_text(json.dumps(results))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 4.5))
    for engine, color in [("autojac", "#c44"), ("autogram", "#2a7")]:
        pts = [r for r in results[engine] if "oom" not in r]
        plt.plot([r["params"] / 1e6 for r in pts], [r["peak_mib"] for r in pts],
                 "o-", label=engine, color=color)
    plt.xlabel("parameters (millions)"); plt.ylabel("peak memory (MiB)")
    plt.title(f"{args.dataset}: peak memory vs model size (UPGrad, batch 32)")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(out / f"scaling_{args.dataset}.png", dpi=160)
    print(f"Saved {out / f'scaling_{args.dataset}.png'}")


# ----------------------------------------------------------------------------- decompose
SEGMENTS = ["forward", "gramian_pass", "weighting_qp", "backward_step"]


def cmd_decompose(args):
    """Split one autogram JD step into four timed segments, across an (m, width)
    grid. m is varied via batch size (IWRM: m = batch size).

    Segments (autogram path -- per-objective backward work and Gramian
    accumulation are fused into one pass by design; that fused pass is
    'gramian_pass'):
      forward        model(x) + per-instance CE losses
      gramian_pass   engine.compute_gramian(losses)   [vjp work + G accumulation]
      weighting_qp   UPGradWeighting(G)               [m QPs, CPU, incl. transfer]
      backward_step  losses.backward(w) + SGD step
    """
    device = device_from_arg(args.device)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    X, Y = load_data(args.dataset, args.data_root, args.subset, args.seed, device)

    results = []
    for w in args.widths:
        model = make_model(args.dataset, w)
        model.load_state_dict(init_state(args.dataset, args.seed, out, w))
        model.to(device)
        n_params = sum(p.numel() for p in model.parameters())
        for m in args.batch_sizes:
            model.load_state_dict(init_state(args.dataset, args.seed, out, w))
            opt = torch.optim.SGD(model.parameters(), lr=args.lr)
            engine = Engine(model, batch_dim=0)
            weighting = UPGradWeighting()

            def seg_step(x, y, acc=None):
                if acc is None:
                    losses = loss_fn(model(x), y)
                    g = engine.compute_gramian(losses)
                    losses.backward(weighting(g))
                    opt.step(); opt.zero_grad()
                    return
                sync(device); t0 = time.perf_counter()
                losses = loss_fn(model(x), y)
                sync(device); t1 = time.perf_counter()
                g = engine.compute_gramian(losses)
                sync(device); t2 = time.perf_counter()
                wts = weighting(g)
                sync(device); t3 = time.perf_counter()
                losses.backward(wts)
                opt.step(); opt.zero_grad()
                sync(device); t4 = time.perf_counter()
                for k, dt in zip(SEGMENTS, (t1 - t0, t2 - t1, t3 - t2, t4 - t3)):
                    acc[k] += dt

            # warmup (untimed), then timed epochs with per-segment accumulation
            for ep in range(args.warmup):
                for idx in batch_order(len(X), m, ep, args.seed):
                    seg_step(X[idx], Y[idx])
            acc = {k: 0.0 for k in SEGMENTS}
            for ep in range(args.warmup, args.warmup + args.timed):
                for idx in batch_order(len(X), m, ep, args.seed):
                    seg_step(X[idx], Y[idx], acc)
            per_epoch = {k: v / args.timed for k, v in acc.items()}
            total = sum(per_epoch.values())
            qp_share = per_epoch["weighting_qp"] / total
            results.append({"width": w, "params": n_params, "m": m,
                            "seconds_per_epoch": per_epoch, "qp_share": qp_share})
            segs = " | ".join(f"{k} {v:.3f}s" for k, v in per_epoch.items())
            print(f"[w={w} ({n_params/1e6:.2f}M) m={m:3d}] {segs} "
                  f"|| QP share = {qp_share:5.1%}")

    (out / f"decompose_{args.dataset}.json").write_text(json.dumps(results, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [f"w={r['width']}\nm={r['m']}" for r in results]
    colors = {"forward": "#bbb", "gramian_pass": "#2a7",
              "weighting_qp": "#c44", "backward_step": "#47c"}
    plt.figure(figsize=(max(7, 1.1 * len(results)), 4.6))
    bottom = [0.0] * len(results)
    for seg in SEGMENTS:
        vals = [r["seconds_per_epoch"][seg] for r in results]
        plt.bar(labels, vals, bottom=bottom, label=seg, color=colors[seg])
        bottom = [b + v for b, v in zip(bottom, vals)]
    plt.ylabel("s / epoch (batch=m; #steps varies with m)")
    plt.title(f"{args.dataset}: autogram step time decomposition")
    plt.legend(); plt.grid(alpha=0.3, axis="y"); plt.tight_layout()
    plt.savefig(out / f"decompose_{args.dataset}.png", dpi=160)
    print(f"Saved {out / f'decompose_{args.dataset}.png'}")
    print("NOTE: epochs at small m contain more steps (1024/m), so compare the "
          "SHARES within a bar and how they shift with m and width, not bar heights.")
    
def cmd_profile(args):
    device = device_from_arg(args.device)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    X, Y = load_data(args.dataset, args.data_root, args.subset, args.seed, device)
 
    if max(args.widths) > 4:
        print(f"** WARNING: --widths includes {max(args.widths)} (>4). Profiler overhead "
              f"stacks on top of your machine's known width=16 DPC_WATCHDOG crash. Watch "
              f"`nvidia-smi -l 1` in another terminal, or lower --widths. **")
 
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    seg_names = set(SEGMENTS)
 
    summary_rows = []
    for w in args.widths:
        args.width = w
        model = make_model(args.dataset, w)
        model.load_state_dict(init_state(args.dataset, args.seed, out, w))
        model.to(device)
        opt = torch.optim.SGD(model.parameters(), lr=args.lr)
        engine = Engine(model, batch_dim=0)
        weighting = UPGradWeighting()
        n_params = sum(p.numel() for p in model.parameters())
 
        idx_pool = batch_order(len(X), args.batch_size, 0, args.seed)
        x, y = X[idx_pool[0]], Y[idx_pool[0]]
 
        def profiled_step():
            with record_function("forward"):
                losses = loss_fn(model(x), y)
            with record_function("gramian_pass"):
                g = engine.compute_gramian(losses)
            with record_function("weighting_qp"):
                wts = weighting(g)
            with record_function("backward_step"):
                losses.backward(wts)
                opt.step(); opt.zero_grad()
 
        for _ in range(args.warmup):                     # untimed, real -- burns in
            profiled_step()                               # cuDNN autotune / lazy CUDA init
        sync(device)                                      # (a) clean boundary before timing
 
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
 
        sync(device); t_wall0 = time.perf_counter()        # (c) synced wall-clock bracket
        with profile(activities=activities, record_shapes=True,
                     profile_memory=True, with_stack=False) as prof:
            for _ in range(args.active_steps):
                profiled_step()
        sync(device); t_wall1 = time.perf_counter()
 
        wall_ms = 1000 * (t_wall1 - t_wall0) / args.active_steps
        peak_mib = (torch.cuda.max_memory_allocated(device) / 2**20
                   if device.type == "cuda" else float("nan"))
 
        time_sort = "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
        mem_sort = "self_cuda_memory_usage" if device.type == "cuda" else "self_cpu_memory_usage"
 
        # ---- Table A: operator time (self-time sorted, CPU+GPU) ----
        table_time = prof.key_averages().table(sort_by=time_sort, row_limit=20)
        # ---- Table B: operator memory (which ops allocate the most) ----
        table_mem = prof.key_averages().table(sort_by=mem_sort, row_limit=15)
        # ---- Table C: grouped by input tensor shape -- ties directly to the
        # source-confirmed finding that JacobianComputer materializes one
        # [m, P_layer] tensor per module; this shows WHICH shape dominates ----
        table_shape = prof.key_averages(group_by_input_shape=True).table(
            sort_by=time_sort, row_limit=15)
        # ---- Table D: segment breakdown, from our record_function labels ----
        seg_rows = {e.key: e for e in prof.key_averages() if e.key in seg_names}
        seg_summary = {
            k: {"cpu_us": e.self_cpu_time_total / args.active_steps,
               "cuda_us": (e.self_cuda_time_total / args.active_steps
                          if device.type == "cuda" else float("nan"))}
            for k, e in seg_rows.items()
        }
 
        (out / f"profile_w{w}_time.txt").write_text(table_time)
        (out / f"profile_w{w}_memory.txt").write_text(table_mem)
        (out / f"profile_w{w}_shapes.txt").write_text(table_shape)
        trace_path = out / f"profile_w{w}_trace.json"
        prof.export_chrome_trace(str(trace_path))
 
        print(f"\n{'='*70}\nwidth={w} ({n_params/1e6:.2f}M params) m={args.batch_size} "
              f"device={device}\n{'='*70}")
        print(f"wall time/step: {wall_ms:.2f} ms   peak memory: {peak_mib:.1f} MiB")
        print("\n-- segment breakdown (per step, from record_function labels) --")
        for seg in SEGMENTS:
            r = seg_summary.get(seg, {"cpu_us": float("nan"), "cuda_us": float("nan")})
            print(f"  {seg:15s} cpu={r['cpu_us']:9.1f}us  cuda={r['cuda_us']:9.1f}us")
        print("\n-- Table A: top operators by self time --")
        print(table_time)
        print(f"Trace saved: {trace_path}  (open in https://ui.perfetto.dev)")
 
        summary_rows.append({"width": w, "params": n_params, "wall_ms": wall_ms,
                             "peak_mib": peak_mib, "segments": seg_summary})
 
    args.width = 1
    (out / f"profile_{args.dataset}_summary.json").write_text(json.dumps(summary_rows, indent=2))
 
    # ---- Table E: width-sweep summary ----
    print(f"\n{'='*70}\nTable E -- profile sweep summary across widths\n{'='*70}")
    print(f"{'width':>6} | {'params(M)':>10} | {'wall ms/step':>13} | {'peak MiB':>9} | "
          f"{'gramian %':>10} | {'qp %':>7}")
    for r in summary_rows:
        seg = r["segments"]
        total = sum(v["cpu_us"] for v in seg.values() if math.isfinite(v["cpu_us"]))
        gram_pct = 100 * seg.get("gramian_pass", {}).get("cpu_us", 0) / total if total else float("nan")
        qp_pct = 100 * seg.get("weighting_qp", {}).get("cpu_us", 0) / total if total else float("nan")
        print(f"{r['width']:>6} | {r['params']/1e6:>10.2f} | {r['wall_ms']:>13.2f} | "
              f"{r['peak_mib']:>9.1f} | {gram_pct:>9.1f}% | {qp_pct:>6.1f}%")
    print(f"\nSaved {out / f'profile_{args.dataset}_summary.json'}")




# ----------------------------------------------------------------------------- main
def main():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dataset", default="cifar10", choices=["cifar10", "svhn", "synthetic"])
    common.add_argument("--data-root", default="./data")
    common.add_argument("--out", default="./results")
    common.add_argument("--device", default="auto")
    common.add_argument("--seed", type=int, default=1)     # paper D.2: seeds are 1..8
    common.add_argument("--subset", type=int, default=1024)
    common.add_argument("--batch-size", type=int, default=32)
    common.add_argument("--lr", type=float, default=0.03)
    common.add_argument("--aggs", nargs="+", default=["Mean", "UPGrad", "PCGrad", "MGDA"],
                        choices=list(AGGREGATORS))
 
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
 
    sp = sub.add_parser("preflight", parents=[common])
    sp.add_argument("--k-steps", type=int, default=3)
 
    sf = sub.add_parser("figure2", parents=[common])
    sf.add_argument("--epochs", type=int, default=None, help="default: paper horizon")
    sf.add_argument("--sweep", action="store_true", help="per-aggregator lr sweep (paper D.1)")
    sf.add_argument("--lr-grid", default="0.003,0.01,0.03,0.1,0.3")
    sf.add_argument("--smooth", type=float, default=0.05, help="EMA alpha for the plot")
 
    for name in ("table7", "engines"):
        st = sub.add_parser(name, parents=[common])
        st.add_argument("--warmup", type=int, default=3)
        st.add_argument("--timed", type=int, default=10)
 
    ss = sub.add_parser("scaling", parents=[common])
    ss.add_argument("--widths", nargs="+", type=int, default=[1, 2, 4, 8])
    ss.add_argument("--warmup", type=int, default=1)
    ss.add_argument("--timed", type=int, default=3)
 
    sd = sub.add_parser("decompose", parents=[common])
    sd.add_argument("--widths", nargs="+", type=int, default=[1, 4])
    sd.add_argument("--batch-sizes", nargs="+", type=int, default=[4, 8, 16, 32, 64])
    sd.add_argument("--warmup", type=int, default=1)
    sd.add_argument("--timed", type=int, default=3)
 
    # LAPTOP-SAFE DEFAULTS: widths [1,2], not [1,4] or [1,2,4,8] like the
    # commands above -- profiler tracing overhead adds real thermal load on
    # top of the compute itself. Raise --widths deliberately, not by habit.
    spf = sub.add_parser("profile", parents=[common])
    spf.add_argument("--widths", nargs="+", type=int, default=[1, 2])
    spf.add_argument("--warmup", type=int, default=3)
    spf.add_argument("--active-steps", type=int, default=5)
 
    args = p.parse_args()
    set_seed(args.seed)
    {"preflight": cmd_preflight, "figure2": cmd_figure2, "table7": cmd_table7,
     "engines": cmd_engines, "scaling": cmd_scaling, "decompose": cmd_decompose,
     "profile": cmd_profile}[args.cmd](args)
 
 
if __name__ == "__main__":
    main()