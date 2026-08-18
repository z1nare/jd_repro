"""Before/after figure for the router fix (v12).

Reads the L4 phase decomposition from v12 verification run directories when
they are available, and otherwise falls back to the measured snapshot below so
the figure can be drawn before the results have been pulled off the cluster.

    python bench/router_fix_figure.py --out fig/                    # snapshot
    python bench/router_fix_figure.py results/routerfix --out fig/  # real dirs

The snapshot is the A5000 run of 2026-08-11 (v12), transcribed from the live
log while the sweep was still going. It is labelled as such on the figure --
a number typed in by hand is not the same kind of evidence as one read off
disk, and the chart should not pretend otherwise.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

SURFACE = "#fcfcfb"
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
CRITICAL = "#d03b3b"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
TARGET = 1.5

# m -> phase ms, per route. compute_gramian / weighting_qp / final_backward /
# optimizer_step. A Jacobian-Descent step is their sum; the single-objective
# baseline is final_backward + optimizer_step.
SNAPSHOT = {
    2:  {"auto":   (87.730, 0.401, 68.113, 2.255),
         "tfirst": (87.988, 0.399, 68.386, 2.251),
         "dfirst": (123.099, 0.399, 68.449, 2.251)},
    4:  {"auto":   (228.643, 0.461, 131.841, 2.254),
         "tfirst": (239.955, 0.460, 132.723, 2.254),
         "dfirst": (196.603, 0.463, 133.010, 2.252)},
    8:  {"auto":   (612.421, 0.588, 256.559, 2.257),
         "tfirst": (751.024, 0.593, 257.785, 2.257),
         "dfirst": (344.863, 0.592, 259.177, 2.256)},
    16: {"auto":   (1062.250, 1.015, 499.961, 2.258),
         "tfirst": (2567.204, 1.021, 507.082, 2.255),
         "dfirst": (644.883, 1.013, 509.914, 2.259)},
}


def total_ms(phases):
    cg, qp, fb, opt = phases
    return cg + qp + fb + opt


def ratio(phases, denom=None):
    """Step cost relative to a single-objective step.

    ``denom`` forces a COMMON baseline across routes. Dividing each route by
    its own (final_backward + optimizer_step) looks natural and is wrong here:
    that phase is route-invariant work -- the routes agree numerically to
    5e-6, so the weighted backward is identical -- yet it drifts monotonically
    auto < tfirst < dfirst at every m (up to 2.0% at m=16) because the three
    are timed sequentially inside one invocation. Own-denominators therefore
    credit that drift to whichever route ran last, which is dfirst, the one
    being argued for. Every comparison here uses the auto row's baseline.
    """
    cg, qp, fb, opt = phases
    base = denom if denom is not None else (fb + opt)
    return (cg + qp + fb + opt) / base if base > 0 else math.nan


def from_dirs(paths):
    """Pull the same four phases out of real v12 run directories."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from acceptance import load_run, l4_phases
    out = {}
    for p in paths:
        for d in ([p] if (p / "rows.csv").exists() else sorted(p.glob("*"))):
            if not (d / "rows.csv").exists():
                continue
            rows, _ = load_run(d)
            for (route, m, T), ph in l4_phases(rows).items():
                if str(T) != "512":
                    continue
                try:
                    key = int(m)
                except (TypeError, ValueError):
                    continue

                def g(name):
                    return ph.get(name, {}).get("ms", math.nan)
                vals = (g("compute_gramian"), g("weighting_qp"),
                        g("final_backward"), g("optimizer_step"))
                if any(math.isnan(v) for v in vals):
                    continue
                slot = out.setdefault(key, {})
                # keep the fastest observation of each (m, route)
                if route not in slot or total_ms(vals) < total_ms(slot[route]):
                    slot[route] = vals
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="*", type=Path)
    ap.add_argument("--out", type=Path, default=Path("fig"))
    args = ap.parse_args()

    data, live = SNAPSHOT, False
    if args.runs:
        found = from_dirs(args.runs)
        if found:
            data, live = found, True
            print(f"using measured run directories: m={sorted(found)}")
    if not live:
        print("using the transcribed snapshot (no v12 run dirs given/found)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["figure.facecolor"] = SURFACE

    ms = sorted(data)
    xs = np.arange(len(ms), dtype=float)
    w = 0.26
    # One common baseline per m -- the auto row's fb+opt. See ratio()'s note.
    den = {m: data[m]["auto"][2] + data[m]["auto"][3] for m in ms}
    old = [ratio(data[m]["auto"], den[m]) for m in ms]
    tf = [ratio(data[m]["tfirst"], den[m]) for m in ms]
    df = [ratio(data[m]["dfirst"], den[m]) for m in ms]
    best = [min(a, b) for a, b in zip(tf, df)]
    # The saving quoted is the reduction in actual step MILLISECONDS, which is
    # what a user feels; quoting the change in ratio instead reads ~0.8 pp
    # more favourable at every m.
    save = [1 - min(total_ms(data[m]["tfirst"]), total_ms(data[m]["dfirst"]))
            / total_ms(data[m]["auto"]) for m in ms]

    fig, ax = plt.subplots(figsize=(10.5, 5.9))
    for off, vals, colour, lab in (
            (-w, old, S2, "the shipped rule (what we had)"),
            (0.0, best, S1, "always taking the faster route (the fix's target)"),
            (w, tf, MUTED, "always tfirst, for reference")):
        bars = ax.bar(xs + off, vals, width=w * 0.9, color=colour,
                      edgecolor=SURFACE, linewidth=1.5, zorder=2, label=lab)
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, v),
                        ha="center", va="bottom", fontsize=9,
                        fontweight="bold", color=INK,
                        xytext=(0, 3), textcoords="offset points")

    for i, sv in enumerate(save):
        if sv > 0.02:
            ax.annotate(f"-{100 * sv:.0f}%", (xs[i] - w / 2, max(old[i], best[i])),
                        ha="center", va="bottom", fontsize=11,
                        fontweight="bold", color=CRITICAL,
                        xytext=(0, 20), textcoords="offset points")

    ax.axhline(TARGET, ls=(0, (5, 4)), c=CRITICAL, lw=1.8, zorder=1)
    ax.annotate(f"{TARGET}x budget", (0.995, TARGET),
                xycoords=("axes fraction", "data"), va="bottom", ha="right",
                color=CRITICAL, fontsize=10, fontweight="bold",
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.6,
                          alpha=0.88), zorder=5)
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=10, length=0)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(m) for m in ms], fontsize=11)
    ax.set_xlabel("objectives (m)", color=INK2, fontsize=11)
    ax.set_ylabel("step time vs one objective\n(common baseline)",
                  color=INK2, fontsize=11)
    ax.set_title("What fixing the route choice is worth, per objective count",
                 color=INK, fontsize=12.5, loc="left", pad=12)
    ax.legend(frameon=False, fontsize=10, labelcolor=INK2, loc="upper left")
    ax.set_ylim(0, max(tf) * 1.16)
    args.out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    # Provenance goes below the axes, not inside them -- placed in the plot
    # area it lands behind the bars at exactly the objective counts that
    # matter most.
    fig.text(0.008, 0.012,
             ("GPT-2 124M, T=512, phase decomposition (L4), measured from v12 "
              "run directories."
              if live else
              "GPT-2 124M, T=512, phase decomposition (L4), transcribed from "
              "the live v12 log while the sweep was still running -- "
              "regenerate from run directories once pulled."),
             color=MUTED, fontsize=9, ha="left")
    p = args.out / "figM_router_fix.png"
    fig.savefig(p, dpi=200, facecolor=SURFACE)
    plt.close(fig)

    print(f"\n{'m':>4}{'shipped':>10}{'best route':>12}{'saving':>9}")
    for m, o, b in zip(ms, old, best):
        print(f"{m:>4}{o:>10.3f}{b:>12.3f}{100 * (1 - b / o):>8.1f}%")
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
