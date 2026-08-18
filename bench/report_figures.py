"""Presentation figures for the v11 campaign -- the ones you talk through.

`bench/acceptance.py --figures` writes the diagnostic plots that belong beside
its tables. This writes the *narrative* set: one figure per point you actually
want to make out loud, sized and labelled so it can be read from across a room
rather than squinted at.

    python bench/report_figures.py results/v11-pull/* --out fig/

Every figure answers exactly one question, and its title states the answer
rather than the topic -- a slide titled "Time ratio vs m" makes the reader do
the work; "No engine reaches the budget" does not.

Reduction rules match acceptance.py and matter for every number drawn here:
peak memory is deterministic (bit-identical across all 312 replicate groups in
this campaign) so any replicate is THE value; step time is not, because
contention only ever adds time and lands asymmetrically on the engine cell
versus the control cell measured in the same process -- so timing is always the
minimum over replicates, the least-contended observation.
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

from acceptance import (TARGET, acceptance_from_l11, clean_cells, floor_from_l4,
                        l0_kernels, load_run)

# Reference data-viz palette, light-mode categorical slots, documented order.
# That order is the colour-blind-safety mechanism, not decoration: slots 1-3
# clear the all-pairs gate, 1-4 the adjacent-pair gate. Nothing here puts more
# than four series on one axis, and the scatter-like forms stay at three.
SURFACE = "#fcfcfb"
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
CRITICAL, GOOD = "#d03b3b", "#0ca30c"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]
ENGINE_COLOUR = {"jdgram": S1, "autogram": S2, "autojac": S3}
MLIST = [1, 2, 3, 4, 6, 8, 12, 16]


def style(ax, xlabel=None, ylabel=None, title=None, logx=False):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=10, length=0)
    if logx:
        ax.set_xscale("log", base=2)
        ax.set_xticks(MLIST)
        ax.set_xticklabels([str(m) for m in MLIST])
        ax.minorticks_off()
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2, fontsize=11)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=11)
    if title:
        ax.set_title(title, color=INK, fontsize=12.5, loc="left", pad=12)


def budget(ax, horizontal=True, ha="right", x=0.995):
    # The budget line crosses bars in most of these charts, so the label gets
    # a surface-coloured backing plate. Without it the one number every reader
    # looks for lands on top of a saturated bar and becomes unreadable in
    # exactly the charts where it matters most.
    plate = dict(facecolor=SURFACE, edgecolor="none", pad=1.6, alpha=0.88)
    if horizontal:
        ax.axhline(TARGET, ls=(0, (5, 4)), c=CRITICAL, lw=1.8, zorder=1)
        ax.annotate(f"{TARGET}x budget", (x, TARGET),
                    xycoords=("axes fraction", "data"), va="bottom", ha=ha,
                    color=CRITICAL, fontsize=10, fontweight="bold", bbox=plate,
                    zorder=5)
    else:
        ax.axvline(TARGET, ls=(0, (5, 4)), c=CRITICAL, lw=1.8, zorder=1)
        ax.annotate(f"{TARGET}x budget", (TARGET, 1.005),
                    xycoords=("data", "axes fraction"), va="bottom",
                    ha="center", color=CRITICAL, fontsize=10,
                    fontweight="bold", bbox=plate, zorder=5)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("fig"))
    ap.add_argument("--aggregator", default="UPGrad")
    args = ap.parse_args()

    dirs = []
    for p in args.runs:
        dirs.extend(sorted(x for x in ([p] if (p / "rows.csv").exists()
                                       else p.glob("*")) if (x / "rows.csv").exists()))
    if not dirs:
        print("no run directories with rows.csv found")
        return 1

    acc, floors, l0, ooms = [], [], {}, []
    for d in dirs:
        rows, man = load_run(d)
        cfg = man.get("config") or {}
        route = cfg.get("force_route") or "auto"
        acc.extend(acceptance_from_l11(rows, d.name, route, ooms,
                                       ",".join(cfg.get("levels") or [])))
        floors.extend(floor_from_l4(rows, d.name))
        l0_kernels(rows, l0, d)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["figure.facecolor"] = SURFACE

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    made = []

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(out / name, dpi=200, facecolor=SURFACE)
        plt.close(fig)
        made.append(name)

    agg = args.aggregator
    eng_best = {e: clean_cells(acc, e, agg, "duplicate", 512)
                for e in ("jdgram", "autogram", "autojac")}
    jd_t = clean_cells(acc, "jdgram", agg, "duplicate", 512, "tfirst")
    jd_d = clean_cells(acc, "jdgram", agg, "duplicate", 512, "dfirst")

    # ================================================================ A. VERDICT
    # The single slide that answers the question that was asked.
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    engines = ["jdgram", "autogram", "autojac"]
    xs = np.arange(len(engines), dtype=float)
    w = 0.34
    for ax, idx, lab, ttl in (
            (axes[0], 0, "step time vs one objective",
             "Time: every engine is roughly DOUBLE the budget"),
            (axes[1], 1, "peak memory vs one objective",
             "Memory: jdgram is comfortably inside it")):
        for k, (mm, colour) in enumerate(((2, SEQ[3]), (3, SEQ[5]))):
            vals = [eng_best[e].get(mm, (math.nan,) * 4)[idx] for e in engines]
            bars = ax.bar(xs + (k - 0.5) * w, vals, width=w * 0.9, color=colour,
                          edgecolor=SURFACE, linewidth=1.5, zorder=2,
                          label=f"m = {mm} objectives")
            for b, v in zip(bars, vals):
                if not math.isnan(v):
                    ax.annotate(f"{v:.2f}x", (b.get_x() + b.get_width() / 2,
                                              v), ha="center", va="bottom",
                                fontsize=10, color=INK, fontweight="bold",
                                xytext=(0, 3), textcoords="offset points")
        budget(ax, ha="left" if idx else "right", x=0.02 if idx else 0.995)
        style(ax, None, lab, ttl)
        ax.set_xticks(xs)
        ax.set_xticklabels(engines, fontsize=11, color=INK2)
        ax.legend(frameon=False, fontsize=10, labelcolor=INK2, loc="upper left")
        ax.set_ylim(0, max(3.4, ax.get_ylim()[1] * 1.18))
    fig.suptitle('"With two or three objectives, is it 1.5x -- or 2-3x?"',
                 color=INK, fontsize=14, x=0.008, ha="left", y=0.995)
    save(fig, "figA_verdict.png")

    # ============================================== B. TRADE-OFF SPACE
    # Where each engine actually sits, and which way it moves as m grows.
    fig, ax = plt.subplots(figsize=(9.6, 6.6))
    ax.add_patch(plt.Rectangle((0, 0), TARGET, TARGET, color=GOOD, alpha=0.08,
                               zorder=0))
    ax.annotate("both budgets met\n(nothing is here yet)", (TARGET - 0.05, 0.35),
                ha="right", color=GOOD, fontsize=10, fontweight="bold",
                zorder=1)
    # Only the endpoints get labels. Labelling every rung turned this into an
    # unreadable knot: all three engines start within 0.3x of each other, so
    # the interesting part is the direction each one travels, not the rungs.
    ends = {"jdgram": (22, -30), "autogram": (10, 12), "autojac": (-10, 12)}
    for e in engines:
        d = eng_best[e]
        ms = [m for m in sorted(d) if not math.isnan(d[m][1])]
        if not ms:
            continue
        ax.plot([d[m][0] for m in ms], [d[m][1] for m in ms], "-o",
                color=ENGINE_COLOUR[e], lw=2, ms=7, mec=SURFACE, mew=1.5,
                label=e, zorder=3)
        ax.annotate(f"{e}\nm={ms[-1]}", (d[ms[-1]][0], d[ms[-1]][1]),
                    color=ENGINE_COLOUR[e], fontsize=10, fontweight="bold",
                    ha="right" if e == "autojac" else "left",
                    va="center", xytext=ends[e], textcoords="offset points")
        ax.plot([d[ms[0]][0]], [d[ms[0]][1]], "o", ms=13, mfc="none",
                mec=ENGINE_COLOUR[e], mew=1.6, zorder=2)
    # Both budget lines are drawn but only labelled once, by the shaded corner
    # they bound. A second "1.5x budget" tag on the vertical line has nowhere
    # to sit that does not collide with either the title or the data.
    ax.axvline(TARGET, ls=(0, (5, 4)), c=CRITICAL, lw=1.8, zorder=1)
    ax.axhline(TARGET, ls=(0, (5, 4)), c=CRITICAL, lw=1.8, zorder=1)
    ax.annotate("dashed lines = the 1.5x budget on each axis", (0.035, 0.80),
                xycoords="axes fraction", color=CRITICAL, fontsize=9.5,
                fontweight="bold")
    ax.annotate("open rings mark a single objective", (0.035, 0.755),
                xycoords="axes fraction", color=MUTED, fontsize=9.5)
    style(ax, "step time vs one objective", "peak memory vs one objective",
          "As objectives grow, jdgram travels along the floor -- autojac "
          "leaves the chart")
    ax.legend(frameon=False, fontsize=10, labelcolor=INK2, loc="upper left")
    ax.set_xlim(0.75, 6.9)
    ax.set_ylim(0, 4.55)
    save(fig, "figB_tradeoff_space.png")

    # ============================================ C. THE COST OF ONE STEP
    # A waterfall at m=2 -- the exact cell the budget names.
    f512 = [r for r in floors if r["T"] == 512 and r["route"] == "dfirst"]
    at = {}
    for r in f512:
        if r["m"] not in at or r["time_ratio"] < at[r["m"]]["time_ratio"]:
            at[r["m"]] = r
    if 2 in at:
        r = at[2]
        b = r["baseline_ms"]
        parts = [("one ordinary\ntraining step", 1.0, S1),
                 ("+ a SECOND\nforward pass", r["second_forward_ms"] / b, S2),
                 ("+ building the\nGramian", r["gramian_machinery_ms"] / b, S3),
                 ("+ the aggregator\nsolve", r["qp_ms"] / b, S4)]
        fig, ax = plt.subplots(figsize=(10.5, 5.8))
        run = 0.0
        for i, (lab, val, colour) in enumerate(parts):
            ax.bar(i, val, bottom=run, width=0.62, color=colour,
                   edgecolor=SURFACE, linewidth=1.5, zorder=2)
            # A segment worth 0.006x draws as nothing at all. Leaving a bare
            # number floating over an empty column reads as a broken chart, so
            # say why the column is empty -- the negligible size IS the point.
            txt = f"{val:.2f}x" if i == 0 else f"+{val:.2f}x"
            if val < 0.02:
                txt += "\n(too small to draw)"
            ax.annotate(txt, (i, run + val), ha="center", va="bottom",
                        fontsize=11, fontweight="bold", color=INK,
                        xytext=(0, 4), textcoords="offset points")
            run += val
            if i < len(parts) - 1:
                ax.plot([i + 0.31, i + 1 - 0.31], [run, run], color=MUTED,
                        lw=1, ls=":", zorder=1)
        ax.bar(len(parts), run, width=0.62, color=MUTED, alpha=0.35,
               edgecolor=SURFACE, linewidth=1.5, zorder=2)
        ax.annotate(f"{run:.2f}x", (len(parts), run), ha="center", va="bottom",
                    fontsize=12, fontweight="bold", color=INK,
                    xytext=(0, 4), textcoords="offset points")
        budget(ax)
        style(ax, None, "cost, relative to one ordinary step",
              "What two objectives actually cost, and where it goes")
        ax.set_xticks(range(len(parts) + 1))
        ax.set_xticklabels([p[0] for p in parts] + ["TOTAL"], fontsize=10,
                           color=INK2)
        ax.set_ylim(0, run * 1.3)
        save(fig, "figC_cost_of_a_step.png")

    # ======================================== D. THE PATH TO THE BUDGET
    # What would have to change, in order, to land inside 1.5x.
    if 4 in at:
        r = at[4]
        stages = [("today", r["time_ratio"], S2),
                  ("stop doing the\nsecond forward", r["fused_ratio"], S4),
                  (f"+ make the Gramian\n{r['fused_speedup_needed']:.1f}x faster",
                   TARGET, S3),
                  ("(a free Gramian\nwould reach)", r["fused_floor_ratio"], S1)]
        fig, ax = plt.subplots(figsize=(10.0, 5.8))
        for i, (lab, val, colour) in enumerate(stages):
            ax.bar(i, val, width=0.6, color=colour, edgecolor=SURFACE,
                   linewidth=1.5, zorder=2)
            ax.annotate(f"{val:.2f}x", (i, val), ha="center", va="bottom",
                        fontsize=12, fontweight="bold", color=INK,
                        xytext=(0, 4), textcoords="offset points")
        # autogram already runs one forward and keeps the graph -- it IS the
        # fused design, built by someone else. Marking where it actually lands
        # turns the middle bar from a projection into a checked prediction.
        ref = eng_best.get("autogram", {}).get(4, (math.nan,) * 4)[0]
        if not math.isnan(ref):
            ax.plot([0.6, 1.4], [ref, ref], color=INK, lw=1.8,
                    ls=(0, (4, 3)), zorder=4)
            ax.annotate(f"autogram already fuses, and measures {ref:.2f}x",
                        (1.45, ref), color=INK, fontsize=9.5,
                        fontweight="bold", va="center")
        budget(ax)
        style(ax, None, "step time vs one objective",
              "What it would take to get inside the budget (m=4)")
        ax.set_xticks(range(len(stages)))
        ax.set_xticklabels([s[0] for s in stages], fontsize=10, color=INK2)
        ax.set_ylim(0, max(s[1] for s in stages) * 1.28)
        save(fig, "figD_path_to_budget.png")

    # ================================== E. OVERHEAD AMORTIZES WITH OBJECTIVES
    g = defaultdict(list)
    for r in acc:
        if (r["engine"] == "jdgram" and r["aggregator"] == agg
                and r["objective_mode"] == "duplicate" and r["T"] == 512
                and r["route"] == "dfirst"):
            g[r["m"]].append(r)
    pts = []
    for m in sorted(g):
        best = min(g[m], key=lambda x: x["time_ratio"])
        pts.append((m, best["ms_per_step"], best["baseline_ms"],
                    best["time_ratio"]))
    if len(pts) > 2:
        # Absolute milliseconds, not ratios. An earlier version of this figure
        # plotted a finite-difference "marginal cost" line, which was both
        # noisy (it differences unevenly spaced m) and hard to explain out
        # loud. Both curves here are straight lines in m, so the whole cost
        # model is visible directly: a fixed overhead per step (the intercept)
        # plus a per-objective cost (the slope). The ratio of slopes is the
        # value the total ratio decays toward, which is what the ratio plot
        # was trying and failing to show.
        def fit(xs, ys):
            n = len(xs)
            sx, sy = sum(xs), sum(ys)
            sxx = sum(x * x for x in xs)
            sxy = sum(x * y for x, y in zip(xs, ys))
            slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
            return slope, (sy - slope * sx) / n

        use = [p for p in pts if p[0] >= 2]      # m=1 skips work; excluded
        xs = [p[0] for p in use]
        sj, ij = fit(xs, [p[1] for p in use])
        sb, ib = fit(xs, [p[2] for p in use])
        fig, ax = plt.subplots(figsize=(10.0, 5.9))
        line = np.linspace(0, max(xs) * 1.06, 50)
        for slope, inter, colour, name in (
                (sj, ij, S1, "Jacobian Descent (ours)"),
                (sb, ib, S2, "ordinary training")):
            ax.plot(line, slope * line + inter, "-", color=colour, lw=1.4,
                    alpha=0.45, zorder=2)
            ax.annotate(f"{slope:.0f} ms per objective\n+ {inter:.0f} ms fixed",
                        (max(xs), slope * max(xs) + inter), color=colour,
                        fontsize=10, fontweight="bold", va="center",
                        xytext=(10, 0), textcoords="offset points")
        ax.plot(xs, [p[1] for p in use], "o", color=S1, ms=8, mec=SURFACE,
                mew=1.5, label="Jacobian Descent (ours)", zorder=3)
        ax.plot(xs, [p[2] for p in use], "o", color=S2, ms=8, mec=SURFACE,
                mew=1.5, label="ordinary training (the reference)", zorder=3)
        style(ax, "objectives (m)", "milliseconds per training step",
              f"Both grow in a straight line: each extra objective costs us "
              f"{sj / sb:.2f}x what it costs ordinary training")
        ax.legend(frameon=False, fontsize=10, labelcolor=INK2, loc="upper left")
        ax.set_xlim(0, max(xs) * 1.42)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(int(x)) for x in xs])
        save(fig, "figE_cost_model.png")

    # ============================ F. THE RELATIONSHIP BETWEEN OBJECTIVES IS FREE
    modes = ["duplicate", "conflicting", "independent"]
    md = {m: clean_cells(acc, "jdgram", agg, m, 512, "dfirst") for m in modes}
    common = sorted(set.intersection(*(set(v) for v in md.values() if v)) - {1})
    if common:
        fig, ax = plt.subplots(figsize=(9.8, 5.4))
        xs = np.arange(len(common), dtype=float)
        w = 0.26
        names = {"duplicate": "identical objectives",
                 "conflicting": "directly opposed objectives",
                 "independent": "unrelated objectives"}
        for k, (mode, colour) in enumerate(zip(modes, (S1, S2, S3))):
            vals = [md[mode].get(m, (math.nan,) * 4)[0] for m in common]
            ax.bar(xs + (k - 1) * w, vals, width=w * 0.9, color=colour,
                   edgecolor=SURFACE, linewidth=1.5, zorder=2,
                   label=names[mode])
        budget(ax)
        style(ax, "objectives (m)", "step time vs one objective",
              "Conflicting objectives cost the same as identical ones")
        ax.set_xticks(xs)
        ax.set_xticklabels([str(m) for m in common])
        ax.legend(frameon=False, fontsize=10, labelcolor=INK2, loc="upper right")
        ax.set_ylim(0, 3.6)
        save(fig, "figF_objective_relationship.png")

    # ================================================ G. ROUTE CROSSOVER
    if jd_t and jd_d:
        fig, ax = plt.subplots(figsize=(9.8, 5.8))
        ms = sorted(set(jd_t) & set(jd_d))
        ax.axvspan(3.5, 9.0, color=CRITICAL, alpha=0.07, zorder=0)
        for data, colour, name in ((jd_t, S2, "strategy A (tfirst)"),
                                   (jd_d, S1, "strategy B (dfirst)")):
            ax.plot(ms, [data[m][0] for m in ms], "-o", color=colour, lw=2,
                    ms=7, mec=SURFACE, mew=1.5, label=name, zorder=3)
        band = [m for m in ms if 3.5 < m < 9.0]
        if band:
            wm = max(band, key=lambda m: jd_t[m][0] / jd_d[m][0])
            ax.annotate(
                f"the engine keeps choosing A across this band,\n"
                f"while B is up to {jd_t[wm][0] / jd_d[wm][0]:.2f}x faster",
                xy=(5.6, 0.97), xycoords=("data", "axes fraction"),
                ha="center", va="top", color=CRITICAL, fontsize=10,
                fontweight="bold")
        budget(ax)
        style(ax, "objectives (m)", "step time vs one objective",
              "The two strategies swap places at m~3.5 -- the engine switches "
              "at m=9", logx=True)
        ax.legend(frameon=False, fontsize=10, labelcolor=INK2, loc="upper left")
        save(fig, "figG_strategy_crossover.png")

    # ============================================ H. KERNEL-LEVEL ROUTE MISS
    shapes = [s for s in l0 if {"tfirst", "dfirst"} <= set(l0[s])]
    shapes.sort(key=lambda s: l0[s]["tfirst"] / max(l0[s]["dfirst"], 1e-9))
    if shapes:
        y = np.arange(len(shapes), dtype=float)
        h = 0.36
        fig, ax = plt.subplots(figsize=(10.5, 0.55 * len(shapes) + 2.6))
        ax.barh(y + h / 2, [l0[s]["tfirst"] for s in shapes], height=h * 0.9,
                color=S2, edgecolor=SURFACE, linewidth=1.5,
                label="strategy A (tfirst)", zorder=2)
        ax.barh(y - h / 2, [l0[s]["dfirst"] for s in shapes], height=h * 0.9,
                color=S1, edgecolor=SURFACE, linewidth=1.5,
                label="strategy B (dfirst)", zorder=2)
        bad = 0
        for i, s in enumerate(shapes):
            t, d = l0[s]["tfirst"], l0[s]["dfirst"]
            pick = l0[s].get("router")
            if not pick:
                continue
            chosen, other = (t, d) if pick == "tfirst" else (d, t)
            if chosen > other * 1.05:
                bad += 1
                # .1f, not .0f: large-P is 1.35x slower and rounded to
                # "chose the 1x slower one", which reads as "chose an
                # identical option" and invites the reader to discount a
                # real miss.
                ax.annotate(f"chose the {chosen / other:.2f}x slower one",
                            xy=(max(t, d), i), textcoords="offset points",
                            xytext=(9, -3), color=CRITICAL, fontsize=9.5,
                            fontweight="bold", va="center")
            else:
                ax.annotate("chose correctly", xy=(max(t, d), i),
                            textcoords="offset points", xytext=(9, -3),
                            color=MUTED, fontsize=9.5, va="center")
        ax.set_xscale("log")
        style(ax, "time for one layer's Gramian (ms, log scale)", None,
              f"Layer by layer, in isolation: the wrong strategy is chosen "
              f"{bad} times out of {len(shapes)}")
        ax.set_yticks(list(y))
        ax.set_yticklabels(shapes, fontsize=10)
        ax.set_xlim(right=max(l0[s]["tfirst"] for s in shapes) * 70)
        ax.legend(frameon=False, fontsize=10, labelcolor=INK2, loc="lower right")
        save(fig, "figH_layer_strategy.png")

    # ================================================ I. MEMORY & THE OOM WALL
    fig, ax = plt.subplots(figsize=(9.8, 5.8))
    for e in engines:
        d = eng_best[e]
        ms = [m for m in sorted(d) if not math.isnan(d[m][1])]
        if not ms:
            continue
        ax.plot(ms, [d[m][1] for m in ms], "-o", color=ENGINE_COLOUR[e], lw=2,
                ms=7, mec=SURFACE, mew=1.5, label=e, zorder=3)
    died = sorted({r["m"] for r in ooms
                   if r["engine"] == "autojac" and r["T"] == 512
                   and not r["baseline_oom"]})
    if died:
        ax.axvline(died[0], color=S3, ls=":", lw=1.6, zorder=1)
        ax.annotate(f"autojac runs out of\nmemory from m={died[0]}",
                    xy=(died[0], 0.55), xycoords=("data", "axes fraction"),
                    ha="left", va="center", color=S3, fontsize=10,
                    fontweight="bold", xytext=(8, 0),
                    textcoords="offset points")
    budget(ax, x=0.30)
    style(ax, "objectives (m)", "peak memory vs one objective",
          "Memory is where jdgram wins, and the gap widens with objectives",
          logx=True)
    ax.legend(frameon=False, fontsize=10, labelcolor=INK2, loc="upper left")
    save(fig, "figI_memory_wall.png")

    # ================================================ J. REPRODUCIBILITY
    # "Identical" has to mean identical. Two runs of the same (m, T, mode,
    # aggregator, engine) are NOT repeats of each other if they used different
    # contraction routes, or ran a different list of benchmark levels
    # beforehand (which consumes a different amount of the global RNG stream
    # before training starts). Grouping without those two fields produced a
    # spurious "our engine is non-deterministic, up to 1.11 nats" result that
    # was really a route-to-route comparison -- and one that no other engine
    # could ever fail, because the route flag only affects ours. With both
    # fields pinned, every engine agrees to ~1e-4 nats.
    rep = defaultdict(list)
    for r in acc:
        if math.isnan(r["val_ce"]):
            continue
        rep[(r["engine"], r["aggregator"], r["objective_mode"], r["m"],
             r["T"], r["route"], r.get("levels", ""))].append(r["val_ce"])
    worst = defaultdict(float)
    for (e, a, *_), v in rep.items():
        if len(v) > 1:
            worst[(e, a)] = max(worst[(e, a)], max(v) - min(v))

    # Second panel's data: the same configuration run down the two
    # mathematically-equivalent contraction routes. Any difference here is
    # pure floating-point ordering, so it measures how much each aggregator
    # AMPLIFIES numerical noise. Only our engine has a route to vary.
    byroute = defaultdict(dict)
    for r in acc:
        if math.isnan(r["val_ce"]) or r["engine"] != "jdgram":
            continue
        byroute[(r["aggregator"], r["objective_mode"], r["m"],
                 r["T"])][r["route"]] = r["val_ce"]
    amp = defaultdict(float)
    for (a, *_), d in byroute.items():
        if len(d) > 1:
            amp[a] = max(amp[a], max(d.values()) - min(d.values()))
    if worst:
        # LINEAR scale, deliberately. The first version used a log axis with
        # exact zeros clamped to 1e-6, which drew a full-height bar for every
        # value that was in fact identically zero -- it made perfect
        # reproducibility look like a small amount of noise, and exaggerated
        # the one real result by burying it among fake ones. On a linear axis
        # a zero is a zero: the single non-zero bar is the entire finding.
        aggs = sorted({a for _, a in worst})
        fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.4))
        xs = np.arange(len(aggs), dtype=float)
        w = 0.26
        ax = axes[0]
        for k, e in enumerate(engines):
            vals = [worst.get((e, a), 0.0) for a in aggs]
            ax.bar(xs + (k - 1) * w, vals, width=w * 0.9,
                   color=ENGINE_COLOUR[e], edgecolor=SURFACE, linewidth=1.5,
                   zorder=2, label=e)
        style(ax, None, "worst gap between identical reruns (nats)",
              "Repeating a run: all three engines are equally stable")
        ax.set_xticks(xs)
        ax.set_xticklabels(aggs, fontsize=11, color=INK2)
        ax.legend(frameon=False, fontsize=10, labelcolor=INK2, loc="upper left")
        ax.annotate("PCGrad shuffles internally without a fixed seed --\n"
                    "expected, documented, and true of every engine",
                    (0.03, 0.66), xycoords="axes fraction", ha="left",
                    va="top", color=MUTED, fontsize=9.5)
        nonp = max((v for (e, a), v in worst.items() if a != "PCGrad"),
                   default=0.0)
        ax.annotate(f"every other aggregator, every engine:\n"
                    f"reproduces to within {nonp:.0e} nats",
                    (0.03, 0.50), xycoords="axes fraction", ha="left",
                    va="top", color=INK2, fontsize=9.5, fontweight="bold")

        ax = axes[1]
        vals = [amp.get(a, 0.0) for a in aggs]
        ax.bar(xs, vals, width=0.5, color=S4, edgecolor=SURFACE, linewidth=1.5,
               zorder=2)
        for xi, v in zip(xs, vals):
            ax.annotate(f"{v:.2f}", (xi, v), ha="center", va="bottom",
                        fontsize=10, fontweight="bold", color=INK,
                        xytext=(0, 3), textcoords="offset points")
        style(ax, None, "worst gap between the two routes (nats)",
              "Same maths, different arithmetic order:\nMGDA and PCGrad "
              "amplify it, UPGrad and Mean do not")
        ax.set_xticks(xs)
        ax.set_xticklabels(aggs, fontsize=11, color=INK2)
        ax.set_ylim(0, max(max(vals), 0.01) * 1.25)
        fig.suptitle("Reproducibility: two different questions, two different "
                     "answers", color=INK, fontsize=13, x=0.006, ha="left",
                     y=0.995)
        save(fig, "figJ_reproducibility.png")

    # ======================== L. DO THESE OBJECTIVES EVER ACTUALLY CONFLICT?
    # The premise check nothing else in the campaign performs, drawn against
    # both synthetic extremes. An earlier version plotted only the real data
    # over an empty red half-plane labelled "nothing gets here", which showed
    # an absence and invited the obvious objection: maybe the measurement
    # simply cannot register conflict. It can -- the conflicting probe reads
    # exactly -1.000 -- so the two probes belong on the axis as calibration.
    align = defaultdict(dict)
    for r in acc:
        if (r["engine"] == "jdgram" and r["T"] == 512
                and not math.isnan(r["min_offdiag_cos"])):
            align[r["objective_mode"]][r["m"]] = r["min_offdiag_cos"]
    if align.get("independent"):
        real = align["independent"]
        ms = sorted(real)
        fig, ax = plt.subplots(figsize=(10.2, 6.0))
        # Band labels sit at heights chosen to miss the data, not at each
        # band's midpoint: the real series occupies 0.66-0.86, so a label
        # centred in the "agree" band lands on top of it.
        for lo, hi, y, colour, lab in (
                (0.30, 1.05, 0.44, S3, "objectives AGREE -- nothing to resolve"),
                (-0.30, 0.30, 0.0, S4, "unrelated"),
                (-1.05, -0.30, -0.62, CRITICAL,
                 "objectives CONFLICT -- the case Jacobian Descent exists for")):
            ax.axhspan(lo, hi, color=colour, alpha=0.09, zorder=0)
            ax.annotate(lab, (0.985, y),
                        xycoords=("axes fraction", "data"), ha="right",
                        va="center", color=colour, fontsize=10,
                        fontweight="bold", zorder=1)
        for mode, colour, style_, lab in (
                ("duplicate", MUTED, (0, (5, 4)),
                 "probe: identical objectives (must be +1)"),
                ("conflicting", MUTED, (0, (2, 3)),
                 "probe: opposed objectives (must be -1)")):
            d = align.get(mode)
            if d:
                xs2 = sorted(d)
                ax.plot(xs2, [d[m] for m in xs2], ls=style_, color=colour,
                        lw=2, marker="o", ms=6, mec=SURFACE, mew=1.2,
                        label=lab, zorder=3)
        ax.plot(ms, [real[m] for m in ms], "-o", color=S1, lw=2.6, ms=9,
                mec=SURFACE, mew=1.6, zorder=4,
                label="REAL objectives (different passages of the corpus)")
        for m in ms:
            ax.annotate(f"{real[m]:+.2f}", (m, real[m]), ha="center",
                        va="bottom", fontsize=9.5, color=INK,
                        fontweight="bold", xytext=(0, 9),
                        textcoords="offset points")
        ax.axhline(0, color=AXIS, lw=1.2, zorder=1)
        style(ax, "objectives (m)",
              "how opposed the two most-opposed objectives are\n"
              "(gradient cosine: +1 identical, 0 unrelated, -1 opposed)",
              "The measurement CAN see conflict -- these objectives simply "
              "never have any", logx=True)
        ax.set_ylim(-1.18, 1.18)
        # The band between -0.3 and +0.3 is empty at every m, so the legend
        # goes there rather than on top of the -1 probe line.
        ax.legend(frameon=False, fontsize=9.5, labelcolor=INK2,
                  loc="center left", ncol=1)
        save(fig, "figL_objective_alignment.png")

    # ================================================ K. CAMPAIGN COVERAGE
    cov = defaultdict(int)
    for r in acc:
        cov[(r["T"], r["m"])] += 1
    if cov:
        Ts = sorted({t for t, _ in cov}, reverse=True)
        Ms = sorted({m for _, m in cov})
        grid = np.array([[cov.get((t, m), 0) for m in Ms] for t in Ts],
                        dtype=float)
        fig, ax = plt.subplots(figsize=(13.6, 4.8))
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list("seq", ["#ffffff"] + SEQ)
        im = ax.imshow(grid, cmap=cmap, aspect="auto")
        for i in range(len(Ts)):
            for j in range(len(Ms)):
                v = int(grid[i, j])
                if v:
                    ax.text(j, i, str(v), ha="center", va="center",
                            fontsize=10,
                            color="#ffffff" if v > grid.max() * 0.55 else INK)
        ax.set_xticks(range(len(Ms)))
        ax.set_xticklabels([str(m) for m in Ms], fontsize=10)
        ax.set_yticks(range(len(Ts)))
        ax.set_yticklabels([f"T={t}" for t in Ts], fontsize=10)
        ax.tick_params(colors=MUTED, length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xlabel("objectives (m)", color=INK2, fontsize=11)
        # The grid's shape is not arbitrary and the earlier version left the
        # reader to guess: the filled band is the headline ladder (T fixed at
        # 512, objective count varying) and the sparse diagonal is the pinned
        # ladder, where m*T is held at 2048 so every cell processes the same
        # number of tokens. Saying so turns a mostly-empty chart into the
        # experiment design.
        ax.set_title(f"What was measured: {int(grid.sum())} timed cells "
                     f"(one cell = one engine x aggregator x strategy x repeat)",
                     color=INK, fontsize=12.5, loc="left", pad=12)
        if 512 in Ts:
            r512 = Ts.index(512)
            # Bottom-left is empty; putting this past the right edge (as a
            # first cut did) hides it behind the colour bar entirely.
            ax.annotate("the headline ladder: sequence length fixed at 512,\n"
                        "objective count varying -- sections 1 to 9",
                        (-0.35, len(Ts) - 1), ha="left", va="center",
                        color=INK2, fontsize=9.5, fontweight="bold")
        diag = [(Ts.index(t), Ms.index(mm)) for t in Ts for mm in Ms
                if (t, mm) in cov and t != 512 and t * mm == 2048]
        if diag:
            ax.plot([c for _, c in diag], [r for r, _ in diag], ls=(0, (4, 3)),
                    color=MUTED, lw=1.6, zorder=3)
            r, c = diag[0]
            ax.annotate("the pinned ladder: m x T held at 2048,\n"
                        "so every cell sees the same number of tokens",
                        (c + 0.4, r), ha="left", va="center", color=MUTED,
                        fontsize=9.5, xytext=(8, 0), textcoords="offset points")
        cb = fig.colorbar(im, ax=ax, pad=0.015)
        cb.set_label("measured cells", color=INK2, fontsize=10)
        cb.ax.tick_params(colors=MUTED, length=0)
        cb.outline.set_visible(False)
        save(fig, "figK_coverage.png")

    print(f"figures -> {out}")
    for n in made:
        print(f"  {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
