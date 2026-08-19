"""Turn a v10 campaign into the acceptance answer, on the laptop.

Rui's bar: with two or three objectives, time and memory should sit near 1.5x a
single-objective run, not 2-3x. This reads the run directories a campaign left
behind and reports the ratio against that bar, plus the decomposition that says
whether the gap is reachable by tuning or not.

    python bench/acceptance.py results/v10_*        # table to stdout
    python bench/acceptance.py results/v10_* --csv acceptance.csv --figures fig/

Three things it is careful about, because each one has already produced a wrong
number in this project:

1. THE BASELINE. The only true single-objective control in the harness is L11's
   ``sgd_erm`` arm -- one forward, one backward, one optimiser step, same data
   and seed as every other cell in its run. Where a run has it, it is used.
   Where it does not, the L4 phase decomposition
   (``final_backward + optimizer_step``) is used as a proxy and the row is
   flagged ``proxy``. At 124M the proxy tracked the measured ratio to under 1%,
   but a flagged number and an unflagged one should never be mixed silently.

2. THE LADDER. ``m`` is the batch dimension in this harness -- per_sequence_losses
   returns one scalar per row -- so a ladder over ``m`` with independent windows
   grows the data as well as the objective count. Ladder A (objective_mode =
   duplicate) holds the data fixed and is the one that answers the question that
   was actually asked. Ladders are kept separate and never averaged together.

3. THE FLOOR. jdgram runs two forwards per step: one inside compute_gramian, one
   for the weighted backward. Substituting a zero-cost Gramian gives the floor
   that no amount of kernel work can go below. If the floor is already near the
   budget, the remaining headroom is the whole story and it belongs in the table.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

TARGET = 1.5  # Rui's stated budget, time and memory alike


# ----------------------------------------------------------------- loading
def load_run(d: Path) -> tuple[list[dict], dict]:
    rows_p, man_p = d / "rows.csv", d / "manifest.json"
    rows = []
    if rows_p.exists():
        with open(rows_p, newline="") as f:
            rows = list(csv.DictReader(f))
    man = {}
    if man_p.exists():
        try:
            man = json.load(open(man_p))
        except (OSError, json.JSONDecodeError):
            pass
    return rows, man


def fnum(r, key="value"):
    try:
        v = float(r.get(key, "nan"))
    except (TypeError, ValueError):
        return math.nan
    return v


def is_true(s):
    return str(s).strip().lower() in ("true", "1", "yes")


# ----------------------------------------------------- L11 measured ratios
def l11_cells(rows):
    """(engine, aggregator, objective_mode, m, T) -> {metric: value} for L11."""
    out = defaultdict(dict)
    for r in rows:
        if r.get("level") != "L11":
            continue
        if r.get("test") != "train":
            continue  # skip the per-step curve rows
        key = (r.get("engine", ""), r.get("driver", ""),
               r.get("objective_mode", ""), r.get("m", ""), r.get("T", ""))
        out[key][r.get("metric", "")] = fnum(r)
    return out


def acceptance_from_l11(rows, run_name, run_route="auto", oom_out=None,
                        run_levels=""):
    """Measured ratios against the real sgd_erm control.

    `run_route` is the run directory's --force-route, read from manifest.json
    by the caller. rows.csv's own per-row "route" column is not trustworthy at
    L11 -- it reads the literal string "default" on every L11 row regardless
    of what was actually forced, because that column is populated by the L4
    phase-timing code path and L11's training loop never sets it. The one
    thing that IS true for every row in a single invocation is that the whole
    process ran under one forced route, so route has to come from the
    manifest, not from this table. Getting this wrong previously merged
    tfirst and dfirst measurements into one "cell" and produced a false 170%
    replicate-disagreement alarm at m=16 (tfirst ~6.07x and dfirst ~2.26x,
    each internally tight on its own, read as if they disagreed).
    """
    cells = l11_cells(rows)
    baselines = {}
    for (eng, agg, mode, m, T), met in cells.items():
        if eng == "sgd_erm":
            baselines[(mode, m, T)] = met
    results = []
    for (eng, agg, mode, m, T), met in sorted(cells.items()):
        if eng == "sgd_erm":
            continue
        base = baselines.get((mode, m, T))
        if not base:
            continue
        bt, bp = base.get("ms_per_step"), base.get("peak_mib")
        ct, cp = met.get("ms_per_step"), met.get("peak_mib")
        # An OOM'd cell writes NaN, and `nan` is TRUTHY in Python -- so the
        # obvious `if not (bt and ct ...)` guard waves it straight through and
        # a NaN ratio lands in the table. It then poisons anything that does
        # not itself filter NaN: the per-engine cell counts, `all(time_ok)`
        # (NaN <= 1.5 is False, so a dead cell reads as a budget failure), and
        # plotted series, where the last point of a line is an invisible NaN.
        # Record these as what they are -- the engine could not run this shape.
        if ct is None or bt is None or math.isnan(ct) or math.isnan(bt) or bt <= 0:
            if oom_out is not None and (ct is None or math.isnan(ct)):
                # Whether the sgd_erm CONTROL also died at this shape is the
                # whole difference between "this engine cannot handle m
                # objectives" and "this batch does not fit on this card at
                # all". At m=24/T=512 on a 24 GiB A5000 the plain
                # single-objective baseline OOMs too -- quoting that as a
                # jdgram capability limit would be simply false.
                oom_out.append({"engine": eng, "aggregator": agg,
                                "objective_mode": mode or "independent",
                                "m": int(m or -1), "T": int(T or -1),
                                "route": run_route, "run": run_name,
                                "baseline_oom": bt is None or math.isnan(bt)})
            continue
        mem_ratio = (cp / bp) if (bp and bp > 0 and cp is not None) else math.nan
        results.append({
            "run": run_name, "source": "L11 measured", "engine": eng,
            "aggregator": agg, "objective_mode": mode or "independent",
            "route": run_route, "levels": run_levels,
            "m": int(m or -1), "T": int(T or -1),
            "ms_per_step": ct, "baseline_ms": bt, "time_ratio": ct / bt,
            "peak_mib": cp, "baseline_peak_mib": bp,
            "mem_ratio": mem_ratio,
            "val_ce": met.get("val_ce", math.nan),
            "val_perplexity": met.get("val_perplexity", math.nan),
            "min_offdiag_cos": met.get("gramian_min_offdiag_cos", math.nan),
            "time_ok": (ct / bt) <= TARGET,
            "mem_ok": bool(not math.isnan(mem_ratio) and mem_ratio <= TARGET),
        })
    return results


# -------------------------------------------------- L0 isolated kernels
def l0_kernels(rows, into, run_dir=None):
    """shape -> {tfirst, dfirst, router} from the L0 identity-kernel sweep.

    L0 runs the identity kernels with no model, no hooks and no autograd, so
    it times the route choice itself with nothing else in the frame. Note the
    route lives in the ``driver`` column at this level (the ``route`` column
    is unset), and a third pseudo-driver ``measured`` carries the harness's
    own faster/leaner verdict flags rather than a timing.

    ``router`` -- which route the live router actually chose -- is read from
    the run's ``violations.json`` rather than recomputed here. The rule is
    ``tfirst if m*T^2 < P_layer`` today, but a copy of it in this file would
    be free to drift out of sync with :mod:`jdgram.engine.router` and would
    then quietly mislabel exactly the shapes this analysis exists to find.
    The run directory already records what the real router decided; use that.
    """
    for r in rows:
        if r.get("level") != "L0" or r.get("metric") != "ms":
            continue
        route = r.get("driver", "")
        if route not in ("tfirst", "dfirst"):
            continue
        into.setdefault(r.get("test", ""), {})[route] = fnum(r)
    vp = (run_dir / "violations.json") if run_dir else None
    if vp and vp.exists():
        try:
            checks = json.load(open(vp))
        except (OSError, json.JSONDecodeError):
            checks = []
        for c in checks if isinstance(checks, list) else []:
            name = str(c.get("check", ""))
            if not name.startswith("L0_router_picks_faster@"):
                continue
            shape = name.split("@", 1)[1]
            for tok in str(c.get("detail", "")).split():
                if tok.startswith("router="):
                    into.setdefault(shape, {})["router"] = tok.split("=", 1)[1]
    return into


# ------------------------------------------------- L4 phase decomposition
def l4_phases(rows):
    """(route, m, T) -> {phase: {ms, peak_mib}} from the L4 breakdown."""
    out = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        if r.get("level") != "L4":
            continue
        test = r.get("test", "")
        if not test.startswith("phases/"):
            continue
        phase = test.split("/", 1)[1]
        key = (r.get("route", ""), r.get("m", ""), r.get("T", ""))
        out[key][phase][r.get("metric", "")] = fnum(r)
    return out


def floor_from_l4(rows, run_name):
    """The ratio a *free* Gramian would still cost.

    A JD step is compute_gramian + weighting_qp + final_backward + optimizer_step.
    compute_gramian contains its own forward, so replacing it with forward_only
    models an engine whose capture, reverse and identity arithmetic are all free
    but which still pays for the second forward. That is the floor; the budget is
    1.5x. What is left between them is the entire optimisation budget.
    """
    out = []
    for (route, m, T), ph in sorted(l4_phases(rows).items()):
        def ms(p):
            return ph.get(p, {}).get("ms", math.nan)

        cg, fo = ms("compute_gramian"), ms("forward_only")
        qp, fb, op = ms("weighting_qp"), ms("final_backward"), ms("optimizer_step")
        if any(math.isnan(x) for x in (cg, fo, qp, fb, op)):
            continue
        base = fb + op
        if base <= 0:
            continue
        full = cg + qp + fb + op
        floor = fo + qp + fb + op
        # What fusing the two forwards would buy, from data already on disk.
        #
        # final_backward is "fresh forward + weighted backward", and forward_only
        # measures that forward on its own, so the weighted backward alone is
        # their difference. A fused engine -- one forward, graph retained, both
        # backward traversals over it -- pays compute_gramian (which contains the
        # single surviving forward) + qp + weighted-backward + step.
        #
        # The consequence is sharper than "fusion helps". Substituting a free
        # Gramian into the FUSED step leaves qp + final_backward + optimizer_step,
        # i.e. the baseline plus a QP that measures well under 1% -- so the fused
        # floor is ~1.0x. The entire 1.36-1.41x structural floor IS the redundant
        # forward. Remove it and the whole overhead budget is available to the
        # identity machinery instead of 18% of it.
        weighted_backward = fb - fo
        fused = cg + qp + weighted_backward + op
        fused_floor = qp + fb + op          # free Gramian, fused
        machinery = cg - fo
        allowed_after_fusion = TARGET * base - fused_floor
        out.append({
            "run": run_name, "route": route, "m": int(m or -1), "T": int(T or -1),
            "jd_step_ms": full, "baseline_ms": base,
            "time_ratio": full / base,
            "floor_ratio": floor / base,
            "second_forward_ms": fo,
            "gramian_machinery_ms": machinery,
            "qp_ms": qp,
            # Carried so the caller can detect the pre-fix artifact: this phase
            # used to time SGD over None gradients and read ~0.04 ms regardless
            # of model size. See the schema check further down.
            "optimizer_step_ms": op,
            "qp_pct_of_step": 100.0 * qp / full,
            "headroom_to_target": TARGET - (floor / base),
            "gramian_speedup_needed": (
                machinery / (TARGET * base - floor)
                if (TARGET * base - floor) > 0 else math.inf),
            # --- the fusion projection ---
            "fused_ratio": fused / base,
            "fused_floor_ratio": fused_floor / base,
            "fused_speedup_needed": (machinery / allowed_after_fusion
                                     if allowed_after_fusion > 0 else math.inf),
            "fusion_reaches_target": (fused / base) <= TARGET,
        })
    return out


# --------------------------------------------------------------- rendering
def table(rows, cols, title):
    if not rows:
        return f"\n## {title}\n  (no rows)\n"
    w = {c: max(len(c), *(len(fmt(r.get(c))) for r in rows)) for c in cols}
    out = [f"\n## {title}", "  " + "  ".join(c.ljust(w[c]) for c in cols),
           "  " + "  ".join("-" * w[c] for c in cols)]
    for r in rows:
        out.append("  " + "  ".join(fmt(r.get(c)).ljust(w[c]) for c in cols))
    return "\n".join(out) + "\n"


def median_of(xs):
    xs = sorted(x for x in xs if isinstance(x, (int, float))
                and not math.isnan(x) and not math.isinf(x))
    if not xs:
        return math.nan
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def fmt(v):
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "NO"
    if isinstance(v, float):
        if math.isnan(v):
            return "-"
        if math.isinf(v):
            return "inf"
        return f"{v:.3f}" if abs(v) < 1000 else f"{v:.0f}"
    return str(v)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--csv", type=Path, default=None,
                    help="write the joined long-format table here")
    ap.add_argument("--figures", type=Path, default=None,
                    help="write PNG figures into this directory (needs matplotlib)")
    ap.add_argument("--engine", default="jdgram",
                    help="engine to headline (default jdgram)")
    ap.add_argument("--aggregator", default="UPGrad")
    args = ap.parse_args()

    dirs = []
    for p in args.runs:
        dirs.extend(sorted(x for x in ([p] if (p / "rows.csv").exists()
                                       else p.glob("*")) if (x / "rows.csv").exists()))
    if not dirs:
        print("no run directories with rows.csv found")
        return 1

    acc, floors, env, l0, ooms = [], [], {}, {}, []
    for d in dirs:
        rows, man = load_run(d)
        if man and not env:
            env = man.get("env", {})
        run_route = (man.get("config") or {}).get("force_route") or "auto"
        acc.extend(acceptance_from_l11(
            rows, d.name, run_route, ooms,
            ",".join((man.get("config") or {}).get("levels") or [])))
        floors.extend(floor_from_l4(rows, d.name))
        l0_kernels(rows, l0, d)
    args._l0 = l0

    print("=" * 78)
    print(f"ACCEPTANCE REPORT   target = {TARGET}x on both time and memory")
    if env:
        print(f"  {env.get('gpu','?')} | torch {env.get('torch','?')} | "
              f"torchjd {env.get('torchjd','?')} | host {env.get('host','?')}")
    print(f"  {len(dirs)} run dir(s), {len(acc)} measured cells, "
          f"{len(floors)} phase decompositions")
    print("=" * 78)

    head = [r for r in acc
            if r["engine"] == args.engine and r["aggregator"] == args.aggregator]

    for mode in sorted({r["objective_mode"] for r in head}):
        sub = sorted([r for r in head if r["objective_mode"] == mode],
                     key=lambda r: (r["m"], r["T"], r["route"]))
        note = {
            "duplicate": "objectives vary, DATA HELD FIXED -- answers the question asked",
            "independent": "objectives AND data both grow -- the v9 framing",
            "scaled": "positively collinear objectives",
            "conflicting": "anti-collinear; QP actually projects. Read cost, not loss.",
        }.get(mode, "")
        print(table(sub,
                    ["m", "T", "route", "ms_per_step", "baseline_ms", "time_ratio",
                     "time_ok", "peak_mib", "mem_ratio", "mem_ok",
                     "min_offdiag_cos"],
                    f"{args.engine}+{args.aggregator}, objective_mode={mode}"
                    + (f"   [{note}]" if note else "")))

    # Replicate disagreement on the HEADLINE table itself, not just the L4 floor.
    # The floor-side detector below was added first and it genuinely works for
    # what it watches -- but it watches L4 phase decompositions, and the outlier
    # that actually corrupted a night of acceptance numbers (m=4 r3 reading
    # 3.265x against two replicates at 2.465/2.466x) was an L11 ms_per_step
    # anomaly that landed in exactly this table and produced zero warnings,
    # because nothing was watching this table. A Phase-B sweep started mid-write
    # into that run directory; L4 runs first in each invocation so it finished
    # clean, and only the later L11 phase was hit -- which is precisely why a
    # floor-only check cannot catch an L11-only contamination event.
    if head:
        from collections import defaultdict as _dd2
        by_head_cfg = _dd2(list)
        for r in head:
            by_head_cfg[(r["objective_mode"], r["m"], r["T"], r["route"])].append(r)
        head_noisy = []
        for cfg, rs in sorted(by_head_cfg.items()):
            if len(rs) < 2:
                continue
            vals = [r["time_ratio"] for r in rs]
            spread = (max(vals) - min(vals)) / max(min(vals), 1e-9)
            if spread > 0.10:
                head_noisy.append((cfg, len(rs), min(vals), max(vals), spread))
        if head_noisy:
            print(f"\n  !! HEADLINE REPLICATE DISAGREEMENT -- {len(head_noisy)} "
                  f"config(s) in the {args.engine}+{args.aggregator} table above "
                  f"disagree by >10% on time_ratio across replicates (route held "
                  f"fixed within each group -- this is not a tfirst/dfirst split):")
            for (mode, m, T, route), n, lo, hi, sp in head_noisy:
                print(f"     objective_mode={mode} m={m} T={T} route={route}: "
                      f"{n} runs, time_ratio {lo:.3f}-{hi:.3f} ({100*sp:.0f}% spread)")
            print(f"     At least one of these replicates is contended -- likely a "
                  f"concurrent job landing on the same card mid-run. Drop the "
                  f"outlier(s) before quoting a mean or median for this cell; do "
                  f"not average a clean reading with a contended one.")

    if floors:
        fl = sorted(floors, key=lambda r: (r["route"], r["m"], r["T"]))
        print(table(fl,
                    ["route", "m", "T", "time_ratio", "floor_ratio",
                     "headroom_to_target", "second_forward_ms",
                     "gramian_machinery_ms", "qp_ms", "qp_pct_of_step",
                     "gramian_speedup_needed"],
                    "Floor decomposition (L4). floor_ratio = cost with a FREE Gramian"))
        # Headline off the MEDIAN, not the max.
        #
        # Taking max() here made the report's strongest claim -- "the floor is at
        # or above budget, no kernel work can reach the target" -- rest entirely
        # on whichever single cell was noisiest, and on this campaign that was
        # literally the run whose L4-vs-L11 gap is corrupted by host
        # interference (m=3 run A reads 1.579; its own repeat reads 1.389, a 14%
        # swing on a quantity that should barely move). A conclusion that flips on
        # the least trustworthy cell in the set is not a conclusion.
        fr = sorted(r["floor_ratio"] for r in floors)
        med = median_of(fr)
        worst = max(floors, key=lambda r: r["floor_ratio"])
        best_f = min(floors, key=lambda r: r["floor_ratio"])
        print(f"  Floor ratio: median {med:.3f}x, range {best_f['floor_ratio']:.3f}"
              f"-{worst['floor_ratio']:.3f}x across {len(floors)} cell(s). "
              f"Max at m={worst['m']} T={worst['T']} route={worst['route']}.")
        if med >= TARGET:
            print(f"  >> The MEDIAN floor is at or above the {TARGET}x budget. No "
                  f"amount of identity or kernel optimisation reaches the target "
                  f"while the engine runs two forwards per step.")
        else:
            pct = 100 * (med - 1) / (TARGET - 1)
            print(f"  >> The two-forward floor consumes {pct:.0f}% of the entire "
                  f"allowed overhead before any identity kernel runs.")
            if worst["floor_ratio"] >= TARGET:
                print(f"  >> NOTE: {sum(1 for r in floors if r['floor_ratio'] >= TARGET)}"
                      f" individual cell(s) do exceed {TARGET}x, but the median does "
                      f"not. Check those cells for contention before quoting them.")

        # Replicate disagreement is the contention detector. Two runs of an
        # identical (route, m, T) should agree closely; peak_mib does agree to the
        # byte across the m=3 repeats while the timings move 28-53%, which is the
        # signature of host interference rather than anything in the engine.
        from collections import defaultdict as _dd
        by_cfg = _dd(list)
        for r in floors:
            by_cfg[(r["route"], r["m"], r["T"])].append(r)
        noisy = []
        for cfg, rs in sorted(by_cfg.items()):
            if len(rs) < 2:
                continue
            vals = [r["time_ratio"] for r in rs]
            spread = (max(vals) - min(vals)) / max(min(vals), 1e-9)
            if spread > 0.10:
                noisy.append((cfg, len(rs), min(vals), max(vals), spread))
        if noisy:
            print(f"\n  !! REPLICATE DISAGREEMENT -- {len(noisy)} config(s) measured "
                  f"more than once disagree by >10% on time_ratio:")
            for (route, m, T), n, lo, hi, sp in noisy:
                print(f"     route={route} m={m} T={T}: {n} runs, time_ratio "
                      f"{lo:.3f}-{hi:.3f} ({100*sp:.0f}% spread)")
            print(f"     Identical configs should not disagree by this much. Treat "
                  f"absolute ms from these cells as contended; ratios within a "
                  f"single run still cancel shared contention.")

        # The optimizer_step artifact splits the data into two incomparable eras.
        # Before the level4_phases fix, that phase timed SGD over gradients that
        # final_backward had just set to None -- an empty walk over the parameter
        # list, ~0.04 ms instead of ~2.8 ms at 124M. `base` is
        # final_backward + optimizer_step, so a pre-fix run understates its own
        # baseline and overstates every ratio derived from it. Mixing the two eras
        # in one invocation silently puts incomparable rows side by side.
        stale = [r for r in floors if r.get("optimizer_step_ms", float("nan")) < 0.5]
        if stale and len(stale) != len(floors):
            print(f"\n  !! MIXED optimizer_step SCHEMA: {len(stale)} of {len(floors)} "
                  f"cell(s) carry the pre-fix ~0.04 ms artifact and the rest do not. "
                  f"Their floor/time ratios are NOT comparable to each other. Re-run "
                  f"the affected configs, or analyse the two eras separately.")
        elif stale:
            print(f"\n  NOTE: all {len(stale)} cell(s) predate the optimizer_step fix "
                  f"(that phase timed SGD on None gradients, reading ~0.04 ms instead "
                  f"of ~2.8 ms at 124M). Baselines are understated, so these ratios "
                  f"are upper bounds -- roughly 2% high at m=1.")

        print(table(fl, ["route", "m", "T", "time_ratio", "fused_ratio",
                         "fused_floor_ratio", "gramian_speedup_needed",
                         "fused_speedup_needed", "fusion_reaches_target"],
                    "Fusion projection -- one forward, graph retained, both "
                    "backwards over it"))
        best = min(floors, key=lambda r: r["fused_floor_ratio"])
        print(f"  Fused floor collapses to {best['fused_floor_ratio']:.3f}x "
              f"(from {best['floor_ratio']:.3f}x). The structural floor IS the "
              f"redundant forward -- remove it and essentially the whole overhead "
              f"budget goes to the identity machinery instead of ~18% of it.")
        need_now = median_of(r["gramian_speedup_needed"] for r in floors)
        need_fused = median_of(r["fused_speedup_needed"] for r in floors)
        print(f"  Identity speedup needed to reach {TARGET}x: "
              f"{need_now:.1f}x as built, {need_fused:.1f}x after fusion "
              f"(median across cells).")
        if any(r["fusion_reaches_target"] for r in floors):
            n = sum(1 for r in floors if r["fusion_reaches_target"])
            print(f"  >> {n} cell(s) reach {TARGET}x on fusion ALONE, with no "
                  f"kernel work at all.")

    # cross-engine memory, the comparison that currently favours us
    mem = [r for r in acc if r["aggregator"] == args.aggregator]
    if mem:
        by_eng = defaultdict(list)
        for r in mem:
            by_eng[r["engine"]].append(r)
        def median(xs):
            xs = sorted(x for x in xs if not math.isnan(x))
            if not xs:
                return math.nan
            n = len(xs)
            return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2

        # Every boolean column in this report reads "yes" = within budget. The
        # first cut had `any_mem_over_target`, whose False rendered as "NO" in
        # the same column style as a time_ok failure -- identical glyph, opposite
        # meaning, in a table meant to be read aloud in a meeting.
        summary = [{
            "engine": e,
            "cells": len(v),
            "median_time_ratio": median(x["time_ratio"] for x in v),
            "median_mem_ratio": median(x["mem_ratio"] for x in v),
            "time_within_budget": all(x["time_ok"] for x in v),
            "mem_within_budget": all(x["mem_ok"] for x in v),
        } for e, v in sorted(by_eng.items())]
        print(table(summary,
                    ["engine", "cells", "median_time_ratio", "median_mem_ratio",
                     "time_within_budget", "mem_within_budget"],
                    f"Engine comparison at aggregator={args.aggregator}"
                    "   [yes = inside the 1.5x budget]"))

    # Where each engine stops being able to run at all. This is a capability
    # boundary, not a speed result, and it belongs next to the speed table:
    # an engine that is 2x faster at m=8 and cannot run m=12 has not won.
    if ooms:
        # Shapes where even the control died are a property of the card, not
        # of any engine; they are excluded from the engine boundary and
        # reported separately.
        device_limit = sorted({(r["T"], r["m"]) for r in ooms if r["baseline_oom"]})
        engine_ooms = [r for r in ooms if not r["baseline_oom"]]
        reach = defaultdict(lambda: {"ran": set(), "died": set()})
        for r in acc:
            reach[(r["engine"], r["T"])]["ran"].add(r["m"])
        for r in engine_ooms:
            reach[(r["engine"], r["T"])]["died"].add(r["m"])
        lines = []
        for (e, T), v in sorted(reach.items()):
            lines.append({
                "engine": e, "T": T,
                "largest_m_that_ran": max(v["ran"]) if v["ran"] else 0,
                "smallest_m_that_OOMed": min(v["died"]) if v["died"] else None,
                "OOM_cells": len([x for x in engine_ooms
                                  if x["engine"] == e and x["T"] == T])})
        if any(x["smallest_m_that_OOMed"] for x in lines):
            print(table(lines, ["engine", "T", "largest_m_that_ran",
                                "smallest_m_that_OOMed", "OOM_cells"],
                        "Capability boundary -- where an engine runs out of "
                        "memory while the sgd_erm control still fits"))
            clean_e = sorted({x["engine"] for x in lines
                              if not x["smallest_m_that_OOMed"]}
                             - {x["engine"] for x in lines
                                if x["smallest_m_that_OOMed"]})
            if clean_e:
                print(f"  Ran every shape the control could run: "
                      f"{', '.join(clean_e)}.")
        if device_limit:
            shapes = ", ".join(f"m={m}/T={T}" for T, m in device_limit)
            print(f"\n  Device limit (NOT an engine result): at {shapes} the "
                  f"sgd_erm control OOMs too, so nothing runs there. That is "
                  f"the card's batch ceiling, not a Gramian-engine limit, and "
                  f"it must not be quoted as one.")

    if args.csv:
        allrows = ([dict(r, kind="acceptance") for r in acc]
                   + [dict(r, kind="floor") for r in floors])
        keys = sorted({k for r in allrows for k in r})
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(allrows)
        print(f"\nwrote {len(allrows)} rows -> {args.csv}")

    if args.figures:
        try:
            make_figures(head, floors, acc, args)
        except ImportError:
            print("\n(matplotlib not available; skipped figures)")
    return 0


# --------------------------------------------------------------- figures
#
# Colours are the reference data-viz palette's light-mode categorical slots,
# used in their documented order. That order is the colour-blind-safety
# mechanism, not decoration: slots 1-3 clear the all-pairs gate and 1-4 clear
# the adjacent-pair gate, so no chart here puts more than four series on one
# axis and the scatter-like forms stay at three.
SURFACE = "#fcfcfb"
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
CRITICAL = "#d03b3b"          # reserved status colour -- the budget line only
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
MLIST = [1, 2, 3, 4, 6, 8, 12, 16]


def clean_cells(acc, engine, agg, mode, T, route=None):
    """m -> (time_ratio, mem_ratio, n_reps, spread) with contention removed.

    Timing and memory need different estimators, and mixing them up is how a
    contended run gets quoted as a result:

    * peak_mib is deterministic -- across all 312 replicate groups in the v11
      campaign not one varied by even 0.1%. Any replicate's memory is THE
      memory.
    * ms_per_step is not. Contention only ever ADDS time, and it lands on the
      jdgram cell without necessarily landing on the sgd_erm cell measured in
      the same process, so it inflates the ratio asymmetrically. The minimum
      over replicates is therefore the least-contended estimate, and the
      spread is reported alongside so a noisy cell is visible rather than
      silently averaged in.
    """
    by_m = defaultdict(list)
    for r in acc:
        if (r["engine"] == engine and r["aggregator"] == agg
                and r["objective_mode"] == mode and r["T"] == T
                and (route is None or r["route"] == route)):
            by_m[r["m"]].append(r)
    out = {}
    for m, v in by_m.items():
        times = [x["time_ratio"] for x in v]
        mems = [x["mem_ratio"] for x in v
                if x["mem_ratio"] is not None and not math.isnan(x["mem_ratio"])]
        lo, hi = min(times), max(times)
        out[m] = (lo, min(mems) if mems else math.nan, len(v),
                  (hi - lo) / max(lo, 1e-9))
    return out


def _style(ax, xlabel=None, ylabel=None, title=None, logx=True):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    if logx:
        ax.set_xscale("log", base=2)
        ax.set_xticks(MLIST)
        ax.set_xticklabels([str(m) for m in MLIST])
        ax.minorticks_off()
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=10)
    if title:
        ax.set_title(title, color=INK, fontsize=11, loc="left", pad=10)


def _budget(ax, label_x=0.015, ha="left"):
    ax.axhline(TARGET, ls=(0, (5, 4)), c=CRITICAL, lw=1.6, zorder=1)
    ax.annotate(f"{TARGET}x budget", (label_x, TARGET),
                xycoords=("axes fraction", "data"), va="bottom", ha=ha,
                color=CRITICAL, fontsize=9, fontweight="bold")


def make_figures(head, floors, acc, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir = args.figures
    outdir.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["figure.facecolor"] = SURFACE
    eng, agg = args.engine, args.aggregator
    made = []

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(outdir / name, dpi=200, facecolor=SURFACE)
        plt.close(fig)
        made.append(name)

    # ---------------------------------------------------------------- FIG 1
    # The acceptance ladder. ONE objective_mode, ONE T, one line per ROUTE.
    #
    # The first version of this plot drew every replicate of every route and
    # every T against x=m and joined them with a line, so a single "curve"
    # zig-zagged between configurations that are not comparable -- at m=16 it
    # connected tfirst's 6.09x to dfirst's 2.26x as though they were two
    # readings of one quantity. Conditioning on route and T is what makes the
    # line mean one thing.
    tf = clean_cells(acc, eng, agg, "duplicate", 512, "tfirst")
    df = clean_cells(acc, eng, agg, "duplicate", 512, "dfirst")
    if tf and df:
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
        for ax, idx, lab in (
                (axes[0], 0, "step time / single-objective step"),
                (axes[1], 1, "peak memory / single-objective step")):
            for data, colour, name in ((tf, S2, "tfirst"), (df, S1, "dfirst")):
                ms = sorted(data)
                ax.plot(ms, [data[m][idx] for m in ms], "-o", color=colour,
                        lw=2, ms=7, mec=SURFACE, mew=1.5, label=name, zorder=3)
            _budget(ax)
            _style(ax, "objectives (m)", lab)
            ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="best")
        axes[0].set_title("Time: dfirst wins from m=4 up", color=INK,
                          fontsize=11, loc="left", pad=10)
        axes[1].set_title("Memory: both routes stay inside budget at T=512",
                          color=INK, fontsize=11, loc="left", pad=10)
        fig.suptitle(
            f"Ladder A -- {eng}+{agg}, duplicate objectives, T=512 "
            f"(data held fixed, objective count varies)",
            color=INK, fontsize=12.5, x=0.008, ha="left", y=0.99)
        save(fig, "fig1_acceptance_ladder.png")

    # ---------------------------------------------------------------- FIG 2
    # Route crossover, and the band where the router picks the loser.
    if tf and df:
        fig, ax = plt.subplots(figsize=(8.2, 4.8))
        ms = sorted(set(tf) & set(df))
        ax.axvspan(3.5, 9.0, color=CRITICAL, alpha=0.07, zorder=0)
        for data, colour, name in ((tf, S2, "tfirst"), (df, S1, "dfirst")):
            ax.plot(ms, [data[m][0] for m in ms], "-o", color=colour, lw=2,
                    ms=7, mec=SURFACE, mew=1.5, label=name, zorder=3)
        _budget(ax)
        _style(ax, "objectives (m)", "step time / single-objective step")
        ax.set_title("Route crossover sits at m~3.5; the router switches at m=9",
                     color=INK, fontsize=11, loc="left", pad=10)
        # The penalty quoted has to come from INSIDE the shaded band. Taking
        # the max over all m pulls in m=16, where the router already picks
        # dfirst correctly -- that number describes the routes' separation,
        # not the router's mistake, and putting it on this annotation would
        # overstate the finding by 1.6x.
        band = [m for m in ms if 3.5 < m < 9.0]
        if band:
            worst = max(band, key=lambda m: tf[m][0] / df[m][0])
            ax.annotate(
                f"router picks tfirst across this band,\nbut dfirst is up to "
                f"{tf[worst][0] / df[worst][0]:.2f}x faster (at m={worst})",
                xy=(5.6, 0.97), xycoords=("data", "axes fraction"),
                ha="center", va="top", color=CRITICAL, fontsize=9,
                fontweight="bold")
        ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper left")
        save(fig, "fig2_route_crossover.png")

    # ---------------------------------------------------------------- FIG 3
    # Three engines, each at its own best route -- the fair comparison.
    eng_data = {e: clean_cells(acc, e, agg, "duplicate", 512)
                for e in ("jdgram", "autogram", "autojac")}
    if any(eng_data.values()):
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
        for ax, idx, lab in (
                (axes[0], 0, "step time / single-objective step"),
                (axes[1], 1, "peak memory / single-objective step")):
            for (name, data), colour in zip(eng_data.items(), (S1, S2, S3)):
                if not data:
                    continue
                ms = sorted(data)
                ax.plot(ms, [data[m][idx] for m in ms], "-o", color=colour,
                        lw=2, ms=7, mec=SURFACE, mew=1.5, label=name, zorder=3)
                if name == "autojac" and ms:
                    ax.annotate("OOM >", (ms[-1], data[ms[-1]][idx]),
                                textcoords="offset points", xytext=(9, -3),
                                color=INK2, fontsize=9, fontweight="bold")
            _budget(ax)
            _style(ax, "objectives (m)", lab)
            ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="best")
        axes[0].set_title("Time: autogram edges ahead; autojac blows up",
                          color=INK, fontsize=11, loc="left", pad=10)
        axes[1].set_title("Memory: jdgram is the only engine inside budget",
                          color=INK, fontsize=11, loc="left", pad=10)
        fig.suptitle(
            f"Engine comparison at aggregator={agg}, duplicate objectives, "
            f"T=512, each engine at its best route",
            color=INK, fontsize=12.5, x=0.008, ha="left", y=0.99)
        save(fig, "fig3_engine_comparison.png")

    # ---------------------------------------------------------------- FIG 4
    # What a step is actually made of -- and how little of it is the QP.
    fl = [r for r in floors if r["T"] == 512 and r["route"] == "dfirst"]
    best_by_m = {}
    for r in fl:
        if r["m"] not in best_by_m or r["time_ratio"] < best_by_m[r["m"]]["time_ratio"]:
            best_by_m[r["m"]] = r
    if best_by_m:
        ms = sorted(best_by_m)
        x = range(len(ms))
        base = [1.0] * len(ms)
        second = [best_by_m[m]["second_forward_ms"] / best_by_m[m]["baseline_ms"] for m in ms]
        mach = [best_by_m[m]["gramian_machinery_ms"] / best_by_m[m]["baseline_ms"] for m in ms]
        qp = [best_by_m[m]["qp_ms"] / best_by_m[m]["baseline_ms"] for m in ms]
        fig, ax = plt.subplots(figsize=(9.5, 5.0))
        b1 = [0.0] * len(ms)
        b2 = base
        b3 = [a + b for a, b in zip(base, second)]
        b4 = [a + b for a, b in zip(b3, mach)]
        # The QP's share is folded into its own legend label rather than set
        # as a free-floating annotation: it is a sliver too thin to see at
        # this scale (that IS the finding), so the number has to travel with
        # the swatch or it reads as a missing series.
        qp_lab = (f"aggregator QP -- {100 * min(qp):.2f}-{100 * max(qp):.2f}% "
                  f"of the step, too thin to see")
        for vals, bot, colour, lab in (
                (base, b1, S1, "baseline: forward + backward + optimiser step"),
                (second, b2, S2, "redundant SECOND forward"),
                (mach, b3, S3, "capture + reverse + identity kernels"),
                (qp, b4, S4, qp_lab)):
            ax.bar(x, vals, bottom=bot, width=0.68, color=colour,
                   edgecolor=SURFACE, linewidth=1.5, label=lab, zorder=2)
        _budget(ax, label_x=0.995, ha="right")
        _style(ax, "objectives (m)", "cost / single-objective step", logx=False)
        ax.set_xlim(-0.65, len(ms) - 0.35)
        ax.set_xticks(list(x))
        ax.set_xticklabels([str(m) for m in ms])
        ax.set_title("Where a Jacobian-Descent step goes (dfirst, T=512)",
                     color=INK, fontsize=11, loc="left", pad=10)
        ax.legend(frameon=False, fontsize=9, labelcolor=INK2,
                  loc="upper center", ncol=2, columnspacing=1.4,
                  handlelength=1.6)
        ax.set_ylim(0, max(b4) * 1.52)
        save(fig, "fig4_step_composition.png")

    # ---------------------------------------------------------------- FIG 5
    # What fusing the two forwards would buy.
    if best_by_m:
        ms = sorted(best_by_m)
        import numpy as np
        x = np.arange(len(ms), dtype=float)
        w = 0.27
        built = [best_by_m[m]["time_ratio"] for m in ms]
        fused = [best_by_m[m]["fused_ratio"] for m in ms]
        floor = [best_by_m[m]["fused_floor_ratio"] for m in ms]
        fig, ax = plt.subplots(figsize=(9.5, 4.8))
        for off, vals, colour, lab in (
                (-w, built, S1, "as built (two forwards)"),
                (0.0, fused, S2, "fused (one forward, graph retained)"),
                (w, floor, S3, "fused + a free Gramian = the true floor")):
            ax.bar(x + off, vals, width=w * 0.92, color=colour,
                   edgecolor=SURFACE, linewidth=1.5, label=lab, zorder=2)
        _budget(ax, label_x=0.995, ha="right")
        _style(ax, "objectives (m)", "step time / single-objective step",
               logx=False)
        ax.set_xlim(-0.62, len(ms) - 0.38)
        ax.set_xticks(list(x))
        ax.set_xticklabels([str(m) for m in ms])
        ax.set_title(
            "Fusion projection: the floor collapses to ~1.0x, so the whole "
            "overhead budget\nbecomes available to the identity kernels",
            color=INK, fontsize=11, loc="left", pad=10)
        ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper right")
        save(fig, "fig5_fusion_projection.png")

    # ---------------------------------------------------------------- FIG 6
    # The router mismatch, measured on isolated kernels (L0).
    if getattr(args, "_l0", None):
        l0 = args._l0
        shapes = [s for s in l0 if "tfirst" in l0[s] and "dfirst" in l0[s]]
        shapes.sort(key=lambda s: l0[s]["tfirst"] / max(l0[s]["dfirst"], 1e-9))
        if shapes:
            import numpy as np
            y = np.arange(len(shapes), dtype=float)
            h = 0.36
            fig, ax = plt.subplots(figsize=(9.5, 0.52 * len(shapes) + 2.4))
            ax.barh(y + h / 2, [l0[s]["tfirst"] for s in shapes], height=h * 0.92,
                    color=S2, edgecolor=SURFACE, linewidth=1.5, label="tfirst",
                    zorder=2)
            ax.barh(y - h / 2, [l0[s]["dfirst"] for s in shapes], height=h * 0.92,
                    color=S1, edgecolor=SURFACE, linewidth=1.5, label="dfirst",
                    zorder=2)
            # Annotate ONLY the shapes where the router actually chose the
            # slower side. An earlier cut of this figure flagged every shape
            # whose tfirst was slower than its dfirst, which marked 7 of 9
            # shapes as router errors when the router had in fact picked
            # correctly on all but two -- it agrees with the measurement
            # everywhere except large-P and vocab-head-50k. The shapes it gets
            # right are the control group and have to read as such.
            n_bad = 0
            for i, s in enumerate(shapes):
                t, d = l0[s]["tfirst"], l0[s]["dfirst"]
                pick = l0[s].get("router")
                if not pick:
                    continue
                chosen, other = (t, d) if pick == "tfirst" else (d, t)
                if chosen > other * 1.05:
                    n_bad += 1
                    ax.annotate(
                        f"router picked {pick} ({chosen / other:.1f}x slower)",
                        xy=(max(t, d), i), textcoords="offset points",
                        xytext=(9, -3), color=CRITICAL, fontsize=8.5,
                        fontweight="bold", va="center")
                else:
                    ax.annotate(f"router picked {pick}", xy=(max(t, d), i),
                                textcoords="offset points", xytext=(9, -3),
                                color=MUTED, fontsize=8.5, va="center")
            ax.set_xscale("log")
            _style(ax, "kernel time (ms, log scale)", None, logx=False)
            ax.set_yticks(list(y))
            ax.set_yticklabels(shapes, fontsize=9)
            ax.set_xlim(right=max(l0[s]["tfirst"] for s in shapes) * 60)
            ax.set_title(
                f"Identity kernels in isolation (L0): the router picks the "
                f"slower route on {n_bad} of {len(shapes)} shapes",
                color=INK, fontsize=11, loc="left", pad=10)
            ax.legend(frameon=False, fontsize=9, labelcolor=INK2,
                      loc="lower right")
            save(fig, "fig6_router_kernels.png")

    print(f"\nfigures -> {outdir}")
    for n in made:
        print(f"  {n}")


if __name__ == "__main__":
    raise SystemExit(main())
