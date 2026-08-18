"""Read the per-objective loss curves and the held-out numbers, on the laptop.

Two things Rui asked for that nothing in the repo currently reads.

1. THE TWO CURVES. "the curve of two losses should be pretty straightforward to
   see, okay, if they go too far." L11 writes every objective's loss at every
   step to ``l11_per_objective_curves.json`` beside ``rows.csv``, and until now
   no tool opened that file, so the check was logged and never performed.

   The check is unusually sharp here because the harness constructs a KNOWN
   relationship between the objectives instead of hoping for one. Under
   ``objective_mode=duplicate`` the m objectives are row 0 repeated m times with
   unit coefficients: the m loss curves are not merely close, they are the same
   number at every step. Any spread at all is an engine or harness bug, not
   noise, and this report says so loudly rather than printing "0.000".
   ``scaled`` and ``conflicting`` fix the objectives up to a known coefficient
   vector, so the same equality check applies once that vector is divided out.

2. THE OTHER METRIC. "you only look at one thing, the perplexity." val_perplexity
   is logged as a row now, but a logged row nobody prints is not a reported
   number. The perplexity tables put it beside val_ce, next-token accuracy and
   the final train loss, against the sgd_erm control, and add the check those
   enable: for one aggregator at one shape every engine computes the SAME
   mathematical object, so they must land on the same held-out CE. A spread
   there is a correctness bug in whichever engine is the odd one out, and it is
   completely invisible from timing data.

    python bench/curves.py results/solo/v10_*
    python bench/curves.py results/solo/v10_* --csv curves.csv --json curves.json

Exit status: 0 clean, 2 a check failed, 1 no data to read.

Deliberately not here: plotting. The --json dump carries the per-step spread and
mean series so a notebook can draw them; the raw per-objective curves stay in
the run directories where the harness wrote them rather than being copied into
a second location that can silently go stale.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from collections import defaultdict
from pathlib import Path

# Under `duplicate` the m objectives are the same tensor rows through the same
# batched kernels, and every cell measured so far came back at EXACTLY 0.0 --
# not "small", zero. So the tolerance is not a noise budget, it is a guard
# against a float that is zero in every bit that matters; anything above it is
# reported as a finding.
DUP_TOL = 1e-9

# `scaled` and `conflicting` reach equality only after dividing by c_i, which is
# a real fp32 multiply, so those get a relative tolerance instead of an absolute
# one. Losses here are O(10) nats, so 1e-6 relative is still far below anything
# a training difference could produce.
PROP_TOL = 1e-6

# Cross-engine held-out CE spread. Three engines running the same aggregator on
# the same seed and data differ only in how they build the same Gramian, so
# their val_ce should agree to well past this. Rui would call 1e-3 nats noise;
# above it, one of the engines is computing something else.
AGREE_TOL = 1e-3


# ----------------------------------------------------------------- loading
def expand(patterns: list[str]) -> list[Path]:
    """Resolve command-line arguments to run directories.

    Wildcards are expanded in here rather than left to the shell. bash expands
    ``results/solo/v10_*`` before exec; PowerShell hands it through verbatim to
    a native executable. This script is the laptop-side half of a cluster
    workflow, so the same command line has to work in both.
    """
    found: list[Path] = []
    for pat in patterns:
        hits = ([Path(h) for h in sorted(glob.glob(pat))]
                if any(c in pat for c in "*?[") else [Path(pat)])
        for p in hits:
            if (p / "rows.csv").exists() or (p / CURVES_NAME).exists():
                found.append(p)
            elif p.is_dir():
                found.extend(sorted(x for x in p.glob("*")
                                    if (x / "rows.csv").exists()))
    seen, uniq = set(), []
    for p in found:
        key = str(p.resolve()).lower()
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    return uniq


CURVES_NAME = "l11_per_objective_curves.json"


def load_run(d: Path) -> tuple[list[dict], list[dict], dict]:
    """rows.csv, the per-objective sidecar, and the manifest -- each optional.

    A run that died mid-campaign still has a partial rows.csv and a sidecar
    covering the cells that finished; both are read for what they have rather
    than being rejected for what they are missing.
    """
    rows: list[dict] = []
    rows_p = d / "rows.csv"
    if rows_p.exists():
        with open(rows_p, newline="") as f:
            rows = list(csv.DictReader(f))
    curves: list[dict] = []
    cur_p = d / CURVES_NAME
    if cur_p.exists():
        try:
            with open(cur_p) as f:
                curves = json.load(f)
        except (OSError, json.JSONDecodeError, ValueError):
            curves = []
    man: dict = {}
    man_p = d / "manifest.json"
    if man_p.exists():
        try:
            with open(man_p) as f:
                man = json.load(f)
        except (OSError, json.JSONDecodeError):
            man = {}
    return rows, curves, man


def fnum(r, key="value"):
    try:
        return float(r.get(key, "nan"))
    except (TypeError, ValueError):
        return math.nan


def as_int(v) -> int:
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return -1


def cell_key(engine, aggregator, qp, m, T):
    """The join between rows.csv and the sidecar.

    rows.csv calls the aggregator ``driver`` and the QP backend ``route``, and
    stores m and T as strings; the sidecar calls them ``aggregator`` and ``qp``
    and stores ints. Normalising in one place keeps that mismatch from silently
    producing empty joins that look like missing data.
    """
    return (str(engine), str(aggregator), str(qp), as_int(m), as_int(T))


def l11_cells(rows: list[dict]) -> dict:
    """key -> {objective_mode, metrics} for the L11 training cells.

    ``test == "train/curve"`` rows carry the scalar mean-loss curve, one row per
    step, and would overwrite the summary metrics with step values if let
    through -- so only ``test == "train"`` is taken.
    """
    out: dict = defaultdict(lambda: {"objective_mode": "", "metrics": {}})
    for r in rows:
        if r.get("level") != "L11" or r.get("test") != "train":
            continue
        key = cell_key(r.get("engine", ""), r.get("driver", ""),
                       r.get("route", ""), r.get("m", ""), r.get("T", ""))
        cell = out[key]
        cell["objective_mode"] = r.get("objective_mode", "") or cell["objective_mode"]
        cell["metrics"][r.get("metric", "")] = fnum(r)
    return dict(out)


# ------------------------------------------------- A. per-objective spread
def expected_coeffs(mode: str, m: int):
    """The coefficient vector ``apply_objective_mode`` multiplied the losses by.

    Copied from bench/profile_suite.py rather than imported: that module imports
    torch at module scope and this one has to run on a laptop that may not have
    a GPU stack installed. The copy is the obvious failure mode, so every row
    below records which mode it assumed and where it read that mode from --
    a stale coefficient vector shows up as a whole mode failing at once, not as
    one quietly wrong number.

    ``independent`` draws m different corpus windows, so there is no
    relationship to check and no verdict is issued for it.
    """
    if mode == "duplicate":
        return [1.0] * m
    if mode == "scaled":
        return [1.0 + 0.1 * i for i in range(m)]
    if mode == "conflicting":
        return [1.0 if i % 2 == 0 else -1.0 for i in range(m)]
    return None


def _mean_pairwise(v: list[float]) -> float:
    n = len(v)
    if n < 2:
        return 0.0
    tot = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            tot += abs(v[i] - v[j])
    return tot / (n * (n - 1) / 2.0)


def curve_stats(curve: list[list[float]], coeffs) -> dict:
    """Spread across the m trajectories, raw and after dividing out c_i.

    ``curve`` is steps x m as written by L11. Three numbers are asked for and
    all three are kept: the worst pairwise gap at the FINAL step (did they end
    apart), the mean pairwise gap over the whole run (did they travel apart),
    and the per-step spread max-min (WHEN did they come apart). The last is a
    series, not a scalar; its max and its argmax go in the table and the whole
    series goes in the JSON dump, because "at which step" is the question that
    turns a divergence number into a bug report.
    """
    steps = len(curve)
    m = len(curve[0]) if steps else 0
    raw_spread = [max(s) - min(s) for s in curve]
    stats = {
        "steps": steps,
        "m_in_curve": m,
        "final_max_pairwise": raw_spread[-1] if steps else math.nan,
        "mean_pairwise": (sum(_mean_pairwise(s) for s in curve) / steps
                          if steps else math.nan),
        "max_step_spread": max(raw_spread) if steps else math.nan,
        "mean_step_spread": (sum(raw_spread) / steps) if steps else math.nan,
        "worst_step": (max(range(steps), key=lambda i: raw_spread[i])
                       if steps else -1),
        "final_mean_loss": (sum(curve[-1]) / m) if steps and m else math.nan,
        "_spread_curve": raw_spread,
        "_mean_curve": [sum(s) / m for s in curve] if m else [],
    }

    # Normalised view: L_i / c_i is the SAME base loss under every mode except
    # independent, so this is the column that carries the verdict. For
    # duplicate (all c_i = 1) it is numerically identical to the raw column,
    # which is deliberate -- one verdict rule covers all three modes.
    if coeffs and m == len(coeffs) and all(c != 0 for c in coeffs):
        norm = [[s[i] / coeffs[i] for i in range(m)] for s in curve]
        nsp = [max(s) - min(s) for s in norm]
        scale = max(abs(sum(norm[-1]) / m), 1e-12)
        stats.update({
            "norm_final_spread": nsp[-1] if steps else math.nan,
            "norm_max_spread": max(nsp) if steps else math.nan,
            "norm_mean_pairwise": (sum(_mean_pairwise(s) for s in norm) / steps
                                   if steps else math.nan),
            "rel_max_spread": (max(nsp) / scale) if steps else math.nan,
            "_norm_spread_curve": nsp,
        })
    else:
        stats.update({
            "norm_final_spread": math.nan, "norm_max_spread": math.nan,
            "norm_mean_pairwise": math.nan, "rel_max_spread": math.nan,
            "_norm_spread_curve": [],
        })

    # conflicting with even m mirrors: +L, -L, +L, -L sums to zero exactly.
    # Reported relative to the mass of the vector so it means the same thing at
    # any loss scale.
    if steps and m % 2 == 0 and coeffs and coeffs[:2] == [1.0, -1.0]:
        fin = curve[-1]
        mass = sum(abs(x) for x in fin) or 1e-12
        stats["mirror_residual"] = abs(sum(fin)) / mass
    else:
        stats["mirror_residual"] = math.nan
    return stats


def divergence_rows(run_id, run_name, curves, cells, man_mode) -> tuple[list, list]:
    """One row per cell that has a per-objective curve, plus the ones skipped."""
    out, skipped = [], []
    for rec in curves:
        key = cell_key(rec.get("engine", ""), rec.get("aggregator", ""),
                       rec.get("qp", ""), rec.get("m"), rec.get("T"))
        curve = rec.get("curve_per_objective") or []
        m = as_int(rec.get("m"))

        cell = cells.get(key)
        mode, mode_src = resolve_mode(cell["objective_mode"] if cell else "",
                                      man_mode)

        if not curve or m < 2:
            skipped.append({
                "run": run_name, "run_id": run_id, "engine": rec.get("engine"),
                "aggregator": rec.get("aggregator"), "m": m,
                "why": "m<2, nothing to compare" if curve else "empty curve",
            })
            continue

        st = curve_stats(curve, expected_coeffs(mode, m))
        row = {
            "kind": "divergence", "run": run_name, "run_id": run_id,
            "source": CURVES_NAME, "objective_mode": mode,
            "mode_source": mode_src,
            "aggregator": rec.get("aggregator", ""), "engine": rec.get("engine", ""),
            "qp": rec.get("qp", ""), "m": m, "T": as_int(rec.get("T")),
            "V": as_int(rec.get("V")),
            "val_ce": rec.get("val_ce", math.nan),
            "val_perplexity": rec.get("val_perplexity", math.nan),
        }
        row.update({k: v for k, v in st.items() if not k.startswith("_")})

        expect, ok = {
            "duplicate": ("identical", st["norm_max_spread"] <= DUP_TOL),
            "scaled": ("proportional", st["rel_max_spread"] <= PROP_TOL),
            "conflicting": ("mirrored", st["rel_max_spread"] <= PROP_TOL),
        }.get(mode, ("unconstrained", None))
        row["expected"] = expect
        row["ok"] = ok
        row["exact_zero"] = st["max_step_spread"] == 0.0
        row["_series"] = {k[1:]: v for k, v in st.items() if k.startswith("_")}
        out.append(row)
    return out, skipped


# ------------------------------------------------------ B. held-out metrics
def resolve_mode(cell_mode: str, man_mode: str) -> tuple[str, str]:
    """objective_mode, and where it came from.

    The sidecar does not carry it, so it has to be recovered -- and it is the
    field that decides what "correct" means for the whole of part A. rows.csv
    has it per cell (L11 puts it in every row's base); manifest.json has it per
    run. Prefer the per-cell source, and always say which was used: a verdict
    computed against the wrong mode is worse than no verdict at all.
    """
    if cell_mode:
        return cell_mode, "rows.csv"
    if man_mode:
        return man_mode, "manifest.json"
    return "unknown", "none"


def metric_rows(run_id, run_name, cells, curves, man_mode,
                run_route="?", run_levels="?") -> list[dict]:
    """The four held-out/quality numbers per cell, from rows.csv.

    rows.csv is the source, not the sidecar: it is the only one that carries
    val_next_token_acc, final_train_loss and objective_mode. Where the sidecar
    also has val_ce the two are differenced, so a join that silently paired the
    wrong cells would show up as a nonzero delta instead of as a plausible
    table.
    """
    side = {cell_key(r.get("engine", ""), r.get("aggregator", ""), r.get("qp", ""),
                     r.get("m"), r.get("T")): r for r in curves}
    out = []
    for key, cell in cells.items():
        eng, agg, qp, m, T = key
        met = cell["metrics"]
        vc = met.get("val_ce", math.nan)
        s = side.get(key)
        delta = math.nan
        if s is not None and not math.isnan(vc):
            try:
                delta = abs(vc - float(s.get("val_ce")))
            except (TypeError, ValueError):
                delta = math.nan
        mode, mode_src = resolve_mode(cell["objective_mode"], man_mode)
        out.append({
            "kind": "metrics", "run": run_name, "run_id": run_id,
            "route": run_route, "levels": run_levels,
            "source": "rows.csv", "objective_mode": mode, "mode_source": mode_src,
            "m": m, "T": T, "aggregator": agg, "engine": eng, "qp": qp,
            "val_ce": vc,
            "val_perplexity": met.get("val_perplexity", math.nan),
            "val_next_token_acc": met.get("val_next_token_acc", math.nan),
            "final_train_loss": met.get("final_train_loss", math.nan),
            "mean_last10": met.get("mean_last10", math.nan),
            "ms_per_step": met.get("ms_per_step", math.nan),
            "has_curves": s is not None,
            "sidecar_val_ce_delta": delta,
        })
    return sorted(out, key=lambda r: (r["m"], r["T"], r["objective_mode"],
                                      r["aggregator"], r["engine"]))


def pivot_by_engine(metrics: list[dict], field: str, prefix: str):
    """(run, m, T, mode, aggregator) -> one column per engine, plus the control.

    The sgd_erm arm is not an aggregator -- it is one forward, one backward, the
    plain mean of the losses -- so it never joins an aggregator group. It is
    carried alongside as the control column, which is what makes the aggregator
    numbers readable: 50.47 means nothing until the control next to it says
    50.47 too.
    """
    groups: dict = defaultdict(dict)
    controls: dict = {}
    for r in metrics:
        gk = (r["run_id"], r["m"], r["T"], r["objective_mode"])
        if r["engine"] == "sgd_erm":
            controls[gk] = r[field]
            continue
        groups[(gk, r["aggregator"])][r["engine"]] = r[field]

    engines = sorted({e for v in groups.values() for e in v})
    rows = []
    for (gk, agg), by_eng in sorted(groups.items(),
                                    key=lambda kv: (kv[0][0][1], kv[0][0][2],
                                                    kv[0][0][3], kv[0][1])):
        run_id, m, T, mode = gk
        vals = [v for v in by_eng.values()
                if isinstance(v, float) and not math.isnan(v) and not math.isinf(v)]
        row = {"run_id": run_id, "m": m, "T": T, "objective_mode": mode,
               "aggregator": agg}
        for e in engines:
            row[f"{prefix}_{e}"] = by_eng.get(e, math.nan)
        row[f"{prefix}_control"] = controls.get(gk, math.nan)
        row["spread"] = (max(vals) - min(vals)) if len(vals) >= 2 else math.nan
        row["n_engines"] = len(vals)
        rows.append(row)
    return rows, engines


def agreement_rows(metrics: list[dict]) -> list[dict]:
    """Do the engines that compute the same thing reach the same held-out CE?

    Grouped WITHIN a run directory. Two run dirs with the same config are two
    separate trainings; folding them together would turn a re-run into a fake
    cross-engine disagreement.

    PCGrad is measured but not judged. Its weighting is genuinely
    nondeterministic: PCGradWeighting.forward calls torch.randperm(dimension)
    once per objective on the global CPU RNG (TorchJD 0.17.0,
    src/torchjd/aggregation/_pcgrad.py:31), and the harness seeds only once at
    model construction. The three engines consume different amounts of CPU RNG
    on the way there, so they draw different projection orders and PCGrad's
    projection is order-dependent -- three different answers from the same
    Gramian is expected behaviour, not an engine bug. Flagging it would train
    the reader to ignore this column.
    """
    rows, _ = pivot_by_engine(metrics, "val_ce", "ce")
    out = []
    for r in rows:
        checked = r["aggregator"] != "PCGrad"
        sp = r["spread"]
        out.append(dict(
            r, kind="agreement", checked=checked,
            ok=(None if (not checked or r["n_engines"] < 2 or math.isnan(sp))
                else sp <= AGREE_TOL),
            note=("" if checked else "PCGrad: unseeded randperm, excluded"),
        ))
    return out


def repeat_rows(metrics: list[dict]) -> list[dict]:
    """Same config, two run directories: how reproducible is a whole training?

    Not part of the ask, but the solo campaign contains a repeated cell and a
    reader comparing two rows with identical (m, T, mode, aggregator, engine)
    needs to be told they are separate trainings rather than a duplicate print.
    It also calibrates the agreement tolerance above with a number from this
    machine instead of a guess.
    """
    groups: dict = defaultdict(dict)
    for r in metrics:
        groups[(r["m"], r["T"], r["objective_mode"], r["aggregator"],
                r["engine"], r.get("route", "?"),
                r.get("levels", "?"))][r["run_id"]] = r["val_ce"]
    out = []
    for (m, T, mode, agg, eng, route, levels), by_run in sorted(groups.items()):
        vals = [v for v in by_run.values()
                if isinstance(v, float) and not math.isnan(v)]
        if len(vals) < 2:
            continue
        out.append({"kind": "repeat", "m": m, "T": T, "objective_mode": mode,
                    "aggregator": agg, "engine": eng, "route": route,
                    "levels": levels, "runs": len(vals),
                    "run_ids": ",".join(sorted(by_run)),
                    "ce_spread": max(vals) - min(vals)})
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
        return f"{v:.4f}" if abs(v) < 1000 else f"{v:.0f}"
    return str(v)


def sci(v):
    """Scientific notation for the divergence columns.

    fmt()'s fixed decimals render 1e-9 as "0.0000", which is precisely the
    failure this report exists to catch: a real nonzero divergence printed as a
    clean zero. Exact zero prints as a bare "0" so "identical" is visually
    distinct from "very small", which is the distinction the duplicate mode is
    for.
    """
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    if isinstance(v, float) and math.isinf(v):
        return "inf"
    if v == 0:
        return "0"
    return f"{v:.3e}"


def as_display(rows, sci_cols):
    """Pre-format the scientific columns; table() passes strings through."""
    return [dict(r, **{c: sci(r.get(c)) for c in sci_cols if c in r})
            for r in rows]


def strip_private(r):
    return {k: v for k, v in r.items() if not k.startswith("_")}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+",
                    help="run directories (wildcards expanded here, so an "
                         "unexpanded PowerShell glob also works)")
    ap.add_argument("--csv", type=Path, default=None,
                    help="write every row of every table here, long format")
    ap.add_argument("--json", type=Path, default=None,
                    help="same, plus the per-step spread series for plotting")
    ap.add_argument("--agree-tol", type=float, default=AGREE_TOL,
                    help=f"cross-engine val_ce spread to flag (default {AGREE_TOL})")
    args = ap.parse_args()

    dirs = expand(args.runs)
    if not dirs:
        print("no run directories found (looked for rows.csv / " + CURVES_NAME + ")")
        return 1

    div, metrics, skipped, legend, env = [], [], [], [], {}
    no_sidecar = []
    for i, d in enumerate(dirs, 1):
        run_id = f"r{i}"
        rows, curves, man = load_run(d)
        if man and not env:
            env = man.get("env", {})
        cfg = man.get("config", {}) if man else {}
        man_mode = cfg.get("objective_mode", "")
        cells = l11_cells(rows)
        dv, sk = divergence_rows(run_id, d.name, curves, cells, man_mode)
        div.extend(dv)
        skipped.extend(sk)
        metrics.extend(metric_rows(
            run_id, d.name, cells, curves, man_mode,
            cfg.get("force_route") or "auto",
            ",".join(cfg.get("levels") or [])))
        if not curves:
            no_sidecar.append(d.name)
        legend.append({
            "run_id": run_id, "dir": d.name,
            "m": cfg.get("m", "?"), "T": cfg.get("T", "?"),
            "objective_mode": man_mode or "?",
            "route": cfg.get("force_route", "?"), "steps": cfg.get("steps", "?"),
            "cells": len(cells), "curves": len(curves),
        })

    print("=" * 78)
    print("PER-OBJECTIVE CURVES AND HELD-OUT METRICS")
    if env:
        print(f"  {env.get('gpu','?')} | torch {env.get('torch','?')} | "
              f"torchjd {env.get('torchjd','?')} | host {env.get('host','?')}")
    print(f"  {len(dirs)} run dir(s), {len(metrics)} L11 cells, "
          f"{len(div)} with a comparable per-objective curve")
    print("=" * 78)
    print(table(legend, ["run_id", "m", "T", "objective_mode", "route", "steps",
                         "cells", "curves", "dir"],
                "Run legend -- run_id is used in every table below"))
    if no_sidecar:
        print(f"  !! no {CURVES_NAME} in: {', '.join(no_sidecar)}")

    # ------------------------------------------------------------- part A
    print("\n" + "=" * 78)
    print("A. PER-OBJECTIVE DIVERGENCE")
    print("=" * 78)
    sci_cols = ["final_max_pairwise", "mean_pairwise", "max_step_spread",
                "mean_step_spread", "norm_max_spread", "rel_max_spread",
                "mirror_residual"]
    for mode in sorted({r["objective_mode"] for r in div}):
        sub = sorted([r for r in div if r["objective_mode"] == mode],
                     key=lambda r: (r["m"], r["T"], r["aggregator"], r["engine"]))
        note = {
            "duplicate": "m copies of ONE sequence, unit coeffs -- the curves "
                         "must be IDENTICAL, not merely close",
            "scaled": "one sequence, coeffs 1, 1.1, 1.2, ... -- L_i/c_i must be "
                      "identical",
            "conflicting": "coeffs +1,-1,... -- L_i/c_i identical, and for even "
                           "m the losses sum to 0. Diverging loss is BY DESIGN.",
            "independent": "m different windows -- no relationship is forced, so "
                           "no verdict is issued",
        }.get(mode, "")
        cols = ["run_id", "m", "T", "aggregator", "engine",
                "final_max_pairwise", "mean_pairwise", "max_step_spread",
                "worst_step", "norm_max_spread"]
        # The raw spread under conflicting is large BY CONSTRUCTION (+L against
        # -L), so a reader scanning that column would see the worst numbers in
        # the report on the one mode where they are expected. mirror_residual is
        # the column that actually carries information there: |sum L_i| over
        # sum|L_i|, which is zero when the objectives really do cancel.
        if mode == "conflicting":
            cols.append("mirror_residual")
        cols += ["expected", "ok", "mode_source"]
        print(table(as_display(sub, sci_cols), cols,
                    f"objective_mode={mode}" + (f"   [{note}]" if note else "")))

    judged = [r for r in div if r["ok"] is not None]
    bad = [r for r in div if r["ok"] is False]
    if bad:
        print("  " + "!" * 74)
        print(f"  !! {len(bad)} cell(s) FAILED the objective-relationship check.")
        for r in bad:
            print(f"  !!   {r['run_id']} m={r['m']} {r['aggregator']}/{r['engine']}"
                  f" mode={r['objective_mode']}: normalised spread "
                  f"{sci(r['norm_max_spread'])} peaks at step {r['worst_step']} "
                  f"(expected {r['expected']})")
        print("  !! Under a constructed mode the objectives are the same "
              "function up to a known")
        print("  !! constant. A nonzero spread is an engine or harness defect, "
              "not training noise.")
        print("  " + "!" * 74)
    elif judged:
        exact = sum(1 for r in judged if r["exact_zero"])
        print(f"  All {len(judged)} judged cell(s) pass; {exact} of them are "
              f"EXACTLY zero at every step.")
        print("  The m objective curves lie on top of each other, which is what "
              "duplicate mode")
        print("  demands -- the engines are not leaking one objective's gradient "
              "into another's loss.")
    if skipped:
        by_why = defaultdict(int)
        for s in skipped:
            by_why[s["why"]] += 1
        print("  skipped: " + ", ".join(f"{n} x {w}" for w, n in sorted(by_why.items())))

    # ------------------------------------------------------------- part B
    print("\n" + "=" * 78)
    print("B. HELD-OUT METRICS")
    print("=" * 78)
    print(table(metrics,
                ["run_id", "m", "T", "objective_mode", "aggregator", "engine",
                 "val_ce", "val_perplexity", "val_next_token_acc",
                 "final_train_loss", "has_curves", "sidecar_val_ce_delta"],
                "Per cell (source: rows.csv; sidecar_val_ce_delta cross-checks "
                "the join against " + CURVES_NAME + ")"))

    joined = [r for r in metrics if r["has_curves"]
              and not math.isnan(r["sidecar_val_ce_delta"])]
    if joined:
        worst = max(joined, key=lambda r: r["sidecar_val_ce_delta"])
        print(f"  rows.csv and {CURVES_NAME} agree on val_ce to "
              f"{sci(worst['sidecar_val_ce_delta'])} at worst over "
              f"{len(joined)} joined cells.")
    orphan = [r for r in metrics if not r["has_curves"]]
    if orphan:
        print(f"  {len(orphan)} cell(s) in rows.csv have no per-objective curve "
              f"(OOM, error, or resumed from an earlier run).")

    ppl, engines = pivot_by_engine(metrics, "val_perplexity", "ppl")
    if ppl:
        print(table(ppl,
                    ["run_id", "m", "T", "objective_mode", "aggregator"]
                    + [f"ppl_{e}" for e in engines] + ["ppl_control", "spread"],
                    "Validation perplexity by engine, against the sgd_erm control"))

    agree = agreement_rows(metrics)
    # agreement_rows judges against the module default so it stays usable as a
    # library call; --agree-tol re-judges here rather than being threaded
    # through, so the tolerance that produced the verdict is set in one place.
    for r in agree:
        r["ok"] = (None if (r["ok"] is None) else r["spread"] <= args.agree_tol)
        r["tolerance"] = args.agree_tol
    if agree:
        print(table(as_display(agree, ["spread"]),
                    ["run_id", "m", "T", "objective_mode", "aggregator"]
                    + [f"ce_{e}" for e in engines]
                    + ["spread", "n_engines", "checked", "ok", "note"],
                    f"CROSS-ENGINE AGREEMENT on val_ce   [flag if spread > "
                    f"{args.agree_tol:g} nats]"))
        checked = [r for r in agree if r["ok"] is not None]
        fails = [r for r in checked if r["ok"] is False]
        if fails:
            print("  " + "!" * 74)
            print(f"  !! {len(fails)} of {len(checked)} checked group(s) disagree "
                  f"by more than {args.agree_tol:g} nats.")
            for r in sorted(fails, key=lambda r: -r["spread"]):
                cols = ", ".join(f"{e}={fmt(r.get(f'ce_{e}'))}" for e in engines
                                 if not math.isnan(r.get(f"ce_{e}", math.nan)))
                print(f"  !!   {r['run_id']} m={r['m']} {r['aggregator']}: "
                      f"spread {sci(r['spread'])} nats  [{cols}]")
            print("  !! These engines build the SAME Gramian from the same seed "
                  "and the same data.")
            print("  !! A held-out CE spread is a correctness concern in one of "
                  "them, and timing")
            print("  !! data cannot see it.")
            # Which engine is the odd one out, across all the failing groups.
            # A spread number says the group disagrees; it does not say who is
            # wrong. If one engine is the extreme in every failing group that is
            # a systematic bias in that engine, and it is the difference between
            # "tighten the tolerance" and "go read that engine's code".
            tally = defaultdict(int)
            for r in fails:
                vals = {e: r.get(f"ce_{e}") for e in engines}
                vals = {e: v for e, v in vals.items()
                        if isinstance(v, float) and not math.isnan(v)}
                if len(vals) >= 2:
                    tally[max(vals, key=vals.get)] += 1
            if tally:
                top, n = max(tally.items(), key=lambda kv: kv[1])
                print(f"  !! Highest val_ce in {n}/{len(fails)} failing group(s): "
                      f"{top}.")
            print("  " + "!" * 74)
        elif checked:
            worst = max(checked, key=lambda r: r["spread"])
            print(f"  All {len(checked)} checked group(s) agree; worst spread "
                  f"{sci(worst['spread'])} nats "
                  f"({worst['run_id']} m={worst['m']} {worst['aggregator']}).")
        excluded = sorted({r["aggregator"] for r in agree if not r["checked"]})
        if excluded:
            print(f"  excluded from the check: {', '.join(excluded)} "
                  f"(unseeded torch.randperm in its weighting -- see "
                  f"agreement_rows docstring).")

    rep = repeat_rows(metrics)
    if rep:
        print(table(as_display(rep, ["ce_spread"]),
                    ["m", "T", "objective_mode", "aggregator", "engine",
                     "route", "levels", "runs",
                     "run_ids", "ce_spread"],
                    "Repeat runs of the same config -- separate trainings, not "
                    "duplicate rows"))
        nonpc = [r for r in rep if r["aggregator"] != "PCGrad"]
        if nonpc:
            w = max(nonpc, key=lambda r: r["ce_spread"])
            print(f"  Run-to-run val_ce spread on identical configs: "
                  f"{sci(w['ce_spread'])} nats at worst (excluding PCGrad). "
                  f"That is the real noise floor for the check above.")

    # ------------------------------------------------------------- part C
    if args.csv:
        allrows = ([strip_private(r) for r in div]
                   + [dict(r, kind="metrics") for r in metrics]
                   + [dict(r, kind="perplexity") for r in ppl]
                   + [strip_private(r) for r in agree]
                   + rep)
        keys = sorted({k for r in allrows for k in r})
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(allrows)
        print(f"\nwrote {len(allrows)} rows -> {args.csv}")

    if args.json:
        # The series live here and nowhere else in the output: they are what a
        # laptop notebook needs to draw "did the two curves go too far", and
        # they are useless in a CSV cell.
        payload = {
            "env": env, "runs": legend,
            "tolerances": {"duplicate": DUP_TOL, "proportional": PROP_TOL,
                           "cross_engine_val_ce": args.agree_tol},
            "divergence": [dict(strip_private(r), series=r["_series"]) for r in div],
            "divergence_skipped": skipped,
            "metrics": metrics, "perplexity": ppl, "agreement":
                [strip_private(r) for r in agree], "repeats": rep,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=1)
        print(f"wrote {len(div)} curve records + {len(metrics)} cells "
              f"-> {args.json}")

    # 0 = clean, 2 = report produced and a check failed, 1 = could not run at
    # all. Kept distinct on purpose: a campaign script that treats "found a
    # divergence" and "found no data" as the same exit status will eventually
    # report a missing results directory as a passing suite.
    return 2 if (bad or any(r["ok"] is False for r in agree)) else 0


if __name__ == "__main__":
    raise SystemExit(main())
