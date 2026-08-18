"""profile_stats.py -- turn a profiling run into compact, reviewable statistics.

Raw traces are the wrong unit of analysis. A single chrome trace is hundreds of
MB of individual events; reading it directly costs far more attention than it
returns, and none of that attention is spent on the question you opened it with.
What actually localises a problem is the aggregate: which op owns the time, how
many launches produced it, how often the GPU had to stop and wait for the host,
and where the memory went.

So this reads whatever a run produced --

    rows.csv               the suite's own long-format measurements
    profiler_ops.csv       torch.profiler key_averages (compact, always written)
    profiler_trace.json    chrome trace, raw or .gz (optional, richest)
    memory_snapshot.pickle CUDA allocation history (optional)

-- and emits a bounded report: per-op and per-kernel distributions
(count/mean/median/min/max/p95/total/share), the CPU-vs-GPU split, GPU idle
fraction, launch-boundedness, host-device synchronisation points with the op that
caused each, top allocation sites, measured scaling exponents, and a ranked
FINDINGS list from explicit thresholds.

Output is capped by --top so it stays readable no matter how large the input.

Usage
  python bench/profile_stats.py results/v5_..._baseline_abc1234
  python bench/profile_stats.py results/v5_... --top 25 --json stats.json
  python bench/profile_stats.py results/A results/B --compare
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import pickle
import statistics as st
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

MiB = 1024**2
US = 1e-6

# --------------------------------------------------------------------------
# Ops whose implementation forces the host to wait for the device. Each one
# drains the pipeline: the CPU cannot run ahead, so launch latency stops being
# hidden and every subsequent kernel pays for it. This is the "GPU exits to CPU"
# question -- a data-dependent `if`, a `.item()` in a loop, a python-side
# comparison on a tensor value all land here.
# --------------------------------------------------------------------------
SYNC_OPS = {
    "aten::_local_scalar_dense": ".item() / float() / int() on a device tensor",
    "aten::item": ".item() on a device tensor",
    "aten::equal": "tensor == tensor used as a python bool",
    "aten::nonzero": "data-dependent output shape; host must learn the size",
    "aten::masked_select": "data-dependent output shape",
    "aten::_unique2": "data-dependent output shape",
    "aten::unique_consecutive": "data-dependent output shape",
    "aten::allclose": "reduction read back to the host",
    "aten::is_nonzero": "python truthiness of a device tensor",
    "aten::_assert_async": "device-side assert readback",
}
SYNC_RUNTIME = {
    "cudaStreamSynchronize": "explicit stream sync",
    "cudaDeviceSynchronize": "explicit device sync",
    "cudaEventSynchronize": "event wait",
    "cudaMemcpyAsync": "memcpy (device-to-host is blocking in practice)",
    "cudaMemcpy": "blocking memcpy",
    "cudaHostAlloc": "pinned allocation mid-stream",
    "cudaFree": "allocator returning blocks to the driver (implicit sync)",
    "cudaMalloc": "allocator growing the pool (implicit sync)",
}
#: Kernels below this are dominated by launch overhead rather than work.
LAUNCH_BOUND_US = 8.0
#: A run spending more than this fraction of wall time with no kernel resident.
IDLE_FRACTION_WARN = 0.35


def pct(x: float, total: float) -> float:
    return 100.0 * x / total if total else 0.0


def summarize(values: list[float]) -> dict:
    """The distribution, not just the mean. A stable mean with a large spread and
    a p95 far from the median is a different problem from a slow mean."""
    if not values:
        return {}
    vs = sorted(values)
    n = len(vs)
    return {
        "count": n,
        "total": sum(vs),
        "mean": sum(vs) / n,
        "median": st.median(vs),
        "min": vs[0],
        "max": vs[-1],
        "p95": vs[min(n - 1, int(0.95 * (n - 1)))],
        "stdev": st.stdev(vs) if n > 1 else 0.0,
        "cv": (st.stdev(vs) / (sum(vs) / n)) if n > 1 and sum(vs) else 0.0,
    }


@dataclass
class Finding:
    severity: str            # "high" | "medium" | "info"
    kind: str
    message: str
    evidence: str = ""

    def line(self) -> str:
        tag = {"high": "!!", "medium": " !", "info": "  "}[self.severity]
        out = f"{tag} [{self.kind}] {self.message}"
        if self.evidence:
            out += f"\n       {self.evidence}"
        return out


@dataclass
class Report:
    run: str
    sections: dict = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)

    def add(self, severity: str, kind: str, message: str, evidence: str = "") -> None:
        self.findings.append(Finding(severity, kind, message, evidence))


# ============================================================ suite rows.csv
def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _f(row: dict, key: str) -> float:
    try:
        v = float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return float("nan")
    return v


def analyse_rows(rows: list[dict], rep: Report, top: int) -> None:
    """Aggregate the suite's own measurements, grouped the way the questions are
    asked: by level/test/engine/driver/route/metric."""
    if not rows:
        return
    groups: dict[tuple, list[float]] = defaultdict(list)
    ooms: list[dict] = []
    for r in rows:
        if str(r.get("oom", "")).lower() == "true":
            ooms.append(r)
            continue
        v = _f(r, "value")
        if math.isnan(v):
            continue
        key = (r.get("level", ""), r.get("test", ""), r.get("engine", ""),
               r.get("driver", ""), r.get("route", ""), r.get("metric", ""))
        groups[key].append(v)

    agg = {}
    for key, vals in groups.items():
        agg["|".join(key)] = summarize(vals)
    rep.sections["suite_measurements"] = agg
    rep.sections["oom_rows"] = [
        {k: r.get(k) for k in ("level", "test", "engine", "driver", "route",
                               "m", "T", "V", "n_embd", "note")}
        for r in ooms
    ]

    # ---- driver comparison: the axis that decides jdgram's memory profile ----
    peaks = {}
    for r in rows:
        if r.get("metric") != "peak_mib" or r.get("engine") != "jdgram":
            continue
        d = r.get("driver", "")
        v = _f(r, "value")
        if d and not math.isnan(v):
            peaks.setdefault(d, []).append(v)
    if len(peaks) > 1:
        summary = {d: summarize(v)["max"] for d, v in peaks.items()}
        rep.sections["driver_peak_max_mib"] = summary
        if "squashed" in summary:
            for other in ("batched", "loop"):
                if other in summary and summary["squashed"] > 0:
                    ratio = summary[other] / summary["squashed"]
                    if ratio > 1.5:
                        rep.add("info", "driver",
                                f"driver '{other}' peaks {ratio:.1f}x higher than 'squashed'",
                                f"{other}={summary[other]:.0f} MiB, squashed={summary['squashed']:.0f} MiB")
                    elif ratio < 0.9:
                        rep.add("high", "driver",
                                f"'squashed' is NOT the leanest driver ({ratio:.2f}x vs {other})",
                                f"expected squashed to win; check the block-diagonal assumption")

    # ---- route sensitivity, WITHIN each driver.
    # Comparing routes across drivers is meaningless: whichever driver dominates
    # peak is route-invariant by construction, so a max over all drivers reports
    # "routes identical" no matter what the routes actually do. Group first.
    by_dr: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in rows:
        if r.get("metric") == "peak_mib" and r.get("engine") == "jdgram":
            v = _f(r, "value")
            if not math.isnan(v):
                by_dr[(r.get("driver", ""), r.get("route", ""))].append(v)
    spreads = {}
    for driver in sorted({d for d, _ in by_dr}):
        routed = {
            rt: summarize(by_dr[(driver, rt)])["max"]
            for rt in ("tfirst", "dfirst")
            if by_dr.get((driver, rt))
        }
        if len(routed) != 2:
            continue
        hi = max(routed.values())
        spread = (hi - min(routed.values())) / hi if hi else 0.0
        spreads[driver] = {"max_by_route": routed, "spread": round(spread, 4)}
        if spread < 0.02:
            # Only a problem for the driver that ships. squashed is the default;
            # batched and loop are the controls, and route-invariance there is the
            # expected finding, not a regression -- flagging it 'high' buries the
            # one line that would matter.
            sev = "high" if driver in ("squashed", "", "default") else "info"
            tail = ("" if sev == "high"
                    else "  (expected: this is a control arm, its driver dominates peak)")
            rep.add(sev, "route",
                    f"[driver={driver}] peak is identical across tfirst/dfirst -- the "
                    f"contraction workspace is NOT the peak for this driver{tail}",
                    f"tfirst={routed['tfirst']:.0f} MiB dfirst={routed['dfirst']:.0f} MiB")
        else:
            rep.add("info", "route",
                    f"[driver={driver}] routes differ in peak by {spread * 100:.0f}% -- "
                    f"the router is controlling memory",
                    str({k: round(v, 1) for k, v in routed.items()}))
    if spreads:
        rep.sections["route_peak_spread_by_driver"] = spreads

    # ---- measured vs theoretical workspace ----
    theo = {}
    meas = {}
    for r in rows:
        k = (r.get("test"), r.get("driver"), r.get("m"), r.get("T"))
        if r.get("metric") == "theoretical_mib":
            theo[k] = _f(r, "value")
        elif r.get("metric") == "peak_mib" and r.get("level") == "L0":
            meas[k] = _f(r, "value")
    overheads = []
    for k, t in theo.items():
        mv = meas.get(k)
        if mv and t and not math.isnan(mv) and not math.isnan(t) and t > 0:
            overheads.append((mv / t, k, mv, t))
    if overheads:
        overheads.sort(reverse=True)
        rep.sections["identity_overhead_vs_theory"] = [
            {"ratio": round(r, 2), "test": k[0], "route": k[1],
             "measured_mib": round(mv, 1), "theoretical_mib": round(t, 1)}
            for r, k, mv, t in overheads[:top]
        ]
        worst = overheads[0]
        if worst[0] > 3.0:
            rep.add("medium", "identity",
                    f"identity peak is {worst[0]:.1f}x its theoretical workspace at "
                    f"{worst[1][0]}/{worst[1][1]}",
                    f"measured {worst[2]:.1f} MiB vs theory {worst[3]:.1f} MiB")

    # ---- scaling exponents ----
    exps = {}
    for r in rows:
        met = r.get("metric", "")
        if met.startswith(("peak_exponent_", "time_exponent_")):
            key = f"{r.get('driver')}|{r.get('route')}|{met}"
            exps[key] = _f(r, "value")
    if exps:
        rep.sections["scaling_exponents"] = exps
        for key, val in exps.items():
            if math.isnan(val):
                continue
            if key.endswith("peak_exponent_T") and val > 1.6:
                rep.add("medium", "scaling",
                        f"peak grows as T^{val:.2f} for {key.split('|')[0]}/"
                        f"{key.split('|')[1]} -- superlinear in sequence length",
                        "T-first workspace is m^2 T^2; a linear-in-T route should read ~1.0")
            if key.endswith("peak_exponent_m") and val > 1.5:
                rep.add("medium", "scaling",
                        f"peak grows as m^{val:.2f} for {key.split('|')[0]}/"
                        f"{key.split('|')[1]}",
                        "an m-fold replication of reverse state reads ~2.0; "
                        "per-objective captures alone read ~1.0")

    if ooms:
        rep.add("info", "oom", f"{len(ooms)} configuration(s) hit OOM",
                "; ".join(f"{r.get('level')}/{r.get('engine')}/{r.get('driver')} "
                          f"m={r.get('m')} T={r.get('T')} V={r.get('V')}"
                          for r in ooms[:8]))


# ================================================== torch.profiler key_averages
def analyse_ops_csv(path: Path, rep: Report, top: int) -> None:
    if not path.exists():
        return
    with open(path, newline="") as f:
        ops = list(csv.DictReader(f))
    if not ops:
        return

    def num(o: dict, k: str) -> float:
        try:
            return float(o.get(k) or 0.0)
        except ValueError:
            return 0.0

    total_self_cuda = sum(num(o, "self_cuda_time_total_us") for o in ops)
    total_self_cpu = sum(num(o, "self_cpu_time_total_us") for o in ops)

    def rank(field_name: str, total: float) -> list[dict]:
        ranked = sorted(ops, key=lambda o: num(o, field_name), reverse=True)
        out = []
        for o in ranked[:top]:
            v = num(o, field_name)
            if v <= 0:
                break
            cnt = int(num(o, "count")) or 1
            out.append({
                "name": o["name"],
                "count": cnt,
                "total_ms": round(v * US * 1e3, 3),
                "mean_us": round(v / cnt, 2),
                "share_pct": round(pct(v, total), 1),
            })
        return out

    rep.sections["top_ops_self_cuda"] = rank("self_cuda_time_total_us", total_self_cuda)
    rep.sections["top_ops_self_cpu"] = rank("self_cpu_time_total_us", total_self_cpu)
    rep.sections["op_totals"] = {
        "self_cuda_ms": round(total_self_cuda * US * 1e3, 2),
        "self_cpu_ms": round(total_self_cpu * US * 1e3, 2),
        "distinct_ops": len(ops),
        "total_op_calls": int(sum(num(o, "count") for o in ops)),
    }

    # concentration: is time in a few big kernels, or smeared over many small ones?
    top5 = sum(num(o, "self_cuda_time_total_us")
               for o in sorted(ops, key=lambda o: num(o, "self_cuda_time_total_us"),
                               reverse=True)[:5])
    if total_self_cuda:
        share = pct(top5, total_self_cuda)
        rep.sections["cuda_concentration_top5_pct"] = round(share, 1)
        if share < 40:
            rep.add("medium", "dispatch",
                    f"top-5 ops hold only {share:.0f}% of device time -- work is smeared "
                    f"across many small ops, which is the signature of a python-level "
                    f"per-layer loop rather than a few large GEMMs")

    # sync-forcing ops
    syncs = [(o["name"], int(num(o, "count")), SYNC_OPS[o["name"]])
             for o in ops if o["name"] in SYNC_OPS and num(o, "count") > 0]
    if syncs:
        rep.sections["sync_forcing_ops"] = [
            {"op": n, "count": c, "why": why} for n, c, why in sorted(syncs, key=lambda x: -x[1])
        ]
        worst = max(syncs, key=lambda x: x[1])
        rep.add("high" if worst[1] > 20 else "medium", "gpu-sync",
                f"{sum(c for _, c, _ in syncs)} host-device synchronisation(s) forced by "
                f"{len(syncs)} distinct op(s)",
                "; ".join(f"{n} x{c} ({why})" for n, c, why in sorted(syncs, key=lambda x: -x[1])[:5]))

    # memory attribution
    allocs = [(o["name"], num(o, "self_cuda_memory_usage")) for o in ops]
    allocs = [(n, v) for n, v in allocs if v > 0]
    if allocs:
        allocs.sort(key=lambda x: -x[1])
        rep.sections["top_ops_by_cuda_alloc"] = [
            {"name": n, "alloc_mib": round(v / MiB, 2)} for n, v in allocs[:top]
        ]


# ======================================================= chrome trace analysis
def load_trace(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:  # type: ignore[operator]
        data = json.load(f)
    return data.get("traceEvents", data if isinstance(data, list) else [])


def analyse_trace(path: Path, rep: Report, top: int) -> None:
    """The richest source: individual kernels, launches, memcpys and gaps."""
    if not path.exists():
        return
    events = [e for e in load_trace(path) if e.get("ph") == "X"]
    if not events:
        return

    by_cat: dict[str, list[dict]] = defaultdict(list)
    for e in events:
        by_cat[e.get("cat", "?")].append(e)
    rep.sections["trace_event_counts"] = {k: len(v) for k, v in sorted(
        by_cat.items(), key=lambda kv: -len(kv[1]))}

    kernels = by_cat.get("kernel", [])
    memcpy = by_cat.get("gpu_memcpy", [])
    memset = by_cat.get("gpu_memset", [])
    runtime = by_cat.get("cuda_runtime", [])

    # ---- launch geometry and occupancy ----
    # Every kernel event carries an "args" dict with grid, block, registers per
    # thread, shared memory and the profiler's occupancy estimate. This was
    # parsed for cat/name/ts/dur only and args was dropped, so the one piece of
    # hardware-level evidence already on disk in every --trace-raw run was being
    # thrown away and then described as "not available without ncu".
    # The occupancy key is spelled differently across torch versions -- 2.10
    # emits "est. achieved occupancy %" (0-100), older builds emit
    # "est. achieved occupancy" (0-1). Accept both and normalise to a fraction,
    # because this analysis has to read traces produced on the cluster (2.4.1)
    # and on a laptop (2.10) and silently reporting None on one of them is how
    # the whole section came to be believed unavailable in the first place.
    OCC_KEYS = ("est. achieved occupancy %", "est. achieved occupancy",
                "est_achieved_occupancy")

    def _occupancy(a):
        for key in OCC_KEYS:
            if key in a:
                try:
                    v = float(a[key])
                except (TypeError, ValueError):
                    return None
                return v / 100.0 if v > 1.5 else v
        return None

    geom: dict[str, dict] = {}
    for k in kernels:
        a = k.get("args") or {}
        if not a:
            continue
        name = k.get("name", "?")
        g = geom.setdefault(name, {"launches": 0, "occ": [], "regs": None,
                                   "smem": None, "grid": None, "block": None,
                                   "bpsm": None, "wpsm": None})
        g["launches"] += 1
        occ = _occupancy(a)
        if occ is not None:
            g["occ"].append(occ)
        for src, dst in (("registers per thread", "regs"),
                         ("shared memory", "smem"),
                         ("grid", "grid"), ("block", "block"),
                         ("blocks per SM", "bpsm"), ("warps per SM", "wpsm")):
            if g[dst] is None and a.get(src) is not None:
                g[dst] = a[src]

    if geom:
        # One pass for durations; the per-name generator inside the loop was
        # O(kernels^2) and a real trace carries tens of thousands of events.
        dur_by_name: dict[str, float] = defaultdict(float)
        for k in kernels:
            dur_by_name[k.get("name", "?")] += float(k.get("dur", 0.0))
        rows = []
        for name, g in geom.items():
            occs = g["occ"]
            rows.append({
                "kernel": name[:110],
                "launches": g["launches"],
                "device_ms": round(dur_by_name[name] * US * 1e3, 3),
                "mean_occupancy": round(sum(occs) / len(occs), 4) if occs else None,
                "registers_per_thread": g["regs"],
                "shared_memory": g["smem"],
                "blocks_per_sm": g["bpsm"],
                "warps_per_sm": g["wpsm"],
                "grid": g["grid"],
                "block": g["block"],
            })
        rows.sort(key=lambda r: -r["device_ms"])
        rep.sections["kernel_launch_geometry"] = rows[:top]

        # An occupancy floor only matters on a kernel that holds real time.
        # 0.3 is the conventional bar below which a kernel is usually limited by
        # registers or block shape rather than by arithmetic.
        busy = sum(r["device_ms"] for r in rows) or 1.0
        low = [r for r in rows
               if r["mean_occupancy"] is not None
               and r["mean_occupancy"] < 0.3
               and r["device_ms"] / busy > 0.05]
        if low:
            worst = low[0]
            rep.add(
                "medium", "occupancy",
                f"{len(low)} kernel(s) holding >5% of device time run below 0.3 "
                f"occupancy; worst is {worst['mean_occupancy']:.2f} on "
                f"{worst['kernel'][:60]} ({worst['device_ms']:.1f} ms). Low "
                f"occupancy on a hot kernel points at block shape or register "
                f"pressure, not at the arithmetic.",
                "; ".join(f"{r['kernel'][:40]}={r['mean_occupancy']:.2f}"
                          for r in low[:6]),
            )

    # ---- per-kernel distribution ----
    kstats: dict[str, list[float]] = defaultdict(list)
    for k in kernels:
        kstats[k.get("name", "?")].append(float(k.get("dur", 0.0)))
    ranked = sorted(kstats.items(), key=lambda kv: -sum(kv[1]))
    total_kernel_us = sum(sum(v) for v in kstats.values())
    rep.sections["kernel_totals"] = {
        "distinct_kernels": len(kstats),
        "launches": len(kernels),
        "device_busy_ms": round(total_kernel_us * US * 1e3, 3),
        "memcpy_count": len(memcpy),
        "memset_count": len(memset),
    }
    rep.sections["top_kernels"] = []
    for name, durs in ranked[:top]:
        s = summarize(durs)
        rep.sections["top_kernels"].append({
            "kernel": name[:110],
            "launches": s["count"],
            "total_ms": round(s["total"] * US * 1e3, 3),
            "share_pct": round(pct(s["total"], total_kernel_us), 1),
            "mean_us": round(s["mean"], 2),
            "median_us": round(s["median"], 2),
            "min_us": round(s["min"], 2),
            "max_us": round(s["max"], 2),
            "p95_us": round(s["p95"], 2),
            "cv": round(s["cv"], 3),
        })

    # ---- launch-bound detection ----
    tiny = [(n, len(d), sum(d) / len(d)) for n, d in kstats.items()
            if sum(d) / len(d) < LAUNCH_BOUND_US]
    tiny_launches = sum(c for _, c, _ in tiny)
    if kernels:
        frac = tiny_launches / len(kernels)
        rep.sections["launch_bound"] = {
            "tiny_kernel_launches": tiny_launches,
            "fraction_of_launches": round(frac, 3),
            "threshold_us": LAUNCH_BOUND_US,
            "worst": [{"kernel": n[:90], "launches": c, "mean_us": round(mu, 2)}
                      for n, c, mu in sorted(tiny, key=lambda x: -x[1])[:top]],
        }
        if frac > 0.5 and tiny_launches > 200:
            rep.add("high", "launch-bound",
                    f"{frac * 100:.0f}% of {len(kernels)} launches run for under "
                    f"{LAUNCH_BOUND_US} us each -- the run is dispatch-bound, not "
                    f"compute-bound; batching or fusing the per-layer loop is the lever",
                    f"{tiny_launches} tiny launches across {len(tiny)} distinct kernels")

    # ---- GPU idle: wall span vs union of kernel intervals ----
    if kernels:
        iv = sorted((float(k["ts"]), float(k["ts"]) + float(k.get("dur", 0.0)))
                    for k in kernels)
        span_lo, span_hi = iv[0][0], max(e for _, e in iv)
        busy, cur_s, cur_e = 0.0, iv[0][0], iv[0][1]
        for s, e in iv[1:]:
            if s > cur_e:
                busy += cur_e - cur_s
                cur_s, cur_e = s, e
            else:
                cur_e = max(cur_e, e)
        busy += cur_e - cur_s
        wall = span_hi - span_lo
        idle = 1.0 - (busy / wall) if wall > 0 else 0.0
        rep.sections["gpu_utilisation"] = {
            "wall_ms": round(wall * US * 1e3, 3),
            "busy_ms": round(busy * US * 1e3, 3),
            "idle_fraction": round(idle, 3),
        }
        if idle > IDLE_FRACTION_WARN:
            rep.add("high", "gpu-idle",
                    f"GPU is idle {idle * 100:.0f}% of the profiled span",
                    "either the host cannot launch fast enough (dispatch-bound) or it "
                    "keeps stopping to wait for results (see gpu-sync findings)")

    # ---- explicit synchronisation from the runtime layer ----
    sync_counts = Counter()
    sync_time = defaultdict(float)
    for r in runtime:
        n = r.get("name", "")
        if n in SYNC_RUNTIME:
            sync_counts[n] += 1
            sync_time[n] += float(r.get("dur", 0.0))
    if sync_counts:
        rep.sections["runtime_sync_calls"] = [
            {"call": n, "count": c, "total_ms": round(sync_time[n] * US * 1e3, 3),
             "why": SYNC_RUNTIME[n]}
            for n, c in sync_counts.most_common(top)
        ]
        blocking = sum(c for n, c in sync_counts.items()
                       if n in ("cudaStreamSynchronize", "cudaDeviceSynchronize",
                                "cudaMemcpy", "cudaEventSynchronize"))
        if blocking:
            rep.add("medium" if blocking < 50 else "high", "gpu-sync",
                    f"{blocking} blocking CUDA runtime call(s) in the profiled span",
                    "; ".join(f"{n} x{c}" for n, c in sync_counts.most_common(6)))

    # ---- device-to-host copies: the other way control leaves the GPU ----
    d2h = [e for e in memcpy if "DtoH" in e.get("name", "")]
    if d2h:
        durs = [float(e.get("dur", 0.0)) for e in d2h]
        s = summarize(durs)
        rep.sections["device_to_host_copies"] = {
            "count": s["count"], "total_ms": round(s["total"] * US * 1e3, 3),
            "mean_us": round(s["mean"], 2), "max_us": round(s["max"], 2),
        }
        if s["count"] > 20:
            rep.add("medium", "gpu-sync",
                    f"{s['count']} device-to-host copies -- each one is a point where the "
                    f"host waits on the device",
                    f"total {s['total'] * US * 1e3:.2f} ms")

    # ---- python-side operator frequency (dispatch pressure) ----
    cpu_ops = by_cat.get("cpu_op", [])
    if cpu_ops:
        freq = Counter(e.get("name", "?") for e in cpu_ops)
        rep.sections["most_frequent_cpu_ops"] = [
            {"op": n, "calls": c} for n, c in freq.most_common(top)
        ]
        rep.sections["cpu_op_totals"] = {"calls": len(cpu_ops), "distinct": len(freq)}


# ================================================= CUDA memory snapshot
def analyse_snapshot(path: Path, rep: Report, top: int) -> None:
    if not path.exists():
        return
    try:
        snap = pickle.loads(path.read_bytes())
    except Exception as e:  # noqa: BLE001
        rep.sections["memory_snapshot_error"] = repr(e)[:200]
        return
    sites: dict[str, list[int]] = defaultdict(list)
    for seg in snap.get("segments", []):
        for b in seg.get("blocks", []):
            if b.get("state") != "active_allocated":
                continue
            frames = b.get("frames") or []
            where = " <- ".join(
                f"{(fr.get('filename') or '?').replace(chr(92), '/').split('/')[-1]}"
                f":{fr.get('line', '?')}:{fr.get('name', '?')}"
                for fr in frames[:3]
            ) or "unknown"
            sites[where].append(int(b.get("size", 0)))
    if not sites:
        return
    ranked = sorted(sites.items(), key=lambda kv: -sum(kv[1]))
    total = sum(sum(v) for v in sites.values())
    rep.sections["live_allocation_sites"] = [
        {"site": s[:150], "blocks": len(v), "total_mib": round(sum(v) / MiB, 2),
         "share_pct": round(pct(sum(v), total), 1),
         "largest_mib": round(max(v) / MiB, 2)}
        for s, v in ranked[:top]
    ]
    rep.sections["live_allocation_total_mib"] = round(total / MiB, 2)
    if ranked:
        s, v = ranked[0]
        share = pct(sum(v), total)
        if share > 40:
            rep.add("high", "memory",
                    f"one allocation site holds {share:.0f}% of live device memory "
                    f"({sum(v) / MiB:.0f} MiB in {len(v)} blocks)", s[:160])


# ============================================================ rendering
def render(rep: Report, top: int) -> str:
    out: list[str] = []
    out.append("=" * 78)
    out.append(f"PROFILE STATISTICS -- {rep.run}")
    out.append("=" * 78)

    if rep.findings:
        order = {"high": 0, "medium": 1, "info": 2}
        out.append("\n## FINDINGS (ranked)\n")
        for f in sorted(rep.findings, key=lambda f: order[f.severity]):
            out.append(f.line())
    else:
        out.append("\n## FINDINGS\n   none triggered")

    def table(title: str, rows: list[dict]) -> None:
        if not rows:
            return
        out.append(f"\n## {title}\n")
        cols = list(rows[0].keys())
        widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
        out.append("  " + "  ".join(c.ljust(widths[c]) for c in cols))
        out.append("  " + "  ".join("-" * widths[c] for c in cols))
        for r in rows[:top]:
            out.append("  " + "  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))

    def block(title: str, obj) -> None:
        if obj in (None, {}, []):
            return
        out.append(f"\n## {title}\n")
        out.append("  " + json.dumps(obj, indent=2).replace("\n", "\n  "))

    s = rep.sections
    block("run totals", s.get("op_totals"))
    block("kernel totals", s.get("kernel_totals"))
    block("gpu utilisation", s.get("gpu_utilisation"))
    table("top kernels by device time", s.get("top_kernels", []))
    table("top ops by self CUDA time", s.get("top_ops_self_cuda", []))
    table("top ops by self CPU time", s.get("top_ops_self_cpu", []))
    table("most frequent CPU ops (dispatch pressure)", s.get("most_frequent_cpu_ops", []))
    block("launch-boundedness", s.get("launch_bound"))
    block("kernel launch geometry / occupancy", s.get("kernel_launch_geometry"))
    table("sync-forcing ops (GPU exits to host)", s.get("sync_forcing_ops", []))
    table("blocking CUDA runtime calls", s.get("runtime_sync_calls", []))
    block("device-to-host copies", s.get("device_to_host_copies"))
    table("top ops by CUDA allocation", s.get("top_ops_by_cuda_alloc", []))
    table("live allocation sites", s.get("live_allocation_sites", []))
    block("driver peak (max MiB)", s.get("driver_peak_max_mib"))
    block("route peak spread", s.get("route_peak_spread_by_driver"))
    block("scaling exponents", s.get("scaling_exponents"))
    table("identity peak vs theoretical workspace",
          s.get("identity_overhead_vs_theory", []))
    if s.get("oom_rows"):
        table("OOM configurations", s["oom_rows"])

    meas = s.get("suite_measurements", {})
    if meas:
        out.append(f"\n## suite measurements ({len(meas)} groups; showing the "
                   f"{top} with the widest spread)\n")
        ranked = sorted(meas.items(), key=lambda kv: -(kv[1].get("cv") or 0))[:top]
        rows = [{"group": k[:70], "n": v["count"], "mean": round(v["mean"], 3),
                 "median": round(v["median"], 3), "min": round(v["min"], 3),
                 "max": round(v["max"], 3), "p95": round(v["p95"], 3),
                 "cv": round(v["cv"], 3)} for k, v in ranked]
        cols = list(rows[0].keys())
        widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
        out.append("  " + "  ".join(c.ljust(widths[c]) for c in cols))
        out.append("  " + "  ".join("-" * widths[c] for c in cols))
        for r in rows:
            out.append("  " + "  ".join(str(r[c]).ljust(widths[c]) for c in cols))

    return "\n".join(out)


def analyse_run(run_dir: Path, top: int) -> Report:
    rep = Report(run=str(run_dir))
    manifest = run_dir / "manifest.json"
    if manifest.exists():
        man = json.loads(manifest.read_text())
        rep.sections["run_identity"] = {
            "tag": man.get("tag"), "git_sha": man.get("git", {}).get("sha"),
            "dirty": man.get("git", {}).get("dirty"),
            "host": man.get("env", {}).get("host"),
            "gpu": man.get("env", {}).get("gpu"),
            "torch": man.get("env", {}).get("torch"),
        }
        if man.get("import_errors"):
            rep.add("high", "environment",
                    "some project imports failed; affected levels reported SKIPPED "
                    "and produced no data",
                    json.dumps(man["import_errors"])[:300])

    analyse_rows(load_rows(run_dir / "rows.csv"), rep, top)
    analyse_ops_csv(run_dir / "profiler_ops.csv", rep, top)
    for candidate in ("profiler_trace.json", "profiler_trace.json.gz"):
        p = run_dir / candidate
        if p.exists():
            analyse_trace(p, rep, top)
            break
    analyse_snapshot(run_dir / "memory_snapshot.pickle", rep, top)

    viol = run_dir / "violations.json"
    if viol.exists():
        items = json.loads(viol.read_text())
        failed = [i for i in items if not i["ok"]]
        rep.sections["violations"] = {
            "passed": len(items) - len(failed), "total": len(items),
            "failed": [{"check": i["check"], "detail": i["detail"],
                        "severity": i["severity"]} for i in failed],
        }
        for i in failed:
            rep.add("high" if i["severity"] == "error" else "medium",
                    "violation", i["check"], i["detail"])
    return rep


def compare(reports: list[Report]) -> str:
    """Before/after on the quantities that decide whether a fix worked."""
    out = ["\n" + "=" * 78, "COMPARISON", "=" * 78]
    keys = ["driver_peak_max_mib", "route_peak_spread_by_driver", "scaling_exponents",
            "gpu_utilisation", "kernel_totals", "op_totals",
            "cuda_concentration_top5_pct", "live_allocation_total_mib"]
    for k in keys:
        present = [(r.run.split("/")[-1].split("\\")[-1], r.sections.get(k))
                   for r in reports if r.sections.get(k) not in (None, {}, [])]
        if len(present) < 2:
            continue
        out.append(f"\n## {k}")
        for name, val in present:
            out.append(f"  {name}:")
            out.append("    " + json.dumps(val, indent=2).replace("\n", "\n    "))
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("run_dirs", nargs="+", type=Path,
                   help="one or more results/<tag>/ directories")
    p.add_argument("--top", type=int, default=15, help="rows per table")
    p.add_argument("--json", type=Path, default=None,
                   help="also write the full structured report here")
    p.add_argument("--compare", action="store_true",
                   help="emit a before/after section across the given runs")
    args = p.parse_args()

    reports = []
    for d in args.run_dirs:
        if not d.exists():
            print(f"[skip] {d} does not exist", file=sys.stderr)
            continue
        rep = analyse_run(d, args.top)
        reports.append(rep)
        print(render(rep, args.top))

    if args.compare and len(reports) > 1:
        print(compare(reports))

    if args.json:
        payload = [{"run": r.run, "sections": r.sections,
                    "findings": [vars(f) for f in r.findings]} for r in reports]
        args.json.write_text(json.dumps(payload, indent=2))
        print(f"\n[wrote] {args.json}")

    if not reports:
        sys.exit(1)


if __name__ == "__main__":
    main()
