"""
run_all.py -- execute the full Phase 2.5 experiment suite in order and bundle
the results for analysis. Cross-platform (tested logic on Linux; paths via
pathlib, subprocess via sys.executable, so it runs the same on Windows).

Usage:
    python run_all.py            # full suite (CIFAR-10, ~60-90 min on a 5070 Ti)
    python run_all.py --quick    # 5-10 min sanity pass (reduced epochs/sweeps)
    python run_all.py --only engines decompose   # run a subset

Produces:
    results/           all PNGs, JSONs, tables, per-step logs, env.json
    results_bundle.zip everything zipped -- this is the file to share back.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
RESULTS = HERE / "results"
BENCH = str(HERE / "iwrm_bench.py")


def dump_env():
    RESULTS.mkdir(parents=True, exist_ok=True)
    info = {"python": sys.version}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            info["gpu"] = {"name": p.name, "total_memory_MiB": p.total_memory // 2**20,
                           "sm": f"{p.major}.{p.minor}", "SMs": p.multi_processor_count}
            info["cuda_runtime"] = torch.version.cuda
    except Exception as e:  # noqa: BLE001
        info["torch_error"] = repr(e)
    try:
        import importlib.metadata as md
        info["torchjd"] = md.version("torchjd")
    except Exception:
        pass
    (RESULTS / "env.json").write_text(json.dumps(info, indent=2))
    print("Environment:", json.dumps(info, indent=2))


def run(name: str, cmd_args: list[str]) -> bool:
    log_dir = RESULTS / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    full = [sys.executable, "-u", BENCH, *cmd_args, "--out", str(RESULTS)]
    print(f"\n{'='*70}\n[{name}] {' '.join(full)}\n{'='*70}", flush=True)
    t0 = time.perf_counter()
    with open(log_dir / f"{name}.log", "w", encoding="utf-8") as log:
        log.write(f"$ {' '.join(full)}\n\n")
        proc = subprocess.Popen(full, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
        proc.wait()
        dt = time.perf_counter() - t0
        log.write(f"\n--- exit {proc.returncode} in {dt:.1f}s ---\n")
    if proc.returncode != 0:
        print(f"[{name}] FAILED (exit {proc.returncode}) -- see results/logs/{name}.log")
        return False
    print(f"[{name}] OK in {dt:.1f}s")
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="reduced settings, ~5-10 min")
    p.add_argument("--dataset", default="cifar10", choices=["cifar10", "svhn", "synthetic"])
    p.add_argument("--only", nargs="*", default=None,
                   help="subset of: preflight figure2 table7 engines decompose scaling")
    args = p.parse_args()

    d = ["--dataset", args.dataset]
    if args.quick:
        suite = {
            "preflight": ["preflight", *d, "--k-steps", "2"],
            "figure2":   ["figure2", *d, "--epochs", "5", "--aggs", "Mean", "UPGrad"],
            "table7":    ["table7", *d, "--warmup", "1", "--timed", "3"],
            "engines":   ["engines", *d, "--warmup", "1", "--timed", "3"],
            "decompose": ["decompose", *d, "--widths", "1", "--batch-sizes", "8", "32",
                          "--warmup", "1", "--timed", "2"],
            "scaling":   ["scaling", *d, "--widths", "1", "2", "--warmup", "0", "--timed", "1"],
        }
    else:
        suite = {
            "preflight": ["preflight", *d],
            # Deliverable A: per-aggregator lr sweep + full paper horizon (20 epochs)
            "figure2":   ["figure2", *d, "--sweep"],
            # Deliverable B: Table 7 ratios
            "table7":    ["table7", *d, "--warmup", "3", "--timed", "10"],
            # Deliverable C: 4-way engine comparison incl. optimize_gramian_computation
            "engines":   ["engines", *d, "--warmup", "3", "--timed", "10"],
            # Phase-3 gate data: step decomposition across m and width
            "decompose": ["decompose", *d, "--widths", "1", "4", "8",
                          "--batch-sizes", "4", "8", "16", "32", "64",
                          "--warmup", "1", "--timed", "3"],
            # Stretch: memory scaling; w=16 may OOM autojac on 12 GB -- that's the point
            "scaling":   ["scaling", *d, "--widths", "1", "2", "4", "8", "16",
                          "--warmup", "1", "--timed", "3"],
        }

    dump_env()
    names = args.only if args.only else list(suite)
    status = {}
    for name in names:
        if name not in suite:
            print(f"unknown step {name!r}; choices: {list(suite)}"); continue
        status[name] = run(name, suite[name])
        if name == "preflight" and not status[name]:
            print("Preflight FAILED -- aborting; nothing downstream is trustworthy.")
            break

    (RESULTS / "status.json").write_text(json.dumps(status, indent=2))
    bundle = shutil.make_archive(str(HERE / "results_bundle"), "zip", RESULTS)
    print(f"\n{'='*70}\nDone. Status: {status}\nBundle to share: {bundle}\n{'='*70}")


if __name__ == "__main__":
    main()
