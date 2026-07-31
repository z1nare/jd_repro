"""Experiment identity and provenance. Import from every harness.

Solves the "the cluster is full of directories and nobody knows which code made
them" problem by making an untagged artifact impossible to produce:

* every run gets a tag ``v{VERSION}_{YYYYMMDD-HHMM}_{name}_{gitsha}{+dirty}``
* every artifact lands under ``{out_root}/{tag}/`` -- never a shared directory
* ``manifest.json`` in that directory records the git SHA, the full diff when the
  tree is dirty, the environment, the GPU, and argv
* ``{out_root}/runs_index.csv`` gets one row per run, so ``tail runs_index.csv``
  says what every directory on the box actually is

``VERSION`` is bumped **by hand** whenever engine semantics change (a fix, a new
route, a dtype default). The date disambiguates repeats; the git SHA is ground
truth; ``+dirty`` plus the stored diff makes even an uncommitted run reproducible.

Usage::

    from runtag import RunContext
    rc = RunContext(version=5, name="leakfix", notes="squashed driver")
    rc.path("rows.csv")          # {out_root}/{tag}/rows.csv
    rc.save_manifest({"m": 8})
    rc.finalize(status="ok")
"""

from __future__ import annotations

import csv
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

DEFAULT_OUT_ROOT = Path(os.environ.get("JDGRAM_RESULTS", "results"))
MAX_DIFF_CHARS = 200_000
#: Refuse to start a run below this, rather than dying halfway through a sweep.
MIN_FREE_GIB = 2.0


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return ""


def git_state() -> dict:
    """SHA, branch, and -- when dirty -- enough to reconstruct the exact tree.

    ``git status --porcelain`` reports untracked files as ``??``, so a run with
    new-but-uncommitted scripts is correctly marked dirty. That matters: a
    ``git commit -am`` does not stage untracked files, so a tag that ignored them
    would claim a provenance it does not have.
    """
    sha = _git("rev-parse", "--short", "HEAD")
    porcelain = _git("status", "--porcelain")
    lines = porcelain.splitlines()
    dirty = bool(porcelain.strip())
    if not sha:
        # No .git here. That is the normal case on a compute node synced by
        # rsync, and it silently defeats the whole point of tagging: every run
        # reads "nogit" and nothing records which code produced which number.
        # Pass the sha from the machine that has the repo:
        #   JDGRAM_GIT_SHA=$(git rev-parse --short HEAD) bash scripts/...
        sha = os.environ.get("JDGRAM_GIT_SHA", "").strip() or "nogit"
        dirty = os.environ.get("JDGRAM_GIT_DIRTY", "").strip().lower() in (
            "1", "true", "yes"
        )
        if sha == "nogit":
            print("[runtag] WARNING: no git metadata and JDGRAM_GIT_SHA unset -- "
                  "this run cannot be traced back to a code version.")
    return {
        "sha": sha,
        "dirty": dirty,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty_files": [ln[3:] for ln in lines][:200],
        "untracked_count": sum(1 for ln in lines if ln.startswith("??")),
        "diff": _git("diff")[:MAX_DIFF_CHARS] if dirty else "",
    }


def env_state() -> dict:
    info = {
        "host": socket.gethostname(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            info["gpu"] = p.name
            info["gpu_total_MiB"] = p.total_memory // 2**20
            info["gpu_count"] = torch.cuda.device_count()
            info["cuda_runtime"] = torch.version.cuda
            info["capability"] = f"{p.major}.{p.minor}"
    except Exception as e:  # noqa: BLE001
        info["torch_error"] = repr(e)
    for pkg in ("torchjd", "numpy", "pandas", "triton", "matplotlib"):
        try:
            import importlib.metadata as md

            info[pkg] = md.version(pkg)
        except Exception:
            pass
    return info


def free_gib(path: Path) -> float:
    try:
        return shutil.disk_usage(path).free / 2**30
    except Exception:
        return float("inf")


@dataclass
class RunContext:
    version: int
    name: str
    notes: str = ""
    root: Path = field(default_factory=lambda: DEFAULT_OUT_ROOT)
    require_free_gib: float = MIN_FREE_GIB
    _t0: float = field(default_factory=time.time, init=False)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        free = free_gib(self.root)
        if free < self.require_free_gib:
            raise RuntimeError(
                f"only {free:.2f} GiB free under {self.root.resolve()}; need "
                f"{self.require_free_gib} GiB. Point --out-root at a roomier "
                f"filesystem or free space before starting a sweep."
            )
        g = git_state()
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        dirty_mark = "+dirty" if g["dirty"] else ""
        self.tag = f"v{self.version}_{stamp}_{self.name}_{g['sha']}{dirty_mark}"
        self.git = g
        self.env = env_state()
        self.out_dir = self.root / self.tag
        self.out_dir.mkdir(parents=True, exist_ok=True)
        if g["dirty"]:
            print(
                f"[runtag] tree is DIRTY ({len(g['dirty_files'])} paths, "
                f"{g['untracked_count']} untracked). Full diff of tracked changes is in "
                f"{self.out_dir / 'manifest.json'} so this run stays reproducible."
            )
        print(f"[runtag] RUN_TAG = {self.tag}")
        print(f"[runtag] out_dir = {self.out_dir.resolve()}  ({free:.1f} GiB free)")

    # ---- artifact paths: nothing should ever write outside these -------------
    def path(self, *parts: str) -> Path:
        p = self.out_dir.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def save_manifest(self, config: dict | None = None) -> Path:
        man = {
            "tag": self.tag,
            "version": self.version,
            "name": self.name,
            "notes": self.notes,
            "started": datetime.fromtimestamp(self._t0).isoformat(timespec="seconds"),
            "argv": sys.argv,
            "cwd": os.getcwd(),
            "git": self.git,
            "env": self.env,
            "config": config or {},
        }
        p = self.path("manifest.json")
        p.write_text(json.dumps(man, indent=2))
        return p

    def finalize(self, status: str = "ok", summary: dict | None = None) -> None:
        """Append one row to the global index so every directory is identifiable."""
        dur = time.time() - self._t0
        row = {
            "tag": self.tag,
            "version": self.version,
            "name": self.name,
            "date": datetime.fromtimestamp(self._t0).isoformat(timespec="seconds"),
            "git_sha": self.git["sha"],
            "dirty": self.git["dirty"],
            "host": self.env.get("host", ""),
            "gpu": self.env.get("gpu", ""),
            "torch": self.env.get("torch", ""),
            "duration_s": round(dur, 1),
            "status": status,
            "notes": self.notes,
            "summary": json.dumps(summary or {})[:1000],
        }
        index = self.root / "runs_index.csv"
        index.parent.mkdir(parents=True, exist_ok=True)
        new = not index.exists()
        with open(index, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)
        print(
            f"[runtag] finalized: status={status} duration={dur:.1f}s "
            f"({free_gib(self.root):.1f} GiB free) -> {index}"
        )


def list_runs(n: int = 30, root: Path | str = DEFAULT_OUT_ROOT) -> None:
    """``python bench/runtag.py`` -- what is on this box?"""
    index = Path(root) / "runs_index.csv"
    if not index.exists():
        print(f"no runs indexed at {index}")
        return
    with open(index) as f:
        rows = list(csv.DictReader(f))
    print(f"{'date':<20} {'ver':<5} {'name':<20} {'sha':>9} {'':6} {'status':<8} {'dur':>8}  notes")
    for r in rows[-n:]:
        flag = " DIRTY" if r.get("dirty") == "True" else ""
        print(
            f"{r['date']:<20} v{r['version']:<4} {r['name']:<20} {r['git_sha']:>9}{flag:6} "
            f"{r['status']:<8} {r['duration_s']:>7}s  {r['notes'][:40]}"
        )


if __name__ == "__main__":
    list_runs(root=sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUT_ROOT)
