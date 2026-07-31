"""Preflight: prove the cluster is running the code you think it is.

An rsync that transfers nothing looks identical to an rsync that transferred
everything, and a partially-synced tree fails in ways that look like results --
an old engine happily produces numbers, they are just the wrong numbers. This
checks the specific things that must be true before a benchmark run is worth
starting, and names the file to re-sync when one is not.

    python bench/preflight.py            # check
    python bench/preflight.py --strict   # exit non-zero on any failure

Every check is cheap and CPU-only.
"""

from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_ROOT), str(_ROOT / "src"), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

CHECKS: list[tuple[str, str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "", fix: str = "", required: bool = True) -> bool:
    CHECKS.append((name, detail, ok, fix))
    mark = "PASS" if ok else ("FAIL" if required else "WARN")
    print(f"  [{mark}] {name:<44} {detail}")
    if not ok and fix:
        print(f"         -> {fix}")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    print("=" * 74)
    print(f"PREFLIGHT  root={_ROOT}")
    print("=" * 74)
    print(f"  python {sys.version.split()[0]}")

    # ---- environment -------------------------------------------------------
    try:
        import torch

        check("torch importable", True, torch.__version__)
        check("CUDA available", torch.cuda.is_available(),
              torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU",
              "export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}")
    except Exception as e:  # noqa: BLE001
        check("torch importable", False, repr(e)[:70])
        torch = None  # type: ignore[assignment]

    try:
        import torchjd  # noqa: F401

        check("torchjd importable", True)
    except Exception as e:  # noqa: BLE001
        check("torchjd importable", False, repr(e)[:70])

    # ---- the engine fixes: each maps to one file that must have synced -----
    print("\n  engine version:")
    try:
        from jdgram.engine.hooks import compute_gramian

        params = inspect.signature(compute_gramian).parameters
        check("compute_gramian has driver=", "driver" in params,
              f"params: {'driver' if 'driver' in params else 'MISSING'}",
              "re-sync src/jdgram/engine/hooks.py")
        src = inspect.getsource(compute_gramian.__wrapped__
                                if hasattr(compute_gramian, "__wrapped__")
                                else compute_gramian)
        check("squashed is the default driver", "squashed" in src,
              "", "re-sync src/jdgram/engine/hooks.py")
    except Exception as e:  # noqa: BLE001
        check("compute_gramian importable", False, repr(e)[:70],
              "re-sync src/jdgram/engine/hooks.py")

    try:
        from jdgram.engine.accumulate import GramianAccumulator, SharedGroup  # noqa: F401

        check("streaming accumulator present", True, "jdgram.engine.accumulate")
    except Exception as e:  # noqa: BLE001
        check("streaming accumulator present", False, repr(e)[:70],
              "re-sync src/jdgram/engine/accumulate.py (NEW FILE)")

    try:
        from jdgram.identities import linear as lin

        s = inspect.getsource(lin.sequence_gramian)
        # Match the computation, not the prose: the new docstring quotes the old
        # einsum to explain what was removed, so searching for that string finds
        # the fixed version too.
        body = s.split('"""')[-1]
        blocked = "for i in range(m)" in body and "(k_a * k_x)" in body
        check("linear T-first is blocked (13x memory fix)", blocked,
              "blocked over objective index" if blocked
              else "still the whole-kernel einsum",
              "re-sync src/jdgram/identities/linear.py")
    except Exception as e:  # noqa: BLE001
        check("linear identity importable", False, repr(e)[:70])

    try:
        from jdgram.identities import embedding as emb

        s = inspect.getsource(emb.sequence_gramian_dfirst)
        check("embedding d-first avoids einsum clones", "reshape(m, -1)" in s,
              "", "re-sync src/jdgram/identities/embedding.py")
    except Exception as e:  # noqa: BLE001
        check("embedding identity importable", False, repr(e)[:70])

    try:
        from models.configs import forward_logits

        p = inspect.signature(forward_logits).parameters
        ok = "batched_positions" in p and p["batched_positions"].default is True
        check("forward_logits batches positions", ok,
              "", "re-sync models/configs.py")
    except Exception as e:  # noqa: BLE001
        check("models.configs importable", False, repr(e)[:70])

    # ---- benchmark harness -------------------------------------------------
    print("\n  harness:")
    for mod, path in (("runtag", "bench/runtag.py"),
                      ("profile_suite", "bench/profile_suite.py"),
                      ("profile_stats", "bench/profile_stats.py"),
                      ("qp_backends", "bench/qp_backends.py")):
        try:
            __import__(mod)
            check(f"{mod} importable", True)
        except Exception as e:  # noqa: BLE001
            check(f"{mod} importable", False, repr(e)[:70], f"re-sync {path}")

    try:
        import profile_suite as ps

        check("profile_suite has L10+L11", "L11" in ps.ALL_LEVELS,
              f"levels: {ps.ALL_LEVELS[-3:]}", "re-sync bench/profile_suite.py")
        check("profile_suite imports cleanly", not ps.IMPORT_ERRORS,
              str(ps.IMPORT_ERRORS)[:60] if ps.IMPORT_ERRORS else "no import errors")
    except Exception:  # noqa: BLE001
        pass

    # ---- jacopt (optional arm) --------------------------------------------
    print("\n  jacopt (optional):")
    try:
        import jacopt
        from jacopt.qp import _admm_qp_batched  # noqa: F401

        check("jacopt importable", True, Path(jacopt.__file__).parent.as_posix())
        check("jacopt has the batched dual-cone solve", True,
              "one Cholesky for all m rows")
    except SyntaxError as e:  # noqa: BLE001
        check("jacopt importable", False, f"SyntaxError {e.filename}:{e.lineno}",
              "re-sync jacopt/src/jacopt/_backend/_loops.py (PEP 695 -> TypeVar) "
              "and jacopt/pyproject.toml (requires-python >=3.10)", required=False)
    except ImportError as e:  # noqa: BLE001
        check("jacopt importable", False, repr(e)[:70],
              "pip install -e ~/jacopt  (or add ~/jacopt/src to PYTHONPATH)",
              required=False)
    except Exception as e:  # noqa: BLE001
        check("jacopt batched solve present", False, repr(e)[:70],
              "re-sync jacopt/src/jacopt/qp.py", required=False)

    # ---- line endings ------------------------------------------------------
    # The repo is edited on Windows and run on Linux. A shell script that arrives
    # with CRLF fails on its first line with "invalid option name" or
    # "$'\r': command not found", which reads like a corrupt script rather than a
    # transfer problem. Cheap to check, expensive to diagnose at 3am.
    print("\n  line endings:")
    crlf = []
    for pat in ("scripts/*.sh", "bench/*.py", "gates/*.py", "src/jdgram/**/*.py"):
        for f in _ROOT.glob(pat):
            try:
                if b"\r\n" in f.read_bytes():
                    crlf.append(f.relative_to(_ROOT).as_posix())
            except OSError:
                pass
    check("no CRLF in scripts or sources", not crlf,
          f"{len(crlf)} file(s) with CRLF" if crlf else "all LF",
          f"sed -i 's/\\r$//' {' '.join(crlf[:4])}" if crlf else "")

    # ---- data --------------------------------------------------------------
    print("\n  data:")
    for split in ("train", "val"):
        p = _ROOT / "data" / "shakespeare_char" / f"{split}.bin"
        check(f"shakespeare {split}.bin", p.exists(),
              f"{p.stat().st_size / 2**20:.1f} MiB" if p.exists() else "missing",
              "python bench/prepare_shakespeare_char.py", required=False)

    import shutil

    free = shutil.disk_usage(_ROOT).free / 2**30
    check("disk headroom", free > 2.0, f"{free:.1f} GiB free",
          "rm -rf ~/jd-phase25-archive-*  (after rescuing anything you want)")

    # ---- verdict -----------------------------------------------------------
    failed = [c for c in CHECKS if not c[2]]
    print("\n" + "=" * 74)
    print(f"{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    if failed:
        print("failed: " + ", ".join(c[0] for c in failed))
    print("=" * 74)
    if args.strict and failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
