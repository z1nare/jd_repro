#!/usr/bin/env bash
# Profiling runbook for gala1 (8x RTX A5000 24 GB).
#
# Everything here runs ON THE CLUSTER. Nothing in this file should ever be run on
# a laptop -- the laptop is for `python -m compileall` and the CPU gate suite only.
#
#   ssh dice-gala
#   cd ~/jd-phase25-bench
#   bash scripts/run_profile_cluster.sh gates          # start here, always
#   bash scripts/run_profile_cluster.sh baseline
#   bash scripts/run_profile_cluster.sh stats
#
# Disk note: / was at 94.7% of 1.70 TB. JDGRAM_RESULTS below points results at a
# roomier filesystem if one is available; override it if not. runtag.py refuses to
# start a run with under 2 GiB free rather than dying halfway through a sweep.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="${ROOT}:${ROOT}/src:${ROOT}/bench${PYTHONPATH:+:$PYTHONPATH}"
export JDGRAM_RESULTS="${JDGRAM_RESULTS:-${ROOT}/results}"

# One GPU is enough and keeps the box free for others. Override with GPU=3 etc.
export CUDA_VISIBLE_DEVICES="${GPU:-0}"

# SDPA CUDA stubs on this host have needed this before.
export LD_LIBRARY_PATH="/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

PY="${PY:-python}"
VERSION="${VERSION:-7}"          # bump BY HAND when engine semantics change
STAGE="${1:-help}"
shift || true

banner() { echo; echo "=============== $* ==============="; echo; }

case "$STAGE" in

  env)
    banner "environment"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
    df -h "$JDGRAM_RESULTS" | tail -2
    $PY -c "import torch,sys;print('python',sys.version.split()[0]);print('torch',torch.__version__,'cuda',torch.cuda.is_available())"
    $PY -c "import torchjd,inspect,os;print('torchjd at',os.path.dirname(inspect.getfile(torchjd)))"
    ;;

  verify)
    # Prove the box is running the code you think it is. An rsync that
    # transferred nothing looks exactly like one that transferred everything.
    banner "preflight"
    $PY bench/preflight.py --strict
    ;;

  gates)
    # Correctness before any number is believed. CPU, fp64, seconds.
    banner "gates (must be green before profiling)"
    $PY -m pytest gates/ -q
    ;;

  baseline)
    # The decoupled profile. --isolate gives each level a fresh process so one
    # config's allocator state cannot inflate the next config's peak.
    banner "profile: L0-L5 decoupled"
    $PY bench/profile_suite.py --version "$VERSION" --name baseline \
        --levels L0 L1 L2 L3 L4 L5 --device cuda --dtype fp32 \
        --m 8 --T 512 --V 65 --n-embd 256 --n-layer 4 \
        --isolate --notes "squashed driver baseline" "$@"
    ;;

  scaling)
    # The complexity evidence: exponents in m and T, per driver and per route.
    banner "profile: L6 scaling laws"
    $PY bench/profile_suite.py --version "$VERSION" --name scaling \
        --levels L6 --device cuda --dtype fp32 --notes "exponent fits" "$@"
    ;;

  accuracy)
    banner "profile: L7 convergence"
    $PY bench/profile_suite.py --version "$VERSION" --name accuracy \
        --levels L7 --device cuda --dtype fp32 --steps 200 \
        --m 8 --T 128 --notes "loss curves vs autogram/autojac/sgd" "$@"
    ;;

  trace)
    # Kernel-level detail for profile_stats.py. --trace-raw is the big one; the
    # compact profiler_ops.csv is always written.
    banner "profile: L9 torch.profiler capture"
    $PY bench/profile_suite.py --version "$VERSION" --name trace \
        --levels L9 --device cuda --dtype fp32 \
        --m 8 --T 512 --V 65 --n-embd 256 --n-layer 4 \
        --record-shapes --trace-raw --notes "kernel + sync analysis" "$@"
    ;;

  snapshot)
    banner "profile: L8 CUDA memory snapshot"
    $PY bench/profile_suite.py --version "$VERSION" --name snapshot \
        --levels L8 --device cuda --dtype fp32 \
        --m 8 --T 512 --V 65 --n-embd 256 --n-layer 4 "$@"
    ;;

  vocab)
    # The regime jdgram is supposed to WIN: P_layer large enough that m*T^2 < P.
    banner "profile: large-vocab head (the T-first regime)"
    for V in 8000 32000 50257; do
      $PY bench/profile_suite.py --version "$VERSION" --name "vocab$V" \
          --levels L2 L4 L5 --device cuda --dtype fp32 \
          --m 8 --T 512 --V "$V" --n-embd 256 --n-layer 4 \
          --notes "vocab sweep V=$V" || echo "V=$V did not complete; continuing"
    done
    ;;

  qp)
    # Does the dual-cone QP belong on the GPU, and from which m? Sweeps m and
    # compares TorchJD's CPU-only quadprog projector against jacopt.
    banner "profile: L10 dual-cone QP backends"
    $PY bench/profile_suite.py --version "$VERSION" --name qp \
        --levels L10 --device cuda --dtype fp32 --qp-reps 50 \
        --notes "quadprog vs jacopt across m" "$@"
    ;;

  aggregators)
    # The Phase-B deliverable: every aggregator against every engine, scored on
    # held-out accuracy as well as cost. Needs the Shakespeare corpus for the
    # accuracy columns to mean anything.
    banner "profile: L11 aggregator x engine matrix"
    if [ ! -f "$ROOT/data/shakespeare_char/val.bin" ]; then
      echo "preparing shakespeare corpus first (held-out accuracy needs it)"
      $PY bench/prepare_shakespeare_char.py
    fi
    $PY bench/profile_suite.py --version "$VERSION" --name aggregators \
        --levels L11 --device cuda --dtype fp32 \
        --m 8 --T 128 --V 65 --n-embd 256 --n-layer 4 \
        --steps "${STEPS:-300}" --eval-batches 40 \
        --notes "Mean/UPGrad/MGDA/PCGrad x jdgram/autogram/autojac" "$@"
    ;;

  overnight)
    # Unattended, thorough, resumable. Every stage is guarded so one failure
    # cannot end the night, everything is tee'd to a log, and the run finishes
    # by bundling itself for transfer.
    LOG="${LOG:-$JDGRAM_RESULTS/overnight_v${VERSION}_$(date +%Y%m%d-%H%M%S).log}"
    mkdir -p "$(dirname "$LOG")"
    echo "logging to $LOG"
    {
      echo "=== overnight v$VERSION started $(date) ==="
      free_gib=$(df -BG --output=avail "$JDGRAM_RESULTS" | tail -1 | tr -dc '0-9')
      echo "free space: ${free_gib} GiB"
      if [ "${free_gib:-0}" -lt 4 ]; then
        echo "REFUSING: under 4 GiB free. Clear results/v6_* or old archives first."
        exit 1
      fi

      run_stage() {
        echo; echo "########## $* ##########"; echo
        if ! "$0" "$@"; then
          echo "!!! stage '$*' FAILED (exit $?) -- continuing"
        fi
      }

      run_stage env
      run_stage verify
      run_stage gates
      run_stage baseline
      run_stage scaling
      run_stage trace

      # The dual-cone QP twice: as the box is, and pinned to one CPU thread.
      # The m>=32 cliff on the CPU arm tracks thread count, not problem size, and
      # a single-threaded control is the only way to say that rather than guess.
      run_stage qp
      echo; echo "########## qp (single-threaded control) ##########"; echo
      $PY bench/profile_suite.py --version "$VERSION" --name qp-1thread           --levels L10 --device cuda --dtype fp32 --qp-reps 50 --qp-threads 1           --notes "single-threaded CPU control for the m>=32 cliff"           || echo "!!! qp-1thread FAILED -- continuing"

      # Longer training than the 5-minute campaign affords.
      export STEPS="${STEPS:-1000}"
      run_stage aggregators
      run_stage accuracy

      # The regime jdgram's whole argument lives in: P_layer large enough that
      # m*T^2 < P, i.e. a real vocabulary head. Not covered by any other stage.
      run_stage vocab

      run_stage snapshot
      run_stage stats
      run_stage bundle
      echo; echo "=== overnight v$VERSION finished $(date) ==="
    } 2>&1 | tee -a "$LOG"
    echo
    echo "log: $LOG"
    ;;

  campaign)
    # The full tagged run. Each stage writes to its own results/<tag>/ directory
    # and appends one row to results/runs_index.csv, so nothing can be confused
    # with anything else afterwards.
    "$0" env
    "$0" verify
    "$0" gates
    "$0" baseline
    "$0" scaling
    "$0" qp
    "$0" aggregators
    "$0" trace
    "$0" stats
    ;;

  all)
    "$0" campaign
    ;;

  stats)
    # Compact statistics for analysis. This is the artifact to send back --
    # never the raw trace.
    banner "statistics"
    # Select by VERSION, not by "the last N directories". A campaign with
    # --isolate writes one directory per level (6 for baseline alone) plus one
    # per stage, so a tail of 6 silently drops L0-L3 and the report reads as
    # though those levels were never run. Set N to fall back to a tail.
    if [ -n "${N:-}" ]; then
      mapfile -t RUNS < <(ls -d "$JDGRAM_RESULTS"/v*/ 2>/dev/null | tail -n "$N")
    else
      mapfile -t RUNS < <(ls -d "$JDGRAM_RESULTS"/v${VERSION}_*/ 2>/dev/null)
    fi
    if [ "${#RUNS[@]}" -eq 0 ]; then
      echo "no runs matching v${VERSION}_* under $JDGRAM_RESULTS"
      echo "(set N=<count> to aggregate the last N runs of any version instead)"
      exit 1
    fi
    echo "aggregating ${#RUNS[@]} run(s) for version $VERSION"
    $PY bench/profile_stats.py "${RUNS[@]}" --top 20 \
        --json "$JDGRAM_RESULTS/stats_latest.json" | tee "$JDGRAM_RESULTS/stats_latest.txt"
    echo
    echo "send back:  $JDGRAM_RESULTS/stats_latest.txt"
    echo "            $JDGRAM_RESULTS/stats_latest.json"
    echo "            $JDGRAM_RESULTS/runs_index.csv"
    ;;

  bundle)
    # One archive with everything needed to reconstruct the analysis, and
    # nothing that can be regenerated. Raw chrome traces and memory snapshots are
    # excluded by default -- they are the bulk of the bytes and the compact
    # profiler_ops.csv already carries what profile_stats.py reads. Set
    # WITH_TRACES=1 to include them.
    banner "bundle results for transfer"
    STAMP="$(date +%Y%m%d-%H%M%S)"
    OUT="${BUNDLE_OUT:-$HOME/jdgram-results-v${VERSION}-${STAMP}.tar.gz}"
    if [ ! -d "$JDGRAM_RESULTS" ]; then
      echo "no results directory at $JDGRAM_RESULTS -- nothing to bundle"; exit 1
    fi
    EXCLUDES=(--exclude='*.pickle')
    if [ "${WITH_TRACES:-0}" != "1" ]; then
      EXCLUDES+=(--exclude='profiler_trace.json' --exclude='profiler_trace.json.gz')
    fi
    cd "$JDGRAM_RESULTS"
    mapfile -t DIRS < <(ls -d v${VERSION}_*/ 2>/dev/null)
    if [ "${#DIRS[@]}" -eq 0 ]; then
      echo "no v${VERSION}_* runs to bundle"; exit 1
    fi
    EXTRA=()
    for f in runs_index.csv stats_latest.txt stats_latest.json; do
      [ -f "$f" ] && EXTRA+=("$f")
    done
    tar czf "$OUT" "${EXCLUDES[@]}" "${DIRS[@]}" "${EXTRA[@]}"
    cd "$ROOT"
    echo
    echo "wrote $OUT  ($(du -h "$OUT" | cut -f1), ${#DIRS[@]} run dirs)"
    echo
    echo "pull it to the laptop with:"
    echo "  rsync -av dice-gala:${OUT} ./"
    echo
    echo "verify after transfer:  tar tzf \$(basename ${OUT}) | head"
    ;;

  index)
    $PY bench/runtag.py "$JDGRAM_RESULTS"
    ;;

  *)
    cat <<'EOF'
usage: bash scripts/run_profile_cluster.sh <stage>

  env          GPU / disk / torch / torchjd fingerprint
  verify       preflight: is the synced code the fixed code?
  gates        correctness suite -- run this first, every time
  baseline     L0-L5 decoupled profile, one process per level
  scaling      L6 exponent fits in m and T, per driver and route
  qp           L10 dual-cone QP: TorchJD quadprog vs jacopt, swept over m
  aggregators  L11 Mean/UPGrad/MGDA/PCGrad x jdgram/autogram/autojac, with
               held-out val CE and next-token accuracy
  accuracy     L7 convergence vs autogram / autojac / sgd
  trace        L9 torch.profiler capture (feeds profile_stats.py)
  snapshot     L8 CUDA memory history (large file; disk-guarded)
  vocab        large-vocab sweep -- the regime T-first should win
  campaign     env -> gates -> baseline -> scaling -> qp -> aggregators
               -> trace -> stats   (the fast tagged run, ~5 min)
  overnight    everything campaign does, plus a single-threaded QP
               control, a longer L11, L7 accuracy, the vocab sweep,
               L8 snapshot, and a bundle. Guarded, logged, resumable.
  stats        aggregate the last N runs into compact statistics
  bundle       tar.gz every v$VERSION run + the stats, ready to pull off the box
  index        list every run recorded on this box

Every stage writes to results/<tag>/ where tag is
v{VERSION}_{YYYYMMDD-HHMMSS}_{name}_{gitsha}{+dirty}, and appends a row to
results/runs_index.csv. Decontaminate first: bash scripts/clean_cluster.sh

env vars: GPU=0  VERSION=7  PY=python  JDGRAM_RESULTS=/path  STEPS=300
          N=<count> overrides version-matching with a plain tail
EOF
    ;;
esac
