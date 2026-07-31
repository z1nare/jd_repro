#!/usr/bin/env bash
# Decontaminate the cluster checkout before the tagged benchmark campaign.
#
# The box currently mixes three eras: the pre-Restructure CIFAR harness at the
# repo root (alg.py, hadamard.py, iwrm_bench.py, run_all.py, requirements.txt,
# RESULTS.md, logs_*.log), the CIFAR datasets, and several generations of result
# directories written to shared paths with no run identity. Nothing downstream can
# tell which code produced which number, which is exactly the problem the run tags
# fix -- but only for artifacts written from now on.
#
# DESTRUCTIVE. Dry run by default; nothing is removed until you pass APPLY=1.
#
#   bash scripts/clean_cluster.sh              # show what would go, and its size
#   ARCHIVE=1 bash scripts/clean_cluster.sh    # also show where it would be moved
#   APPLY=1 ARCHIVE=1 bash scripts/clean_cluster.sh   # move to an archive dir
#   APPLY=1 bash scripts/clean_cluster.sh             # delete outright
#
# ARCHIVE=1 moves instead of deleting, which is the safer default on a shared box
# where /  is already at 95%: check the freed space, then remove the archive.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

APPLY="${APPLY:-0}"
ARCHIVE="${ARCHIVE:-0}"
ARCHIVE_DIR="${ARCHIVE_DIR:-$ROOT/../jd-phase25-archive-$(date +%Y%m%d-%H%M%S)}"

# --- what counts as contamination -------------------------------------------
# Pre-Restructure root files: superseded by src/jdgram + bench/, and iwrm_bench.py
# does not even import (its `from hadamard import algorithm3` predates the move).
STALE_ROOT=(
  alg.py hadamard.py iwrm_bench.py run_all.py run_report.py
  microbench_workspace.py
  requirements.txt RESULTS.md RESULTS_NANOGPT.md RESULTS_NANOGPT_V2.md
  BENCHMARK_PLAN.md
  bench/nanoGPT.py bench/nanogpt_v2.py scripts/run_nanogpt_cluster.sh
  legacy
  logs_engines.log logs_figure.log logs_scaling.log logs_accuracy.log
  nanogpt_report_bundle.zip
)
# The Phase-B runners (run_report.py, bench/nanoGPT.py, bench/nanogpt_v2.py) and
# the CIFAR harness have since been removed from the repo entirely -- they are
# superseded by bench/profile_suite.py and remain in git history. They are listed
# above so a cluster checkout that predates that removal is cleaned too.

# CIFAR-era data. Large, regenerable, and irrelevant to the nanoGPT/Qwen line.
CIFAR_DATA=(
  data/cifar-10-batches-py
  data/cifar-10-python.tar.gz
  data/train_32x32.mat
  data/cifar10_subset1024_seed1.pt
)

# CIFAR-era findings and every untagged result directory.
STALE_RESULTS=(
  results/cifar_fuji2
  results/scaling_cifar10.json
  results/nanogpt_report
  results/nanogpt_gala1
  results/nanogpt_v2
  results/transformer
  results/nanogpt_report_bundle.zip
  results/logs
  results/status.json
  results/env.json
)

# Tagged runs (results/v*/) are NEVER listed here. They carry a manifest with the
# git SHA and environment, which is the whole point of the tagging scheme; delete
# them by hand once they are bundled and pulled off the box.

# NEVER touched: the tokenised Shakespeare corpus (L7/L11 need it for held-out
# accuracy and re-preparing it means a download), the git metadata, the source.
KEEP_NOTE=(
  "data/shakespeare_char/   (held-out accuracy data -- kept)"
  ".git/                    (history -- kept)"
  "src/ bench/ gates/ models/ docs/ scripts/   (kept)"
  "results/v*/              (already tagged -- kept)"
  "results/runs_index.csv   (the run index -- kept)"
)

size_of() { du -sh "$1" 2>/dev/null | cut -f1 || echo "?"; }

echo "=============================================================="
echo " cluster decontamination   ROOT=$ROOT"
echo " mode: $([ "$APPLY" = 1 ] && echo APPLY || echo 'DRY RUN')"\
"$([ "$ARCHIVE" = 1 ] && echo ' + ARCHIVE' || echo '')"
echo "=============================================================="

total=0
declare -a TARGETS=()
for group_name in STALE_ROOT CIFAR_DATA STALE_RESULTS; do
  eval "items=(\"\${${group_name}[@]}\")"
  echo
  echo "--- $group_name ---"
  found=0
  for item in "${items[@]}"; do
    if [ -e "$item" ]; then
      printf '  %-46s %s\n' "$item" "$(size_of "$item")"
      TARGETS+=("$item")
      found=1
    fi
  done
  [ "$found" = 0 ] && echo "  (nothing present)"
done

echo
echo "--- explicitly preserved ---"
for k in "${KEEP_NOTE[@]}"; do echo "  $k"; done

if [ "${#TARGETS[@]}" -eq 0 ]; then
  echo
  echo "Nothing to clean. The checkout is already decontaminated."
  exit 0
fi

echo
echo "disk before:"; df -h "$ROOT" | tail -1

if [ "$APPLY" != 1 ]; then
  echo
  echo "DRY RUN -- nothing removed. ${#TARGETS[@]} path(s) matched."
  echo "Re-run with:  APPLY=1 ARCHIVE=1 bash scripts/clean_cluster.sh   (move)"
  echo "          or: APPLY=1 bash scripts/clean_cluster.sh             (delete)"
  exit 0
fi

if [ "$ARCHIVE" = 1 ]; then
  mkdir -p "$ARCHIVE_DIR"
  echo
  echo "moving ${#TARGETS[@]} path(s) -> $ARCHIVE_DIR"
  for item in "${TARGETS[@]}"; do
    mkdir -p "$ARCHIVE_DIR/$(dirname "$item")"
    mv "$item" "$ARCHIVE_DIR/$item"
    echo "  moved $item"
  done
  echo
  echo "Archived. Verify, then free the space with:"
  echo "  rm -rf '$ARCHIVE_DIR'"
else
  echo
  echo "deleting ${#TARGETS[@]} path(s)"
  for item in "${TARGETS[@]}"; do
    rm -rf "$item"
    echo "  removed $item"
  done
fi

echo
echo "disk after:"; df -h "$ROOT" | tail -1
echo
echo "Next: bash scripts/run_profile_cluster.sh env"
