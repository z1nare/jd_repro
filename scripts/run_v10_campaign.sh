#!/usr/bin/env bash
# v10 campaign -- GPT-2 124M, acceptance criteria against Rui's 1.5x budget.
#
# WHAT THIS ANSWERS, AND WHY THE LADDERS ARE SHAPED THIS WAY
# ----------------------------------------------------------
# Rui set the bar as: with two or three objectives, time and memory should be
# about 1.5x a single-objective run, not 2-3x.  His stated reason is that extra
# objectives are mostly parallelisable.
#
# In a real multi-objective setup you have ONE batch and several loss functions.
# Adding an objective adds no data.  But models/configs.py::per_sequence_losses
# returns one scalar per batch ROW, so in this harness m IS the batch dimension:
# going m=1 -> m=2 doubles the tokens as well as adding the objective.  Every v9
# ratio therefore measures "one more objective AND twice the data", which is not
# the quantity Rui budgeted.  A 2x slowdown for 2x the work is not a finding.
#
# So this campaign runs three ladders and reports all three:
#
#   A  duplicate  fixed T, m varies, every row the SAME sequence.
#                 Data held constant, objective count varies.  This is the
#                 closest thing the harness can express to Rui's question and
#                 it is the one to quote.
#   B  independent fixed T, m varies, m distinct windows.
#                 The v9 framing, kept so the gap between A and B is visible
#                 rather than argued about.
#   C  pinned     m*T held constant.  Total tokens fixed, but note the attention
#                 term scales as m*T^2, so the baseline itself cheapens as T
#                 falls -- read within-run ratios, never absolute ms.
#
# Every ladder runs at FORCED routes, not auto.  m*T^2 is the router's input, so
# along any ladder that varies T the router silently reassigns strategy
# mid-ladder and the objective-count effect gets mixed with a route-reassignment
# effect.  Pinning the route is what makes the ladder mean one thing.
#
# m=1 is included everywhere.  It exists nowhere in v9, and it is the only cell
# that isolates fixed architectural cost: at m=1 the Gramian is [1,1], the
# aggregator returns weight 1, and the QP is trivial, so everything jdgram
# spends over sgd_erm is the two-forward design plus hook bookkeeping.
#
# EVERYTHING HERE RUNS ON THE CLUSTER.  Nothing in this file should be run on a
# laptop beyond --help.
#
# Usage:  bash scripts/run_v10_campaign.sh <stage>
#         bash scripts/run_v10_campaign.sh all
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1
ROOT="$PWD"
export PYTHONPATH="$ROOT:$ROOT/src:$ROOT/bench:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"

PY="${PY:-python}"
VERSION="${VERSION:-10}"
OUT="${JDGRAM_RESULTS:-$ROOT/results}"
LOG="$OUT/v${VERSION}_campaign_$(date +%Y%m%d-%H%M%S).log"
mkdir -p "$OUT"

# ---- the model. Full GPT-2 124M; head_dim = 768/12 = 64. -------------------
NL="${NL:-12}"; NH="${NH:-12}"; NE="${NE:-768}"; VOC="${VOC:-50257}"
GPT2=(--n-layer "$NL" --n-head "$NH" --n-embd "$NE" --V "$VOC")
COMMON=(--version "$VERSION" --device cuda --dtype fp32 --driver squashed
        --out-root "$OUT" --max-alloc-gib "${MAXGIB:-22}")

STEPS="${STEPS:-100}"
EVALB="${EVALB:-10}"
REPS="${REPS:-1}"          # repeat each cost cell; fixed seeds, so this gives
                           # timing spread, not statistical replication

# probe: never let one OOM kill the campaign. At this scale an OOM is a RESULT.
probe() {
  local name="$1"; shift
  echo ""
  echo "=============================================================================="
  echo ">> $name"
  echo "   $PY bench/profile_suite.py --name $name $*"
  echo "=============================================================================="
  "$PY" bench/profile_suite.py --name "$name" "$@"
  local rc=$?
  [ $rc -ne 0 ] && echo "   (exit $rc -- recorded, continuing)"
  return 0
}

stage="${1:-help}"

case "$stage" in

env)
  nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version \
             --format=csv 2>/dev/null || echo "nvidia-smi unavailable"
  df -h "$OUT" | tail -1
  "$PY" - <<'EOF'
import torch, sys
print("python  ", sys.version.split()[0])
print("torch   ", torch.__version__, "cuda", torch.version.cuda)
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("gpu     ", p.name, p.total_memory // 2**20, "MiB", "sm",
          f"{p.major}.{p.minor}", p.multi_processor_count, "SMs")
for mod in ("torchjd", "triton", "numpy", "quadprog"):
    try:
        m = __import__(mod)
        print(f"{mod:<8}", getattr(m, "__version__", "present"))
    except Exception as e:
        print(f"{mod:<8} MISSING ({type(e).__name__})")
EOF
  ;;

verify)
  "$PY" bench/preflight.py --strict
  NL="$NL" NH="$NH" NE="$NE" VOC="$VOC" "$PY" - <<'EOF'
# Assert the model really is GPT-2 124M before spending GPU hours on it. A
# silently-wrong n_head (4 heads of 192 instead of 12 of 64) produced a whole
# campaign's worth of unusable numbers once already, and the run was labelled
# "nanoGPT 124M" the entire time.
import os, torch
from profile_suite import build_model
nl, nh = int(os.environ["NL"]), int(os.environ["NH"])
ne, voc = int(os.environ["NE"]), int(os.environ["VOC"])
model, modules, ov, sh = build_model(torch.device("cpu"), n_layer=nl, n_head=nh,
                                     n_embd=ne, T=128, V=voc, seed=0)
gc = model.config
assert (gc.n_layer, gc.n_head, gc.n_embd, gc.vocab_size) == (nl, nh, ne, voc), gc
assert gc.n_embd % gc.n_head == 0, "n_embd not divisible by n_head"
head_dim = gc.n_embd // gc.n_head
assert head_dim == 64, f"head_dim {head_dim}, expected 64"
n = sum(p.numel() for p in model.parameters())
print(f"params        {n:,}")
print(f"head_dim      {head_dim}   ({gc.n_head} heads x {head_dim})")
print(f"hooked modules {len(modules)}")
print(f"tied          {'lm_head' in ''.join(map(str, sh.keys())) or bool(sh)}")
for m in (1, 2, 4, 8, 16):
    print(f"  autojac [m,P] fp32 at m={m:<3} = {4 * m * n / 2**30:7.2f} GiB")
EOF
  ;;

data)
  # Must run on the cluster: the fetch is stdlib urllib over https and a Windows
  # conda env fails it on the certificate store. Without train.bin/val.bin,
  # _batch_from falls back to synthetic tokens where idx and tgt are drawn
  # independently -- zero mutual information, so val_ce sits at ln(V) forever and
  # every quality result is a green light that proves nothing.
  if [ -f "$ROOT/data/shakespeare_char/train.bin" ]; then
    echo "corpus already present:"
    ls -la "$ROOT/data/shakespeare_char/"
  else
    "$PY" bench/prepare_shakespeare_char.py
  fi
  ;;

gates)
  # Correctness before cost. Every ms below is only meaningful for a Gramian
  # that is exact; without this the campaign measures the speed of a wrong answer.
  "$PY" -m pytest gates/ -q
  ;;

# ---- LADDER A: objectives vary, DATA HELD CONSTANT. The one to quote. -------
ladder-dup)
  for R in ${ROUTES:-tfirst dfirst}; do
    for M in ${MLIST:-1 2 4 8 16}; do
      for rep in $(seq 1 "$REPS"); do
        probe "a-dup-m${M}-${R}-r${rep}" "${COMMON[@]}" "${GPT2[@]}" \
          --levels L4 L5 L11 --m "$M" --T "${TFIX:-512}" \
          --objective-mode duplicate --force-route "$R" \
          --steps "$STEPS" --eval-batches "$EVALB" \
          --notes "ladder A: m objectives on ONE sequence, route=$R"
      done
    done
  done
  ;;

# ---- LADDER B: v9 framing -- objectives AND data both grow. ----------------
ladder-indep)
  for R in ${ROUTES:-tfirst dfirst}; do
    for M in ${MLIST:-1 2 4 8 16}; do
      probe "b-indep-m${M}-${R}" "${COMMON[@]}" "${GPT2[@]}" \
        --levels L4 L5 L11 --m "$M" --T "${TFIX:-512}" \
        --objective-mode independent --force-route "$R" \
        --steps "$STEPS" --eval-batches "$EVALB" \
        --notes "ladder B: m objectives AND m windows, route=$R"
    done
  done
  ;;

# ---- LADDER C: pinned total tokens. m*T constant. --------------------------
ladder-pinned)
  # colon-separated so an env override survives word splitting
  for R in ${ROUTES:-tfirst dfirst}; do
    for RUNG in ${RUNGS:-1:2048 2:1024 4:512 8:256 16:128}; do
      M="${RUNG%%:*}"; T="${RUNG##*:}"
      probe "c-pin-m${M}-T${T}-${R}" "${COMMON[@]}" "${GPT2[@]}" \
        --levels L4 L5 L11 --m "$M" --T "$T" --force-route "$R" \
        --steps "$STEPS" --eval-batches "$EVALB" \
        --notes "ladder C: m*T pinned at 2048, route=$R"
    done
  done
  ;;

# ---- objective relationships: does conflict cost more? ---------------------
modes)
  # What this is NOT for, measured rather than assumed: conflict does not cost
  # meaningfully more. At m=4, 60 steps, the three modes came in at 24.0 / 25.6 /
  # 25.9 ms per step for independent / duplicate / conflicting, and L4 puts the
  # dual-cone solve at 0.7-0.9 ms of a ~25 ms step either way. An earlier 2x
  # reading came from a 4-step run in which warmup dominated. The QP is ~3% of a
  # step and closing that avenue is itself a result.
  #
  # What it IS for: the correctness ladder Rui asked for. Duplicated objectives
  # must give a rank-1 Gramian and every engine must agree; conflicting ones must
  # give a strictly negative off-diagonal and a near-cancelled update.
  # gramian_min_offdiag_cos records what each mode actually produced, so a mode
  # that silently failed to conflict is visible instead of assumed from its label.
  # L4 as well as L11: L4's weighting_qp phase solves on the real Gramian, so it
  # is the only place the QP cost under conflict is isolated from everything
  # else in the step. L11 gives loss curves, per-objective trajectories and the
  # sgd_erm control; L4 says where the extra time went.
  #
  # Routes are swept here too. Route is a per-layer contraction order and has
  # nothing to do with objective relationships, but the two interact through
  # cost, and pinning one route would leave that unmeasured.
  for MODE in ${MODES:-independent duplicate scaled conflicting}; do
    for R in ${ROUTES:-tfirst dfirst}; do
      for M in ${MODE_MLIST:-1 2 4 8}; do
        probe "d-mode-${MODE}-m${M}-${R}" "${COMMON[@]}" "${GPT2[@]}" \
          --levels L4 L11 --m "$M" --T "${TFIX:-512}" \
          --objective-mode "$MODE" --force-route "$R" \
          --steps "$STEPS" --eval-batches "$EVALB" \
          --notes "objective relationship = $MODE, route = $R"
      done
    done
  done
  ;;

# ---- kernel-level traces. v9 never passed --trace-raw, so every ------------
# ---- trace-derived section of profile_stats has been empty. ---------------
traces)
  for R in tfirst dfirst; do
    probe "e-trace-${R}" "${COMMON[@]}" "${GPT2[@]}" \
      --levels L9 --m "${TRACE_M:-4}" --T "${TRACE_T:-512}" \
      --force-route "$R" --record-shapes --trace-raw \
      --notes "kernel trace, forced $R -- per-layer-type route evidence"
  done
  # A trace at the ladder's own operating point, so kernel attribution and the
  # acceptance number describe the same run.
  probe "e-trace-ladder-point" "${COMMON[@]}" "${GPT2[@]}" \
    --levels L9 --m 2 --T "${TFIX:-512}" --objective-mode duplicate \
    --record-shapes --trace-raw \
    --notes "trace at ladder A m=2, the cell Rui's budget names"
  ;;

# ---- per-shape route crossover, isolated from the model -------------------
kernels)
  probe "f-kernels" "${COMMON[@]}" "${GPT2[@]}" --levels L0 \
    --m 4 --T 512 --notes "identity kernels alone: no model, no hooks, no autograd"
  ;;

# ---- route sweep at fixed m,T across the shape grid ------------------------
routes)
  for R in auto tfirst dfirst; do
    local_extra=()
    [ "$R" != "auto" ] && local_extra=(--force-route "$R")
    probe "g-route-${R}" "${COMMON[@]}" "${GPT2[@]}" \
      --levels L2 L4 --m "${ROUTE_M:-4}" --T "${ROUTE_T:-512}" \
      "${local_extra[@]}" --notes "route sweep: $R"
  done
  ;;

farm)
  # fuji2 has four idle A5000s. Each 124M cell peaks around 12 GiB, so one job
  # per card and never two on the same card.
  #
  # Each GPU takes a slice of the stage list and runs ITS slice sequentially,
  # all four slices in parallel. Each worker writes into its own results
  # subdirectory: runtag appends to runs_index.csv, and four processes appending
  # to one file is a race that silently loses rows. 'stats' globs two levels
  # deep so the split is invisible downstream.
  read -r -a GPUS_LIST <<< "${GPUS:-0 1 2 3}"
  read -r -a STAGES_LIST <<< "${FARM_STAGES:-ladder-dup ladder-indep ladder-pinned modes traces routes kernels}"
  n=${#GPUS_LIST[@]}
  echo "farming ${#STAGES_LIST[@]} stages across $n GPU(s): ${GPUS_LIST[*]}"
  for gi in "${!GPUS_LIST[@]}"; do
    (
      g="${GPUS_LIST[$gi]}"
      mkdir -p "$OUT/gpu$g"
      wlog="$OUT/farm-gpu$g.log"
      : > "$wlog"
      for ((i = gi; i < ${#STAGES_LIST[@]}; i += n)); do
        s="${STAGES_LIST[$i]}"
        echo "[gpu$g] $(date -Is) START $s" | tee -a "$wlog"
        GPU="$g" JDGRAM_RESULTS="$OUT/gpu$g" bash "$0" "$s" >> "$wlog" 2>&1
        echo "[gpu$g] $(date -Is) DONE  $s (exit $?)" | tee -a "$wlog"
      done
    ) &
  done
  wait
  echo ""
  echo "all workers finished. per-GPU logs:"
  ls -la "$OUT"/farm-gpu*.log
  ;;

stats)
  # maxdepth 2 so both the sequential layout (results/v10_*) and the farmed one
  # (results/gpu0/v10_*) are picked up by the same command.
  mapfile -t RUNS < <(find "$OUT" -maxdepth 2 -type d -name "v${VERSION}_*" | sort)
  if [ ${#RUNS[@]} -eq 0 ]; then echo "no v${VERSION}_* runs in $OUT"; exit 1; fi
  echo "analysing ${#RUNS[@]} run directories"
  "$PY" bench/profile_stats.py "${RUNS[@]}" --top 25 \
      --json "$OUT/stats_v${VERSION}.json" | tee "$OUT/stats_v${VERSION}.txt"
  ;;

bundle)
  # Two bundles. The light one is for a slow link and carries every number.
  # The heavy one carries the chrome traces, which are the whole point of the
  # kernel work and are several MB each -- pull it when the link allows.
  # Globs cover both layouts: sequential (v10_*) and farmed (gpu*/v10_*).
  ( cd "$OUT" && tar czf "jdgram-v${VERSION}-light-$(date +%Y%m%d-%H%M%S).tar.gz" \
      --exclude='*.pickle' --exclude='profiler_trace.json*' \
      v${VERSION}_*/ gpu*/v${VERSION}_*/ gpu*/runs_index.csv runs_index.csv \
      "stats_v${VERSION}.txt" "stats_v${VERSION}.json" farm-gpu*.log \
      2>/dev/null || true )
  ( cd "$OUT" && tar czf "jdgram-v${VERSION}-traces-$(date +%Y%m%d-%H%M%S).tar.gz" \
      v${VERSION}_*/profiler_trace.json* v${VERSION}_*/profiler_ops.csv \
      gpu*/v${VERSION}_*/profiler_trace.json* gpu*/v${VERSION}_*/profiler_ops.csv \
      2>/dev/null || true )
  ls -lh "$OUT"/jdgram-v${VERSION}-*.tar.gz
  ;;

all)
  free_gib=$(df -BG --output=avail "$OUT" | tail -1 | tr -dc '0-9')
  if [ "${free_gib:-0}" -lt 20 ]; then
    echo "only ${free_gib} GiB free in $OUT; traces need room. Aborting."; exit 1
  fi
  {
    echo "v${VERSION} campaign starting $(date -Is)"
    for s in env data verify gates ladder-dup ladder-indep ladder-pinned \
             modes kernels routes traces stats bundle; do
      echo ""; echo "########## STAGE: $s ##########"
      bash "$0" "$s" || echo "   stage $s returned non-zero -- continuing"
    done
    echo ""; echo "v${VERSION} campaign finished $(date -Is)"
  } 2>&1 | tee "$LOG"
  echo ""
  echo "log: $LOG"
  ;;

*)
  cat <<EOF
v10 campaign -- GPT-2 124M against Rui's 1.5x acceptance budget.

  bash scripts/run_v10_campaign.sh <stage>

Stages, in the order 'all' runs them:
  env             GPU, driver, torch/torchjd/triton versions
  data            fetch tinyshakespeare (cluster-side; Windows SSL fails it)
  verify          preflight + assert the model really is 124M with head_dim 64
  gates           pytest gates/ -- correctness before any cost number
  ladder-dup      LADDER A: m varies, ONE sequence. The number to quote.
  ladder-indep    LADDER B: m varies, m windows. The v9 framing, for contrast.
  ladder-pinned   LADDER C: m*T held at 2048.
  modes           independent / duplicate / scaled / conflicting
  kernels         L0 identity kernels in isolation
  routes          L2+L4 route sweep at one (m,T)
  traces          L9 with --trace-raw at forced tfirst and dfirst
  stats           profile_stats over every v${VERSION}_* run (both layouts)
  bundle          two tarballs: light (numbers) and traces (chrome JSON)
  all             everything above, sequential on one GPU, tee'd to a log
  farm            the heavy stages spread across GPUS, one job per card

Recommended on a 4-GPU box:
  bash scripts/run_v10_campaign.sh env
  bash scripts/run_v10_campaign.sh data
  bash scripts/run_v10_campaign.sh verify
  bash scripts/run_v10_campaign.sh gates
  bash scripts/run_v10_campaign.sh farm     # ~4x wall clock
  bash scripts/run_v10_campaign.sh stats
  bash scripts/run_v10_campaign.sh bundle

Environment overrides:
  GPU=0            CUDA_VISIBLE_DEVICES
  VERSION=10       run-tag version
  STEPS=100        training steps per L11 cell
  REPS=1           repeats per ladder-A cell (timing spread; seeds are fixed)
  MLIST="1 2 4 8 16"          objective counts for ladders A and B
  ROUTES="tfirst dfirst"      forced routes (never 'auto' on a ladder)
  TFIX=512                    fixed T for ladders A and B
  RUNGS="1:2048 2:1024 ..."   m:T pairs for ladder C, colon-separated
  MAXGIB=22                   per-allocation guard

Why forced routes: m*T^2 is the router's input, so a ladder that varies T
reassigns the route mid-ladder and mixes the objective-count effect with a
route effect. Run each ladder at a pinned route and compare across routes.
EOF
  ;;
esac
