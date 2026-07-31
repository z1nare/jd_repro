#!/usr/bin/env bash
# GPT-2 124M ("full nanoGPT") campaign for gala1 (8x RTX A5000, 24 GiB each).
#
# Everything here runs ON THE CLUSTER. The laptop is for `python -m compileall`
# and the CPU gate suite only.
#
#   ssh dice-gala
#   cd ~/jd-phase25-bench
#   bash scripts/run_gpt2_124m_cluster.sh verify     # always start here
#   bash scripts/run_gpt2_124m_cluster.sh overnight  # the whole campaign, guarded
#
# WHY THIS SCRIPT EXISTS, SEPARATELY FROM run_profile_cluster.sh
# --------------------------------------------------------------
# Every number in REPORT_CIFAR_TO_NANOGPT.md was measured at <= 16.03M parameters
# on a 4-layer, 256-wide model. Two of the report's central claims are therefore
# arithmetic rather than measurement, and both of them are claims a reviewer will
# go after first:
#
#   1. "The Gramian trade only pays at a size where autojac cannot run, and we
#      are below that size." Nothing here has ever been run at a size where
#      autojac dies. At 124M its [m, P] Jacobian is 0.464*m GiB in fp32, so it
#      should die somewhere around m=32. That crossover is the single most
#      valuable measurement in this file -- stage `crossover`.
#
#   2. "The router picks the wrong route at vocabulary scale, costing 2.76x."
#      That was measured with a 12.9M-parameter head bolted onto a small trunk.
#      At 124M the head is the same 38.6M-parameter object but the trunk around
#      it is 40x bigger, so the memory the router is trying to protect is an even
#      smaller fraction of peak. Stage `router` re-runs that comparison where it
#      actually matters.
#
# THE MODEL
# ---------
# n_layer=12, n_head=12, n_embd=768, vocab=50257 -- karpathy's GPT-2 124M. The
# harness derives n_head as n_embd//64, which is GPT-2's ratio at every size, but
# --n-head is passed explicitly below so a mismatch is a hard error rather than a
# run silently profiling 4 heads of 192. Verified parameter count with tying on:
# 124,439,808.
#
# THE MEMORY BUDGET (fp32, 24 GiB card), so a stage that dies is diagnosable:
#   params + grads          124.44M * 4 * 2   = 0.93 GiB   (fixed)
#   autojac [m, P]          0.464 * m GiB                  m=8: 3.7  m=32: 14.8
#   logits m*T*V            m*T*50257*4       T=512, m=8:  0.77 GiB (x2-3 for CE)
#   attn weights            m*12*T^2*4*12L    T=512, m=8:  1.2 GiB (less w/ flash)
#   jdgram head, d-first    m*V*d*4           m=8:         1.24 GiB
#   jdgram head, T-first    3*m*T^2*4         m=8, T=512:  0.025 GiB
# The last two lines are the whole argument: 50x less workspace for the head, on
# the route only this engine can express.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="${ROOT}:${ROOT}/src:${ROOT}/bench${PYTHONPATH:+:$PYTHONPATH}"
export JDGRAM_RESULTS="${JDGRAM_RESULTS:-${ROOT}/results}"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export LD_LIBRARY_PATH="/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Fragmentation is the difference between "fits" and "OOM at 21 GiB used" when a
# single allocation is 4 GiB, which is exactly the autojac Jacobian. Expandable
# segments cost nothing and remove a whole class of false OOM.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PY="${PY:-python}"
VERSION="${VERSION:-9}"          # bump BY HAND when engine semantics change
STAGE="${1:-help}"
shift || true

# --- the model, in one place -------------------------------------------------
NL=12; NH=12; NE=768; VOC=50257
GPT2=(--n-layer "$NL" --n-head "$NH" --n-embd "$NE" --V "$VOC")

banner() { echo; echo "=============== $* ==============="; echo; }

# Run one profile_suite invocation, tolerating failure. At this scale an OOM is a
# RESULT, not an accident -- the point of the crossover stage is to find where
# things stop fitting -- so a non-zero exit must never end the campaign.
probe() {
  local name="$1"; shift
  echo; echo "-------- $name --------"
  if $PY bench/profile_suite.py --version "$VERSION" --name "$name" \
       --device cuda --dtype fp32 "${GPT2[@]}" "$@"; then
    echo "[ok] $name"
  else
    echo "[DIED] $name (exit $?) -- recorded as a boundary, continuing"
  fi
}

case "$STAGE" in

  env)
    banner "environment"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
    df -h "$JDGRAM_RESULTS" | tail -2
    $PY -c "import torch,sys;print('python',sys.version.split()[0]);print('torch',torch.__version__,'cuda',torch.cuda.is_available())"
    $PY -c "import torchjd,inspect,os;print('torchjd at',os.path.dirname(inspect.getfile(torchjd)))"
    ;;

  verify)
    banner "preflight + model identity"
    $PY bench/preflight.py --strict
    # Prove the config really is GPT-2 124M before spending a night on it. A run
    # labelled 124M that quietly built 4 heads of 192 is worse than no run.
    $PY - <<PYEOF
import sys, torch
sys.path[:0] = ["$ROOT", "$ROOT/src", "$ROOT/bench"]
import profile_suite as ps
model, *_ = ps.build_model(torch.device("cpu"), n_layer=$NL, n_head=$NH,
                           n_embd=$NE, T=1024, V=$VOC)
c, p = model.config, ps.count_params(model)
assert (c.n_layer, c.n_head, c.n_embd, c.vocab_size) == ($NL, $NH, $NE, $VOC), c
assert c.n_embd // c.n_head == 64, "head_dim must be 64 for GPT-2"
print(f"OK  GPT-2 124M: {c.n_layer}L {c.n_head}H {c.n_embd}d V={c.vocab_size}")
print(f"OK  trainable parameters (tied): {p:,}")
print(f"    autojac [m,P] fp32 would be {4*p/2**30:.3f} GiB per objective")
PYEOF
    ;;

  gates)
    banner "gates (CPU, fp64) -- must be green before any GPU time"
    $PY -m pytest gates/ -q
    ;;

  # ----------------------------------------------------------------- headline
  headline)
    # THE experiment: m=8 at the full 1024 context, where the gap is structural
    # rather than marginal. Modelled costs on this card:
    #     jdgram 11.8 GiB    autogram 14.1 GiB    autojac 43.8 GiB
    # autojac is 1.9x over the card and jdgram has 3.7x headroom, so the result
    # survives even a large modelling error. This is the rung to quote.
    #
    # Note it is deliberately NOT m=8/T=512. There autojac needs ~21.8 GiB
    # against a ~21.5 GiB usable budget -- a coin flip, and a coin flip is not an
    # experiment. That rung is run below as a boundary bracket instead.
    banner "GPT-2 124M: headline, m=8 T=1024"
    probe "124m-headline-m8-T1024" --levels L5 --m 8 --T 1024 \
          --max-alloc-gib 22 \
          --notes "headline: autojac ~43.8 GiB vs jdgram ~11.8 GiB on a 24 GiB card"
    ;;

  # ---------------------------------------------------------------- crossover
  crossover)
    # Bracket the boundary rather than assert it. Sweep m at T=512 across the
    # modelled autojac crossing (~m=8, since the [m,P] Jacobian is 0.4636*m GiB
    # and the vmapped [m,m,T,V] logits gradient adds ~296 MiB*m^2), so whichever
    # side it actually lands on is recorded with its neighbours either side.
    #
    # Both outcomes are publishable; what is not publishable is a single rung
    # that could go either way with no bracket around it.
    banner "GPT-2 124M: engine crossover in m (T=${T_CROSS:-512})"
    for M in ${M_LADDER:-4 6 7 8 9 12}; do
      probe "124m-cross-m$M" --levels L5 --m "$M" --T "${T_CROSS:-512}" \
            --max-alloc-gib 22 \
            --notes "crossover bracket m=$M: [m,P] Jacobian alone is $(awk "BEGIN{printf \"%.2f\", 0.4636*$M}") GiB"
    done
    ;;

  # ------------------------------------------------------------------- router
  router)
    # Re-run the report's #1 open item at real scale. L4 decomposes the step, so
    # this gives compute_gramian per iteration for the router's choice against
    # each forced route. --driver squashed keeps it to the shipping driver; the
    # levels used to sweep all three regardless of the flag, which at this size
    # would spend the night OOM-ing on drivers we no longer ship.
    banner "GPT-2 124M: router choice vs forced routes"
    for ROUTE in auto tfirst dfirst; do
      ARGS=(--levels L4 --m "${M_ROUTER:-4}" --T "${T_ROUTER:-512}" --driver squashed)
      [ "$ROUTE" != auto ] && ARGS+=(--force-route "$ROUTE")
      probe "124m-route-$ROUTE" "${ARGS[@]}" \
            --notes "router validation at 124M, route=$ROUTE"
    done
    echo
    echo "Read off compute_gramian ms for each. If auto tracks tfirst and tfirst"
    echo "is slower than dfirst, the cost-model bug reproduces at GPT-2 scale."
    ;;

  # ------------------------------------------------------------------- ladder
  ladder)
    # Shape sweep. Ordered smallest-first so an OOM at the top still leaves every
    # rung below it measured. L2 gives peak/held, L4 the phase decomposition.
    banner "GPT-2 124M: (m, T) ladder"
    # Pairs are m:T rather than "m T" so an RUNGS= override from the environment
    # survives word splitting. Quoted pairs only work as a literal default.
    for RUNG in ${RUNGS:-2:256 4:256 8:256 4:512 8:512 16:512 8:1024}; do
      M="${RUNG%%:*}"; TT="${RUNG##*:}"
      probe "124m-m${M}-T${TT}" --levels L2 L4 --m "$M" --T "$TT" --driver squashed \
            --notes "ladder rung m=$M T=$TT"
    done
    ;;

  # -------------------------------------------------------------- aggregators
  aggregators)
    # Mean/UPGrad/MGDA/PCGrad x jdgram/autogram/autojac, scored on held-out CE.
    # Small m and T: this trains for real, so it is the most expensive stage.
    banner "GPT-2 124M: aggregator x engine matrix"
    if [ ! -f "$ROOT/data/shakespeare_char/val.bin" ]; then
      echo "preparing shakespeare corpus first"
      $PY bench/prepare_shakespeare_char.py
    fi
    # NOTE: shakespeare_char has a 65-token vocabulary. Training GPT-2's 50257-wide
    # head on it wastes 99.9% of the head, but it keeps this comparable with the
    # V=65 matrix in the report and the point here is engine cost, not language
    # modelling. Read val CE as an agreement check between engines, nothing more.
    probe "124m-aggregators" --levels L11 --m "${M_AGG:-4}" --T "${T_AGG:-128}" \
          --steps "${STEPS:-200}" --eval-batches 20 \
          --notes "aggregator matrix at GPT-2 124M"
    ;;

  accuracy)
    banner "GPT-2 124M: L7 convergence"
    probe "124m-accuracy" --levels L7 --m "${M_ACC:-4}" --T 128 \
          --steps "${STEPS:-200}" --notes "loss curves at 124M"
    ;;

  trace)
    banner "GPT-2 124M: L9 kernel trace"
    probe "124m-trace" --levels L9 --m 4 --T 256 --driver squashed \
          --record-shapes --notes "dispatch analysis at 124M"
    ;;

  # ---------------------------------------------------------------- overnight
  overnight)
    LOG="${LOG:-$JDGRAM_RESULTS/gpt2_124m_v${VERSION}_$(date +%Y%m%d-%H%M%S).log}"
    mkdir -p "$(dirname "$LOG")"
    echo "logging to $LOG"
    {
      echo "=== GPT-2 124M campaign v$VERSION started $(date) ==="
      free_gib=$(df -BG --output=avail "$JDGRAM_RESULTS" | tail -1 | tr -dc '0-9')
      echo "free space: ${free_gib} GiB"
      if [ "${free_gib:-0}" -lt 6 ]; then
        echo "REFUSING: under 6 GiB free."; exit 1
      fi

      run_stage() {
        echo; echo "########## $* ##########"; echo
        if ! "$0" "$@"; then echo "!!! stage '$*' FAILED -- continuing"; fi
      }

      run_stage env
      run_stage verify
      run_stage gates
      # Cheapest evidence first, so a night that dies early still answers the
      # two questions the report actually needs answered.
      run_stage router
      run_stage headline
      run_stage crossover
      run_stage ladder
      export STEPS="${STEPS:-200}"
      run_stage aggregators
      run_stage accuracy
      run_stage trace
      run_stage stats
      run_stage bundle
      echo; echo "=== finished $(date) ==="
    } 2>&1 | tee -a "$LOG"
    echo; echo "log: $LOG"
    ;;

  stats)
    banner "statistics"
    mapfile -t RUNS < <(ls -d "$JDGRAM_RESULTS"/v${VERSION}_*/ 2>/dev/null)
    if [ "${#RUNS[@]}" -eq 0 ]; then
      echo "no runs matching v${VERSION}_* under $JDGRAM_RESULTS"; exit 1
    fi
    echo "aggregating ${#RUNS[@]} run(s) for version $VERSION"
    $PY bench/profile_stats.py "${RUNS[@]}" --top 20 \
        --json "$JDGRAM_RESULTS/stats_124m.json" | tee "$JDGRAM_RESULTS/stats_124m.txt"
    echo
    echo "send back:  $JDGRAM_RESULTS/stats_124m.txt"
    ;;

  bundle)
    banner "bundle"
    STAMP="$(date +%Y%m%d-%H%M%S)"
    OUT="${BUNDLE_OUT:-$HOME/jdgram-gpt2-124m-v${VERSION}-${STAMP}.tar.gz}"
    [ -d "$JDGRAM_RESULTS" ] || { echo "no results dir"; exit 1; }
    cd "$JDGRAM_RESULTS"
    mapfile -t DIRS < <(ls -d v${VERSION}_*/ 2>/dev/null)
    [ "${#DIRS[@]}" -eq 0 ] && { echo "no v${VERSION}_* runs"; exit 1; }
    EXTRA=()
    for f in runs_index.csv stats_124m.txt stats_124m.json; do
      [ -f "$f" ] && EXTRA+=("$f")
    done
    tar czf "$OUT" --exclude='*.pickle' --exclude='profiler_trace.json*' \
        "${DIRS[@]}" "${EXTRA[@]}"
    cd "$ROOT"
    echo "wrote $OUT ($(du -h "$OUT" | cut -f1), ${#DIRS[@]} run dirs)"
    echo "pull with:  rsync -av dice-gala:${OUT} ./"
    ;;

  *)
    cat <<'EOF'
usage: bash scripts/run_gpt2_124m_cluster.sh <stage>

The GPT-2 124M campaign: 12 layers, 12 heads, 768 wide, vocab 50257.
Everything in the current report was measured at <= 16.03M parameters; this
script exists to test the two claims that scale was too small to settle.

  env          GPU / disk / torch fingerprint
  verify       preflight + assert the config really is GPT-2 124M
  gates        correctness suite (CPU, fp64) -- run first, every time
  router       #1 open item at real scale: auto vs forced tfirst vs dfirst
  headline     THE experiment: m=8 T=1024, where autojac needs ~44 GiB on a
               24 GiB card and jdgram needs ~12
  crossover    bracket the m at which autojac stops fitting, at T=512
  ladder       (m, T) shape sweep, smallest first so an OOM keeps prior rungs
  aggregators  Mean/UPGrad/MGDA/PCGrad x jdgram/autogram/autojac at 124M
  accuracy     L7 loss curves at 124M
  trace        L9 kernel + dispatch capture
  overnight    all of the above, guarded and logged, cheapest evidence first
  stats        aggregate v$VERSION runs into compact statistics
  bundle       tar.gz the runs for transfer

An OOM in `crossover` is a RESULT, not a failure -- every stage is guarded so the
boundary gets recorded and the campaign continues.

env vars: GPU=0  VERSION=9  PY=python  STEPS=200
          M_LADDER="4 6 7 8 9 12"        crossover bracket
          T_CROSS=512  M_ROUTER=4  T_ROUTER=512  M_AGG=4  T_AGG=128
          RUNGS="2:256 4:256 8:512"      ladder rungs, as m:T pairs
          JDGRAM_BRUTE_BUDGET_GIB=6.0    float64 brute-force anchor budget;
                                         8*m*P bytes, so m=4 at 124M is 4.0 GiB
EOF
    ;;
esac
