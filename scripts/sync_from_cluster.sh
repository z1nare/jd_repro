#!/usr/bin/env bash
# PLACEHOLDER -- pull results back from the cluster.
#
# The excludes are the point. A previous pull tried to bundle ~280 GB because
# regenerable init checkpoints and profiler traces were included; the archive
# step produced a 125 GB temp file before it was killed. Anything regenerable
# or trace-shaped stays on the cluster.
set -euo pipefail

REMOTE="${1:?usage: sync_from_cluster.sh user@host:/path/to/results}"
LOCAL="${2:-results/transformer}"

rsync -avz --progress \
  --exclude 'init_*.pt' \
  --exclude '*_trace.json' \
  --exclude '*.tar.gz' \
  --exclude '__pycache__' \
  "$REMOTE" "$LOCAL"
