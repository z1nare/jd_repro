#!/usr/bin/env bash
# PLACEHOLDER -- cluster launcher for bench/crossover.py.
#
# Replaces the six one-off .sbatch files from the CIFAR era. Keep launchers
# parameterized rather than adding a new file per experiment.
#
# Fill in the partition/gres lines to match the target cluster; fuji2 is direct
# SSH with no scheduler, landonia is SLURM. See docs/notes/cluster.md.
set -euo pipefail

python bench/crossover.py --out results/transformer
