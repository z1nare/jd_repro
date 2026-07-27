# Cluster notes

PLACEHOLDER — carry the working cheat sheet over here.

Two environments, different access models:

**fuji2** — 4× RTX A5000 (24 GB), direct SSH, no scheduler. Launch with
`nohup ... & disown` or `tmux`; pick a GPU with `CUDA_VISIBLE_DEVICES`.
Produced the results in [`../results_cifar.md`](../results_cifar.md).

**landonia** — SLURM. Submit with `sbatch`; see `scripts/crossover.sbatch`.

## Lessons that cost time, worth keeping

- `/mnt/raid0sata1` needs admin provisioning; without it, write results to the
  home directory rather than fighting the permission error.
- Init checkpoints (`init_*.pt`) are regenerable cache and grow to hundreds of
  GB across a width sweep. Never include them in an archive or an rsync —
  a `zip -r` over them produced a 125 GB temp file before it was killed.
  `scripts/sync_from_cluster.sh` has the excludes.
- Parallel per-GPU jobs writing identically named output files silently
  overwrite each other. Give each job its own output directory and merge
  afterwards.
- Profiler traces are large and not needed off-cluster; inspect them in place
  or on Perfetto, do not sync them.
