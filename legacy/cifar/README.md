# Legacy CIFAR harness (frozen)

`iwrm_bench.py` and `run_all.py` are the harness that produced
[`docs/results_cifar.md`](../../docs/results_cifar.md) and the artifacts in
[`results/cifar_fuji2/`](../../results/cifar_fuji2/). They are kept **unchanged**
so that report stays reproducible, and they are not developed further.

The Gramian engine they call has moved. `hadamard.py` was split into
`src/jdgram/` during the Phase-0 restructure:

| was | now |
|---|---|
| `hadamard.algorithm3` | `jdgram.engine.sequential.sequential_gramian` |
| CE seed (inline) | `jdgram.seeds.softmax_cross_entropy` |
| Linear branch | `jdgram.identities.linear.rank1_gramian` |
| Conv2d branch | `jdgram.identities.conv` |
| ELU / MaxPool / Flatten | `jdgram.identities.propagation` |

The split is behaviour-preserving and `gates/test_legacy_cifar.py` re-runs the
old preflight gate 4 against the relocated code to keep it that way.

To run this harness again, the import at the top of `iwrm_bench.py`
(`from hadamard import algorithm3`) needs repointing at
`jdgram.engine.sequential`. That edit is deliberately **not** made here: the
files are frozen as the artifact of record. Do it in a working copy.

New work goes in `src/jdgram/`, `gates/` and `bench/`.
