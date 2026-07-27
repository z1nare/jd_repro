# jdgram — exact Gramian engines for Jacobian Descent

Computing `G = J Jᵀ` for multi-objective optimisation without materialising the
`[m, P]` Jacobian, so Jacobian Descent runs at LLM scale.

The deliverable is **one exact engine that picks the cheapest correct identity
per layer**: a closed form where materialising `J_ℓ` would be the memory wall
(the vocab head, tied embeddings, or large `m`), and plain `J_ℓ J_ℓᵀ`
materialisation where it would not (interior linears at small `m`). Both routes
produce the same `G`, so the choice is engineering, and the crossover comes from
measurement rather than taste.

Design doc: **[`docs/design/gramian_engines.md`](docs/design/gramian_engines.md)**.

## Status

**Proven.** The CIFAR/IWRM path. The Hadamard-factorised engine matches
TorchJD's `autogram` to 2.8e-14 at float64, is the fastest of the three engines
at paper scale, and reaches 1.65B parameters on a 24 GB A5000 against TorchJD's
134M ceiling — about 12×. Full report:
[`docs/results_cifar.md`](docs/results_cifar.md).

That result was earned at `m=32` with rank-1 per-layer gradients. The transformer
target has `m=2–8` and rank-up-to-`BT` gradients, which changes which engine wins
per layer — hence the hybrid framing rather than "Hadamard everywhere".

**In progress.** Transformer identities, one gated layer at a time, in a nanoGPT
sandbox. Nothing ships until its gate passes against brute-force autograd. Gate
status per operator: [`docs/operator_table.md`](docs/operator_table.md).

## Layout

| Path | What |
|---|---|
| `src/jdgram/identities/` | one module per layer family |
| `src/jdgram/engine/` | hook plumbing, registry, route selection |
| `gates/` | equivalence tests vs brute-force autograd — the preflight |
| `bench/` | timing, capacity, cost-model crossover measurement |
| `models/nanogpt/` | pinned upstream `model.py`, the gate sandbox |
| `legacy/cifar/` | frozen harness that produced the CIFAR report |
| `docs/` | design doc, identity index, operator table, results |
| `results/` | `cifar_fuji2/` (frozen) and `transformer/` (new) |

## Getting started

```bash
pip install -e ".[torchjd,dev]"
pytest gates/ -v
```

Only the legacy CIFAR gate is implemented today; the transformer gates skip
themselves until their step lands.

## Working rules

These are what made the CIFAR numbers credible, and they carry over:

- Every gate is a committed script, and no step starts before the previous gate
  passes.
- `dropout = 0.0` in gates, or the brute-force and hooked passes see different
  networks and the comparison proves nothing.
- fp64 for gates; fp32 for real-shape runs.
- No `torch.compile` until all gates pass eager — it is a performance knob, not
  a correctness tool.
- On a gate failure, shrink the hooked-module set before touching the maths.
