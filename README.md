# jdgram

Exact Gramians for Jacobian Descent, without materializing the Jacobian.

Jacobian Descent optimises several objectives at once by aggregating their
gradients — UPGrad, MGDA, PCGrad and friends all take the same input: the
Gramian `G = J Jᵀ` of the `[m, P]` per-objective Jacobian. `G` is `[m, m]`. `J`
is the size of the model times the number of objectives, and materializing it is
what makes the method expensive.

This library computes `G` exactly without ever forming `J`. Each layer family
contributes a closed form in two quantities the backward pass already has — the
upstream gradient `A` at the layer's output and its input `X` — so the
`[m, P_layer]` gradient block need not exist either.

```
G_ij = Σ_layers ⟨ ∂L_i/∂W , ∂L_j/∂W ⟩_F
```

## Status

Built and gated against brute-force `autograd.grad` in float64 at
`atol=1e-10`: Linear (both contraction orders), Linear bias, LayerNorm/RMSNorm,
token and positional embeddings, and the **tied embedding/head**. Attention
needed no new mathematics — it holds no parameters of its own, and
softmax/SDPA/GELU/residual contribute no Gramian terms at all.

**71 gates, all passing.** Nothing ships until its gate does.

```bash
pytest gates/
```

## What is here that TorchJD does not have

[TorchJD](https://github.com/TorchJD/torchjd) is the reference implementation
and this project takes its reverse-pass design directly: one ones-seeded
backward, each module's contribution computed inside its own backward hook, its
state released as soon as it is squared. Two things differ.

**Tied weights.** Modern LLMs tie the token embedding to the LM head — one
parameter reached through two modules. Its per-objective gradient is the sum
over both sites, so the Frobenius product has four terms. Summing per-module
Gramians computes two of them. `autogram` creates one Gramian computer per
module with no coupling between them, and measurably deviates from brute-force
autograd on a tied model. This library computes all four terms, and refuses to
run on a tied model without an explicit handler rather than return a
structurally plausible wrong answer.

**Two contraction orders.** Writing the Gramian with all four indices,

```
G_ij = Σ_{t,s,p,q}  A_i[t,p] X_i[t,q] A_j[s,p] X_j[s,q]
```

you may contract positions `(t,s)` first or features `(p,q)` first, and no third
pairing stays exact. `autogram` builds `[m, P_module]` explicitly and squares
it, which is the second of these and always costs `m · P_layer`. The first costs
`3 m T²` and never forms that block — which is the cheaper one whenever
`P_layer` is large, as it is for a vocabulary head. `engine/router.py` chooses
per layer.

The choice is made from measured per-kernel timings fitted on the target card
(`jdgram/costmodel.py`, calibrated by `bench/calibrate_router.py`); with no
calibration loaded it falls back to comparing workspace alone. Both routes are
numerically identical, so the choice costs time, never accuracy.

## Layout

```
src/jdgram/
  identities/   one module per layer family, each returning an [m,m] block
  engine/       hooks.py is the entry point; router.py picks the contraction
                order; accumulate.py holds tied groups until every site is seen;
                residual.py covers parameters with no closed form
  costmodel.py  measured per-kernel cost model behind the router's choice
gates/          71 correctness tests against brute-force autograd, float64
bench/          profiling harness; profile_suite.py has levels L0-L11
scripts/        cluster runbooks (these assume a GPU box, not a laptop)
models/         karpathy's nanoGPT, vendored verbatim — see nanogpt/UPSTREAM.txt
docs/           operator table and the derivation for the T=1 collapse
```

## Running

```bash
pip install -e ".[bench]"
pytest gates/
```

The benchmarks need a GPU. Every run writes to a tagged directory recording the
git SHA, the environment and the full diff if the tree is dirty:

```bash
python bench/profile_suite.py --version 1 --name baseline \
    --levels L0 L2 L4 L5 --device cuda --m 8 --T 512
python bench/profile_stats.py results/v1_*/
```

The cluster runbooks wrap this — `scripts/run_profile_cluster.sh` for the
general campaign, `scripts/run_gpt2_124m_cluster.sh` for GPT-2 124M.

## Results

Campaign results are build artifacts, not repository contents: run directories
are written under `results/`, and `bench/make_evidence.py` and
`bench/report_figures.py` turn them into tables and figures. Both are
regenerable from a run directory plus the script, and neither is tracked.

## Credits

`engine/node.py`, `engine/edges.py` and the hook-injection and gradient-edge
bookkeeping in `engine/hooks.py` are ported from TorchJD 0.17's autogram engine
(MIT), as is the reverse-pass strategy. `models/nanogpt/` is karpathy's nanoGPT,
vendored verbatim.

The method is from [*Jacobian Descent for Multi-Objective
Optimization*](https://arxiv.org/abs/2406.16232).
