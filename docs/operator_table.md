# Operator table

Which operators support an exact Gramian identity, at what cost, and what backs
the claim. Parameter-free ops need no identity at all: ordinary autograd
propagates the upstream gradient `A` through them, which is why supporting a new
architecture is re-registration rather than new mathematics.

Gate status is against brute-force `autograd.grad` on a tiny nanoGPT
(`gates/`), float64, `rtol=0, atol=1e-10`, each test run twice for the `bias`
and `no-bias` fixtures.

| Operator | Parameterized | Identity | Structure | Route | Gate |
|---|---|---|---|---|---|
| `nn.Linear` (QKV, proj, MLP, `lm_head`) | yes | `linear.sequence_gramian` | rank up to `T` under per-sequence losses | both routes; router picks | 5a, 5b — pass |
| `nn.Linear` bias | yes | bias term `Σ_t A`, then outer product | `[m, d_out]` | d-first | 5b, 5c — pass |
| `nn.Embedding` (token, `wte`) | yes | `embedding.sequence_gramian` (index-equality kernel) | same contraction shape as Linear | both routes | 5d — pass |
| `nn.Embedding` (positional, `wpe`) | yes | `positional_embedding_gramian` | diagonal in `t` | d-first | 5d — pass |
| LayerNorm / RMSNorm | yes | `norm.norm_gramian` | `[m, d]` for `γ`/`β` | d-first (`d` is small) | 5c — pass |
| **Tied `wte` = `lm_head`** | yes, shared | `tied.tied_gramian`, **four terms** | `G_hh + G_ee + G_he + G_ehᵀ` | closed form + index-gather cross | **5e — pass** |
| Softmax / SDPA / FlashAttention | no | none | — | autograd only | transitively, 5b–5e |
| GELU / SiLU | no | none | — | autograd only | transitively, 5b–5e |
| Residual add, Dropout (`p=0`) | no | none | — | autograd only | transitively |
| RoPE | no | none | — | autograd only | not exercised (no RoPE model in-repo) |
| Reverse-driver equivalence | — | — | 3 strategies must agree | — | 5f — pass |
| Route equivalence | — | — | 2 contraction orders must agree | — | 5g — pass |
| BatchNorm | yes | **out of scope** | couples batch elements, so per-instance objectives are not independent | — | rejected by the registry |

## Notes

1. **Exactness.** Every implemented identity is algebraically exact for
   `G = J Jᵀ`. Nonlinearities are not approximated — they contribute no Gramian
   terms whatsoever, and only shape how `A` propagates.

2. **The head is not rank-1 here.** `per_sequence_losses` averages over tokens,
   so the LM-head gradient has rank up to `T`. This is exactly the assumption
   that the earlier convolutional work (one objective, one position) got for
   free and a transformer does not.

3. **Tied weights are the case that distinguishes this engine.** A parameter
   reached through two modules has a per-objective gradient that is the *sum*
   over its sites, so the Frobenius product carries four terms. Summing
   per-module Gramians computes two of them. The engine computes all four and
   **refuses to run** on a tied model without an explicit shared handler rather
   than return a structurally plausible wrong number.

4. **Route selection is a known weak point.** `engine/router.py` decides on
   workspace alone (`tfirst iff m·T² < P_layer`). Measured at vocabulary scale
   that picks the slower route for the LM head, for a memory saving that does
   not materialise at model scale. See `jdgram/costmodel.py` for the mechanism
   and the numbers. Both routes are numerically identical, so this costs time
   only.

5. **Not implemented.** `seeds.grpo`; a module called more than once per forward
   (the engine raises `NotImplementedError`; TorchJD's `remaining_counter` is
   the model to follow); LoRA registration, though the identity needs no change
   since `W_eff = W + BA` is the same form with `d_out → r`.
