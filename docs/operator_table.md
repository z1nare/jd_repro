# Operator table

The human-readable view of `src/jdgram/engine/registry.py`. Every operator the
engine can meet, the identity that covers it, the route the cost model should
pick, and — the column that matters — whether a gate has actually verified it.

`status: pending` means the algebra is written down and nothing more. Nothing
ships on the strength of a derivation.

## Parameterized operators

| Operator | Identity | Per-objective rank at a layer | Route | Gate | Status |
|---|---|---|---|---|---|
| `nn.Linear` (rank-1, one position per objective) | Hadamard $(AA^\top)\odot(XX^\top)$ | 1 | closed form | legacy gate 4 | **proven** |
| `nn.Conv2d` (grouped/depthwise) | batched per-group $JJ^\top$ | — | materialize (small $P_\ell$) | legacy gate 4 | **proven** |
| `nn.Linear` (sequence, $U=BT$ positions) | II.1 $\langle A_iA_j^\top, XX^\top\rangle_F$ | up to $U$ | cost model | 5a, 5b | pending |
| Linear bias | II.1 bias term, $[m,d_\text{out}]$ materialization | — | materialize | 5c | pending |
| `c_attn` (fused QKV) | II.1, `out = 3d` | up to $U$ | cost model | 5b | pending |
| `attn.c_proj`, `mlp.c_fc`, `mlp.c_proj` | II.1 | up to $U$ | materialize at $m\le8$ | 5b | pending |
| `lm_head` (vocab head) | II.1; I.4 shortcut under GRPO seeds | up to $U$ | **closed form** | 5a, 6 | pending |
| `wte` token embedding | II.2 indicator kernel | up to $U$ | closed form | 5d | pending |
| `wpe` positional embedding | II.3 diagonal shortcut | up to $U$ | closed form | 5d | pending |
| tied `wte` = `lm_head` | II.4, four permutation terms | up to $U$ | closed form | 5e | pending |
| `nn.LayerNorm` $\gamma,\beta$ | II.5, $[m,d]$ materialization | — | materialize | 5c | pending |
| `RMSNorm` $\gamma$ | II.5 | — | materialize | nanochat port | pending |
| `nn.BatchNorm*` | none — cross-instance coupling breaks the decomposition | — | **out of scope** | — | excluded |

## Parameter-free operators (II.6 — contribute no Gramian terms)

| Operator | Handling | Gate | Status |
|---|---|---|---|
| `nn.ELU` | explicit propagation rule (legacy engine) | legacy gate 4 | **proven** |
| `nn.MaxPool2d` | explicit propagation rule (legacy engine) | legacy gate 4 | **proven** |
| `nn.Flatten` / reshapes | explicit propagation rule (legacy engine) | legacy gate 4 | **proven** |
| Softmax / SDPA / Flash-Attention | autograd propagates $A$; no registration | 5b | pending |
| GELU / SiLU | autograd propagates $A$; no registration | 5b | pending |
| Residual adds | autograd propagates $A$; no registration | 5b | pending |
| RoPE | autograd propagates $A$; no registration | nanochat port | pending |

Attention appears here rather than above because it holds no parameters of its
own — its parameters are the four Linears. This is why the port to nanochat is
re-registration rather than new math.

## Cost column

Left empty deliberately. It gets filled from `bench/crossover.py` measurements,
not from the analytic estimates in the design doc — those are in
`src/jdgram/costmodel.py` as starting hypotheses to confirm or refute.
