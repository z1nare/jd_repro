# Identities — index and gate status

One entry per identity, pointing at the derivation, the implementation and the
gate. The full derivations live in
[`design/gramian_engines.md`](design/gramian_engines.md) Part II; the CIFAR-era
derivation and its provenance live in
[`design/hadamard_cifar_derivation.md`](design/hadamard_cifar_derivation.md).

Every objective's loss is a mean over positions $u = (b,t)$, $U = BT$. $A_i$ is
objective $i$'s upstream gradient at a layer; $X$ is the layer input, shared
across objectives because there is one forward pass.

| # | Identity | Derivation | Implementation | Gate | Status |
|---|---|---|---|---|---|
| — | Rank-1 Hadamard, $(AA^\top)\odot(XX^\top)$ | `hadamard_cifar_derivation.md` | `jdgram.identities.linear.rank1_gramian` | `gates/test_legacy_cifar.py` | **proven**, 2.8e-14 vs autogram |
| — | Grouped Conv2d | `hadamard_cifar_derivation.md` | `jdgram.identities.conv` | `gates/test_legacy_cifar.py` | **proven** |
| II.1 | Linear sequence contraction | Part II.1 | `jdgram.identities.linear.sequence_gramian` | 5a, 5b, 5f | pending |
| II.2 | Token embedding indicator kernel | Part II.2 | `jdgram.identities.embedding` | 5d | pending |
| II.3 | Positional embedding diagonal | Part II.3 | `jdgram.identities.embedding` | 5d | pending |
| II.4 | Tied weights, four cross terms | Part II.4 | `jdgram.identities.tied` | 5e | pending |
| II.5 | LayerNorm / RMSNorm | Part II.5 | `jdgram.identities.norm` | 5c | pending |
| II.6 | Parameter-free ops | Part II.6 | `jdgram.identities.propagation` | inherited | **proven** for ELU/MaxPool |
| I.4 | GRPO head-seed shortcut | Part I.4 | `jdgram.seeds.grpo` | 6 | pending |

## Two things worth restating

**II.1 subsumes the CIFAR result.** The rank-1 Hadamard form is the $U = 1$
special case of the sequence contraction, not a separate tier of mathematics.
The LM head is not mathematically special either — it is special only because
of weight tying (II.4) and the GRPO seed structure (I.4).

**Pending means pending.** These are exact algebra, but the discipline that
made the CIFAR result credible was that gates 1–4 were run and reported, not
that the derivations looked right. Same rule here.
