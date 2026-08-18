# Research plan — building the understanding to find a novel contribution

Not an implementation plan. This is what to read, what to be able to derive
unaided, and which cheap probes build the intuition that generates ideas.

Assembled from a parallel literature sweep (four lenses) plus an adversarial
novelty review. **Verify every citation before you use it** — the reviewer
claims to have checked each arXiv ID, but treat that as a starting point, not
as confirmation.

---

## 0. The uncomfortable fact, first

**Your headline finding is already in the literature.** "Gradient conflict is
largely absent in LM pretraining" is not new:

- **PiKE** (arXiv:2502.06244) builds an entire data-mixing method on exactly
  that premise.
- **"SFT Conflicts, RL Coexists"** (arXiv:2608.03573) reports RL updates are
  sparse and approximately orthogonal across tasks.
- **Wu et al., "Imbalanced Gradients in RL Post-Training of Multi-Task LLMs"**
  (arXiv:2510.…) argues the RL problem is magnitude imbalance, not conflict.
- **Sener & Koltun (2018)** already noted, as a throwaway justification for an
  assumption, that a singular Jacobian means the tasks are linearly related and
  no trade-off is needed.

You are not discovering the no-conflict regime. **You are measuring it exactly,
for the first time, with a calibrated instrument.** That is a methods
contribution, and it is still worth something — but reposition now, deliberately,
rather than after a reviewer does it for you.

**Second methodological correction.** The novelty argument "I could find no
paper" failed four times under checking during this review. It is not an
argument. For every direction you pursue, name the specific paper that would
kill it and confirm you have read that paper.

---

## 1. Week one: two instrument checks that could invalidate everything

Do these before anything else. Both are offline, need no Qwen, no GPU-week.
Either one coming back positive reframes the whole project.

### Check A — does the aggregator's choice survive Adam?

Every convergence guarantee in this literature is for **plain gradient
descent**. Nobody trains LLMs that way. "Non-conflicting" is defined against the
raw gradient `d`, but the update actually applied is `P·d` with Adam's diagonal
preconditioner.

Log, for a few hundred steps, side by side:

```
cos(d_agg, d_mean)          # before the optimiser
cos(Δθ_agg, Δθ_mean)        # after it
```

**If the second collapses toward 1 while the first does not, the aggregator's
decision never reaches the weights** — and every aggregator comparison in your
report is measuring a direction that gets erased. That is one figure and it
reframes the entire project.

Closest prior work you must read first: **MAdam (arXiv:2606.03904)**, which
already proves sign-flip conditions and already contains the phrase that Adam's
adaptive metric "turns aligned objectives into apparent conflicts". What
survives: MAdam preconditions *after* aggregation using an EMA diagonal-Fisher
whose off-diagonals the authors themselves call small and noisy, and it has no
language model at any scale. You have the exact `G_P = J P Jᵀ`.

Nice constraint that shows the idea is grounded in your own code: a per-parameter
Adam preconditioner is not Kronecker-factored, so tfirst cannot absorb it — but
dfirst can (scale B by √P before the Gram). Your cost model already says dfirst
is cheaper at m≥4 and at vocabulary scale, so the constraint is free.

### Check B — put error bars on your cosines

Your +0.86 is a point estimate from noisy minibatch gradients, and the noise
**attenuates it toward zero**:

```
E‖ĝ_i‖² = ‖g_i‖² + tr Σ_i        (denominator inflated)
E[ĝ_i · ĝ_j] = g_i · g_j          (off-diagonal already unbiased, i≠j)
```

Fix: split each objective's batch in half and estimate `‖g_i‖²` by the
cross-half inner product `ĝ_i^a · ĝ_i^b`, which is unbiased. **In your engine
that is one 2m×2m Gramian instead of a second full set of backward passes** —
and your cost model prices it exactly, so you can state the overhead rather than
hand-wave it.

**The correction should push +0.86 *up*, toward +0.95** — making your
no-conflict finding stronger, and pushing the Gramian closer to genuinely
rank-deficient, which matters for everything in §2.

Then apply it to one externally-reported *conflicting* setting and ask whether
the conflict survives debiasing. Caveat: split-half U-statistic correction is
standard in kernel two-sample testing and has been applied at LLM scale
(arXiv:2606.27242) — this is a robustness contribution, not a new estimator.

---

## 2. The thesis direction: the partition, not the aggregator

**This is the strongest idea in the set and the only one the reviewer graded
"strong". It is also the formal version of Rui's question.**

Rui asked: *"what is the best loss function?"* Reframed precisely:

> **What is the best partition of the loss into Jacobian rows?**

The entire aggregator literature takes the objective set as **exogenous**.
TorchJD's IWRM picks per-example rows with no justification for that granularity.
GRAIN does min-norm aggregation over group gradients but takes the grouping as
given. Nobody treats the row partition as a design variable **with a cost
attached** — and you are the only person who can supply a calibrated cost
denominator.

Two claims to establish:

**(1) A coarsening lemma.** If partition Q refines P, the update directions
reachable under P are a strict lower-dimensional slice of those reachable under
Q, because rows of `J_P` are fixed sums of rows of `J_Q`. Summing is
irreversible: **the partition, not the aggregator, sets the ceiling on what can
be expressed.**

**(2) An empirical granularity curve.** Does refinement ever change the
direction actually taken, and by how much per unit of compute? Plot information
gained against FLOPs and *derive* the optimal m rather than asserting it.

**The sharp application:** the advantage-cancellation problem that GDPO
(arXiv:2601.05242) and GD²PO (arXiv:2606.16771) both patch with loss surgery is
a **partition artifact** — it exists only because per-reward advantages are
summed before differentiation, and is structurally impossible under per-reward
rows. Neither paper asks whether the sum was the mistake.

**Why you can do it this week, with zero new training runs.** A saved m=16
Gramian coarsens to m=8/4/2 by block-summing: `G_P = S G_Q Sᵀ` for a 0/1 merge
matrix S. Every granularity is an m×m matmul on data you already have. And your
headline ladder already isolates objective count from data volume by repeating
the same sequence — the control most MTL papers lack.

**First probe (one week, offline):** take the logged m=16 Gramians, build four
coarsening levels, run Mean/UPGrad/MGDA/PCGrad at each, report (a) angle between
resulting directions across granularities, (b) same on your synthetic
conflicting probe as positive control, (c) angle-per-millisecond from the cost
model. Write the lemma.

**Kill criterion:** if directions at m=2 and m=16 agree within 1° in every
regime *including* the deliberately conflicting probe, granularity carries no
information — write a two-page negative note and stop.

---

## 3. The post-training measurement, rescoped

Do the Qwen3 GRPO work — but **not** as "is post-training conflicted" (answered
three times already) and **not** as "JD beats GDPO" (a benchmark race you lose
on compute).

Ask instead: **do the standard approximations change the published conclusions
about RL gradient geometry?**

Everyone measuring this uses LoRA deltas, EMA gradients, sublinear compression,
or module-level proxies. **You have the only exact instrument.** That question
is falsifiable, self-contained, needs no baseline-beating, and makes your engine
load-bearing rather than incidental.

Must read before starting — this is the closest hit in the entire sweep and it
is currently missing from your related work: **"Modular Gradient Surgery"
(arXiv:2602.02301)**, which already measures cosine similarity between objective
gradients during LLM RL post-training including reward-vs-KL, and already applies
conflict-triggered surgery at module granularity.

**Hardware reality:** plan **Qwen3-0.6B, not 1.7B**, with vLLM rollouts in a
separate process. The engine port is genuinely cheap — Qwen3 parameters resolve
to existing Linear/Embedding/RMSNorm handlers, and GQA/SwiGLU add no new
parameterised ops. The unbudgeted cost is the **151,936-token vocabulary, 3× GPT-2's**:
the lm_head Gramian will dominate time and peak memory, and dfirst is the only
viable route there. **Budget three weeks for integration, not one.**

---

## 4. Reading list

### Tier 1 — read this month, in this order

1. **Quinton & Rey, "Jacobian Descent for Multi-Objective Optimization"**
   (arXiv:2406.16232). You implement this; now read it as an adversary. Extract
   three things: the three aggregator axioms (non-conflicting / weighted /
   linear-under-scaling) and why UPGrad is claimed unique in satisfying all
   three; **that their convergence theorem assumes β-smooth AND ⪯-convex with
   bounded Pareto front — none of which a transformer satisfies**, so the
   guarantee you cite does not apply to anything you have run; and that SEJD/SSJD
   (the stochastic variants) have no proof at all.

2. **Hu, Ho & Yu, "A Unified Framework for Gradient Aggregation in MOO"**
   (arXiv:2605.30452). Replaces about six method papers of theory. Gives a
   sufficient alignment condition and proves any direction in *both* the convex
   hull and the dual cone gets O(1/√t) to Pareto stationarity. MGDA, Nash-MTL,
   UPGrad, DualProj get guarantees directly; scalarization, CAGrad, PCGrad,
   IMTL-G only via a separate construction. **The analysis is full-batch
   deterministic and the authors name the stochastic extension as open — your
   engine is arguably the best instrument in existence for probing it.**

3. **Désidéri (2012)** + **Fliege & Svaiter (2000)**. Six and twelve pages; one
   sitting. Read to internalise that the guarantee is Pareto **stationarity** =
   "0 is in the convex hull of the gradients". That reframes your finding
   geometrically: with all cosines ≥ +0.66 the hull is a thin sliver nowhere
   near the origin, so **every aggregator is just gradient descent in a slightly
   rotated direction** — and MGDA's chaos becomes inevitable rather than a bug,
   because the min-norm point over a near-degenerate simplex moves
   discontinuously while the objective barely changes.

4. **Sener & Koltun (arXiv:1810.04650)**. Read for their Theorem 1 assumption
   and its justification — your entire GPT-2 finding, stated in passing in 2018.
   Also their closed-form two-task case; differentiate it by hand to quantify
   your MGDA instability.

5. **Zhou et al. (NeurIPS 2022)** + **MoCo (arXiv:2210.12624)**. The answer to
   "do the guarantees survive minibatching?" — no. Weights `w` are a nonlinear
   function of noisy gradients, so `E[A(Ĵ)] ≠ A(E[J])`, and the bias can be
   severe enough that the expected direction conflicts with *all* full-batch
   gradients. **Consequence for you: when you say MGDA is chaotic, be explicit
   whether you mean solver instability (numerical) or estimator bias
   (statistical). They are different problems with different fixes and
   conflating them will get you correctly criticised.**

### Tier 2 — the empirical case against your whole field

Read as one block. These collectively argue tuned scalarization is hard to beat,
which is the null hypothesis your project must defeat:

- Kurin et al., "In Defense of the Unitary Scalarization" (arXiv:2201.04122)
- Xin et al., "Do Current Multi-Task Optimization Methods Even Help?"
- Hu et al., "Revisiting Scalarization in MTL: A Theoretical Perspective"
- Elich et al., "Examining Common Paradigms in Multi-Task Learning"
- Chen et al., "Three-Way Trade-Off in Multi-Objective Learning" (arXiv:2305.20057)

### Tier 3 — where conflict actually lives

- **Modular Gradient Surgery** (arXiv:2602.02301) — closest prior work, read first
- Gradient Vaccine (ICLR 2021) — multilingual, tracks cosines over training
- CONGRAD — conflicting-gradient filtering for multilingual preference alignment
- Safe RLHF (Dai et al.); Moskovitz et al., "Confronting Reward Model
  Overoptimization with Constrained RLHF" — constrained/Lagrangian framing
- Yuan et al., "Gradient Entanglement" — margin-based alignment pitfall
- Ren & Sutherland, "Learning Dynamics of LLM Finetuning" (ICLR 2025 Outstanding)
- Lopez-Paz & Ranzato, GEM (NeurIPS 2017) — the original conflict-projection idea

### Tier 4 — your engine's prior art under other names

You will be asked "isn't this just the NTK / Gauss-Newton / Fisher?" Have an
answer:

- Martens, "New Insights and Perspectives on the Natural Gradient Method"
- Kunstner, Hennig & Balles, "Limitations of the Empirical Fisher Approximation"
- Dangel et al., **ViViT** — curvature access through GGN low-rank structure;
  closest structural cousin to what you built
- Novak et al., "Fast Finite Width Neural Tangent Kernel"
- Li et al., "LLMs Can Be Strong Differentially Private Learners" — **ghost
  clipping** computes per-sample gradient norms without materialising gradients,
  the closest thing to your trick in a different community
- Gur-Ari, Roberts & Dyer, "Gradient Descent Happens in a Tiny Subspace" —
  why your Gramian is near-rank-deficient in the first place

---

## 5. Concepts you must be able to derive unaided

Test yourself: can you answer each without looking?

| Concept | You have it when you can… |
|---|---|
| Pareto stationarity vs optimality | state why "0 ∈ conv(gradients)" is the actual guarantee, and why cosines ≥ +0.66 put you nowhere near it |
| Dual cone vs convex hull | say why modern guarantees need membership in **both**, and which aggregators fail which |
| MGDA in Gramian form | write `min_{w∈Δ} wᵀGw`, get `γ* = H⁻¹1 / 1ᵀH⁻¹1`, and show the diagonal-G case gives `w_i ∝ 1/‖g_i‖²` in one line |
| Conditioning of the subproblem | explain why near-parallel gradients make the argmin discontinuous while the objective value barely moves |
| Stochastic bias | explain why `E[A(Ĵ)] ≠ A(E[J])` and why that is a *different* problem from solver instability |
| Concentration of measure | explain why you measured +0.86 and not ~0 — and why random high-dimensional vectors would give ~0 |
| Dynamic scalarization | say which "multi-objective" updates are secretly just adaptive loss weighting, and which genuinely are not |
| What G alone recovers | list what the m×m Gramian determines (all pairwise angles, the rank, the reachable direction set) and what it cannot |
| The identity web | distinguish Gramian / empirical NTK / GGN / Fisher / empirical Fisher, and say which of the four your G actually is |
| Gram–Woodbury duality | explain why `JJᵀ` (m×m) and `JᵀJ` (D×D) share nonzero spectra, and what that buys you |
| Lagrangian duality as loss weighting | show that `max R s.t. C ≤ d` has a dual whose multipliers *are* adaptive objective weights — the bridge to Rui's question |

---

## 6. Ordering

| When | What |
|---|---|
| **Week 1** | Check A (does the choice survive Adam) and Check B (debiased cosines). Nothing else until both are done. |
| **Weeks 2–6** | The partition thesis (§2), entirely offline on existing Gramians. Spectral analysis folded in as its analysis section — **not** written up as a theorem; the rank-one and diagonal limits are one-line substitutions into MGDA's published closed form and a reviewer will say so. |
| **Weeks 4–12** | Qwen3-0.6B port in parallel, framed as §3's approximation question. This also generates the data the partition work needs at post-training scale. |
| **Dropped** | The "predict when JD is worth 2.3x" idea. It forecasts an effect your own data says is zero, and it is a grid sweep in disguise. |

---

## 7. One caveat about this document

The "ambitious" idea-generator in the sweep died on a connection error, so the
high-risk/high-reward quadrant is under-explored here. Notably the reviewer
arrived independently at the reframe that agent was meant to find — *use the
Gramian as a measurement instrument rather than an optimiser component* — which
is the thread running through §1, §2 and §3. Worth revisiting deliberately once
the week-one checks are done.
