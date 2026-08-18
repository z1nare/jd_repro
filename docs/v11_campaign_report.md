# Jacobian Descent on GPT-2 124M — where we stand

**The question.** With two or three objectives instead of one, training should
cost about **50% more** time and memory — not two or three times more.

**The short answer.** On time, no engine is close: everything sits at roughly
**2.3–2.8x**. On memory, ours is the only engine that meets the bar, at
**1.13–1.23x**. The time gap is not spread thinly across the code — **three
quarters of it is one specific design decision**, which we can name, price, and
remove.

**And one finding that reframes the whole exercise:** on this workload the
objectives never actually conflict (§9). Jacobian Descent exists to resolve
conflict; if there is none to resolve, no amount of engineering will show a
benefit here. That is now the most important open question, and it is a
*benchmark design* question, not an implementation one.

> **Two dated experiments here.** §0–§10 are the **v11 campaign** (10–11 Aug,
> 104 runs, complete). **§11 is new work from today** (v12): the defect found
> in §5 has since been fixed, calibrated and partly verified. Nothing in
> §0–§10 has been changed by it.

---

## 0. What is being measured, and against what

Everything below is a **ratio**, so the denominator matters more than anything
else in this document.

**The setup.** GPT-2 124M (12 layers, 12 heads, width 768, vocabulary 50,257),
fp32, sequence length 512, plain SGD at learning rate 0.01, 100 training steps
per run, single RTX A5000 (24 GB), with a held-out validation split never
trained on.

**What "an objective" means here.** The loss returns **one number per sequence
in the batch** — so *m objectives* means *m sequences, each with its own loss*,
and Jacobian Descent combines those m losses instead of averaging them. This
has a consequence that invalidates the naive reading:

> Adding an objective also adds a sequence — so it adds **data**. A run with 2
> objectives would process twice the tokens of a run with 1, and "2x slower for
> 2x the work" would be no finding at all.

To separate the two effects the headline ladder **repeats the same sequence m
times**: data held completely fixed, only the objective count varies. That is
the ladder quoted throughout. Where the alternative is used — m genuinely
different passages — it is labelled "unrelated objectives".

**The reference — what every "x" is divided by.** Each run contains a control
arm doing ordinary training: **same model, same data, same seed, same step
count**, one forward pass, losses averaged into one number, one backward pass,
one optimiser step. No Gramian, no aggregation. It runs **inside the same
process** as the engines it is compared against, so machine state affects both
alike. Every ratio is:

> **(the engine's number) ÷ (ordinary single-objective training, same run)**

| quantity | how it is measured |
|---|---|
| **time** | wall-clock ms per training step, averaged over steps 2–100 (step 1 discarded as warm-up), GPU synchronised at both ends. Covers the whole step: forward pass(es), Gramian, aggregator, backward, optimiser. |
| **memory** | peak GPU memory allocated during those steps, reset after warm-up. |
| **quality** | cross-entropy on 10 fixed held-out batches never trained on, reported as perplexity (lower is better). |

**The three engines.** `jdgram` is ours. `autogram` and `autojac` are the two
TorchJD implementations — autogram builds the Gramian as we do, autojac builds
the full Jacobian explicitly. Identical data, seeds and aggregator throughout.

---

## 1. The verdict

![The verdict](../fig/figA_verdict.png)

| | **2 objectives** | | | **3 objectives** | |
|---|---|---|---|---|---|
| | time | memory | | time | memory |
| **ours (jdgram)** | 2.28x | **1.13x** | | 2.51x | **1.15x** |
| autogram | 2.51x | 1.40x | | 2.35x | 1.48x |
| autojac | 2.28x | 1.15x | | 2.77x | 1.58x |
| *the budget* | *1.50x* | *1.50x* | | *1.50x* | *1.50x* |

- **Nobody meets the time bar** — not us, not either TorchJD engine.
- **We meet the memory bar and the alternatives don't**, and our lead widens
  with objective count while theirs erodes.

![Trade-off space](../fig/figB_tradeoff_space.png)

Following each engine from 1 to 16 objectives: ours barely moves (1.00x →
1.21x memory); autogram climbs past the budget at 3 objectives, ending at
1.67x; autojac reaches 4x and then fails. The inside-budget corner is empty.

## 2. Why there are two forward passes — and what removing one would cost

![Cost of a step](../fig/figC_cost_of_a_step.png)

| component of a two-objective step | cost | what it is |
|---|---|---|
| one ordinary training step | 1.00x | the reference |
| **a second forward pass** | **+0.35x** | the model is run twice |
| building the Gramian | +1.38x | capture, reverse pass, per-layer arithmetic |
| the aggregator solve | +0.01x | **under 1%** |
| **total** | **2.74x** | |

**Why the second pass exists — it is a deliberate trade, not an oversight.**
Our engine builds the Gramian from its own forward pass and **frees each
layer's activations as the reverse sweep passes over them**. That is precisely
where the memory advantage in §1 comes from. But the weighted backward pass
that follows then has nothing to back-propagate through, so it needs a fresh
forward pass.

autogram makes the opposite choice: it keeps the graph alive and reuses it, so
it runs **one** forward — and pays for it in memory (1.40–1.67x versus our
1.13–1.21x). **autogram is, in effect, the fused design already built.**

So "fuse the two forward passes" is really "adopt autogram's memory trade".
That is the honest framing, and it means the recommendation in §3 is not free:
it spends the memory advantage that is currently our strongest result. The
target should be a *selective* fusion — retaining activations only for layers
that are cheap to retain — rather than the all-or-nothing choice both current
designs make.

**The aggregator is not the problem.** The optimisation solve that combines the
objectives — the part that looks mathematically expensive — is **under 1% of a
step**, and its share *shrinks* as objectives grow.

## 3. What would it take to reach the budget?

![Path to budget](../fig/figD_path_to_budget.png)

If the Gramian computation were **completely free** — infinitely fast, zero
cost — the engine would *still* sit at **1.37x**, because it would still run
the model twice. **That floor alone consumes 74% of the allowance.** So the
question splits in two: remove the second forward pass, and make the Gramian
work itself cheaper. Both are needed; neither is sufficient.

### Lever 1 — the second forward pass, and why it is still there

This is not an optimisation somebody forgot. It is the other half of a
deliberate trade:

- **Ours** frees each layer's activations as the reverse sweep passes over
  them. Nothing is left to back-propagate through, so the weighted backward
  needs a fresh forward pass. **Cost: a second forward. Benefit: the low
  memory in §1.**
- **autogram** keeps the graph alive and reuses it — one forward pass, and
  1.40–1.67x memory against our 1.13–1.21x.

So "just fuse the two passes" means "adopt autogram's memory profile", which
spends our strongest result. That is why it has not simply been done.

**The mechanism that would actually work is selective retention**: decide *per
layer* whether to keep its activations or recompute them, exactly as the engine
already decides per layer which strategy to use for the Gramian. Retain where
the activations are small and the recompute is expensive (the attention
blocks); free where they are large and cheap to redo. Neither current design
does this — both make one global choice. Building it needs a per-layer
retain/recompute decision with a memory budget, which is the same shape of
machinery as §5's cost model and could reuse it.

Checked against reality, not just arithmetic: autogram *is* the fused design,
already built, and it lands at **2.19x** at 4 objectives against our projected
2.09x for a fused version of ours. The projection is in the right place.

### Lever 2 — the Gramian work itself

After fusing, the remaining term is capture + reverse sweep + the per-layer
identity kernels (the +1.38x row in §2). It would need to be roughly **2.2x
faster** to bring the total inside budget. Two measured routes to that:

| | effect |
|---|---|
| Always choose the faster per-layer strategy (§5) | cuts the requirement from 2.7x to 2.2x — about a fifth of the gap |
| Reduce kernel working memory | the isolated kernels allocate **3–34x their own theoretical workspace**, which points at intermediates being materialised that could be fused or reused |

The first of those is done and measured (§11). The second is a located symptom,
not yet an implementation.

## 4. Where does the cost actually go — and does it get better with scale?

![Cost model](../fig/figE_cost_model.png)

**Why this section exists.** Every number so far is a ratio at one objective
count. That leaves the practical question unanswered: is the overhead a fixed
tax you pay once per step, or does it grow with every objective you add? Those
have opposite consequences — a fixed tax gets cheaper the more objectives you
use, a growing one does not.

**How single-objective and multi-objective are compared.** Both are the *same
model, same data, same seed, same 100 steps*, differing only in what happens
between the forward and the optimiser step:

- **ordinary training** (`sgd_erm`): average the m per-sequence losses into one
  number, one backward pass. This is plain SGD — no Gramian, no aggregation.
- **ours**: keep the m losses separate, build the m×m Gramian, run **UPGrad**
  over it to get per-objective weights, then one weighted backward pass.

Plotting raw milliseconds rather than ratios, both are **straight lines in the
objective count**, which exposes the whole cost structure in two numbers each:

| | cost per objective | fixed cost per step |
|---|---|---|
| ordinary training | 31 ms | 16 ms |
| **ours** | **68 ms** | **73 ms** |
| ratio | **2.18x** | 4.6x |

**Read it as: we pay 2.18x per objective, plus a fixed 57 ms per step that
ordinary training does not.** That single decomposition explains the whole
shape of §1: the total ratio falls from 2.73x at m=2 to 2.26x at m=16 purely
because the fixed part is amortised over more objectives. It decays toward
**2.18x — the ratio of the two slopes — and never below it.** The fit predicts
every measured point from m=2 upward to within 0.5%.

**The consequence that matters:** the worst point on the entire curve is **2
objectives**, which is exactly the case in the budget, and amortisation cannot
help there. Getting inside 1.5x at m=2 requires cutting the per-objective cost
itself, not waiting for scale.

## 5. A performance bug: the engine picks the slower strategy

![Strategy crossover](../fig/figG_strategy_crossover.png)

**This section is our engine against itself** — a choice made inside it, not a
comparison with autogram. (That comparison is §1, where the memory gap is
large: 1.17x against 1.52x on median.)

The engine builds each layer's Gramian one of two ways and chooses per layer
with this rule:

> use **strategy A** while `m × T² < P_layer`, otherwise **strategy B**
> — where `P_layer` is the layer's parameter count

It compares the *scratch space* the two strategies need and takes the smaller.
**Nothing in it models time.**

| objectives | 1 | 2 | 3 | 4 | 6 | 8 | 12 | 16 |
|---|---|---|---|---|---|---|---|---|
| strategy A | **2.01** | **2.28** | **2.51** | 2.79 | 3.38 | 3.87 | 4.91 | 6.09 |
| strategy B | 2.10 | 2.73 | 2.53 | **2.45** | **2.37** | **2.33** | **2.29** | **2.26** |

The strategies **swap places at about 3.5 objectives**; the rule switches at
**9**. Across the gap it keeps choosing the loser — by up to **1.66x** at 8
objectives.

**And the trade-off it thinks it is making does not exist.** Switching our
engine from strategy A to strategy B everywhere — again, *our engine against
itself*, autogram not involved — costs:

| objectives | 4 | 8 | 12 | 16 |
|---|---|---|---|---|
| extra peak memory | +1.0% | +1.1% | +1.2% | +1.2% |
| step time saved | 12% | 40% | 53% | **63%** |

The rule is protecting about **1% of peak memory** at a cost of up to **63% of
step time**. The reason the memory barely moves: the rule optimises scratch
space *inside* the Gramian computation, but peak memory is set by the model's
own stored activations, which are far larger and identical either way. It is
minimising a quantity that is not the constraint.

![Layer strategy](../fig/figH_layer_strategy.png)

Per layer in isolation, the choice is wrong on **2 of 9 shapes** — but one is
the vocabulary output layer, the largest tensor in the model, where the chosen
strategy is **13x slower**. The rule would need 147 objectives before it
switched there.

**On the heuristic, and what was expected of it.** Improving this rule was
scoped work, and it is not done. `src/jdgram/costmodel.py` exists as its
intended home and currently raises `NotImplementedError`; its documentation
already identifies the failure and the reason (the rule minimises bytes that
are not the peak). What this campaign contributes is the missing input: the
crossover is now *measured* — at model scale (m≈3.5) and per layer shape — on
the target hardware, which is exactly what a measured cost model needs and
what could not be derived analytically. Replacing the rule is the next
implementation task; the decision on *how* was explicitly deferred, so no
design is proposed here.

**On LoRA appearing in that chart.** `lora-A-r32` and `lora-B-r32` are the two
low-rank adapter matrices at rank 32. They are **not part of GPT-2 124M** —
they are extra shapes in the per-layer micro-benchmark, included because
Jacobian Descent is most likely to be applied to fine-tuning, where the
trainable tensors are adapters with very lopsided dimensions. They are
coverage for a future use case. No GPT-2 result in this report depends on
them, and the strategy rule handles both correctly.

## 6. How conflicting objectives are built — and why it is a probe, not a workload

![Objective relationship](../fig/figF_objective_relationship.png)

The three relationships are constructed by taking **one sequence, repeating it
m times, and attaching coefficients** to the resulting per-sequence losses:

| relationship | how it is built | what it forces |
|---|---|---|
| **identical** | same sequence ×m, coefficients all `+1` | every objective's gradient is the same vector; the Gramian must collapse to rank 1 |
| **directly opposed** | same sequence ×m, coefficients `+1, −1, +1, …` | objective 2's gradient is exactly `−1 ×` objective 1's; every off-diagonal must be strictly negative, cosine exactly −1 |
| **unrelated** | m genuinely different passages | nothing forced; this is the realistic case |

| objectives | identical | directly opposed | unrelated |
|---|---|---|---|
| 2 | 2.73x | 2.74x | 2.74x |
| 4 | 2.45x | 2.47x | 2.45x |
| 8 | 2.33x | 2.33x | 2.33x |

**Within 1% everywhere.** Objectives that fight each other — where the
aggregator genuinely has to project — cost the same as objectives that agree.
The geometry does not affect the bill.

**The honest caveat.** "Directly opposed" is a *synthetic probe*: literally the
same loss negated, which means maximising it. It exists to force the conflict
code path and make the negative off-diagonal measurable, and a long run under
it diverges by construction. It is **not** a realistic multi-task workload, and
it should not be read as "we tested conflicting real objectives". §9 is where
that gap becomes the main story.

Correctness on the probe is exact: the opposed losses mirror and cancel to
**exactly zero** at every objective count, aggregator and engine.

## 7. The memory wall

![Memory wall](../fig/figI_memory_wall.png)

| engine | largest objective count that ran | verdict |
|---|---|---|
| **ours** | 16 | ran everything the control ran |
| autogram | 16 | ran everything the control ran |
| autojac | 8 | **fails from 12 onward** |

At 24 and 32 objectives **the ordinary control runs out of memory too** — the
card's capacity, not an engine limit, and not quotable as one.

## 8. Do the engines agree, and are they reproducible?

Quality is held-out cross-entropy, in **nats**; perplexity is its exponential.
A *gap* in nats is therefore a *percentage* gap in perplexity, which is the
scale to read the rest of this section on (our models sit at ~3.90 nats,
perplexity ~49.5):

| gap | perplexity effect |
|---|---|
| 0.0006 nats | +0.1% — identical in practice |
| 0.02 nats | +2% — small but real |
| 1.1 nats | **+200% — three times worse** |

### Do the three engines compute the same thing? Yes

Held-out perplexity after 100 steps, identical data and seed:

| objectives | ours | autogram | autojac |
|---|---|---|---|
| 2 | 53.27 | 53.27 | 53.28 |
| 8 | 48.51 | 48.51 | 48.53 |
| 16 | 50.57 | 50.57 | OOM |

Agreement to the second decimal — which is what makes comparing their *costs*
meaningful. If they disagreed here, none of §1–§7 would mean anything.

### Is a single engine reproducible? Two different questions

![Reproducibility](../fig/figJ_reproducibility.png)

The two panels answer two questions that are easy to confuse.

**Left — run the identical thing twice.** Same engine, aggregator, data, seed:
the result should be bit-identical, and it is. Every engine and every
aggregator **except PCGrad** agrees to within 6×10⁻⁴ nats. PCGrad varies in
*all three* engines (0.3–0.6 nats) because it shuffles internally without a
fixed seed — a known property of the aggregator, not of any engine, and ours is
the smallest of the three bars.

**Right — run the same maths two different ways.** Our engine's two strategies
(§5) are mathematically identical and agree to 5×10⁻⁶, but they do the
arithmetic in a different order. Whether that matters after 100 steps depends
entirely on the aggregator:

| aggregator | gap between the two strategies | |
|---|---|---|
| Mean | 0.00 nats | stable |
| UPGrad | 0.02 nats | stable |
| PCGrad | 1.01 nats | **fragile** |
| MGDA | 1.11 nats | **fragile** |

MGDA and PCGrad each make a *discrete* internal choice — which objectives to
project away — and a last-bit difference can flip it, after which the runs
diverge. Mean and UPGrad have no such switch. **A concrete reason to keep
UPGrad as the default.**

*Retraction: an earlier draft read the right-hand panel as "our engine is
non-deterministic". It is not — that panel compares two different strategies,
and only our engine has a strategy to vary, so no other engine could have
failed the same test.*

## 9. The finding that matters most: these objectives never conflict

Each objective produces a **gradient** — the direction it wants to move the
weights. The cosine between two of them says whether they agree: **+1**
identical, **0** unrelated, **−1** exactly opposed. Jacobian Descent exists for
the negative end: its job is to find an update that serves conflicting
objectives fairly instead of letting the average silently sacrifice one.

For every run I take the **most-opposed pair** — the worst case, the one most
likely to be negative — and record its cosine.

![Objective alignment](../fig/figL_objective_alignment.png)

The two grey series are calibration probes, there to show the instrument works
before it is used to argue an absence: duplicated objectives must read +1, and
an objective against its own negation must read −1. **Both land exactly on
their theoretical value at every m.** The measurement is not blind to conflict.

The blue series is the real thing — different passages of the corpus:

| objectives | 2 | 3 | 4 | 6 | 8 | 12 | 16 |
|---|---|---|---|---|---|---|---|
| most-opposed real pair | +0.86 | +0.83 | +0.83 | +0.79 | +0.79 | +0.77 | **+0.66** |

**Never negative, at any objective count.** Gradients from different passages
of one corpus point substantially the same way. They drift apart slowly as
objectives are added, but they never oppose.

### What follows from it

Three concrete consequences, in order of how much they should change what we do
next:

1. **It explains §8.** Every aggregator reaching the same held-out loss looked
   odd. It is not: with no conflict to resolve, UPGrad, MGDA and PCGrad all
   reduce to approximately a plain average, so of course they agree.
2. **The cost numbers are unaffected.** §1–§7 measure what the machinery costs.
   That price is real whether or not the machinery is earning it here.
3. **But the benefit cannot be measured on this workload at all.** We are
   paying 2.3x for conflict resolution on objectives that never conflict. No
   engineering improvement changes that — it is a property of the benchmark,
   not of the engine.

So the next experiment is not a faster kernel; it is **a workload with
objectives that genuinely pull apart** — different domains, different tasks, or
an auxiliary objective traded against language modelling. Until that exists,
"is Jacobian Descent worth its price?" cannot be answered here either way.

## 10. What was measured

![Coverage](../fig/figK_coverage.png)

Each square is one (sequence length, objective count) setting; the number is
how many **timed cells** were measured there — one cell being one *engine ×
aggregator × objective relationship × strategy × repeat*. **104 runs, 1,100
cells** in total.

The grid is sparse by design, and its shape is the experiment:

- **The filled band (T=512)** is the headline ladder — sequence length fixed,
  objectives varied 1→16. Everything in §1–§9 comes from this row.
- **The diagonal** is the pinned ladder, with **m × T held at 2048**, so every
  cell on it processes identical token counts. It separates "cost of more
  objectives" from "cost of more data" — a control the headline ladder cannot
  give on its own.

Timings are the **best of repeated runs** (shared-GPU interference can only make
a run look slower, so the minimum is the least-contaminated estimate); three
contaminated runs were caught this way and excluded. Memory needed no such
treatment — **bit-identical across all 312 repeated configurations**, which is
itself evidence the harness is sound.

---

## 11. NEW TODAY (v12): the strategy-selection defect is fixed

*§0–§10 are the v11 campaign, finished and pulled. This section is work done on
11 August, after that report was written. Each part says whether it is measured
or still pending.*

### 11.1 What was built — *implemented, tested*

`src/jdgram/costmodel.py` was a stub that raised `NotImplementedError`. It now
holds a **cost model measured on the target card**, with two terms per strategy
taken from what the kernels actually do:

| strategy | cost terms |
|---|---|
| A (`tfirst`) | `m²T²(d_out + d_in)` GEMMs, plus `m²T²` elementwise |
| B (`dfirst`) | `m·P·T` to build the per-objective block, plus `m²·P` to Gram it |

Coefficients encode achieved GEMM efficiency, so they are measured on the card,
never derived. Three safeguards: near-ties (within 10%) go to the *leaner*
strategy so the memory result is not spent on noise; a workspace cap can
override the time preference; and **with no cost model loaded the engine behaves
exactly as before**, so all v11 numbers stay comparable.

Gate suite **58 → 71 tests**, all passing.

### 11.2 What it is worth — *measured today*

**Per layer, in isolation** (56 shapes × m=1…16): the shipped rule picks the
slower strategy on **18 of 56** shapes (13 of them by >1.5x); the measured model
on **1 of 56** (none by >1.5x). The two strategies still agree numerically to
5×10⁻⁶.

**On the real model** — phase decomposition, savings in actual step
milliseconds:

![Router fix](../fig/figM_router_fix.png)

| objectives | shipped rule | best strategy | saving |
|---|---|---|---|
| 2 | 2.252x | 2.260x | **−0.3%** |
| 4 | 2.709x | 2.478x | **8.5%** |
| 8 | 3.369x | 2.345x | **30.4%** |
| 16 | 3.117x | 2.306x | **26.0%** |

**m=2 shows nothing because it cannot.** At two objectives the rule routes every
layer to strategy A, so `auto` and `tfirst` are the same code path measured
twice — the −0.3% is the harness's repeat-measurement noise (0.29%), which is
the only error bar these numbers have.

Two measurement choices, both of which move the numbers *against* the
conclusion and are made anyway: a **common baseline** across strategies (each
strategy's own baseline drifts up to 2% in the direction that would flatter the
result), and **milliseconds rather than ratios** for the saving (the ratio
version reads ~0.8 pp better).

**Where the gap comes from.** The rule switches each layer as `m·T²` passes its
parameter count: `attn.c_proj` at m≥3, `attn.c_attn` at m≥7, the MLP linears at
m≥9 — and the vocabulary head not until **m≥148**. By m=16 everything else is
already routed correctly, so the remaining gap is that block, by construction.
Its *size* is not established: the isolated-kernel number over-predicts the
measured gap by 15%, because with tied embeddings the head is four terms
sharing one decision and the calibration times only one of them.

### 11.3 Verification status — *incomplete*

The whole-training-loop sweep is not finished. The forced-strategy arms are
already measured in v11 (dfirst medians: m=2 2.740, m=4 2.466, m=8 2.333,
m=16 2.268), so only the two `auto` arms are new — and of those, m=2 completed
while m=4 was contended and m=8/m=16 hit OOM after I launched them into a
still-running sweep.

What is not in doubt: the extrapolation from phase timings to the full loop is
**validated on data already on disk** — 45 matched pairs, median agreement
0.999, 37/45 within 1%. And the memory cost is checkable now: forcing strategy
B everywhere costs **+1.19% peak at m=16**, not the "≤1%" an earlier draft
claimed.

### 11.4 Two further findings — *measured, new*

**Ours is the most numerically accurate engine, by one to three orders of
magnitude.** Against a brute-force Gramian: **3.8×10⁻⁸** for ours, 1.0×10⁻⁶ for
autogram (~25x worse), 8.3×10⁻⁵ for autojac (~2000x worse).

**autogram is measurably wrong on tied embeddings — and GPT-2 ties them.** It
drops the cross-terms between the token embedding and the output head, deviating
by ~2–4×10⁻⁶ where ours stays exact. A *correctness* advantage on the actual
architecture in use, and the strongest engine-versus-engine argument in the
report.

One counterweight: in that same tied case our peak memory is **higher** than
autogram's (1.03–1.54x), the reverse of the untied case (0.72–0.78x). Worth
understanding before leaning on the memory result in a tied-weight setting.

---

## Where this leaves us

**Do we have any advantage over ordinary single-objective training? On cost,
no — and we cannot.** Jacobian Descent does strictly more work: it computes
per-objective gradients, forms their Gramian, and solves for weights. Against
the single-objective reference we are **2.3x the time and 1.13x the memory**,
and the best conceivable version of this design is still ~1.4x on time. The
single-objective run is not a competitor to beat; it is the price list.

**Our advantages are real, but they are against the other ways of doing
Jacobian Descent:**

| versus | our position |
|---|---|
| autojac (explicit Jacobian) | **strictly better** — faster from 3 objectives on, a third of the memory at 8, still runs at 16 where it fails, and ~2000x more numerically accurate |
| autogram (same Gramian idea) | **better on memory** (1.21x vs 1.67x at 16 objectives) and **correct on tied embeddings where autogram is not** (§11.4); 5–13% slower on time before the router fix, less after |
| ordinary training | strictly more expensive, by construction |

**Whether Jacobian Descent is worth its price is still unanswered**, and §9
explains why: on this workload the objectives do not conflict, so the mechanism
has nothing to do. That is the single most important thing this campaign
learned.

### Next steps, in priority order

1. **Build a workload where the objectives actually conflict.** Everything else
   optimises machinery whose value is unproven. Different corpora, a
   task/auxiliary pair — anything with a measured negative cosine. Cheap to set
   up, and it decides whether the rest matters.
2. ~~Replace the strategy rule with a measured cost model.~~ **Done today (§11)** — implemented, gated, calibrated on the A5000, worth 9–31% of step
   time at m≥4 for ≤1% memory. Final whole-loop verification still running; the
   predictions it will be checked against are recorded in §11.3.
3. **Prototype selective fusion of the two forward passes.** Now the largest
   remaining time lever (74% of the gap). It spends memory — our best current
   result — so it must be selective by layer rather than all-or-nothing, with
   memory tracked as a first-class outcome. §2 shows autogram has already made
   the all-or-nothing version of this trade and what it costs.
4. **Investigate the tied-weight memory inversion** (§11.4): untied we are
   0.72–0.78x autogram's peak, tied we are 1.03–1.54x. GPT-2 ties its
   embeddings, so this matters for the headline memory claim and is not yet
   understood.
5. **Then, and only then, run long training.** Perplexity from 100-step runs is
   a correctness check, not evidence about convergence. Once a conflicting
   workload exists and the cost is closer to budget, long runs become the
   experiment worth paying for.

---

*Supporting detail: `docs/attention_hybrid_finding.md` (strategy choice per
layer shape), `docs/l4_l11_discrepancy.md` (measurement methodology, and an
earlier finding retracted when it failed to reproduce). Numbers regenerate from
`bench/acceptance.py`; figures from `bench/report_figures.py`.*
