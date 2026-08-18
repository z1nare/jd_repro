# Walkthrough — what to say, in order

Companion to `RESULTS.md` (all the tables and figures, no argument). This is the
spoken version. Roughly 12–15 minutes if you go through all of it; the first
four sections are the core if time is short.

Square brackets are stage directions, not things to say.

---

## Opening — 40 seconds

> "The question you set was: with two or three objectives instead of one, does
> training cost about 50% more, or does it cost two to three times more.
>
> I have an answer, and it's split. On **time**, nobody is close — not us, and
> not either of the two TorchJD engines. Everything is around 2.3 to 2.8x. On
> **memory**, we're the only engine that meets the bar.
>
> But the more important thing I found is further down, and it's about the
> benchmark rather than the engine. I'll get to it."

---

## 1. What's being compared — 1.5 minutes

[`RESULTS.md` header line has the setup if he wants to read along]

> "Everything is a ratio, so let me be precise about the denominator.
>
> **The reference is plain SGD doing standard ERM** — empirical risk
> minimisation. Learning rate 0.01, no momentum, no weight decay. It takes the
> m per-sequence losses, averages them into one scalar, does one backward pass,
> one SGD step. That is exactly what normal training does. In the code it's the
> arm called `sgd_erm`.
>
> **The Jacobian Descent arm uses the identical optimiser** — same SGD, same
> learning rate. The only thing that changes is how the m losses become one
> gradient: instead of averaging them, it keeps them separate, builds the m-by-m
> Gramian, runs an aggregator over it to get per-objective weights, and does one
> *weighted* backward. Everything downstream of that is the same SGD step.
>
> So the comparison isolates exactly one thing — the aggregation — because the
> optimiser is identical on both sides. **The aggregator I quote throughout is
> UPGrad.** I also ran Mean, MGDA and PCGrad; they're in the results file, and
> section 6 is about why they all land in the same place.
>
> Same model, same data, same seed, same hundred steps, and the control runs in
> the *same process* as the engines it's compared against, so machine conditions
> hit both equally.
>
> One thing about the setup that took me a while to appreciate: our loss returns
> one number per sequence. So 'm objectives' literally means 'm sequences'.
> Which means adding an objective also adds *data* — and '2x slower for 2x the
> work' would be no result at all. So the headline ladder **repeats the same
> sequence m times**. Data completely fixed, only the objective count moves.
> That's the number I'll quote throughout.
>
> Model is GPT-2 124M, fp32, sequence length 512, on one A5000."

---

## 2. The verdict — 2 minutes

[Show **figA_verdict**]

> "At two objectives: we're 2.28x on time, autogram 2.51, autojac 2.28. At three:
> 2.51, 2.35, 2.77. The budget line is 1.5. Nobody's near it.
>
> Memory is the other story. We're at 1.13 and 1.15. autogram is 1.40 and 1.48.
> autojac 1.15 then 1.58. We're the only one inside the budget."

[Show **figB_tradeoff_space**]

> "This is the same data but following each engine from 1 objective up to 16.
> Watch the direction of travel rather than any single point. We barely move —
> 1.00 to 1.21 on memory across the whole range. autogram climbs steadily and
> crosses the budget at three objectives. autojac shoots off to 4x and then
> fails entirely.
>
> The bottom-left corner is 'inside budget on both'. It's empty."

---

## 3. Where the time goes — 2 minutes

[Show **figC_cost_of_a_step**]

> "This is a two-objective step broken into pieces, against one ordinary step.
>
> Baseline is 1. The second forward pass adds 0.35. Building the Gramian adds
> 1.38. And the aggregator solve — the part that *looks* mathematically
> expensive, the UPGrad projection — adds **0.01**. Under one percent, and its
> share shrinks as objectives grow. If anyone assumes the optimisation solve is
> the bottleneck, it measurably isn't.
>
> Now the second forward pass. That's not something we forgot to optimise — it's
> a deliberate trade. Our engine frees each layer's activations as the reverse
> sweep goes over them, and that's exactly where the memory win comes from. But
> then the weighted backward has nothing left to traverse, so it needs a fresh
> forward.
>
> autogram makes the opposite choice — keeps the graph, one forward, and pays
> 1.40 to 1.67x memory for it. So autogram is effectively the fused design,
> already built."

[Show **figD_path_to_budget**]

> "Here's what closes the gap. If the Gramian were completely *free* — infinitely
> fast — we'd still be at 1.37x, because we'd still run the model twice. That
> floor eats 74% of the entire allowance.
>
> Fuse the forwards and the floor collapses to about 1.0, and the speedup we'd
> then need from the Gramian machinery drops from 8.5x — which isn't happening —
> to 2.2x, which is a normal optimisation target.
>
> The honest catch: fusing means adopting autogram's memory profile, and that's
> our best result. So what I'd actually want is *selective* retention — decide
> per layer whether to keep activations or recompute, the same way the engine
> already decides per layer how to build the Gramian. Neither design does that
> today; both make one global choice."

---

## 4. Fixed cost versus per-objective cost — 1.5 minutes

[Show **figE_cost_model**]

> "This is raw milliseconds instead of ratios, and it's the cleanest way to see
> the structure. Both are straight lines in the objective count.
>
> Ordinary training: 31 milliseconds per objective, plus 16 fixed per step.
> Ours: 68 per objective, plus 73 fixed.
>
> So we pay **2.18x per objective**, plus a fixed 57 milliseconds a step that
> plain training doesn't. That one decomposition explains the whole shape of the
> earlier chart — the total ratio falls from 2.73 at two objectives to 2.26 at
> sixteen purely because the fixed part gets amortised. It decays toward 2.18
> and never goes below.
>
> Which means the worst point on the entire curve is **two objectives** — exactly
> the case in your budget. Scale doesn't rescue that one; only cutting the
> per-objective cost does."

---

## 5. A bug I found and fixed — 2 minutes

[Show **figG_strategy_crossover**]

> "Our engine builds each layer's Gramian one of two ways, and picks per layer
> using a rule that compares only *working memory*. It never looks at speed.
>
> The two strategies swap places at about three and a half objectives. The rule
> switches at nine. So across that whole band it keeps picking the loser — by up
> to 1.66x at eight objectives.
>
> And the trade-off it thinks it's making doesn't exist. Switching to the faster
> strategy everywhere costs about **1% more peak memory** and saves up to **63%
> of step time**. The reason memory barely moves: the rule is optimising scratch
> space inside the Gramian computation, but peak memory is set by the model's
> own activations, which are far bigger and identical either way. It's
> minimising something that isn't the constraint."

[Show **figH_layer_strategy**]

> "Per layer, in isolation, it picks wrong on two of nine shapes — but one of
> them is the vocabulary head, the biggest tensor in the model, where the chosen
> strategy is thirteen times slower. The rule would need 148 objectives before
> it switched there."

> "So I fixed it. `costmodel.py` was a stub that raised NotImplementedError —
> it's now a real cost model, with the terms taken from what the kernels
> actually do, and the coefficients **measured on the A5000** rather than
> derived. Calibrated on 56 shapes: the old rule picks the slower strategy on 18
> of them, 13 by more than 1.5x. The new model: one, none of them badly.
>
> On the real model that's worth about 8.5% at four objectives, 30% at eight,
> 26% at sixteen. Nothing at two, because at two objectives the old rule was
> already right.
>
> Gate suite went from 58 tests to 71, all passing. And the two strategies still
> agree numerically to five times ten to the minus six — a routing change can
> only ever cost time or memory, never change a result."

[If asked about verification status:]

> "Phase-level is measured. The full training-loop sweep isn't finished — I
> launched it into a still-running job and three of the four arms either got
> contended or ran out of memory. That's on me. The extrapolation from phase
> timings to the full loop is validated separately though — 45 matched pairs
> already on disk agree to a median of 0.999."

---

## 6. The finding that actually matters — 2 minutes

[Show **figL_objective_alignment**]

> "This is the one I'd most like your view on.
>
> Every objective produces a gradient — a direction it wants to move the
> weights. The cosine between two of them says whether they agree: +1 means
> identical, 0 unrelated, −1 exactly opposed. Jacobian Descent exists for the
> negative end — its whole job is resolving conflict between objectives.
>
> The two grey lines are calibration. If I duplicate an objective, the
> measurement must read +1 — it reads exactly +1. If I take an objective and its
> exact negation, it must read −1 — exactly −1. So the instrument works.
>
> The blue line is real objectives — different passages of the corpus. It goes
> from +0.86 down to +0.66. **It never goes negative. Not at any objective
> count.**
>
> Language-model gradients from different bits of the same corpus point
> substantially the same way. There is no conflict here for the aggregator to
> resolve. Which explains something that had puzzled me — every aggregator,
> UPGrad, MGDA, PCGrad, reaches essentially the same held-out loss. Of course
> they do. With nothing to resolve they all reduce to roughly a plain average.
>
> The consequence is uncomfortable: we're paying 2.3x for conflict resolution on
> objectives that never conflict. That's a property of the *benchmark*, not the
> engine — no amount of kernel work changes it.
>
> So I think the next experiment isn't a faster kernel. It's a workload where
> the objectives genuinely pull apart. Different domains, or a task-plus-auxiliary
> pair. Until that exists I don't think we can answer whether Jacobian Descent is
> worth its price, either way."

---

## 7. Where we stand versus the alternatives — 1 minute

> "To be direct about the comparison you'd care about:
>
> Against **plain SGD/ERM — the single-objective reference** — we have no
> advantage and can't
> have one — we do strictly more work by construction. We're 2.3x the time and
> 1.13x the memory. That run isn't a competitor, it's the price list.
>
> Against **autojac** we're strictly better — faster from three objectives on, a
> third of the memory at eight, still running at sixteen where it fails, and
> about 2000x more numerically accurate.
>
> Against **autogram** it's mixed and I want to be straight about it. We're much
> better on memory, 1.21 against 1.67 at sixteen objectives. We're 5 to 13%
> *slower* on time — that's the second forward pass. But there's one thing in
> our favour I only found today: **autogram is measurably wrong on tied
> embeddings, and GPT-2 ties them.** It drops the cross-terms between the token
> embedding and the output head. We stay exact. That's a correctness difference
> on the actual architecture, not a performance one."

---

## 8. What I'd do next — 45 seconds

> "In priority order:
>
> One — build a workload where the objectives actually conflict. Everything else
> is optimising machinery whose value is unproven, and this is cheap to set up.
>
> Two — the strategy fix is done; it needs its verification run finished
> cleanly.
>
> Three — prototype selective fusion of the two forward passes. Biggest
> remaining time lever, but it spends memory, so it has to be per-layer rather
> than all-or-nothing.
>
> Four — there's a memory inversion in the tied-weight case I don't understand
> yet: untied we use less memory than autogram, tied we use more. GPT-2 is tied,
> so that matters for the headline claim.
>
> Five — and only then, long training runs. What I have is 100 steps, which is a
> correctness check, not evidence about convergence."

---

## Questions you should expect

**"What exactly is the baseline — what optimiser?"**
> "Plain SGD, learning rate 0.01, no momentum, no weight decay, doing standard
> ERM: average the m losses into one scalar, one backward, one step. The
> Jacobian Descent arm uses the *identical* optimiser — the only difference is
> that it weights the objectives via the Gramian instead of averaging them. So
> the ratio isolates the aggregation and nothing else."

**"Which aggregator are these numbers?"**
> "UPGrad throughout. Mean, MGDA and PCGrad are all in the results file — and
> section 6 explains why they all reach the same held-out loss."

**"Why is the perplexity the same everywhere? Is the aggregator doing anything?"**
> "Correct, and that's section 6 — the objectives don't conflict, so every
> aggregator reduces to roughly an average. It's the benchmark, not the code."

**"These are only 100 steps. What does the perplexity tell us?"**
> "Only that the three engines compute the same thing — it's a correctness
> check. It is not evidence about convergence and I'm not claiming it is."

**"How confident are you in the timings?"**
> "Memory is bit-identical across all 312 repeated configurations. Timing I take
> as the minimum over repeats, because shared-GPU interference can only make a
> run look slower. Three contaminated runs were caught that way and dropped. The
> phase-level numbers are ten-iteration timings with no dispersion recorded, so
> I wouldn't read past the second digit."

**"Is the second forward pass just a bug?"**
> "No — it's what buys the memory result. Removing it means taking autogram's
> memory profile. The interesting version is doing it per layer."

**"Why did you not just always use the faster strategy?"**
> "That's what the fix does, via a measured cost model rather than a hard-coded
> preference — because at one and two objectives the *other* strategy genuinely
> is faster, so a blanket rule would trade one wrong answer for another."

---

## If you have five minutes instead of fifteen

Sections 2, 3, 6 — the verdict, where the time goes, and the alignment finding.
Skip the cost model, the bug, and the comparison table; they're all in
`RESULTS.md` if he wants them.
