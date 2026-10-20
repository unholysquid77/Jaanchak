# What changed from the original methodology, and why

**For:** review of the ARC project against
`Adaptive_Representation_Control_Latest_Methodology.docx` and its engineering
translation `docs/ARC_SPEC.md` (v0.1, 2026-10-08).
**Status:** written 2026-10-20, from the measurements in `docs/RESULTS_SUBSPACE.md`
(ResNet arm) and `docs/RESULTS_VIT.md` (ViT arm). Every number below is on the
validation split; no test-split number exists yet, by design.

The short version: **the original plan's central question was answered, and the answer
was "no".** Not "no, the expensive controller loses to the cheap one" — the more awkward
"no, there is nothing for either controller to control." The paper that now exists is a
different paper, built on what the failure of the original one revealed. This document
says exactly which parts were kept, which were abandoned, and what the evidence was in
each case, so the change can be judged rather than taken on trust.

---

## 1. What the original methodology proposed

From `ARC_SPEC.md` §0 and §3–§7:

Every layer `l` gets a **representation state** `s_l = [U_l, D_l, T_l, S_l, N_l]` —
utility (a one-step meta-gradient), drift (1 − CKA), topological mismatch (persistence
landscapes, sliced Wasserstein), stability (Hessian `λ_max`), novelty (BatchNorm
statistic distance). A **controller** maps `s_l` to an adaptation intensity
`a_l ∈ [0,1]`:

* **Controller A** — a lightweight differentiable gate.
* **Controller B** — an FBSDE stochastic-optimal-control policy.

> §0: "The central empirical question is not 'does ARC beat fine-tuning' alone — it is
> **does the expensive controller buy anything over the cheap one?**"

The ladder (§7) was Level 0 baselines → Level 1 gate on `U,D` → Level 2 gate on all
five signals → Level 3 FBSDE (explicit stretch goal). The claim to be earned by
Levels 0–2 (§11):

> "Layer-wise, representation-aware control of adaptation intensity improves few-shot
> classification under domain shift relative to uniform fine-tuning strategies, while
> modifying a smaller fraction of pretrained parameters."

---

## 2. What was kept

Quite a lot, and it is the infrastructure that made the pivot possible at all.

| Kept | Where it ended up |
|---|---|
| The gate formulation `h_l = (1−a_l)·frozen_l(h) + a_l·adapted_l(h)` | Still the mechanism for every experiment. Its bit-exact endpoints (`a=0` reproduces the frozen network exactly) are the project's correctness anchor and are asserted by test. |
| Episodic protocol, 5-way, ProtoNet head, group-disjoint splits | Unchanged. §7's leakage discipline ("split by source specimen, never by image") is enforced in code. |
| **Fraction of parameters substantially modified** (§7 metrics) | Became the single most useful diagnostic in the project. It is reported per episode and catches a selector that silently opens the wrong parameters — which happened twice, each time reporting a plausible accuracy. |
| Level 0 baselines | Complete and reportable. |
| Run-survival discipline (`setsid nohup`, no `tmux`) | Unchanged and vindicated: detached jobs survived 20+ SSH channel kills. |
| The claim-hygiene rule in §11 (Lyapunov as measurement, not theorem) | Generalised into the practice described in §5 below. |

---

## 3. What was abandoned, and on what evidence

### 3.1 The controller — Levels 1, 2 and 3

**Level 1 was built, tuned, and is null** (`docs/RESULTS_LEVEL1.md`): the adaptive gate,
correctly implemented, gives no improvement over uniform strategies. On the harder
dataset the signals are **actively harmful**.

Four probes then explained why (`docs/RESULTS_SUBSPACE.md`), and this is the part that
matters, because a null result from a possibly-buggy implementation is worth little:

1. **Adaptation is ~90% a per-channel affine**, and near rank-1 at the top of the
   network across 200 tasks. There is very little structure for a layer-wise policy to
   exploit.
2. **Neither Level-1 signal predicts, within an episode, which unit is worth adapting.**
   The controller's input is uninformative about its own decision.
3. **Units are redundant** and every parameter group plateaus at ~49.5% accuracy — a
   **0.75-point band across a 778× range** of trainable parameter counts, replicated on
   both datasets.
4. **0.113% of parameters gets 93% of the available gain.**

Taken together: the thing Controller A was designed to decide — *which layers deserve
more adaptation intensity* — has almost no effect on the outcome, because nearly any
sufficient subset works about equally well. That is why Controller B was dropped too.
**Level 3 would have been a more expensive way to answer a question whose answer does
not depend on the policy.** Running it would have burned the remaining budget to
produce a second null with a harder implementation.

This is a genuine negative result about the original hypothesis, not a failure to
implement it. §5b of `ARC_SPEC.md` records the inner-loop defects found and fixed
*before* this conclusion was drawn, including a transductive-BatchNorm confound that
was corrupting every ResNet measurement until caught.

### 3.2 Three of the five signals

`T_l` (topology) was flagged in §4 as "the main engineering risk" on cost grounds. It,
`S_l` and `N_l` were never needed: Level 1 on `U,D` was null and the probes showed the
null was about the *problem*, not about which signals were used. Building three more
uninformative inputs for a policy that cannot help would have been sunk cost.

### 3.3 ChestX, and the forensic framing

ChestX is unobtainable (`RESULTS_SUBSPACE.md` §8), which removes the third and largest
shift from the ladder. This is a real limitation and is stated as one in the draft
rather than worked around — two datasets cannot establish a monotone relationship
between shift magnitude and anything.

The original "forensic evidence recognition" framing was dropped in favour of standard
BSCD-FSL benchmarks. The honest reason is comparability: SSF, LN-Tune and DIPA report on
these, and a result on a bespoke forensic corpus would not be checkable against them.

### 3.4 Protocol: 600 episodes × 3 seeds → 100 × 3

§7 specifies "600 test episodes × 3 seeds". We run **100 episodes × 3 draws** on
validation. The reason is budget — one 10 GiB RTX 3080, shared, and the depth sweeps
alone are ~25 GPU-hours — and the consequence is honestly a loss of power.

Partly mitigated by something §7 did not anticipate: because our sampler seeds per
episode, every baseline within a draw sees **literally the same episodes**, so
comparisons are *paired*. A paired comparison over 100 shared episodes is far tighter
than an unpaired one over 600. It does not fully substitute, and the final test-split
run should use the specified 600.

---

## 4. What replaced it

The probes that killed the controller also pointed somewhere specific: if *which
parameters* barely matters, what does? The answer required an architecture the original
plan did not use.

**A ViT-B/16 has twelve blocks identical in shape and parameter count**, so "adapt block
`i` and nothing else" varies **position** with size and type held exactly fixed. A
ResNet cannot express that experiment — its stages differ in width, depth and resolution
— which is why the original ResNet-only plan could not have found this, and arguably why
the literature has not.

The result (`RESULTS_VIT.md` §3.3):

* Five interventions identical in size and type span **8.9 points** across depth on
  EuroSAT, against **6.5** for the entire parameter-type axis and **1.2** for
  attention-versus-MLP inside it. **Depth is the larger axis.**
* A single well-placed block beats **full fine-tuning of all 85.8M parameters** by
  4.4–5.7 points; a badly placed block of identical size and type does not beat it at
  all.
* The same sweep in the block MLP reproduces the shape, span and interior peak —
  against a prediction, recorded in the code before the data existed, that it would
  not. So it is depth, not attention.
* On ISIC the curve is **mirrored** — shallow wins, deep loses — with four
  shallow-versus-deep comparisons of opposite sign on the two datasets, all settled at
  three draws.

**The relationship to the original claim.** §11's intended claim was that *layer-wise
control* improves few-shot adaptation while modifying fewer parameters. The second half
survived in a stronger form than intended: a hand-picked 2.75–5.5% of the backbone beats
100% of it. The first half did not: the gain comes from *choosing the right layer once,
offline*, not from a representation-aware policy choosing per task. The ResNet arm's
own null is now readable as **regime-specific** rather than general — it was measured on
ISIC, the strong-shift setting, which is exactly where our ViT results show placement
mattering least.

---

## 5. How the standard of evidence changed

This is the change I would most defend, and it is not in the original spec at all.

The project's early results were wrong in a recurring way, and the record is kept rather
than cleaned up (`RESULTS_VIT.md` §3.3.4, §4.1):

* **Four readings of our own data were published internally and then retracted**, each
  with the cause recorded in the section where the claim was made. Every one was the
  same error: a quantity averaged over an unmeasured axis, read as though the axis were
  not there.
* **Seven single-draw results moved when more draws were taken** — four promoted to
  settled, three demoted. The lesson is counter-intuitive and now a standing rule:
  single draws are uninformative about near-threshold differences *in either direction*,
  so the guess must not be the optimistic one.
* **Combining two confidence intervals in quadrature** — the obvious thing to do —
  assumes independence and understated real margins by 2.6× and 4×, producing one false
  null.
* **A grid edge is not neutral uncertainty**: it hides the peak you went looking for.
  One edge inflated a headline by half a point; another, on a different curve, deflated
  the claim it bore on. Both are now closed on the same rule regardless of which way
  they lean.
* Byte-identical runs differ by **~0.6 points** from GPU non-determinism, so that is a
  floor below which nothing is claimed.

Practices that came out of it, all now enforced in code or in tests: paired comparisons
with the per-seed difference; one wide learning-rate grid at every cell with every edge
extended before any curve is read; predictions written into the code *before* the
measurement exists; `fraction_modified` checked on every row; a `--split` flag defaulting
to validation with a source-level test forbidding a hard-coded test split; and every
figure in the draft generated from the raw result files rather than transcribed — which
caught an eighth error while this document was being written.

Four weeks of the original four-week plan went into learning this. It is the reason the
current results are worth more than the original timeline would have produced.

---

## 6. Honest status

**Established** (validation, three draws, paired): the depth effect and its size; the
interior peak at block 8 on EuroSAT in two parameter types independently; one
well-placed block beating full fine-tuning; the mirror across datasets; one confirmed
mechanism prediction (more optimisation helps the small group and hurts the large one).

**Not established, and labelled as such in the draft:** anything on the test split
(nothing is run); the shape of the curve between the five measured depths; the
type × depth interaction on ISIC (not measured); any monotone claim about shift
magnitude from two datasets; a second architecture family.

**The one thing that could still invalidate the arm** is the test-split run, which is
deliberately held until every design decision is frozen. If those numbers diverge from
validation, no amount of writing fixes it — which is the real argument against
compressing the remaining schedule.
