# What adaptation actually does — four probes on ISIC

**Validation split only. No test numbers here.** ~3 GPU-hours, 2026-10-16.
Raw output: `results/subspace/`, `results/unit_benefit/`, `results/isic_groups/`.
Driver: `scripts/run_probe.sh`. ISIC 5-shot, `group_disjoint` by lesion at
0.1/0.4/0.5, split seed 0, episode seed 10,000, ResNet-50, 20 inner steps.

1-shot is excluded throughout and should stay excluded: every ISIC baseline lands in
27.6–30.9 against a 20% chance floor, so it cannot separate methods.

---

## 0. Why these were run

Level 1's gate is null on EuroSAT and worse than null on ISIC, and `bn_only_ft`
adapts 0.22% of the backbone to tie full fine-tuning (49.44 vs 49.55). Read together
that suggests the binding constraint in few-shot adaptation is the **dimensionality**
of the update, not its location — which the gate never addressed, since it modulates
the magnitude of a 23M-parameter update estimated from 25 images.

BatchNorm affine is a per-channel scale and shift: a *diagonal* affine map on each
unit's activations. So `bn_only_ft` is already an activation-subspace method. These
probes test that reading.

## 1. Adaptation is ~90% a per-channel affine, and near rank-1 at the top

`experiments/diagnose_subspace.py`, 200 episodes. Each unit's measured displacement
(adapted vs pretrained activations on the same probe batch — the query set, under
`no_grad`, never seen by the optimiser) is split into the best per-channel affine plus
a residual. Shift and scale together are exactly what BatchNorm can express.

```
                          full_ft (49.61)    bn_only_ft (48.69)
diagonal-expressible           0.903              0.946
non-diagonal residual          0.097              0.054
mean effective rank           12.84              10.20     (of 200 episodes)
```

Per unit, `full_ft`:

```
unit        shift   scale   resid   eff.rank   top1
stem        0.756   0.097   0.150     13.15   0.164
layer1_2    0.916   0.023   0.071     17.03   0.180
layer2_0    0.900   0.035   0.080     20.24   0.159
layer3_5    0.770   0.100   0.172      8.17   0.341
layer4_0    0.637   0.212   0.209      2.65   0.613
layer4_1    0.538   0.294   0.335      2.93   0.582
layer4_2    0.453   0.437   0.296      1.51   0.813
```

Two things to take from this. **Nine-tenths of what fine-tuning does to the
representation is a per-channel affine** — the mechanism behind the `bn_only_ft` tie,
measured rather than asserted. And **the top of the network is near rank-1 across 200
different tasks**: `layer4_2`, the features the ProtoNet head consumes, has effective
rank 1.51 with 81.3% of displacement energy in a single direction.

Effective rank is the participation ratio `(Σs²)²/Σs⁴`, so no threshold is chosen.
`ncomp` in the raw output is `min(episodes, channels)`, so deep units are
episode-limited rather than channel-limited.

Sanity check the measurement passes: `bn_only_ft`'s residual is 0.054 rather than
0.000 even though it *is* diagonal, because the displacement at unit *l* composes
BatchNorm changes from units 1…*l* through nonlinearities. That is the expected
behaviour, and its absence would have meant the decomposition was wrong.

## 2. Nothing for a layer-wise controller to control

`experiments/diagnose_unit_benefit.py`, 30 episodes. Per-unit realised benefit by
direct intervention: `only` adapts a single unit, `loo` adapts all but one. Signals
are read after 5 warmup steps, because at step 0 the two branches are the same network
and drift is identically zero.

```
adapt nothing 38.22      adapt all 50.22

                  vs standalone   vs marginal (loo)   WITHIN-episode
utility             -0.033            +0.146             +0.039
drift               +0.141            -0.009             -0.038
```

**The within-episode column is the one that matters**, because ranking units *inside*
an episode is exactly what a gate does. Both are zero. The pooled ρ ≈ 0.14 with small
p-values is cross-episode structure, not unit ranking — the sort of thing that looks
like a result and is not.

**Drift is confirmed a depth index: ρ = +0.973 ± 0.023** across 30 episodes. So
`−β·D` is a stage-4 detector, i.e. a fixed "freeze the top" depth schedule wearing a
feedback controller's costume — and the baselines say the top is the most productive
thing to adapt. That is a mechanism for drift's negative marginal, not just its size.
Utility's depth ρ is −0.085 ± 0.506: not a depth proxy, simply unpredictive.

**Units are highly redundant.** Standalone benefits sum to **74.5%** while adapting
everything gains **12.0%**. Removing any single unit costs 0.4–2.4%; adapting almost
any single unit alone buys 2–8%. Layer selection cannot matter much when layers are
this interchangeable — which is a direct measurement against ARC's premise that *where*
you adapt dominates *how much*.

## 3. The parameter-type plateau

`experiments/sweep_lr.py` over four new group selectors, 40 configs, each at its own
tuned learning rate on the 3e-6 … 1e-1 grid.

```
                   acc     params      gain   pts per % param      lr
frozen_proto     38.00     0.000%     +0.00          —             —
bn_scale_ft      48.79     0.113%    +10.79        95.5          3e-2
bn_shift_ft      48.32     0.113%    +10.32        91.3          3e-2
bn_only_ft       49.44     0.226%    +11.44        50.6          3e-2
downsample_ft    49.80    11.811%    +11.80         1.0          1e-3
conv_only_ft     49.64    87.996%    +11.64         0.1          1e-4
full_ft          49.55   100.000%    +11.55         0.1          1e-4
stage4_ft        46.52    63.658%     +8.52         0.1          3e-4
```

**Every group lands in 48.3–49.8.** Half of BatchNorm — 26,560 parameters, 0.113% —
captures 10.79 of the 11.55 points full fine-tuning gets, i.e. **93% of the benefit at
about 1/900th of the parameters**.

Call this a **plateau, not a frontier.** The groups are indistinguishable in accuracy
and differ only in cost. `stage4_ft` being the *worst* at 63.7% of parameters makes the
point sharply: parameter count does not predict accuracy either. This corroborates §2's
redundancy result from a completely different direction — almost any sufficiently
expressive slice of the network reaches the same ~49.5.

### A prediction that failed, recorded as such

§1 shows shift carrying ~85% of the diagonal energy, so the prediction was that
`bn_shift_ft` would clearly beat `bn_scale_ft`. Measured: **48.32 vs 48.79, shift loses
by 0.47.** Within noise, but certainly not the predicted clear win.

So **energy share does not translate into accuracy share**: the large-norm component of
the displacement is not the one doing the classification work. This is a genuine caveat
on the subspace argument and it needs explaining before anything is built on a
basis chosen by energy. One candidate: the cosine ProtoNet head normalises embeddings,
so a large common-mode shift at the top is partly divided out — exactly the component
§1 measures as dominant. If that is right, a basis should be selected by *effect on the
head's decision*, not by displacement norm, and §1's spectrum is measuring the wrong
objective. Untested.

## 4. Where this leaves the project

**Established (val, ISIC, 5-shot):**
- 90% of fine-tuning's representational change is a per-channel affine; at the top of
  the network it is near rank-1 across 200 tasks
- neither Level-1 signal predicts, within an episode, which unit is worth adapting
- drift is a depth index (ρ = 0.973), which explains its harm rather than just sizing it
- units are redundant, and every parameter group plateaus at ~49.5
- 0.113% of parameters gets 93% of the available gain

**The synthesis these support:** on a domain-shifted target, "few-shot adaptation" is
mostly a **low-rank, largely task-independent, per-channel domain correction** rather
than task-specific representation learning. That one claim accounts for all three
nulls: the `bn_only_ft` tie, the irrelevance of layer selection, and the uselessness of
a task-conditional controller.

**Not established, and the next thing to test:** whether the correction can be computed
**once, offline, from the train split** and applied with **zero test-time gradient
steps**. If a fixed affine extracted from train episodes reaches ~49 on val, the
synthesis is demonstrated and the result is positive and useful. If it does not, the
correction is more task-dependent than the rank-1 spectrum suggests, and §3's failed
prediction is the first hint of why. `results/` now carries everything needed to check
either way.

**What these probes do *not* support:** that a cleverer basis beats the diagonal. §3's
plateau says the ~49.5 ceiling is not the diagonal's fault, so "better basis" is
probably not where the win is. The win, if there is one, is in removing test-time
optimisation, not in raising the ceiling.
