# ISIC results — tuned baselines and the adaptive gate

**Validation-split tuning only. Nothing here is a test number yet.** Stage 4 of
`scripts/run_isic.sh` (600 episodes x 3 seeds on test) is deliberately ungated and
unrun; see §5 for the recommendation on whether to spend it.

Corpus: ISIC 2018 Task 3 (= HAM10000), 10,015 images, 7 classes, **7,470 lesions**.
Split `group_disjoint` by lesion at 0.1/0.4/0.5 (`--split-seed 0`), giving val 4,013
images and test 5,002. The fractions are not EuroSAT's 0.2/0.1/0.7 — see §4.

5-way episodes, 15 queries, ResNet-50 ImageNet, ProtoNet cosine head, two-view inner
objective, BatchNorm frozen during adaptation. 100 episodes per baseline config,
75 per gate config.

---

## 1. Why ISIC: the headroom, measured

The `lr=3e-6` column of the baseline sweep is low enough that nothing moves, so it
stands in for a frozen backbone. At 5-shot it reads **~38.0**; the best tuned baseline
reaches **49.55**.

| | frozen | best tuned | headroom |
|---|---|---|---|
| EuroSAT 5-shot | 84.59 | 87.73 | **3.1** |
| ISIC 5-shot | ~38.0 | 49.55 | **11.5** |

A 3.7x wider window for a controller to demonstrate anything. This was the stated
reason for moving off EuroSAT (`docs/HANDOFF.md` §9.3) and it holds up.

At 1-shot the picture is the opposite: every baseline lands in 27.6–30.9, a 3.3-point
band barely above the 20% chance floor. **Treat 1-shot on ISIC as near-uninformative**
— it cannot separate methods, and the gate results there should not be read as
evidence either way.

## 2. Tuned baselines (val)

```
baseline                shot       lr  steps   val acc
stage4_ft                  1    3e-04     20    30.91
progressive_unfreeze       1    1e-04     20    30.71
last_layer_ft              1    3e-03     20    30.48
full_ft                    1    1e-04      5    30.27
bn_only_ft                 1    1e-02     20    29.92

full_ft                    5    1e-04     20    49.55
bn_only_ft                 5    3e-02     20    49.44
progressive_unfreeze       5    1e-04     20    47.15
stage4_ft                  5    3e-04     20    46.52
last_layer_ft              5    3e-03     20    43.64
```

All ten cells have an interior optimum on the merged grid (3e-6 … 3e-1).

### bn_only_ft ties full fine-tuning at 1/400th of the parameters

Stage 1's grid topped out at 1e-2 with `bn_only_ft` still climbing, so stage 2 aborted
and the grid was extended. The peak and the collapse:

```
bn_only_ft 5-shot, 20 steps
  3e-06:38.00  1e-05:38.07  3e-05:38.16  1e-04:38.65  3e-04:40.12  1e-03:43.33
  3e-03:46.32  1e-02:48.65  3e-02:49.44*  1e-01:38.71  3e-01:19.97
```

**49.44 against full fine-tuning's 49.55** — 0.11 points apart, far inside the ±1.5
CI, at **0.22% of parameters versus ~94%**. On the hard dataset, adapting only
BatchNorm affine parameters is statistically indistinguishable from adapting the
entire backbone.

The optimal BatchNorm learning rate is also an order of magnitude higher than on
EuroSAT (3e-2 versus 1e-3–3e-3, where 3e-2 had already collapsed). Consistent with a
larger domain shift needing the normalisation statistics to move further, and a
concrete reason the EuroSAT-tuned settings could never have been transplanted.

## 3. The adaptive gate: the null replicates, and strengthens

Marginal effect of each signal — the mean over all configurations at each coefficient
level, averaging the others out. **Not** the best-config delta: comparing the maximum
over 18 signal-enabled configurations against the maximum over 9 disabled ones favours
the larger set by construction (`docs/HANDOFF.md` §6, landmine 9).

| | EuroSAT 1-shot | ISIC 1-shot | EuroSAT 5-shot | **ISIC 5-shot** |
|---|---|---|---|---|
| marginal utility | -0.05 | -0.18 | +0.04 | **-1.43** |
| marginal drift | -0.10 | -0.30 | -1.20 | **-3.33** |

Both signals are harmful at every shot count on ISIC, and more harmful than on
EuroSAT. Drift is the worse of the two and its damage is concentrated at the high
coefficient:

```
5-shot, drift coefficient:   0 -> 48.16    1 -> 47.80    2 -> 44.83
```

So a *small* drift coefficient is roughly neutral (-0.36) and a large one is
destructive (-3.33). The same shape as EuroSAT, three times the magnitude.

Best overall 5-shot configuration is 50.90 (bias=1, aU=0, bD=1) against 48.94 for the
best signal-free gate. **That +1.96 is the biased comparison, not a result:** it is a
maximum over 18 against a maximum over 9, the CI is ±2.0, and the two best configs
differ in which of them happens to carry `bD=1`. The marginal effects are the unbiased
read, and they are negative.

### No parameter-efficiency win, again — worse than EuroSAT

`fraction_modified` across all 27 gate configurations:

```
ISIC 1-shot     92.5 – 94.7 %
ISIC 5-shot     97.7 – 98.4 %
EuroSAT         84   – 98   %
```

The sigmoid gate never approaches zero, so it modulates *how much* each unit adapts,
never *whether*. Against a `bn_only_ft` that now ties full fine-tuning at 0.22%, this
is the more serious problem of the two: ARC cannot win the accuracy race on ISIC
(nothing can — the ceiling is already held at 1/400th the cost), so parameter
efficiency is the only axis left, and the gate currently produces none.

### A reversal worth putting to Prof. Hemrajani

On EuroSAT at 5-shot, `stage4_ft` (87.73) beat `full_ft` (87.21) — adapting only the
top of the network was better than adapting all of it, which is the observation ARC's
premise rests on: *where* you adapt matters more than *how much*.

On ISIC that reverses at both shot counts: `full_ft` 49.55 > `stage4_ft` 46.52 at
5-shot, and 30.27 vs 30.91 at 1-shot is inside noise. Meanwhile `bn_only_ft`, which
adapts BatchNorm in *every* unit rather than selecting by depth, matches the best.

The axis that pays on ISIC is therefore not layer depth but **parameter type** —
normalisation statistics versus all weights. If that holds up on test, ARC's
layer-wise granularity may simply be the wrong granularity for this kind of shift, and
a controller over parameter *groups* (BN / conv / downsample) would be the more
promising object. That is a redesign, not a tuning change, and it is a question for
him before more compute goes into the current formulation.

## 4. Split fractions: why not EuroSAT's

ISIC's largest class is 58x its smallest (NV 6,705 images vs DF 115), and
`EpisodicSampler` raises when any class in a split holds fewer than `k_shot + n_query`
images. Measured with `experiments/diagnose_splits.py` before any GPU time:

```
0.2/0.1/0.7   val DF  ~11   BLOCKED at every k-shot
0.2/0.2/0.6   val DF   18   BLOCKED from 5-shot
0.1/0.3/0.6   val DF   33   BLOCKED at 20-shot
0.1/0.4/0.5   val DF   43   OK at 1/5/10/20-shot   <- chosen
```

`docs/HANDOFF.md` §11's suggested one-liner used 0.2/0.1/0.7 and would have died at
sampler construction. 10% stays in train, untouched by Levels 0–2, in case Level 3
needs a meta-training pool.

## 5. Recommendation

**Do not spend stage 4 on the gate yet. Do spend it on the baselines.**

The baseline table is a real contribution and needs test numbers: *"on a genuinely
hard cross-domain target, adapting 0.22% of parameters matches adapting all of
them"* is worth reporting on its own, and it is the bar every ARC variant has to
clear. Baseline reporting is roughly 9–10 GPU-hours (600 episodes x 3 seeds x 6
baselines x 2 shots), not the ~20 the driver's note estimates for both halves.

The gate as currently formulated has now produced a null on two datasets, with the
signals actively harmful on the harder one, and no efficiency win on either. Reporting
it on test would buy a more precise measurement of a negative we already have twice.
Of `docs/HANDOFF.md` §9.4's options, the ordering the ISIC numbers imply:

1. **Hard/sparse gate (9.4c).** The only option that attacks the axis ARC can still
   win on. Gumbel-sigmoid or an explicit sparsity penalty, so `fraction_modified` can
   actually fall. Until the gate can reach zero, "modify less" has no mechanism behind
   it, let alone evidence.
2. **A parameter-type controller** rather than a layer controller — the §3 reversal.
   Not in §9.4 because the EuroSAT data did not suggest it; the ISIC data does.
3. **Drift as a loss penalty (9.4b)** rather than a gate input. Drift is clearly
   carrying signal — it is strong enough to cost 3.3 points when weighted — it just
   appears to be the wrong sign or the wrong place.
4. **Inverted drift sign (9.4d).** Cheap, but ISIC weakens the prior: inverting it
   should push adaptation toward the high-drift top of the network, which here is
   `stage4_ft`, and `stage4_ft` *loses* to `full_ft` on this corpus.

### Caveat on cross-comparing the two sweeps

The gate sweep ran 75 episodes per configuration and the baseline sweep 100. Both use
split seed 0 and episode seed 10,000, and episodes are seeded per index, so the gate's
75 are exactly the first 75 of the baselines' 100 — overlapping, but not the same set.
Gate-versus-baseline differences of a point or two are therefore not meaningful, and
the gate's nominal 50.90 should **not** be read as beating `full_ft`'s 49.55.
