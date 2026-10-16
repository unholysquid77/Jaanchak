# ARC — full experiment log

Every run performed, including the void ones. Numbers are exact, pulled from the
result JSON rather than retyped. Void runs are recorded so nobody repeats them and
so the reason each was discarded is on record.

Protocol throughout: ResNet-50 ImageNet, 5-way episodes, ProtoNet head (cosine),
per-episode test-time adaptation on the support set only, query scored under no_grad,
pristine weight reset every episode. Tuning on val, reporting on test.

---

## 1. Level 0 — baselines on EuroSAT (VALID, final)

```
baseline               shot       lr steps seeds episodes  accuracy   ci95  %params
frozen_proto              1    1e-04     0     3     1800     67.67   0.47     0.00
frozen_proto              5    1e-04     0     3     1800     84.59   0.29     0.00
frozen_proto             10    1e-04     0     2     1200     87.28   0.32     0.00
frozen_proto             20    1e-04     0     2     1200     88.99   0.27     0.00
last_layer_ft             1    1e-05    20     3     1800     67.70   0.47    14.72
last_layer_ft             5    1e-04     5     3     1800     84.80   0.29    17.30
last_layer_ft            10    3e-03    20     2     1200     89.11   0.30    18.03
last_layer_ft            20    3e-03    20     2     1200     91.99   0.24    18.40
bn_only_ft                1    1e-03    20     3     1800     68.46   0.49     0.21
bn_only_ft                5    3e-03    20     3     1800     86.83   0.29     0.22
bn_only_ft               10    3e-03    20     2     1200     90.79   0.27     0.22
bn_only_ft               20    3e-03    20     2     1200     93.61   0.20     0.22
stage4_ft                 1    3e-04    20     3     1800     69.06   0.48    59.20
stage4_ft                 5    3e-04    20     3     1800     87.73   0.27    62.37
stage4_ft                10    1e-04    20     2     1200     91.11   0.26    62.36
stage4_ft                20    1e-04    20     2     1200     93.51   0.21    62.52
progressive_unfreeze      1    3e-05     5     3     1800     68.51   0.48    81.95
progressive_unfreeze      5    1e-04    20     3     1800     87.66   0.27    96.42
progressive_unfreeze     10    3e-05    20     2     1200     91.06   0.26    92.96
progressive_unfreeze     20    3e-05    20     2     1200     93.64   0.20    93.21
full_ft                   1    1e-05     5     3     1800     68.37   0.48    62.45
full_ft                   5    1e-05    20     3     1800     87.21   0.28    86.51
full_ft                  10    3e-05    20     2     1200     91.35   0.26    94.03
full_ft                  20    3e-05    20     2     1200     94.02   0.20    94.34
```

## 2. Baseline learning-rate sweeps (VALID)

Staged across several runs; merged by argmax with duplicate configs deduplicated.
All cells ended with an interior optimum on the merged grid.

232 configs run, 222 unique. Validation accuracy (%), best per row marked `*`:

**1-shot**

```
baseline               steps    1e-06    3e-06    1e-05    3e-05    1e-04    3e-04    1e-03    3e-03    1e-02    3e-02
last_layer_ft              5    66.83    66.85    66.96    67.09    67.11    65.43        -        -        -        -
last_layer_ft             20    66.84    66.96    67.15    67.15    65.49    62.08        -        -        -        -
bn_only_ft                 5    66.83    66.84    66.80    66.81    66.96    67.28    67.55    67.32    65.29    58.76
bn_only_ft                20    66.84    66.79    66.85    67.11    67.37    67.44    67.6*    66.32    63.64    57.40
stage4_ft                  5    66.89    67.01    67.27    67.5*    66.39    66.08    66.73    53.28    19.64    20.00
stage4_ft                 20    67.04    67.20    67.31    66.77    66.67    67.71    65.57    17.95    19.67    20.00
progressive_unfreeze       5    66.97    67.13    67.67    68.11    66.73    60.57        -        -        -        -
progressive_unfreeze      20    67.23    67.59    67.67    66.87    66.36    57.72        -        -        -        -
full_ft                    5    67.17    67.35    67.9*    67.37    64.84    57.53        -        -        -        -
full_ft                   20    67.35    67.73    67.55    66.76    64.16    55.68        -        -        -        -
```

**5-shot**

```
baseline               steps    1e-06    3e-06    1e-05    3e-05    1e-04    3e-04    1e-03    3e-03    1e-02    3e-02
last_layer_ft              5    83.77    83.80    83.80    83.87    84.03    83.08        -        -        -        -
last_layer_ft             20    83.80    83.81    83.87    83.9*    83.21    83.20        -        -        -        -
bn_only_ft                 5    83.77    83.81    83.77    83.89    84.04    84.31    85.32    85.84    86.01    81.09
bn_only_ft                20    83.76    83.81    83.89    84.07    84.41    85.55    86.32    86.5*    85.45    79.39
stage4_ft                  5    83.89    84.00    84.52    85.63    85.89    86.51    83.85    65.24    20.00    20.00
stage4_ft                 20    84.16    84.67    85.99    85.96    87.23    87.8*    84.60    18.89    20.00    20.00
progressive_unfreeze       5    83.96    84.35    85.27    86.56    86.6*    78.99        -        -        -        -
progressive_unfreeze      20    84.31    85.36    86.49    87.09    87.67    79.13        -        -        -        -
full_ft                    5    84.16    84.53    85.92    86.2*    86.00    77.09        -        -        -        -
full_ft                   20    84.69    85.95    86.99    86.96    84.96    77.12        -        -        -        -
```

**10-shot**

```
baseline               steps    1e-05    3e-05    1e-04    3e-04    1e-03    3e-03    1e-02
last_layer_ft             20    87.27    87.93    87.68    88.02    88.85    89.30    21.81
bn_only_ft                20    87.00    87.15    87.77    88.84    90.38    91.18    90.26
stage4_ft                 20    89.44    90.60    91.6*    91.22    88.18    19.66    20.00
progressive_unfreeze      20    90.44    91.50    90.83    85.94    58.84    19.96    20.00
full_ft                   20    91.36    91.70    89.92    85.05    25.97    20.00    20.00
```

**20-shot**

```
baseline               steps    1e-05    3e-05    1e-04    3e-04    1e-03    3e-03    1e-02
last_layer_ft             20    89.35    90.38    91.11    91.08    91.96    92.50    23.41
bn_only_ft                20    89.07    89.10    89.90    91.38    92.80    93.72    93.40
stage4_ft                 20    92.12    93.44    94.20    93.60    92.05    19.73    20.00
progressive_unfreeze      20    93.33    94.03    93.33    89.56    64.73    20.00    20.00
full_ft                   20    94.06    94.28    93.53    90.12    21.69    20.00    20.00
```

### Tuned settings used for the Level-0 table
```
baseline                shot       lr  steps   val acc
bn_only_ft                 1    1e-03     20    67.68%
bn_only_ft                 5    3e-03     20    86.56%
bn_only_ft                10    3e-03     20    91.18%
bn_only_ft                20    3e-03     20    93.72%
full_ft                    1    1e-05      5    67.96%
full_ft                    5    1e-05     20    86.99%
full_ft                   10    3e-05     20    91.70%
full_ft                   20    3e-05     20    94.28%
last_layer_ft              1    1e-05     20    67.15%
last_layer_ft              5    1e-04      5    84.03%
last_layer_ft             10    3e-03     20    89.30%
last_layer_ft             20    3e-03     20    92.50%
progressive_unfreeze       1    3e-05      5    68.11%
progressive_unfreeze       5    1e-04     20    87.67%
progressive_unfreeze      10    3e-05     20    91.50%
progressive_unfreeze      20    3e-05     20    94.03%
stage4_ft                  1    3e-04     20    67.71%
stage4_ft                  5    3e-04     20    87.88%
stage4_ft                 10    1e-04     20    91.68%
stage4_ft                 20    1e-04     20    94.20%
```
---

## 3. Level 1 — adaptive gate (VALID, current)

`results/gate_sweep/`. Standardised signals, BatchNorm guard in place. Tuned on val,
75 episodes per configuration. **This is the run the null result rests on.**

Read marginal effects, not the best-config delta: the maximum over 18 signal-enabled
configurations beats the maximum over 9 disabled ones by construction.

**1-shot** — normalised signals, BN guard

```
 bias  aU  bD      acc   ci95    gate    %par
    2   0   2    68.18   2.79   0.822    89.0
    2   1   2    68.12   2.82   0.821    88.7
    1   1   2    68.09   2.88   0.705    89.4
    1   2   0    68.02   2.75   0.651    84.1
    1   0   1    68.00   2.90   0.719    89.1
    1   0   2    68.00   2.85   0.728    88.3
    2   2   0    67.96   2.76   0.778    83.1
    2   2   1    67.96   2.74   0.804    82.9
    2   0   0    67.93   2.74   0.881    82.9
    2   1   1    67.93   2.85   0.851    83.3
    0   0   0    67.91   2.75   0.500    87.3
    2   1   0    67.91   2.66   0.846    82.8
    1   0   0    67.84   2.84   0.731    84.7
    1   2   1    67.84   2.73   0.669    84.3
    0   1   0    67.82   2.78   0.501    85.9
    0   1   1    67.73   2.80   0.503    88.4
    0   2   0    67.73   2.73   0.502    85.4
    0   1   2    67.72   2.79   0.537    88.6
    2   0   1    67.72   2.87   0.848    85.9
    1   1   1    67.68   2.78   0.703    85.7
    1   1   0    67.64   2.78   0.699    84.2
    1   2   2    67.61   2.81   0.664    88.0
    0   0   2    67.50   2.79   0.576    87.6
    2   2   2    67.48   2.73   0.789    85.5
    0   0   1    67.41   2.87   0.523    89.7
    0   2   1    67.40   2.74   0.507    86.1
    0   2   2    67.40   2.84   0.512    89.4
```
- marginal effect of **utility**: -0.05 points
- marginal effect of **drift**: -0.10 points
- stage-1 learning rate chosen: 1e-05

**5-shot** — normalised signals, BN guard

```
 bias  aU  bD      acc   ci95    gate    %par
    2   0   0    88.14   1.27   0.881    94.1
    2   0   1    87.86   1.25   0.846    94.9
    2   1   1    87.70   1.29   0.838    94.1
    2   1   0    87.59   1.32   0.846    94.1
    2   2   0    87.54   1.17   0.780    94.3
    1   0   0    87.54   1.31   0.731    94.9
    2   2   2    87.45   1.41   0.760    94.8
    0   0   0    87.43   1.37   0.500    96.0
    1   2   0    87.40   1.34   0.656    94.7
    1   1   0    87.38   1.28   0.700    94.7
    0   1   0    87.32   1.36   0.504    95.5
    1   0   1    86.93   1.40   0.716    96.5
    1   1   1    86.88   1.34   0.690    95.1
    2   1   2    86.61   1.35   0.798    96.5
    1   2   1    86.56   1.36   0.652    94.7
    2   2   1    86.38   1.44   0.783    94.2
    0   2   0    86.24   1.42   0.510    95.3
    0   1   1    86.19   1.55   0.512    96.5
    1   2   2    85.97   1.44   0.648    96.1
    0   2   1    85.83   1.48   0.515    95.5
    2   0   2    85.80   1.50   0.811    97.5
    0   0   1    85.71   1.62   0.522    97.4
    0   1   2    85.51   1.68   0.546    97.4
    0   0   2    85.33   1.53   0.574    97.1
    1   1   2    85.26   1.60   0.690    97.4
    0   2   2    85.17   1.64   0.524    97.2
    1   0   2    84.43   1.67   0.719    97.6
```
- marginal effect of **utility**: +0.04 points
- marginal effect of **drift**: -1.20 points
- stage-1 learning rate chosen: 3e-05

### Measured signal profile (per unit, 5-shot, one episode)

From `experiments/diagnose_gate.py`. This is the figure the paper should carry:
drift rises monotonically with depth, which is why penalising it freezes the top of
the network.

```
unit            U        D     gate(raw)   gate(standardised)
stem            0.485    0.000       0.997               0.992
layer1_0        0.311    0.000       0.995               0.983
layer1_1        0.422    0.001       0.997               0.959
layer1_2        0.506    0.001       0.998               0.919
layer2_0        0.504    0.001       0.998               0.813
layer2_1        0.397    0.009       0.996               0.942
layer2_2        0.339    0.012       0.995               0.831
layer2_3        0.423    0.016       0.996               0.757
layer3_0        0.491    0.011       0.997               0.765
layer3_1        0.353    0.022       0.995               0.614
layer3_2        0.352    0.022       0.995               0.552
layer3_3        0.387    0.025       0.996               0.728
layer3_4        0.468    0.026       0.997               0.642
layer3_5        0.636    0.032       0.998               0.820
layer4_0        0.583    0.110       0.997               0.156
layer4_1        0.675    0.249       0.997               0.035
layer4_2        0.756    0.329       0.997               0.012
spread          0.445    0.329       0.004               0.980
```
Raw and standardised columns are from separate runs at different coefficients, so
compare the *spread*, not row-by-row values. The point: identical signals give a
gate spread of 0.004 raw and 0.980 standardised.

---

## 4. Void runs — kept so they are not repeated

### 4.1 First baseline LR sweep — `results/sweep_INVALID_bn_pollution/`

**Void: metric saturation + BatchNorm pollution.** Accuracy was identical to four
decimals across a 300x learning-rate range, and `lr=0.0` reproduced it bit for bit
(0.6122666... at 5 steps, 0.5116 at 20). Nothing was learning; the only active
variable was BatchNorm statistic drift, which is why accuracy depended on step count
and not on learning rate.

```
baseline             shot       lr steps      acc
last_layer_ft           1    1e-05     5    61.23
last_layer_ft           1    3e-05     5    61.23
last_layer_ft           1    1e-04     5    61.23
last_layer_ft           1    3e-04     5    61.23
last_layer_ft           1    1e-03     5    61.23
last_layer_ft           1    3e-03     5    61.23
last_layer_ft           1    1e-05    20    51.16
last_layer_ft           1    3e-05    20    51.16
... 12 configs total
```

### 4.2 First gate sweep — `results/gate_sweep_bnbug/`

**Void: BatchNorm pollution via the controller.** `compute_signals()` restored mode
with `.train()`, overriding `bn_mode='eval'`. It fired only when drift was enabled,
so drift appeared actively harmful: every top-8 configuration had `bD=0`, and
enabling drift cost ~9 points at 1-shot. That conclusion was an artefact.

1-shot, best and worst (note every best has bD=0):
```
 bias  aU  bD      acc    %par
    2   2   0    67.98    82.2
    4   4   0    67.91    82.2
    2   4   0    67.88    82.1
    4   2   8    58.31    72.1
    4   2   4    58.29    72.1
```

5-shot, best and worst (note every best has bD=0):
```
 bias  aU  bD      acc    %par
    2   2   0    88.07    93.7
    4   2   0    88.05    93.6
    2   4   0    88.00    93.7
    4   2   4    82.72    91.9
    4   4   4    82.58    91.9
```

### 4.3 Second gate sweep — `results/gate_sweep_saturated/`

**Void: gate saturated into a constant.** BatchNorm was fixed, but signals were fed
to the sigmoid raw. With a bias of 2-4 against signals of magnitude ~0.1, and with
utility and drift both rising with depth so the +U and -D terms cancel, the gate came
out at ~0.997 at every unit. ARC was running as exactly full fine-tuning. The
marginal effects (+0.23 utility, +0.04 drift at 5-shot) are what a gate that never
varies produces.

- 1-shot best 68.20 (bias=2 aU=4 bD=4), gate_mean 0.973, %params 82.2; marginal U +0.00, D -0.01
- 5-shot best 88.32 (bias=4 aU=4 bD=4), gate_mean 0.997, %params 93.6; marginal U +0.23, D +0.04

### 4.4 Discarded Level-0 cells — `results/discarded/`

Ten result files run at untuned or grid-truncated settings after the 10/20-shot
tuning sweep died of an out-of-memory error and the pipeline continued anyway. The
clearest case: `bn_only_ft` at 20-shot scored **90.09** at the untuned `lr=1e-4`
versus **93.61** once retuned to `3e-3`. Keeping them would have made BatchNorm-only
adaptation look like a weak baseline that ARC comfortably beats.

---

## 5. Diagnostic measurements (not sweeps, but load-bearing)

### 5.1 Inner-loop instrumentation — `experiments/diagnose_inner_loop.py`

Run when the first LR sweep returned identical accuracy across a 300x range. Per-step
loss, true (unscaled) gradient norm, parameter displacement and BatchNorm statistic
drift, on one real episode.

**Before the fix** (ResNet-50, lr=3e-3, 5 steps):

```
k=1, BatchNorm frozen        loss 0.000e+00  grad 0.000e+00  param delta 0.000e+00
k=1, BatchNorm training      loss 0.000e+00  grad 0.000e+00  param delta 0.000e+00
                             BUT BatchNorm running-stat drift 1.62e+02
k=5, AMP on                  loss 1.56e-05 -> inf gradients, steps skipped by GradScaler
```

The 1-shot loss was **exactly** zero, not merely small: every support image was its own
prototype, the softmax saturated, and fp32 cross-entropy underflowed. With `lr=0` the
only thing that moved was BatchNorm, which is why accuracy tracked step count and
ignored learning rate entirely.

**After the fix** (cosine metric, two-view objective, BN frozen, lr=1e-4):

```
k=1  loss 3.334e-02 -> 6.703e-04 monotone, grad 1.26e+00 -> 1.29e-02, BN drift 0.0
k=5  loss 1.932e-01 -> 1.355e-02 monotone, grad 1.58e+00 -> 2.92e-01, BN drift 0.0
```

### 5.2 Metric saturation, measured directly

On EuroSAT, 2048-d ResNet-50 features, mean embedding norm 13.8-14.1:

```
squared distance range    5.6 .. 419.4
correct-class distance    10.9 (k=1), 67.0 (k=5)
softmax max probability   1.0000000000
=> cross-entropy          exactly 0.0
cosine similarity range   0.069 .. 0.990      (well-conditioned)
```

### 5.3 ISIC corpus structure

```
10,015 images, 7 classes, 7,470 distinct lesions
images per lesion:  1 -> 5514 lesions,  2 -> 1423,  3 -> 490,  4 -> 34,  5 -> 5,  6 -> 4
=> 4,501 images (45%) share a lesion with at least one other
group_disjoint split: 7,470 groups, 0 lesions spanning splits
per-class counts: NV 6705, MEL 1113, BKL 1099, BCC 514, AKIEC 327, VASC 142, DF 115
```

Those per-class counts match the published HAM10000 distribution exactly, which is an
independent check that the one-hot ground-truth parsing is correct.

### 5.4 ProtoNet cross-validation against easyfsl 1.5.0

Prototypes match exactly; predictions, accuracy and the full per-row class ranking agree
for both metrics at 1/5/20-shot and across 50 independent episodes. Logits differ in
scale by construction (easyfsl returns `-cdist` unsquared with no temperature) but ours
is a strictly monotone transform.

### 5.5 Memory and timing on the RTX 3080 (10 GiB)

```
20-shot without gradient checkpointing   OOM at 8.8 / 9.6 GiB
20-shot with checkpointing               fits; outputs and gradients bit-identical
10-shot                                  fits without checkpointing

per-episode cost (600-episode reporting runs)
  10-shot   ~2.85 s/episode   ~29 min per cell
  20-shot   ~5.76 s/episode   ~58 min per cell
```

---

## 6. Compute spent

```
first baseline LR sweep (void)        ~0.5 h   killed early once diagnosed
baseline LR sweep + extension         ~3.5 h   120 + 32 configs
10/20-shot tuning (v1, OOM)           ~2.5 h   35 of 50 configs, partially reused
10/20-shot tuning (extension)         ~5 h
Level-0 reporting                     ~20 h    60 cells, 600 episodes each
gate sweep 1 (void, BN bug)           ~2.5 h
gate sweep 2 (void, saturated)        ~2.5 h
gate sweep 3 (valid)                  ~2.5 h
```

Roughly 39 hours of GPU time, of which about 8 was spent on runs later discarded. Each
discard traced to a defect that would otherwise have produced a plausible but wrong
number in the paper.

---

## 6b. The data package was never committed (2026-10-15)

`src/arc/data/` — the corpus loaders, the leakage-safe splitter and the episodic
sampler — was matched by the unanchored `data/` line in `.gitignore` and was therefore
absent from every commit in the project's history. Both machines that ever ran an
experiment held the files as untracked work, and the box is a plain tar-synced tree with
no git repo of its own, so nothing warned.

**Resolved.** The original modules were recovered from `~/arc/src/arc/data/` over the
tailnet and committed verbatim, byte-for-byte. **Every number in this log is
re-derivable from the repository again.** The ignore patterns are now anchored, and the
package has 28 tests where it previously had none.

For the record, a reconstruction built from the test suite and the call sites was
committed first, while the box was unreachable. It passed all 82 existing tests and ran
the full pipeline end to end, and still differed from the original in four ways that
would each have shifted the numbers: the evaluation transform, the `group_disjoint`
assignment function, the class-selection order inside an episode, and the handling of a
class with too few images. Worth recording as a calibration: a package can be
API-correct, fully green and still not reproduce the experiment.

---

## 6c. ISIC — baselines and the gate (VALID, val only; 2026-10-15/16)

~8.5 GPU-hours. `results/isic_sweep` (160 configs), `results/isic_sweep_ext` (12),
`results/gate_sweep_isic` (27 x 2 shots + LR selection). Split `group_disjoint` by
lesion at 0.1/0.4/0.5, split seed 0, episode seed 10,000. Headline numbers and the full
grids are in `docs/RESULTS_ISIC.md`; recorded here as the run log.

Stage 1 (160 configs, 5.2h) ended with `bn_only_ft` pinned to the top of the 3e-6..1e-2
grid at both shots, and stage 2's grid-edge check aborted the driver — landmine 8 caught
at the point it matters rather than in the paper. The extension (3e-2/1e-1/3e-1,
12 configs, 23 min) found the optimum and the collapse:

```
bn_only_ft 5-shot 20 steps
  1e-03:43.33  3e-03:46.32  1e-02:48.65  3e-02:49.44*  1e-01:38.71  3e-01:19.97
```

49.44 against `full_ft`'s 49.55, at 0.22% of parameters versus ~94%.

Gate sweep (3h) marginal effects, against the EuroSAT run in §3:

```
                 EuroSAT 1 / 5      ISIC 1 / 5
utility          -0.05 / +0.04      -0.18 / -1.43
drift            -0.10 / -1.20      -0.30 / -3.33
fraction_modified   84-98 %         92.5-94.7 % (1-shot), 97.7-98.4 % (5-shot)
```

**Not yet run on ISIC:** test reporting (stage 4 of `scripts/run_isic.sh`, gated behind
`REPORT=1`), 10- and 20-shot, and `frozen_proto` (non-tunable, so the sweep skips it —
the `lr=3e-6` column stands in for it at ~38.0 on 5-shot).

Cross-sweep caveat: the gate sweep used 75 episodes per config and the baseline sweep
100. Episodes are seeded per index from the same seed, so the gate's 75 are exactly the
first 75 of the baselines' 100 — overlapping but not identical. Differences of a point
or two between the two sweeps are not meaningful.

---

## 6d. Probes: what adaptation actually does (VALID, val only; 2026-10-16)

~3 GPU-hours via `scripts/run_probe.sh`. Headline numbers and interpretation in
`docs/RESULTS_SUBSPACE.md`; recorded here as the run log. All ISIC 5-shot, val.

```
P0a  diagnose_subspace, full_ft,     200 eps   diagonal 0.903  eff.rank 12.84  acc 49.61
P0b  diagnose_subspace, bn_only_ft,  200 eps   diagonal 0.946  eff.rank 10.20  acc 48.69
E2   diagnose_unit_benefit,           30 eps   within-episode rho: U +0.039, D -0.038
                                               drift depth rho +0.973 +/- 0.023
                                               adapt none 38.22, adapt all 50.22
                                               standalone benefits sum 74.5% vs 12.0%
E1   sweep_lr, 4 parameter groups,    40 cfg   see the plateau table below
```

Parameter-type plateau, each at its own tuned LR on the 3e-6..1e-1 grid:

```
bn_scale_ft      48.79   0.113%   3e-2
bn_shift_ft      48.32   0.113%   3e-2
bn_only_ft       49.44   0.226%   3e-2   (from section 6c)
downsample_ft    49.80  11.811%   1e-3
conv_only_ft     49.64  87.996%   1e-4
full_ft          49.55 100.000%   1e-4   (from section 6c)
stage4_ft        46.52  63.658%   3e-4   (from section 6c)
```

**A failed prediction, recorded because it constrains the next step.** P0a shows shift
carrying ~85% of the diagonal energy, so `bn_shift_ft` was predicted to clearly beat
`bn_scale_ft`. It lost, 48.32 to 48.79. Energy share does not translate into accuracy
share, so a basis chosen by displacement norm is choosing on the wrong objective.

**Not run:** precomputed-shift adaptation (the decisive follow-up), any test-split
number, 1-shot (uninformative on ISIC), E3 the hard/sparse gate.

---

## 7. What has never been run

- **ISIC test split** — val tuning complete (§6c); no test numbers, no 10/20-shot.
- **ChestX** — loader written and unit-tested, data not obtainable automatically.
- **Any forensic dataset** — still unconfirmed to exist.
- **Leave-one-out inner objective** — implemented, never ablated against two-view.
- **`bn_mode="train"` (transductive BatchNorm)** — supported, never swept.
- **Learned gate coefficients** (mode ii) — the gate is already on the forward path and
  differentiable, so this is a small change; never attempted.
- **Drift as a loss penalty** (the spec's `λ₁·D_repr`) rather than a gate input.
- **Hard or sparse gates** — the sigmoid never reaches 0, so no parameter-efficiency
  win is currently possible.
- **`oracle_utility`** — the spec's label-using utility, as an upper bound on what a
  perfect utility signal could buy.
