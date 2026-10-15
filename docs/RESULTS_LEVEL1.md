# Level-1 results — the adaptive gate (EuroSAT)

**Result: null.** The adaptive gate, correctly implemented and tuned, gives no
measurable benefit over a constant gate. The best 5-shot configuration has both signal
coefficients at zero.

Produced by `experiments/sweep_gate.py` on the validation split, after fixing the
BatchNorm-pollution and gate-saturation defects documented in `docs/HANDOFF.md` section
6. Earlier gate sweeps are void and retained only for provenance.

Read the marginal effect of each signal, not the best-config delta: comparing the
maximum over 18 signal-enabled configurations against the maximum over 9 disabled ones
favours the larger set by construction, and that biased comparison is what made earlier
runs look mildly positive.

## Summary

| | 1-shot | 5-shot |
|---|---|---|
| marginal effect of utility | -0.05 | +0.04 |
| marginal effect of drift | -0.10 | **-1.20** |
| best configuration | bias=2, aU=0, bD=2 | bias=2, **aU=0, bD=0** |

Caveats, in full, are in `docs/HANDOFF.md` section 5. The two that matter: EuroSAT has
too little headroom to distinguish methods (frozen backbone already reaches 84.6% at
5-shot), and the sigmoid gate never reaches zero, so `%params` stays at 84-98% in every
configuration and no parameter-efficiency win is produced either.

### 1-shot — all 27 gate configurations (75 val episodes)

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
- learning rate chosen in stage 1: 1e-05

### 5-shot — all 27 gate configurations (75 val episodes)

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
- learning rate chosen in stage 1: 3e-05