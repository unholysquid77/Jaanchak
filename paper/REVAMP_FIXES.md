# Three fixes for `arc_revamped`

Generated 2026-10-25 from `paper/figdata/*.dat`, which `paper/make_figures.py`
builds directly from `results/`. Every number below is machine-generated; none
is transcribed by hand.

The revamped manuscript has diverged from `paper/arc.tex` in this repo. The repo
copy already contains all three of these. These fragments port them across.

---

## FIX 1 — ISIC lesion-group counts (page 8). **Hard error.**

**Current text:**

> The group-disjoint partition assigns 4,482, 1,494, and 1,494 lesion groups to
> train, validation, and test respectively.

Those are 7,470 split 0.6 / 0.2 / 0.2 — `make_splits`' **default** fractions.
ISIC ran `--fractions 0.1 0.4 0.5`.

It also contradicts Table 2 in the same section. At 10,015 images over 7,470
lesions (~1.34 images per lesion), 4,482 lesion groups is ~6,000 images, but
Table 2 correctly reports 1,000 training images.

**Replace with:**

```latex
The group-disjoint partition assigns $747$, $2{,}988$, and $3{,}735$ lesion
groups to train, validation, and test respectively.
```

Table 2's image counts (1,000 / 4,013 / 5,002) are correct and need no change.

---

## FIX 2 — Add the ISIC MLP depth curve.

Three draws, seeds 0/1/2, 100 episodes each, already committed under
`results/vit_isicmlppos{0,2,5,8,11}` (block 5 from `_v2`) and
`results/vit_isicmlpposcurve_seeds`.

### 2a. Table 6 — add a fourth column

```latex
\begin{tabular}{crrrr}
\toprule
& \multicolumn{3}{c}{EuroSAT} & \multicolumn{1}{c}{ISIC} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-5}
Block & Attention & MLP & & Attention \\
\midrule
 0 & 78.76 (0.32) & 80.27 (0.96) & & 48.20 (0.78) \\
\end{tabular}
```

becomes a five-column table. Full replacement body:

```latex
\begin{tabular}{crrrr}
\toprule
& \multicolumn{2}{c}{EuroSAT} & \multicolumn{2}{c}{ISIC} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
Block & Attention & MLP & Attention & MLP \\
\midrule
 0 & 78.76 (0.32) & 80.27 (0.96) & 48.20 (0.78) & 49.13 (1.53) \\
 2 & 78.64 (0.29) & 80.68 (0.36) & 47.96 (0.93) & 49.74 (0.91) \\
 5 & 81.78 (0.39) & 86.44 (0.36) & 47.78 (1.25) & 49.21 (0.69) \\
 8 & 87.49 (0.29) & 88.79 (0.11) & 45.84 (1.20) & 46.92 (0.76) \\
11 & 85.95 (0.50) & 84.99 (0.45) & 44.49 (0.48) & 42.78 (0.58) \\
\bottomrule
\end{tabular}
```

Caption stays accurate as written ("mean accuracy, standard deviation across
episode-sampling seeds") — all twenty cells are three-draw.

### 2b. Figure 4 — add the MLP series

If the figure reads `figdata/depth_curves.dat`, add inside the `axis`:

```latex
\addplot[mlpcol, mark=square*, thick, dashed, error bars/.cd, y dir=both, y explicit]
  table[x=depth, y=isic_mlp, y error=isic_mlp_sd] {figdata/depth_curves.dat};
\addlegendentry{block MLP ($5.50\%$)}
```

and relabel the existing series `\addlegendentry{attention ($2.75\%$)}`.

If the numbers are inline in the revamped source instead, the coordinates are:

```
(0,49.13) +-(0,1.53)   (2,49.74) +-(0,0.91)   (5,49.21) +-(0,0.69)
(8,46.92) +-(0,0.76)   (11,42.78) +-(0,0.58)
```

Revised caption:

```latex
\caption{ISIC validation depth profiles for both parameter types, three draws,
error bars one standard deviation across draws. Each is a flat plateau over
blocks $0$--$5$ then a step down to the deep end. Compare the EuroSAT figure:
the ordering is reversed in both types.}
```

### 2c. Mirror table — add an MLP panel

Three MLP links with opposite settled signs on both datasets:

```latex
\addlinespace
\multicolumn{3}{l}{\emph{block MLP} ($5.50\%$)} \\
block 0 $-$ block 11 & $+6.35 \pm 2.37$ & $-4.72 \pm 2.06$ \\
block 2 $-$ block 11 & $+6.96 \pm 1.16$ & $-4.32 \pm 0.38$ \\
block 2 $-$ block 8  & $+2.82 \pm 0.41$ & $-8.11 \pm 0.84$ \\
```

(Label the existing rows `\multicolumn{3}{l}{\emph{attention} ($2.75\%$)}`.)

### 2d. Replacement for §5.4's closing, and the honest exception

```latex
\paragraph{The reversal survives changing the parameter type.} The sweep was
repeated in the block MLP on ISIC --- a second family, twice the size, the same
five depths, the same per-cell learning-rate grid. The ordering reverses there
too, and the three MLP rows of the mirror table are settled on both sides. This
separates two readings the EuroSAT data alone cannot: the reversal could have
been something attention does under large shift, in which case it would not
appear in the MLP. It appears in the MLP, at a larger margin than in attention
($6.96$ points from block 2 to block 11, against $3.47$), so the shift rather
than the parameter type sets the sign.

One link does \emph{not} reverse, and we state it rather than drop it. Block 5
$-$ block 11 in the MLP is $+6.43 \pm 0.63$ on ISIC and $+1.45 \pm 1.93$ on
EuroSAT --- same sign, and unsettled on the EuroSAT side. It is excluded from
the mirror table, and the criterion is one we can defend but did not fix in
advance of seeing the number: a row belongs there only if both of its depths lie
on the same side of the EuroSAT peak as the labels ``shallow'' and ``deep''
claim. Block 11 is \emph{past} the block-8 peak on EuroSAT, so block 5 $-$ block
11 compares two points on the far side of a maximum and is not a
shallow-versus-deep contrast at all. The same is true of block 8 $-$ block 11,
positive on both datasets ($+4.15 \pm 0.76$ on ISIC, $+3.80 \pm 0.97$ on
EuroSAT) --- the fall-off at the last block is the one feature the two curves
share.
```

### 2e. Downstream text that must change with it

- **Abstract.** "The depth profile differs on ISIC" → say it differs in *both*
  families, and that six shallow-versus-deep comparisons take opposite settled
  signs across the two datasets.
- **ISIC span.** §5.4 currently reads "the overall measured span is 3.71 points,
  less than half of the EuroSAT attention span." True of attention. In the MLP
  the spans are **6.96** (ISIC) and **8.52** (EuroSAT) — comparable. The span gap
  is a fact about attention on ISIC, not about ISIC. Add that sentence.
- **Limitations, "Two target datasets" / "One architecture family".** The
  remaining thinness is two parameter types out of five, and two datasets — not
  "MLP is EuroSAT-only."
- **Appendix D, Scope of claims.** The bullet scoping the depth-profile claim
  should now read: supported by paired shallow/deep comparisons over three draws
  on EuroSAT and ISIC, **in attention and MLP**.
- **Contribution 3** ("A dataset-dependent result") can state it holds in both
  measured families.

---

## FIX 3 — Two formatting glitches

- §3.5 "Experimental pipeline" is an empty header immediately followed by §3.6.
  Either give it a sentence or delete it and promote §3.6.
- Page 8, directly under Table 2: a stray orphan line reading `ViT depth sweeps.`
  Looks like a truncated sentence or an escaped caption fragment. Delete.

---

## Not a defect: leave as is

`\ell^2`-SP is declared unrun in §5.7, §7 and Appendix D, consistently. It is
implemented in `src/arc/train/episodic.py` and `scripts/run_vit_l2sp.sh` but has
no numbers, and the manuscript says so. That is the correct treatment.

## Verified unchanged

Every `\dd`-style margin in the revamped PDF was checked against
`paper/figdata/key_numbers.dat` and `depth_curves.dat`: the depth tables, the
EuroSAT/ISIC mirror links, the weight-decay dose-response, the support-size rows.
All match. The manuscript also correctly keeps the three-draw five-shot benchmark
($+6.35$ / $+3.85$) distinct from the matched seed-0 follow-up ($+6.69$ /
$+3.97$) rather than blending them.
