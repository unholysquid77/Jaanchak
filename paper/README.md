# Paper draft

```
python3 make_figures.py      # from the repo root: python3 paper/make_figures.py
pdflatex arc && pdflatex arc # twice, for references
```

Needs `pgfplots` >= 1.16 (texlive-pictures). No Python dependencies beyond the
standard library, and no matplotlib — the figures are pgfplots, so they inherit the
document's fonts and stay vector.

**Every plotted and quoted number comes from `make_figures.py`, which reads
`results/` directly.** Nothing is transcribed by hand. `figdata/key_numbers.dat`
carries the dataset, difference, 95% CI, **draw count** and a settled/unsettled
verdict for each comparison the text quotes, so a caption cannot claim more draws
than exist. Re-run the script after any new measurement and the draft updates.

The lab record behind it is `../docs/RESULTS_VIT.md` (ViT arm) and
`../docs/RESULTS_SUBSPACE.md` (ResNet arm). `../docs/METHODOLOGY_DIVERGENCE.md` is
the accounting of what changed from the original spec and why.

`\TODO{...}` marks what is still open in the text; `grep TODO arc.tex`.
