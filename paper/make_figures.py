#!/usr/bin/env python3
"""Emit pgfplots data tables for the paper, straight from results/.

Run from the repository root:  python3 paper/make_figures.py

Every number in the paper's figures comes from here, so there is one place to check
and no hand-copied values. Written to be runnable with the standard library only --
the figures are pgfplots, not matplotlib, so the paper has no Python dependency at
build time and the plots inherit the document's fonts.

Conventions enforced below, matching docs/RESULTS_VIT.md:
  * three-draw means where three draws exist, single draw otherwise, and the
    DRAW COUNT IS EMITTED so a caption cannot silently claim more than it has
  * per-seed differences with t(2)=4.303 for three draws; per-episode paired
    differences with t(99)=1.9842 for one
  * `_v2` directories supersede their originals (post-grid-extension re-measurements)
"""

from __future__ import annotations

import glob
import json
import math
import os
import statistics as st

OUT = os.path.join(os.path.dirname(__file__), "figdata")
T2, T99 = 4.303, 1.9842

# _v2 supersedes: block 8 attention and block 5 MLP on EuroSAT, block 0 on ISIC, were
# re-measured after their learning-rate grids were extended.
EURO_ATTN = {0: "vit_pos0", 2: "vit_pos2", 5: "vit_pos5", 8: "vit_pos8_v2", 11: "vit_cap1"}
EURO_MLP = {0: "vit_mlppos0", 2: "vit_mlppos2", 5: "vit_mlppos5_v2", 8: "vit_mlppos8",
            11: "vit_mlppos11"}
ISIC_ATTN = {0: "vit_isicpos0", 2: "vit_isicpos2", 5: "vit_isicpos5", 8: "vit_isicpos8",
             11: "vit_isicpos11"}
SEED_DIRS = ["vit_poscurve_seeds", "vit_mlpposcurve_seeds", "vit_isicposcurve_seeds"]


def _load_dir(d: str, acc: dict) -> None:
    """Fill acc[(dataset, baseline, seed)] from every json in a results dir.

    THE KEY MUST INCLUDE THE DATASET. The first version of this file keyed on
    (baseline, seed) alone, and because EuroSAT and ISIC share every baseline NAME --
    full_ft, frozen_proto, attn_block8_ft, ... -- loading both directories made ISIC
    silently overwrite EuroSAT. The symptom was a figure whose EuroSAT column was
    identical to its ISIC column and a quoted gap of +37.29 points, which is how it
    was caught; a subtler collision would have produced a plausible wrong number.

    This is the same failure family as the averaging errors in RESULTS_VIT.md §3.3.4:
    a key or a statistic that omits an axis, so two different things occupy one slot.
    The dataset comes from manifest.config.dataset, which run_baselines.py always
    writes -- not from the filename, which is only conventionally formatted.
    """
    for f in glob.glob(f"results/{d}/*.json"):
        j = json.load(open(f))
        ds = j.get("manifest", {}).get("config", {}).get("dataset")
        if ds is None:
            raise RuntimeError(
                f"{f}: no manifest.config.dataset -- refusing to guess which dataset "
                f"this is, because guessing is what produced the collision this "
                f"function's docstring describes"
            )
        key = (ds, j["baseline"], j.get("seed", 0))
        if key in acc:
            raise RuntimeError(f"{f}: duplicate {key} -- two files claim the same cell")
        acc[key] = [e["accuracy"] * 100 for e in j["episodes"]]


def _episodes() -> dict:
    acc: dict = {}
    for d in (list(EURO_ATTN.values()) + list(EURO_MLP.values())
              + list(ISIC_ATTN.values()) + SEED_DIRS
              + ["vit_2b_eurosat", "vit_2b_isic", "vit_seeds_eurosat", "vit_seeds_isic"]):
        if os.path.isdir(f"results/{d}"):
            _load_dir(d, acc)
    return acc


ACC = _episodes()
# ISIC depth 11 is filed under attn_last1_ft (attn_block11_ft is deliberately
# unregistered -- it duplicates attn_last1_ft); same for the EuroSAT n=1 suffix point.
ALIAS = {"attn_block11_ft": "attn_last1_ft"}


def mean_and_draws(ds: str, base: str):
    """Three-draw mean where available, else the single draw. Returns (mean, n, sd)."""
    base = ALIAS.get(base, base)
    vals = [st.mean(ACC[(ds, base, s)]) for s in (0, 1, 2) if (ds, base, s) in ACC]
    if not vals:
        return None, 0, None
    return st.mean(vals), len(vals), (st.stdev(vals) if len(vals) > 1 else None)


def paired(ds: str, a: str, b: str):
    """(difference, ci95, n_draws) WITHIN one dataset. Per-seed across draws where
    three exist, else per-episode within the single draw."""
    a, b = ALIAS.get(a, a), ALIAS.get(b, b)
    shared = [s for s in (0, 1, 2) if (ds, a, s) in ACC and (ds, b, s) in ACC]
    if len(shared) >= 3:
        d = [st.mean(ACC[(ds, a, s)]) - st.mean(ACC[(ds, b, s)]) for s in shared]
        return st.mean(d), T2 * st.stdev(d) / math.sqrt(len(d)), len(d)
    if not shared:
        return None, None, 0
    s = shared[0]
    d = [x - y for x, y in zip(ACC[(ds, a, s)], ACC[(ds, b, s)])]
    return st.mean(d), T99 * st.stdev(d) / math.sqrt(len(d)), 1


def write(name: str, header: str, rows: list[str]) -> None:
    path = os.path.join(OUT, name)
    with open(path, "w") as fh:
        fh.write(header.rstrip() + "\n")
        for r in rows:
            fh.write(r.rstrip() + "\n")
    print(f"  wrote {path} ({len(rows)} rows)")


DEPTHS = [0, 2, 5, 8, 11]


def fig_depth_curves() -> None:
    """Figure 1: depth curves -- both parameter types on EuroSAT, attention on ISIC."""
    rows, notes = [], []
    for i in DEPTHS:
        an = "attn_last1_ft" if i == 11 else f"attn_block{i}_ft"
        ea, na, sa = mean_and_draws("eurosat", an)
        em, nm, sm = mean_and_draws("eurosat", f"mlp_block{i}_ft")
        ia, ni, si = mean_and_draws("isic", an)
        rows.append(f"{i} {ea:.2f} {sa or 0:.2f} {em:.2f} {sm or 0:.2f} "
                    f"{ia:.2f} {si or 0:.2f}")
        notes.append((i, na, nm, ni))
    write("depth_curves.dat",
          "depth euro_attn euro_attn_sd euro_mlp euro_mlp_sd isic_attn isic_attn_sd",
          rows)
    for i, na, nm, ni in notes:
        print(f"    depth {i:2d}: draws euro_attn={na} euro_mlp={nm} isic_attn={ni}")


def fig_params_vs_acc() -> None:
    """Figure 3: adapted fraction against accuracy, EuroSAT. The figure's point is
    that full_ft sits BELOW single-block interventions 20-40x smaller."""
    pts = [
        ("frozen", 0.0, "frozen_proto"),
        ("LN", 0.045, "ln_only_ft"),
        ("attn~b8", 2.75, "attn_block8_ft"),
        ("attn~b11", 2.75, "attn_last1_ft"),
        ("MLP~b8", 5.50, "mlp_block8_ft"),
        ("attn~all", 33.04, "attn_only_ft"),
        ("MLP~all", 66.05, "mlp_only_ft"),
        ("full", 100.0, "full_ft"),
    ]
    rows = []
    for label, frac, base in pts:
        m, n, sd = mean_and_draws("eurosat", base)
        if m is None:
            print(f"    SKIP {label}: no eurosat data for {base}")
            continue
        rows.append(f"{frac} {m:.2f} {sd or 0:.2f} {n} {label}")
    write("params_vs_acc.dat", "frac acc sd draws label", rows)


def fig_inner_steps() -> None:
    """Figure 4: the inner-step crossover. attn improves with more steps, full_ft
    degrades past 10. Single draw at 5 and 10, so all three use seed 0."""
    def one(d, base):
        f = glob.glob(f"results/{d}/baseline_eurosat_{base}_5shot_seed0.json")
        if not f:
            return None
        return st.mean([e["accuracy"] * 100 for e in json.load(open(f[0]))["episodes"]])
    rows = []
    for steps, d in ((5, "vit_steps5_v2"), (10, "vit_steps10_v2"), (20, "vit_2b_eurosat")):
        a, f_ = one(d, "attn_only_ft"), one(d, "full_ft")
        if a is None or f_ is None:
            print(f"    SKIP steps={steps}")
            continue
        rows.append(f"{steps} {a:.2f} {f_:.2f}")
    write("inner_steps.dat", "steps attn full", rows)


def key_numbers() -> None:
    """Every inline number the paper quotes, with dataset and draw count."""
    comps = [
        ("eurosat", "b8attn_vs_full", "attn_block8_ft", "full_ft"),
        ("eurosat", "b8mlp_vs_full", "mlp_block8_ft", "full_ft"),
        ("eurosat", "b8attn_vs_frozen", "attn_block8_ft", "frozen_proto"),
        ("eurosat", "b8attn_vs_attnall", "attn_block8_ft", "attn_only_ft"),
        ("eurosat", "b8_vs_b11", "attn_block8_ft", "attn_last1_ft"),
        ("eurosat", "attnall_vs_mlpall", "attn_only_ft", "mlp_only_ft"),
        ("eurosat", "b8mlp_vs_b8attn", "mlp_block8_ft", "attn_block8_ft"),
        ("eurosat", "b11mlp_vs_b11attn", "mlp_block11_ft", "attn_last1_ft"),
        ("isic", "b5_vs_b11", "attn_block5_ft", "attn_last1_ft"),
        ("isic", "b2_vs_b11", "attn_block2_ft", "attn_last1_ft"),
        ("isic", "b5_vs_full", "attn_block5_ft", "full_ft"),
        ("isic", "b5_vs_attnall", "attn_block5_ft", "attn_only_ft"),
        ("eurosat", "b5_vs_b11", "attn_block5_ft", "attn_last1_ft"),
        ("eurosat", "b2_vs_b11", "attn_block2_ft", "attn_last1_ft"),
    ]
    rows = []
    for ds, name, a, b in comps:
        d, ci, n = paired(ds, a, b)
        if d is None:
            print(f"    SKIP {ds}/{name}")
            continue
        flag = "settled" if abs(d) > ci and abs(d) > 0.6 else "unsettled"
        rows.append(f"{ds} {name} {d:+.2f} {ci:.2f} {n} {flag}")
    write("key_numbers.dat", "dataset name diff ci95 draws verdict", rows)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    print("figure data:")
    fig_depth_curves()
    fig_params_vs_acc()
    fig_inner_steps()
    key_numbers()
    print("done. Draw counts above; a caption must not claim more draws than listed.")
