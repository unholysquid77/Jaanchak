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

SHOT_DIRS = {1: "vit_shots1", 5: "vit_2b_eurosat", 20: "vit_shots20"}
# k=5's frozen reference is deliberately NOT taken from vit_shotsref5. The main
# corpus (vit_2b_eurosat + vit_seeds_eurosat) already carries frozen_proto at k=5
# over three draws, which is strictly better than shotsref5's one, and loading
# both tripped the duplicate check on ('eurosat', 5, 'frozen_proto', 0) -- the
# guard working as designed on a genuine re-measurement.
#
# Nothing is lost by dropping it: the two runs agree to +0.00 +/- 0.00, paired,
# bit-for-bit over all 100 episodes. frozen_proto takes no gradient steps, so
# that is forward-pass-only reproducibility and it is exact. See RESULTS_VIT.md
# §3.7e -- the ~0.6 floor is entirely in the optimiser path.
SHOT_REF_DIRS = {1: "vit_shotsref1", 20: "vit_shotsref20"}


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
        c = j.get("manifest", {}).get("config", {})
        ds = c.get("dataset")
        if ds is None:
            raise RuntimeError(
                f"{f}: no manifest.config.dataset -- refusing to guess which dataset "
                f"this is, because guessing is what produced the collision this "
                f"function's docstring describes"
            )
        # The key must carry the SHOT COUNT for the same reason it carries the
        # dataset. The k-shot runs are all EuroSAT and reuse every baseline name
        # (full_ft, attn_only_ft, ...), so on the three-field key the k=1 and k=20
        # directories would have collided with the k=5 ones -- a fourth instance of
        # this file's founding bug, on a new axis. It would have been caught rather
        # than silent, because of the duplicate check below, but only if someone
        # happened to load those directories; keying on k means the axis simply
        # exists. k_shot is written as a list by run_baselines.py.
        ks = c.get("k_shot")
        if not isinstance(ks, list) or len(ks) != 1:
            raise RuntimeError(
                f"{f}: manifest.config.k_shot is {ks!r}, expected a one-element list. "
                f"A multi-k run would need one record per k, not one per file."
            )
        key = (ds, ks[0], j["baseline"], j.get("seed", 0))
        if key in acc:
            raise RuntimeError(f"{f}: duplicate {key} -- two files claim the same cell")
        acc[key] = [e["accuracy"] * 100 for e in j["episodes"]]


def _episodes() -> dict:
    acc: dict = {}
    for d in (list(EURO_ATTN.values()) + list(EURO_MLP.values())
              + list(ISIC_ATTN.values()) + SEED_DIRS
              + ["vit_2b_eurosat", "vit_2b_isic", "vit_seeds_eurosat", "vit_seeds_isic"]
              + [d for d in list(SHOT_DIRS.values()) + list(SHOT_REF_DIRS.values())
                 if d != "vit_2b_eurosat"]):
        if os.path.isdir(f"results/{d}"):
            _load_dir(d, acc)
    return acc


ACC = _episodes()
# ISIC depth 11 is filed under attn_last1_ft (attn_block11_ft is deliberately
# unregistered -- it duplicates attn_last1_ft); same for the EuroSAT n=1 suffix point.
ALIAS = {"attn_block11_ft": "attn_last1_ft"}


def mean_and_draws(ds: str, base: str, k: int = 5):
    """Three-draw mean where available, else the single draw. Returns (mean, n, sd).

    k defaults to 5 because every depth, type and capacity result in the paper is
    5-shot; only the support-size section varies it.
    """
    base = ALIAS.get(base, base)
    vals = [st.mean(ACC[(ds, k, base, s)]) for s in (0, 1, 2) if (ds, k, base, s) in ACC]
    if not vals:
        return None, 0, None
    return st.mean(vals), len(vals), (st.stdev(vals) if len(vals) > 1 else None)


def paired(ds: str, a: str, b: str, k: int = 5):
    """(difference, ci95, n_draws) WITHIN one dataset AND one shot count.

    Per-seed across draws where three exist, else per-episode within the single draw.
    Both arms must share k: the episodes are only identical, and the comparison only
    paired, within one (dataset, k) cell.
    """
    a, b = ALIAS.get(a, a), ALIAS.get(b, b)
    shared = [s for s in (0, 1, 2) if (ds, k, a, s) in ACC and (ds, k, b, s) in ACC]
    if len(shared) >= 3:
        d = [st.mean(ACC[(ds, k, a, s)]) - st.mean(ACC[(ds, k, b, s)]) for s in shared]
        return st.mean(d), T2 * st.stdev(d) / math.sqrt(len(d)), len(d)
    if not shared:
        return None, None, 0
    s = shared[0]
    d = [x - y for x, y in zip(ACC[(ds, k, a, s)], ACC[(ds, k, b, s)])]
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


def fig_shots() -> None:
    """Support-size sweep: does the attention advantage grow with k, as predicted?

    Emitted with the frozen reference in its own column precisely because the k=1
    reading is ambiguous without it. k=1 gave attn - full_ft = +1.25 against +2.72 at
    k=5 -- the gap SHRANK where the over-optimisation story said it should grow. But
    "the prediction failed" and "at k=1 nothing adapts, so there is no gap to have"
    are observationally identical from those two rows alone. If attn and full_ft both
    sit near frozen at k=1, it is a floor, not a refutation.

    frozen may be absent (run_vit_shots_ref.sh has not run); the column then reads
    "nan" and the row still carries its draw count, so a reader cannot mistake a
    missing reference for a measured one. Rows for a k whose run has not finished are
    omitted entirely rather than emitted empty.
    """
    rows = []
    for k in sorted(SHOT_DIRS):
        a, na, _ = mean_and_draws("eurosat", "attn_only_ft", k)
        f_, nf, _ = mean_and_draws("eurosat", "full_ft", k)
        if a is None or f_ is None:
            print(f"    SKIP k={k} (no measurement yet)")
            continue
        fr, nfr, _ = mean_and_draws("eurosat", "frozen_proto", k)
        d, ci, nd = paired("eurosat", "attn_only_ft", "full_ft", k)
        frs = f"{fr:.2f}" if fr is not None else "nan"
        rows.append(f"{k} {a:.2f} {f_:.2f} {frs} {d:+.2f} {ci:.2f} {nd} {min(na, nf)}")
    if rows:
        write("shots.dat", "k attn full frozen gap ci95 draws arm_draws", rows)
    else:
        print("    SKIP shots.dat entirely -- no k cells loaded")


def fig_weight_decay() -> None:
    """Weight-decay sweep, each level paired against its own wd=0 arm.

    Loaded by a DEDICATED accumulator rather than through ACC, because the wd runs
    are all EuroSAT k=5 and reuse the same baseline names: on the shared key they
    would collide with vit_2b_eurosat on ('eurosat', 5, 'full_ft', 0). Weight decay
    is a fourth axis, and rather than widen the universal key again for a
    self-contained comparison, it gets its own dict keyed (wd, baseline). The
    comparison never crosses sweeps -- every level is paired against the wd=0 arm
    measured in the SAME sweep, at the same tuned lr.
    """
    acc: dict = {}
    for d in sorted(glob.glob("results/vit_wd_*")):
        wd = os.path.basename(d)[len("vit_wd_"):]
        for f in glob.glob(f"{d}/*.json"):
            j = json.load(open(f))
            key = (wd, j["baseline"])
            if key in acc:
                raise RuntimeError(f"{f}: duplicate {key}")
            acc[key] = (j, [e["accuracy"] * 100 for e in j["episodes"]])
    if ("0", "full_ft") not in acc:
        print("    SKIP weight_decay.dat -- no wd=0 arm to pair against")
        return
    rows = []
    for base in ("full_ft", "attn_only_ft"):
        if ("0", base) not in acc:
            continue
        j0, ref = acc[("0", base)]
        for wd in sorted(k[0] for k in acc if k[1] == base and k[0] != "0"):
            jj, v = acc[(wd, base)]
            if jj["inner_lr"] != j0["inner_lr"]:
                raise RuntimeError(
                    f"wd={wd} {base}: lr {jj['inner_lr']} != wd=0's {j0['inner_lr']}; "
                    f"that is a learning-rate comparison, not a weight-decay one"
                )
            d = [x - y for x, y in zip(v, ref)]
            m = st.mean(d)
            ci = T99 * st.stdev(d) / math.sqrt(len(d))
            rows.append(f"{base} {wd} {st.mean(v):.2f} {m:+.2f} {ci:.2f} "
                        f"{'settled' if abs(m) > ci and abs(m) > 0.6 else 'unsettled'}")
    write("weight_decay.dat", "baseline wd acc diff_vs_wd0 ci95 verdict", rows)


def mech_numbers() -> None:
    """Each method's gain over a frozen backbone, per shot count.

    This is the statistic the over-optimisation account actually predicts, and the
    one the raw attn-minus-full gap cannot stand in for: the gap is a difference of
    two gains that are both floored at k=1 (nothing adapts) and both large at k=20
    (both adapt fully), so it is non-monotone in k for reasons that have nothing to
    do with the mechanism. See RESULTS_VIT.md §3.7d.
    """
    rows = []
    for k in sorted(SHOT_DIRS):
        for base in ("attn_only_ft", "full_ft"):
            d, ci, n = paired("eurosat", base, "frozen_proto", k)
            if d is None:
                print(f"    SKIP mech k={k} {base}")
                continue
            rows.append(f"{k} {base} {d:+.2f} {ci:.2f} {n} "
                        f"{'settled' if abs(d) > ci and abs(d) > 0.6 else 'unsettled'}")
    if rows:
        write("mech_gains.dat", "k baseline gain_over_frozen ci95 draws verdict", rows)


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
        ("isic", "b5_vs_b8", "attn_block5_ft", "attn_block8_ft"),
        ("eurosat", "b8_vs_b5", "attn_block8_ft", "attn_block5_ft"),
        ("eurosat", "b2_vs_b0", "attn_block2_ft", "attn_block0_ft"),
        ("eurosat", "b5mlp_vs_b5attn", "mlp_block5_ft", "attn_block5_ft"),
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
    fig_shots()
    fig_weight_decay()
    mech_numbers()
    key_numbers()
    print("done. Draw counts above; a caption must not claim more draws than listed.")
