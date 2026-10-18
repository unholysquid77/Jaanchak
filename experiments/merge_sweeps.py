#!/usr/bin/env python
"""Merge several sweep outputs into one tuned-settings file.

    python experiments/merge_sweeps.py results/sweep results/sweep_ext \
        --out results/best_lr_eurosat.json

Exists because the learning-rate grid had to be extended for baselines that update few
parameters: BatchNorm-affine adaptation touches ~53k of ResNet-50's 25M weights and
needs a far larger step than full fine-tuning before it moves at all. Rather than re-run
the whole sweep, the extension runs separately and is merged here by argmax over
validation accuracy.

Re-checks the grid-edge condition on the *merged* grid, since a config can peak at the
edge of one sweep while the union has an interior optimum -- or still be truncated after
merging, which the first sweep could not have told us.

Both edges are reported, not just the top. A peak at the BOTTOM of the grid is the
same defect and it has already happened: patch_embed_ft on EuroSAT peaked at 1e-6,
the smallest value tried, declining monotonically above it. That usually means the
group does nothing useful and the best step size is the one closest to frozen -- but
"usually means" is not a measurement, and an unflagged lower edge reads as a settled
optimum in the table.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dirs", nargs="+", help="sweep output directories")
    p.add_argument("--dataset", default="eurosat")
    p.add_argument("--out", default=None)
    p.add_argument("--edges-out", default=None, metavar="PATH",
                   help="also write the edge report as JSON: a list of "
                        "{baseline, k_shot, lr, side}. Lets a driver script decide "
                        "mechanically which baselines are tuned well enough to "
                        "report, instead of a human reading the warning above.")
    args = p.parse_args()

    rows: list[dict] = []
    for d in args.dirs:
        path = Path(d) / f"sweep_{args.dataset}_raw.json"
        if not path.exists():
            raise SystemExit(f"missing {path}")
        loaded = json.loads(path.read_text())
        rows.extend(loaded)
        print(f"{path}: {len(loaded)} configs")

    # Deduplicate: a config repeated across sweeps should not be counted twice, and if
    # two runs disagree the later file wins (sweeps are seeded, so they should not).
    unique: dict[tuple, dict] = {}
    for r in rows:
        unique[(r["baseline"], r["k_shot"], r["lr"], r["inner_steps"])] = r
    rows = list(unique.values())
    print(f"\n{len(rows)} unique configs after merge\n")

    best: dict[str, dict[str, dict]] = defaultdict(dict)
    grids: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        key, k = r["baseline"], str(r["k_shot"])
        grids[(key, k)].append(r["lr"])
        prev = best[key].get(k)
        if prev is None or r["accuracy"] > prev["accuracy"]:
            best[key][k] = r

    truncated: list[tuple[str, str, float, str]] = []
    print(f"{'baseline':<22}{'shot':>5}{'lr':>9}{'steps':>7}{'val acc':>10}")
    for key in sorted(best):
        for k in sorted(best[key], key=int):
            r = best[key][k]
            grid = sorted(grids[(key, k)])
            edge = ""
            if len(grid) > 1 and r["lr"] == grid[-1]:
                edge = " <- at grid edge (top)"
                truncated.append((key, k, r["lr"], "top"))
            elif len(grid) > 1 and r["lr"] == grid[0]:
                edge = " <- at grid edge (bottom)"
                truncated.append((key, k, r["lr"], "bottom"))
            print(f"{key:<22}{k:>5}{r['lr']:>9.0e}{r['inner_steps']:>7}"
                  f"{r['accuracy']*100:>9.2f}%{edge}")

    if truncated:
        print("\nWARNING: these peaked at an edge of the merged grid and are still")
        print("truncated -- their reported setting is a bound, not an optimum:")
        for key, k, lr, side in truncated:
            direction = "extend upward" if side == "top" else "extend downward"
            print(f"  {key} {k}-shot (lr={lr:.0e}, {side} edge -- {direction})")
    else:
        print("\nAll baselines have an interior optimum on the merged grid.")

    out = Path(args.out or f"results/best_lr_{args.dataset}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(best, indent=2))
    print(f"\nwrote {out}")

    if args.edges_out:
        edges = Path(args.edges_out)
        edges.parent.mkdir(parents=True, exist_ok=True)
        edges.write_text(json.dumps(
            [{"baseline": b, "k_shot": int(k), "lr": lr, "side": side}
             for b, k, lr, side in truncated], indent=2))
        print(f"wrote {edges}")


if __name__ == "__main__":
    main()
