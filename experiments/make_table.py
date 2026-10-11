#!/usr/bin/env python
"""Build the Level-0 results table from per-run JSON files.

    python experiments/make_table.py results/level0

Pools episodes across seeds rather than averaging the per-seed means. With a fixed
split and 600 episodes per seed, the seeds differ only in which tasks were drawn, so
the pooled set of 1800 episodes is the sample we actually care about and its CI is the
honest one. Incomplete cells (a seed still running) are reported with the seed count so
a partial table cannot be mistaken for a finished one.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from arc.eval.metrics import mean_ci95

ORDER = [
    "frozen_proto",
    "last_layer_ft",
    "bn_only_ft",
    "stage4_ft",
    "progressive_unfreeze",
    "full_ft",
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results_dir", nargs="?", default="results/level0")
    p.add_argument("--metric", default="accuracy")
    p.add_argument("--markdown", action="store_true")
    args = p.parse_args()

    episodes: dict[tuple[str, int], list[float]] = defaultdict(list)
    frac: dict[tuple[str, int], list[float]] = defaultdict(list)
    seeds: dict[tuple[str, int], set[int]] = defaultdict(set)
    settings: dict[tuple[str, int], tuple] = {}

    for path in sorted(Path(args.results_dir).glob("*.json")):
        d = json.loads(path.read_text())
        key = (d["baseline"], d["k_shot"])
        episodes[key] += [e[args.metric] for e in d["episodes"]]
        frac[key] += [e["fraction_modified"] for e in d["episodes"]]
        seeds[key].add(d.get("seed", 0))
        settings[key] = (d.get("inner_lr"), d.get("inner_steps"), d.get("tuned", False))

    shots = sorted({k for _, k in episodes})
    names = [b for b in ORDER if any((b, k) in episodes for k in shots)]
    names += sorted({b for b, _ in episodes} - set(ORDER))

    sep = " | " if args.markdown else "  "
    head = f"{'baseline':<22}" + sep.join(f"{str(k)+'-shot':>16}" for k in shots)
    print(f"\n{args.metric}, 5-way, mean +/- 95% CI over pooled episodes\n")
    print(head)
    print("-" * len(head))

    for b in names:
        cells = []
        for k in shots:
            vals = episodes.get((b, k))
            if not vals:
                cells.append(f"{'-':>16}")
                continue
            m, ci = mean_ci95(vals)
            n_seeds = len(seeds[(b, k)])
            flag = "" if n_seeds >= 3 else f"({n_seeds})"
            cells.append(f"{m*100:>10.2f}+/-{ci*100:<4.2f}{flag:>2}")
        print(f"{b:<22}" + sep.join(cells))

    print("\n(n) = fewer than 3 seeds pooled, run still in progress\n")

    print(f"{'baseline':<22}{'shot':>6}{'lr':>9}{'steps':>7}{'tuned':>7}"
          f"{'episodes':>10}{'%params moved':>15}")
    for b in names:
        for k in shots:
            if (b, k) not in episodes:
                continue
            lr, steps, tuned = settings[(b, k)]
            lr_s = f"{lr:.0e}" if lr is not None else "-"
            print(f"{b:<22}{k:>6}{lr_s:>9}{steps:>7}{str(tuned):>7}"
                  f"{len(episodes[(b,k)]):>10}{np.mean(frac[(b,k)])*100:>14.2f}%")


if __name__ == "__main__":
    main()
