#!/usr/bin/env python
"""Print the baselines that are tuned well enough to report.

    python experiments/drop_edge_baselines.py results/vit_edges_isic.json \
        --baselines full_ft ln_only_ft attn_only_ft ... --k-shot 5

Reads the ``--edges-out`` report from ``merge_sweeps.py`` and drops any baseline whose
best learning rate sits at the TOP of the merged grid, because that value is a lower
bound rather than an optimum: putting it in a parameter-type table understates the
group by an unknown amount, which is worse than leaving a hole where the reader can
see one.

A BOTTOM edge is kept. It means the group does best with the smallest step tried,
i.e. closest to doing nothing, and the table carries ``frozen_proto`` as the bracket
that makes that readable. The direction matters, so the two are not treated alike.

Baselines that take no gradient steps are never swept and so never appear in the edge
report; they pass through untouched. Names go to stdout, one per line; the reasoning
goes to stderr so a driver can capture the list with a plain command substitution.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("edges", help="JSON written by merge_sweeps.py --edges-out")
    p.add_argument("--baselines", nargs="+", required=True)
    p.add_argument("--k-shot", type=int, default=5)
    p.add_argument("--keep-bottom", action="store_true", default=True,
                   help="keep bottom-edge peaks (default); see module docstring")
    args = p.parse_args()

    report = json.loads(Path(args.edges).read_text())
    dropped = {
        e["baseline"]: e
        for e in report
        if e["k_shot"] == args.k_shot
        and (e["side"] == "top" or not args.keep_bottom)
    }

    kept = [b for b in args.baselines if b not in dropped]
    for b in args.baselines:
        if b in dropped:
            e = dropped[b]
            print(f"DROP {b}: peaks at the {e['side']} of the merged grid "
                  f"(lr={e['lr']:.0e}); extend the sweep before reporting it",
                  file=sys.stderr)
    print(f"reporting {len(kept)}/{len(args.baselines)} baselines", file=sys.stderr)
    print(" ".join(kept))


if __name__ == "__main__":
    main()
