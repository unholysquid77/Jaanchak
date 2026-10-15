#!/usr/bin/env python3
"""Is a corpus's split actually able to run the episodes we intend to ask for?

Answering this before launching a sweep rather than after. On EuroSAT the question
never came up: 10 near-balanced classes of 2,700 images each, so any split supports any
episode shape. The medical corpora are nothing like that. ISIC's largest class is 58x
its smallest (NV 6,705 vs DF 115), and `EpisodicSampler` deliberately *raises* when any
class in a split holds fewer than `k_shot + n_query` images rather than silently
dropping it -- so a fraction/shot combination that looks fine on paper aborts the run
at construction time, after the backbone has loaded.

Reports, per candidate split fraction:

* the corpus's provenance structure (images, specimens, inflation factor)
* per-split per-class image counts, with the binding constraint marked
* for each k-shot, whether `EpisodicSampler` can be built at all, and if not, which
  class blocks it and by how much

Runs on CPU in seconds -- it never builds a backbone, only indexes files.

    python experiments/diagnose_splits.py --dataset isic --k-shot 1 5 10 20
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arc.data.base import EpisodicSampler, make_splits  # noqa: E402
from arc.data.medical import chestx, isic  # noqa: E402
from arc.data.sources import ImageFolderSource, build_transform, eurosat  # noqa: E402

log = logging.getLogger("arc.diagnose_splits")

DEFAULT_ROOTS = {"eurosat": "data/eurosat", "isic": "data/isic2018", "chestx": "data/chestx"}


def build_source(args):
    root = args.data_root or DEFAULT_ROOTS.get(args.dataset)
    if args.dataset == "eurosat":
        return eurosat(root, size=args.image_size)
    if args.dataset == "isic":
        meta = Path(root) / "HAM10000_metadata.csv"
        return isic(root, size=args.image_size, metadata=meta if meta.exists() else None)
    if args.dataset == "chestx":
        return chestx(root, size=args.image_size)
    return ImageFolderSource(
        root, transform=build_transform(args.image_size), group_pattern=args.group_pattern
    )


def describe_corpus(source) -> None:
    labels = np.asarray(source.labels)
    groups = np.asarray(source.groups)
    _, per_group = np.unique(groups, return_counts=True)
    print(f"\nimages {len(source):>7d}   classes {len(source.class_names):>3d}   "
          f"specimens {per_group.size:>7d}")
    print(f"images sharing a specimen: {int(per_group[per_group > 1].sum()):d} "
          f"({100 * per_group[per_group > 1].sum() / max(len(source), 1):.1f}%)   "
          f"largest specimen: {int(per_group.max())} images")
    print("\nper-class image counts (corpus):")
    for i, name in enumerate(source.class_names):
        n = int((labels == i).sum())
        n_spec = np.unique(groups[labels == i]).size
        print(f"  {name:<14} {n:>6d} images  {n_spec:>6d} specimens")


def report(source, fractions, k_shots, n_way, n_query, seed) -> None:
    labels = np.asarray(source.labels)
    print(f"\n{'=' * 78}\nfractions {tuple(fractions)}  split-seed {seed}\n{'=' * 78}")
    splits = make_splits(source, "group_disjoint", tuple(fractions), seed=seed)

    header = f"{'split':<7}{'images':>8}{'classes':>9}" + "".join(
        f"{n[:9]:>10}" for n in source.class_names
    )
    print(header)
    for name, split in splits.items():
        counts = [int((labels[split.indices] == i).sum()) for i in range(len(source.class_names))]
        row = f"{name:<7}{len(split.indices):>8}{split.classes.size:>9}"
        print(row + "".join(f"{c:>10}" for c in counts))

    for name in ("val", "test"):
        split = splits[name]
        counts = {
            source.class_names[i]: int((labels[split.indices] == i).sum())
            for i in split.classes
        }
        print(f"\n  {name} split -- episode feasibility (n_way={n_way}, n_query={n_query}):")
        for k in k_shots:
            need = k + n_query
            thin = {c: n for c, n in counts.items() if n < need}
            try:
                EpisodicSampler(
                    source, split, n_way=n_way, k_shot=k, n_query=n_query,
                    n_episodes=1, seed=0,
                )
            except ValueError as exc:
                short = min(thin.values()) if thin else 0
                print(f"    {k:>3}-shot  needs {need:>3}/class  BLOCKED  "
                      f"thin: {thin or '-'}  (short by {need - short})")
                log.debug("%s", exc)
                continue
            margin = min(counts.values()) - need
            print(f"    {k:>3}-shot  needs {need:>3}/class  OK       "
                  f"tightest class has {min(counts.values())} (margin +{margin})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="isic",
                   choices=["eurosat", "isic", "chestx", "imagefolder"])
    p.add_argument("--data-root", default=None)
    p.add_argument("--group-pattern", default=None)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--n-way", type=int, default=5)
    p.add_argument("--n-query", type=int, default=15)
    p.add_argument("--k-shot", type=int, nargs="+", default=[1, 5, 10, 20])
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--fractions", type=float, nargs=3, action="append", default=None,
                   metavar=("TRAIN", "VAL", "TEST"),
                   help="repeatable; defaults to a ladder of candidates")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s | %(message)s")

    source = build_source(args)
    describe_corpus(source)

    candidates = args.fractions or [
        [0.2, 0.1, 0.7],   # the EuroSAT setting, for comparison
        [0.2, 0.2, 0.6],
        [0.1, 0.3, 0.6],
        [0.1, 0.4, 0.5],
    ]
    for fractions in candidates:
        report(source, fractions, args.k_shot, args.n_way, args.n_query, args.split_seed)

    print("\nNote: a BLOCKED cell aborts EpisodicSampler construction, so the run dies\n"
          "after the backbone has loaded. Pick fractions where every k-shot you intend\n"
          "to report is OK on both val and test, or reduce --n-query.\n", flush=True)


if __name__ == "__main__":
    main()
