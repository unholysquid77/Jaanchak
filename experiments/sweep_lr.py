#!/usr/bin/env python
"""Learning-rate / inner-step sweep for the Level-0 baselines.

    python experiments/sweep_lr.py --dataset eurosat --k-shot 1 5 --episodes 100

Runs on the **validation** split. The test split is never touched here -- the entire
purpose of this sweep is to let the baselines be reported at their own best settings, so
reading test during tuning would defeat it and hand a reviewer an easy objection.

Writes ``results/sweep/best_lr_<dataset>.json``, a mapping

    {baseline: {k_shot: {"lr": ..., "inner_steps": ..., "accuracy": ...}}}

consumed by ``run_baselines.py --tuned``. Baselines that take no gradient steps
(``frozen_proto``) are skipped rather than swept.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arc.backbones.tapped import TappedBackbone  # noqa: E402
from arc.data.base import EpisodicSampler, make_splits  # noqa: E402
from arc.data.sources import ImageFolderSource, build_transform, eurosat  # noqa: E402
from arc.eval.metrics import mean_ci95, summarise  # noqa: E402
from arc.heads.protonet import ProtoHead  # noqa: E402
from arc.train.baselines import BASELINES  # noqa: E402
from arc.train.episodic import EpisodeRunner  # noqa: E402

log = logging.getLogger("arc.sweep")

DEFAULT_LRS = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3]
DEFAULT_STEPS = [5, 20]


def build_source(args: argparse.Namespace):
    if args.dataset == "eurosat":
        return eurosat(args.data_root, size=args.image_size)
    if args.dataset == "imagefolder":
        if not args.data_root:
            raise SystemExit("--data-root is required for --dataset imagefolder")
        return ImageFolderSource(
            args.data_root,
            transform=build_transform(args.image_size),
            group_pattern=args.group_pattern,
        )
    raise SystemExit(f"unknown dataset {args.dataset!r}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="eurosat", choices=["eurosat", "imagefolder"])
    p.add_argument("--data-root", default="data/eurosat")
    p.add_argument("--group-pattern", default=None)
    p.add_argument("--split-mode", default="group_disjoint",
                   choices=["class_disjoint", "group_disjoint"])
    p.add_argument("--fractions", type=float, nargs=3, default=[0.2, 0.1, 0.7])
    p.add_argument("--arch", default="resnet50")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--n-way", type=int, default=5)
    p.add_argument("--k-shot", type=int, nargs="+", default=[1, 5])
    p.add_argument("--n-query", type=int, default=15)
    p.add_argument("--episodes", type=int, default=100,
                   help="episodes per config; tuning tolerates more noise than reporting")
    p.add_argument("--lrs", type=float, nargs="+", default=DEFAULT_LRS)
    p.add_argument("--inner-steps", type=int, nargs="+", default=DEFAULT_STEPS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--baselines", nargs="+", default=None)
    p.add_argument("--out", default="results/sweep")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-amp", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s | %(message)s"
    )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    keys = args.baselines or [k for k, s in BASELINES.items() if s.tunable]
    skipped = [k for k in keys if not BASELINES[k].tunable]
    if skipped:
        log.info("skipping non-tunable baselines: %s", skipped)
        keys = [k for k in keys if BASELINES[k].tunable]

    source = build_source(args)
    splits = make_splits(
        source, mode=args.split_mode, fractions=tuple(args.fractions), seed=args.seed
    )
    log.info("tuning on val split: %d images, %d classes",
             len(splits["val"].indices), len(splits["val"].classes))

    backbone = TappedBackbone(arch=args.arch, pretrained=True)
    runner = EpisodeRunner(
        backbone, head=ProtoHead(), device=args.device, amp=not args.no_amp
    )

    n_configs = len(keys) * len(args.k_shot) * len(args.lrs) * len(args.inner_steps)
    log.info("%d configs x %d episodes", n_configs, args.episodes)

    results: list[dict] = []
    best: dict[str, dict[str, dict]] = defaultdict(dict)
    done = 0
    sweep_started = time.perf_counter()

    for k_shot in args.k_shot:
        sampler = EpisodicSampler(
            source, splits["val"], n_way=args.n_way, k_shot=k_shot,
            n_query=args.n_query, n_episodes=args.episodes,
            seed=args.seed + 10_000,  # distinct stream from the reporting episodes
        )
        # Not materialised: the sampler seeds per episode *index*, so iterating it
        # afresh for each config yields byte-identical tasks. Holding 100 5-shot
        # episodes as float tensors would cost ~6 GB of RAM for no benefit.
        episodes = sampler

        for key in keys:
            for steps in args.inner_steps:
                for lr in args.lrs:
                    controller, select, _ = BASELINES[key].build(backbone, steps)
                    runner.inner_steps = steps
                    runner.inner_lr = lr

                    accs = []
                    started = time.perf_counter()
                    for ep in episodes:
                        r = runner.run_episode(ep, controller, select=select)
                        accs.append(r.metrics.accuracy)
                    mean, ci = mean_ci95(accs)
                    elapsed = time.perf_counter() - started
                    done += 1

                    row = {
                        "baseline": key, "k_shot": k_shot, "lr": lr,
                        "inner_steps": steps, "accuracy": mean, "ci95": ci,
                        "seconds": elapsed,
                    }
                    results.append(row)
                    log.info(
                        "[%3d/%3d] %-20s k=%-2d lr=%-7.0e steps=%-3d  acc %.4f +/- %.4f  (%.0fs)",
                        done, n_configs, key, k_shot, lr, steps, mean, ci, elapsed,
                    )

                    prev = best[key].get(str(k_shot))
                    if prev is None or mean > prev["accuracy"]:
                        best[key][str(k_shot)] = row

            # Flush after each baseline so a crash loses minutes, not hours.
            (out_dir / f"sweep_{args.dataset}_raw.json").write_text(json.dumps(results, indent=2))
            (out_dir / f"best_lr_{args.dataset}.json").write_text(json.dumps(best, indent=2))

    total = (time.perf_counter() - sweep_started) / 60
    print(f"\n=== best settings on val ({args.dataset}, {total:.0f} min) ===")
    for key in keys:
        for k_shot in args.k_shot:
            row = best[key].get(str(k_shot))
            if row:
                print(f"{key:<22} {k_shot}-shot   lr={row['lr']:<8.0e} "
                      f"steps={row['inner_steps']:<3d} val-acc {row['accuracy']*100:5.2f}")

    # A flat LR profile means the sweep range is in the wrong place entirely.
    for key in keys:
        for k_shot in args.k_shot:
            rows = [r for r in results if r["baseline"] == key and r["k_shot"] == k_shot]
            if rows and max(r["lr"] for r in rows if r["accuracy"] == max(x["accuracy"] for x in rows)) == max(args.lrs):
                log.warning(
                    "%s %d-shot peaked at the largest LR tried (%.0e) -- extend the grid upward",
                    key, k_shot, max(args.lrs),
                )

    log.info("wrote %s", out_dir / f"best_lr_{args.dataset}.json")


if __name__ == "__main__":
    main()
