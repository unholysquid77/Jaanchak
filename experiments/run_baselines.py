#!/usr/bin/env python
"""Level-0 baseline sweep.

    python experiments/run_baselines.py --dataset eurosat --k-shot 5 --episodes 600

Writes one JSON per (baseline, k-shot) into ``results/``, each carrying a manifest with
the git SHA, full config and environment, so a number can always be traced back to the
code that produced it.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import socket
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arc.backbones.tapped import TappedBackbone  # noqa: E402
from arc.data.base import EpisodicSampler, make_splits  # noqa: E402
from arc.data.sources import ImageFolderSource, build_transform, eurosat  # noqa: E402
from arc.eval.metrics import format_summary, summarise  # noqa: E402
from arc.heads.protonet import ProtoHead  # noqa: E402
from arc.train.baselines import BASELINES  # noqa: E402
from arc.train.episodic import EpisodeRunner  # noqa: E402

log = logging.getLogger("arc.baselines")


def git_sha() -> str:
    try:
        # The server checkout is a plain file tree, not a clone, so this is expected
        # to fail there; stderr is swallowed to keep run logs readable.
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parent.parent,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def manifest(args: argparse.Namespace) -> dict:
    return {
        "git_sha": git_sha(),
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": vars(args),
    }


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
    p.add_argument("--group-pattern", default=None, help="regex capturing specimen id")
    # group_disjoint is the default because cross-domain few-shot targets are
    # small-vocabulary (EuroSAT has 10 classes; the forensic sets have 8-12). Holding
    # out classes from such a corpus leaves too few to form n-way episodes, and the
    # standard CDFSL protocol does not hold out target classes anyway -- the held-out
    # domain IS the shift. Disjoint image pools still give an honest tuning split.
    p.add_argument("--split-mode", default="group_disjoint",
                   choices=["class_disjoint", "group_disjoint"])
    p.add_argument("--fractions", type=float, nargs=3, default=[0.2, 0.1, 0.7],
                   metavar=("TRAIN", "VAL", "TEST"),
                   help="split proportions; most of a target domain belongs in test")
    p.add_argument("--arch", default="resnet50")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--n-way", type=int, default=5)
    p.add_argument("--k-shot", type=int, nargs="+", default=[1, 5])
    p.add_argument("--n-query", type=int, default=15)
    p.add_argument("--episodes", type=int, default=600)
    p.add_argument("--inner-steps", type=int, default=20)
    p.add_argument("--inner-lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--baselines", nargs="+", default=list(BASELINES))
    p.add_argument("--out", default="results")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-amp", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s"
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("building source %s", args.dataset)
    source = build_source(args)
    splits = make_splits(source, mode=args.split_mode, fractions=tuple(args.fractions), seed=args.seed)
    for name, s in splits.items():
        log.info("split %-5s  %6d images  %2d classes", name, len(s.indices), len(s.classes))

    log.info("building backbone %s", args.arch)
    backbone = TappedBackbone(arch=args.arch, pretrained=True)
    runner = EpisodeRunner(
        backbone,
        head=ProtoHead(),
        device=args.device,
        inner_steps=args.inner_steps,
        inner_lr=args.inner_lr,
        amp=not args.no_amp,
    )

    for k_shot in args.k_shot:
        sampler = EpisodicSampler(
            source,
            splits["test"],
            n_way=args.n_way,
            k_shot=k_shot,
            n_query=args.n_query,
            n_episodes=args.episodes,
            seed=args.seed,
        )
        print(f"\n=== {args.dataset}  {args.n_way}-way {k_shot}-shot  "
              f"({args.episodes} episodes, seed {args.seed}) ===")

        for key in args.baselines:
            spec = BASELINES[key]
            controller, select, steps = spec.build(backbone, args.inner_steps)
            runner.inner_steps = steps

            records: list[dict[str, float]] = []
            started = time.perf_counter()
            for episode in sampler:
                result = runner.run_episode(episode, controller, select=select)
                rec = result.metrics.as_dict()
                rec["fraction_modified"] = result.fraction_modified
                rec["seconds"] = result.seconds
                records.append(rec)

            summary = summarise(records)
            elapsed = time.perf_counter() - started
            print(format_summary(key, summary), f" [{elapsed/60:.1f} min]")

            payload = {
                "manifest": manifest(args),
                "baseline": key,
                "description": spec.description,
                "k_shot": k_shot,
                "summary": summary,
                "episodes": records,
            }
            path = out_dir / f"baseline_{args.dataset}_{key}_{k_shot}shot_seed{args.seed}.json"
            path.write_text(json.dumps(payload, indent=2))
            log.info("wrote %s", path)

    runner.inner_steps = args.inner_steps


if __name__ == "__main__":
    main()
