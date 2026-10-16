#!/usr/bin/env python3
"""Is the extracted correction reproducible? (No.)

experiments/precomputed_shift.py reported its `shift` variant at 42.34 and
experiments/subspace_frontier.py reported 38.55 for what is provably the same
computation: both the extraction code (cosine 1.00000, max abs difference 1e-16 across
all 17 units) and the application code (max abs difference 0.0 on identical input) were
verified bit-identical between the two scripts.

The gap is between *runs*. Extracting a correction means adapting the backbone on 100
episodes and averaging the resulting activation displacement, and that adaptation is
not deterministic. This script extracts twice in one process -- same seeds, same
episodes, same weights -- and reports the per-unit agreement and the downstream accuracy
of each.

Measured on ISIC 5-shot (AMP on, the default): cosine 0.948 to 0.999, relative
difference 0.32 at the stem falling monotonically to 0.04 at layer4_2, and a 1.99-point
spread in the accuracy the two corrections produce. Early-layer displacements are small
in absolute norm (0.11 against 10.3) so noise dominates there -- which is unfortunate,
because the measured gain lives in layer2-3 rather than layer4, so the useful part of
the correction is the least stably estimated part.

The prime suspect is AMP's GradScaler skipping optimiser steps on inf/nan gradients
nondeterministically, which changes how much adaptation actually happened per episode;
EpisodeRunner already counts `skipped_steps` for exactly this reason. Nondeterministic
cuDNN convolution backward is a secondary candidate. Pass --no-amp to separate them.

Consequence for every result downstream: a correction extracted once carries an
uncertainty of order two accuracy points, so shift-derived numbers need a mean and an
error bar over extraction seeds, not only over evaluation episodes. That includes
rank-k, whose coefficients warm-start from the same mean displacement.

    python experiments/diagnose_extraction_variance.py --dataset isic --repeats 2
    python experiments/diagnose_extraction_variance.py --dataset isic --repeats 2 --no-amp
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arc.backbones.tapped import TappedBackbone  # noqa: E402
from arc.data.base import EpisodicSampler, make_splits  # noqa: E402
from arc.data.medical import chestx, isic  # noqa: E402
from arc.data.sources import ImageFolderSource, build_transform, eurosat  # noqa: E402
from arc.eval.metrics import episode_metrics, mean_ci95  # noqa: E402
from arc.heads.protonet import ProtoHead  # noqa: E402
from arc.train.baselines import BASELINES  # noqa: E402
from arc.train.episodic import EpisodeRunner  # noqa: E402

log = logging.getLogger("arc.extraction_variance")
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


def pooled_mean(a: torch.Tensor) -> torch.Tensor:
    a = a.double()
    return a.mean(dim=(0, 2, 3)) if a.ndim == 4 else a.mean(dim=0)


def embed(backbone, x, shift):
    h = x
    for u in backbone.unit_names:
        h = backbone.frozen[u](h)
        if u in shift:
            h = h + shift[u].to(device=h.device, dtype=h.dtype).view(
                1, -1, *([1] * (h.ndim - 2)))
    return torch.flatten(backbone.pool(h), 1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="isic",
                   choices=["eurosat", "isic", "chestx", "imagefolder"])
    p.add_argument("--data-root", default=None)
    p.add_argument("--group-pattern", default=None)
    p.add_argument("--fractions", type=float, nargs=3, default=[0.1, 0.4, 0.5])
    p.add_argument("--arch", default="resnet50")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--n-way", type=int, default=5)
    p.add_argument("--k-shot", type=int, default=5)
    p.add_argument("--n-query", type=int, default=3)
    p.add_argument("--eval-n-query", type=int, default=15)
    p.add_argument("--extract-episodes", type=int, default=100)
    p.add_argument("--eval-episodes", type=int, default=100)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--inner-steps", type=int, default=20)
    p.add_argument("--inner-lr", type=float, default=1e-4)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-amp", action="store_true",
                   help="AMP's GradScaler skipping steps on inf/nan is the prime "
                        "suspect for the nondeterminism; this separates it out")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="results/extraction_variance")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s | %(message)s")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    source = build_source(args)
    splits = make_splits(source, "group_disjoint", tuple(args.fractions), seed=args.split_seed)
    backbone = TappedBackbone(arch=args.arch, pretrained=True)
    head = ProtoHead().to(args.device)
    runner = EpisodeRunner(backbone, head=ProtoHead(), device=args.device,
                           inner_steps=args.inner_steps, inner_lr=args.inner_lr,
                           amp=not args.no_amp)
    ctrl, sel, steps = BASELINES["full_ft"].build(backbone, args.inner_steps)

    def extract(tag: str) -> tuple[dict, int]:
        runner.inner_steps = steps
        sampler = EpisodicSampler(source, splits["train"], n_way=args.n_way,
                                  k_shot=args.k_shot, n_query=args.n_query,
                                  n_episodes=args.extract_episodes, seed=args.seed + 20_000)
        disp, skipped = {}, 0
        for i, ep in enumerate(sampler):
            ep = ep.to(runner.device)
            probe = torch.cat([ep.support_x, ep.query_x])
            backbone.eval()
            with torch.no_grad():
                r = {k: pooled_mean(v.float().cpu())
                     for k, v in backbone.forward_reference(probe).activations.items()}
            res = runner.run_episode(ep, ctrl, select=sel)
            skipped += int(res.extra.get("skipped_steps", 0))
            backbone.eval()
            with torch.no_grad():
                a = {k: pooled_mean(v.float().cpu())
                     for k, v in backbone.forward_adapted(probe).activations.items()}
            for u in backbone.unit_names:
                disp.setdefault(u, []).append(a[u] - r[u])
            if (i + 1) % 50 == 0:
                log.info("%s extract %d/%d", tag, i + 1, args.extract_episodes)
        return {u: torch.stack(v).mean(0) for u, v in disp.items()}, skipped

    def evaluate(shift: dict) -> tuple[float, float]:
        backbone.eval()
        sampler = EpisodicSampler(source, splits["val"], n_way=args.n_way,
                                  k_shot=args.k_shot, n_query=args.eval_n_query,
                                  n_episodes=args.eval_episodes, seed=args.seed + 10_000)
        accs = []
        for ep in sampler:
            ep = ep.to(runner.device)
            with torch.no_grad():
                logits = head(embed(backbone, ep.support_x, shift), ep.support_y,
                              embed(backbone, ep.query_x, shift), ep.n_way)
            accs.append(episode_metrics(logits.float(), ep.query_y, ep.n_way).accuracy)
        return mean_ci95(accs)

    draws, skips = [], []
    for r in range(args.repeats):
        d, s = extract(chr(ord("A") + r))
        draws.append(d)
        skips.append(s)
        log.info("draw %s: %d AMP steps skipped across %d episodes",
                 chr(ord("A") + r), s, args.extract_episodes)

    print(f"\n=== extraction reproducibility, {args.dataset} {args.k_shot}-shot, "
          f"AMP {'off' if args.no_amp else 'on'} ===")
    print(f"{args.repeats} extractions, {args.extract_episodes} episodes each, "
          f"identical seeds\n")
    print(f"{'unit':<12}" + "".join(f"{'|' + chr(ord('A')+i) + '|':>10}"
                                    for i in range(len(draws)))
          + f"{'min cosine':>12}{'max rel.d':>11}")
    per_unit = {}
    for u in backbone.unit_names:
        cos = min(float(torch.nn.functional.cosine_similarity(draws[0][u], d[u], dim=0))
                  for d in draws[1:]) if len(draws) > 1 else 1.0
        rel = max(float((draws[0][u] - d[u]).norm() / draws[0][u].norm().clamp_min(1e-12))
                  for d in draws[1:]) if len(draws) > 1 else 0.0
        per_unit[u] = {"cosine": cos, "rel_diff": rel,
                       "norms": [float(d[u].norm()) for d in draws]}
        print(f"{u:<12}" + "".join(f"{float(d[u].norm()):>10.4f}" for d in draws)
              + f"{cos:>12.5f}{rel:>11.4f}")

    accs = [evaluate(d) for d in draws]
    print()
    for i, (m, c) in enumerate(accs):
        print(f"  shift from extraction {chr(ord('A')+i)}: {100*m:.2f} +/- {100*c:.2f}"
              f"   ({skips[i]} AMP steps skipped)")
    spread = 100 * (max(a[0] for a in accs) - min(a[0] for a in accs))
    print(f"\n  spread across extractions of the same thing: {spread:.2f} points")
    print("  A correction extracted once carries this much uncertainty. Shift-derived")
    print("  numbers need an error bar over extraction, not only over episodes.")

    path = out_dir / f"variance_{args.dataset}_{args.k_shot}shot_amp{int(not args.no_amp)}.json"
    path.write_text(json.dumps({
        "config": vars(args), "spread_points": spread,
        "accuracies": [{"mean": m, "ci95": c} for m, c in accs],
        "amp_steps_skipped": skips, "units": per_unit}, indent=2))
    print(f"\nwrote {path}", flush=True)


if __name__ == "__main__":
    main()
