#!/usr/bin/env python
"""Level 1 -- the adaptive gate, against the tuned Level-0 baselines.

    python experiments/run_gate.py --k-shot 1 5 --episodes 600 --seeds 0 1 2

Runs through the same EpisodeRunner as every baseline, on the same seeded episodes, so
the comparison is matched by construction. The only difference is the controller.

Coefficients are fixed hyperparameters here (mode (iii) of docs/ARC_SPEC.md section 5)
and must be chosen on the validation split, exactly as the baseline learning rates
were -- see sweep_gate.py. Running this without --coefficients uses the defaults, which
are a guess, and the output says so.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arc.backbones.tapped import TappedBackbone  # noqa: E402
from arc.controllers.gate import AdaptiveGate, GateCoefficients  # noqa: E402
from arc.data.base import EpisodicSampler, make_splits  # noqa: E402
from arc.data.medical import isic  # noqa: E402
from arc.data.sources import eurosat  # noqa: E402
from arc.eval.metrics import format_summary, summarise  # noqa: E402
from arc.heads.protonet import ProtoHead  # noqa: E402
from arc.train.episodic import EpisodeRunner, select_all  # noqa: E402

log = logging.getLogger("arc.gate")


def build_source(args):
    if args.dataset == "eurosat":
        return eurosat(args.data_root or "data/eurosat", size=args.image_size)
    if args.dataset == "isic":
        root = args.data_root or "data/isic2018"
        meta = Path(root) / "HAM10000_metadata.csv"
        return isic(root, size=args.image_size, metadata=meta if meta.exists() else None)
    raise SystemExit(f"unknown dataset {args.dataset!r}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="eurosat", choices=["eurosat", "isic"])
    p.add_argument("--data-root", default=None)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--fractions", type=float, nargs=3, default=[0.2, 0.1, 0.7])
    p.add_argument("--arch", default="resnet50")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--n-way", type=int, default=5)
    p.add_argument("--k-shot", type=int, nargs="+", default=[1, 5])
    p.add_argument("--n-query", type=int, default=15)
    p.add_argument("--episodes", type=int, default=600)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--split", default="test", choices=["val", "test"])

    p.add_argument("--inner-steps", type=int, default=20)
    p.add_argument("--inner-lr", type=float, default=3e-4)
    p.add_argument("--cadence", type=int, default=5)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--coefficients", default=None,
                   help="JSON file of gate coefficients from sweep_gate.py")
    p.add_argument("--bias", type=float, default=2.0)
    p.add_argument("--alpha-utility", type=float, default=2.0)
    p.add_argument("--beta-drift", type=float, default=4.0)
    p.add_argument("--no-utility", action="store_true", help="ablation: drop U")
    p.add_argument("--no-drift", action="store_true", help="ablation: drop D")

    p.add_argument("--grad-checkpoint", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="results/gate")
    p.add_argument("--tag", default="gate")
    p.add_argument("--skip-existing", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s"
    )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.coefficients:
        payload = json.loads(Path(args.coefficients).read_text())
        tuned = True
    else:
        payload = {}
        tuned = False
        log.warning(
            "no --coefficients file: using defaults (bias=%g, utility=%g, drift=%g). "
            "These are a guess. Tune them on the validation split with "
            "experiments/sweep_gate.py before reporting anything.",
            args.bias, args.alpha_utility, args.beta_drift,
        )

    source = build_source(args)
    splits = make_splits(
        source, mode="group_disjoint", fractions=tuple(args.fractions), seed=args.split_seed
    )
    log.info("evaluating on the %r split (%d images)", args.split, len(splits[args.split].indices))

    backbone = TappedBackbone(arch=args.arch, pretrained=True)
    backbone.grad_checkpoint = args.grad_checkpoint
    runner = EpisodeRunner(
        backbone,
        head=ProtoHead(),
        device=args.device,
        inner_steps=args.inner_steps,
        inner_lr=args.inner_lr,
        amp=not args.no_amp,
    )

    for seed in args.seeds:
        for k_shot in args.k_shot:
            path = out_dir / f"{args.tag}_{args.dataset}_{k_shot}shot_seed{seed}.json"
            if args.skip_existing and path.exists():
                log.info("skip %s", path.name)
                continue

            setting = payload.get(str(k_shot), {})
            coef = GateCoefficients(
                bias=float(setting.get("bias", args.bias)),
                utility=0.0 if args.no_utility else float(
                    setting.get("utility", args.alpha_utility)
                ),
                drift=0.0 if args.no_drift else float(
                    setting.get("drift", args.beta_drift)
                ),
            )
            runner.inner_lr = float(setting.get("lr", args.inner_lr))
            runner.inner_steps = int(setting.get("inner_steps", args.inner_steps))

            sampler = EpisodicSampler(
                source, splits[args.split], n_way=args.n_way, k_shot=k_shot,
                n_query=args.n_query, n_episodes=args.episodes, seed=seed,
            )

            records = []
            started = time.perf_counter()
            for episode in sampler:
                controller = AdaptiveGate(
                    coefficients=coef,
                    cadence=args.cadence,
                    warmup=args.warmup,
                    use_utility=not args.no_utility,
                    use_drift=not args.no_drift,
                )
                result = runner.run_episode(episode, controller, select=select_all)
                rec = result.metrics.as_dict()
                rec["fraction_modified"] = result.fraction_modified
                rec["seconds"] = result.seconds
                rec.update(result.extra)
                records.append(rec)

            summary = summarise(records)
            elapsed = (time.perf_counter() - started) / 60
            mark = "" if tuned else "  (untuned coefficients)"
            print(
                format_summary(f"{args.tag} {k_shot}-shot seed{seed}", summary),
                f" lr={runner.inner_lr:.0e} bias={coef.bias:g} "
                f"aU={coef.utility:g} bD={coef.drift:g} [{elapsed:.1f} min]{mark}"
            )

            path.write_text(json.dumps({
                "baseline": args.tag,
                "k_shot": k_shot,
                "seed": seed,
                "split": args.split,
                "tuned": tuned,
                "inner_lr": runner.inner_lr,
                "inner_steps": runner.inner_steps,
                "coefficients": coef.as_dict() | {"bias": coef.bias},
                "cadence": args.cadence,
                "warmup": args.warmup,
                "summary": summary,
                "episodes": records,
            }, indent=2))
            log.info("wrote %s", path)


if __name__ == "__main__":
    main()
