#!/usr/bin/env python
"""Print the per-unit signal and gate values the controller actually produces.

The sweep only records episode means, which hides the question that matters: is the
gate *discriminating between layers*, or is it emitting one near-constant value and
acting as a global learning-rate multiplier? Those are very different methods and the
mean cannot tell them apart.

    python experiments/diagnose_gate.py --k-shot 5 --episodes 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arc.backbones.tapped import TappedBackbone  # noqa: E402
from arc.controllers.gate import AdaptiveGate, GateCoefficients  # noqa: E402
from arc.data.base import EpisodicSampler, make_splits  # noqa: E402
from arc.data.sources import eurosat  # noqa: E402
from arc.heads.protonet import ProtoHead  # noqa: E402
from arc.train.episodic import EpisodeRunner, select_all  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default="data/eurosat")
    p.add_argument("--k-shot", type=int, default=5)
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--inner-steps", type=int, default=20)
    p.add_argument("--inner-lr", type=float, default=3e-5)
    p.add_argument("--bias", type=float, default=4.0)
    p.add_argument("--utility", type=float, default=4.0)
    p.add_argument("--drift", type=float, default=4.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    source = eurosat(args.data_root)
    splits = make_splits(source, mode="group_disjoint", fractions=(0.2, 0.1, 0.7), seed=0)
    backbone = TappedBackbone(arch="resnet50", pretrained=True)
    runner = EpisodeRunner(
        backbone, head=ProtoHead(), device=args.device,
        inner_steps=args.inner_steps, inner_lr=args.inner_lr,
    )
    sampler = EpisodicSampler(
        source, splits["val"], n_way=5, k_shot=args.k_shot,
        n_query=15, n_episodes=args.episodes, seed=10_000,
    )

    coef = GateCoefficients(bias=args.bias, utility=args.utility, drift=args.drift)
    print(f"bias={args.bias:g}  aU={args.utility:g}  bD={args.drift:g}  "
          f"lr={args.inner_lr:.0e}  steps={args.inner_steps}\n", flush=True)

    for episode in sampler:
        controller = AdaptiveGate(coefficients=coef, cadence=5, warmup=1)
        runner.run_episode(episode, controller, select=select_all)

        ep = episode.to(runner.device)
        controller._gates = controller._gates  # final gate from the run
        signals = controller.compute_signals()
        gates = controller.gate_from(signals, runner.device)

        u = signals.get("utility", runner.device)
        d = signals.get("drift", runner.device)
        print(f"--- episode {episode.episode_id} ---", flush=True)
        print(f"{'unit':<12}{'U':>9}{'D':>9}{'gate':>9}", flush=True)
        for i, name in enumerate(backbone.unit_names):
            print(f"{name:<12}{u[i]:>9.3f}{d[i]:>9.3f}{gates[i]:>9.3f}", flush=True)
        print(f"{'spread':<12}{u.max()-u.min():>9.3f}{d.max()-d.min():>9.3f}"
              f"{gates.max()-gates.min():>9.3f}", flush=True)
        print(f"{'mean':<12}{u.mean():>9.3f}{d.mean():>9.3f}{gates.mean():>9.3f}\n",
              flush=True)


if __name__ == "__main__":
    main()
