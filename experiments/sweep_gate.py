#!/usr/bin/env python
"""Tune the adaptive gate on the validation split.

Staged rather than a full grid, because the parameters are not independent and a joint
sweep would spend most of its time at learning rates that are simply wrong:

  stage lr     U = D = 0, so the gate is a constant sigmoid(bias) and the controller
               reduces to uniformly scaled fine-tuning. This isolates the learning
               rate, which the smoke run showed matters far more than anything else:
               at the default 3e-4 the gate sat at 0.87 and modified 98.8% of
               parameters, i.e. full fine-tuning at 30x full_ft's tuned 5-shot rate.

  stage gate   At the winning learning rate, sweep (bias, utility, drift). Only here
               do the signals get to do anything, so only here is the comparison
               "does representation feedback help?" actually being asked.

Writes results/gate_sweep/best_gate_<dataset>.json for run_gate.py --coefficients.

Everything runs on the validation split; test is never read during tuning, exactly as
for the baselines.
"""

from __future__ import annotations

import argparse
import itertools
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
from arc.eval.metrics import mean_ci95  # noqa: E402
from arc.heads.protonet import ProtoHead  # noqa: E402
from arc.train.episodic import EpisodeRunner, select_all  # noqa: E402

log = logging.getLogger("arc.sweep_gate")


def build_source(args):
    if args.dataset == "eurosat":
        return eurosat(args.data_root or "data/eurosat", size=args.image_size)
    root = args.data_root or "data/isic2018"
    meta = Path(root) / "HAM10000_metadata.csv"
    return isic(root, size=args.image_size, metadata=meta if meta.exists() else None)


def evaluate(runner, sampler, coef, cadence, warmup, use_u, use_d):
    accs, gates, fracs = [], [], []
    for episode in sampler:
        controller = AdaptiveGate(
            coefficients=coef, cadence=cadence, warmup=warmup,
            use_utility=use_u, use_drift=use_d,
        )
        r = runner.run_episode(episode, controller, select=select_all)
        accs.append(r.metrics.accuracy)
        fracs.append(r.fraction_modified)
        if "mean_gate_mean" in r.extra:
            gates.append(r.extra["mean_gate_mean"])
    mean, ci = mean_ci95(accs)
    return {
        "accuracy": mean,
        "ci95": ci,
        "gate_mean": sum(gates) / len(gates) if gates else float("nan"),
        "fraction_modified": sum(fracs) / len(fracs),
    }


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
    p.add_argument("--episodes", type=int, default=75)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--inner-steps", type=int, default=20)
    p.add_argument("--cadence", type=int, default=5)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--lrs", type=float, nargs="+",
                   default=[1e-5, 3e-5, 1e-4, 3e-4])
    p.add_argument("--biases", type=float, nargs="+", default=[0.0, 2.0, 4.0])
    p.add_argument("--utilities", type=float, nargs="+", default=[0.0, 2.0, 4.0])
    p.add_argument("--drifts", type=float, nargs="+", default=[0.0, 4.0, 8.0])
    p.add_argument("--grad-checkpoint", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="results/gate_sweep")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s | %(message)s")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    source = build_source(args)
    splits = make_splits(
        source, mode="group_disjoint", fractions=tuple(args.fractions), seed=args.split_seed
    )
    log.info("tuning on val: %d images", len(splits["val"].indices))

    backbone = TappedBackbone(arch=args.arch, pretrained=True)
    backbone.grad_checkpoint = args.grad_checkpoint
    runner = EpisodeRunner(
        backbone, head=ProtoHead(), device=args.device,
        inner_steps=args.inner_steps, amp=not args.no_amp,
    )

    rows, best = [], {}

    for k_shot in args.k_shot:
        sampler = EpisodicSampler(
            source, splits["val"], n_way=args.n_way, k_shot=k_shot,
            n_query=args.n_query, n_episodes=args.episodes, seed=args.seed + 10_000,
        )

        # -- stage 1: learning rate, signals disabled ------------------------
        print(f"\n=== {k_shot}-shot, stage 'lr' (gate constant, signals off) ===")
        lr_rows = []
        for lr in args.lrs:
            runner.inner_lr = lr
            coef = GateCoefficients(bias=2.0, utility=0.0, drift=0.0)
            t0 = time.perf_counter()
            r = evaluate(runner, sampler, coef, args.cadence, args.warmup, False, False)
            r |= {"stage": "lr", "k_shot": k_shot, "lr": lr,
                  "bias": 2.0, "utility": 0.0, "drift": 0.0,
                  "seconds": time.perf_counter() - t0}
            rows.append(r); lr_rows.append(r)
            print(f"  lr={lr:<8.0e} acc {r['accuracy']*100:5.2f} +/- {r['ci95']*100:4.2f}"
                  f"  gate {r['gate_mean']:.3f}  params {r['fraction_modified']*100:5.1f}%")
            (out_dir / f"gate_sweep_{args.dataset}_raw.json").write_text(json.dumps(rows, indent=2))

        best_lr = max(lr_rows, key=lambda r: r["accuracy"])["lr"]
        if best_lr == max(args.lrs) or best_lr == min(args.lrs):
            log.warning(
                "%d-shot: learning rate peaked at a grid edge (%.0e) -- extend the grid",
                k_shot, best_lr,
            )
        print(f"  -> best lr {best_lr:.0e}")

        # -- stage 2: the gate itself ----------------------------------------
        print(f"\n=== {k_shot}-shot, stage 'gate' (lr={best_lr:.0e}) ===")
        runner.inner_lr = best_lr
        gate_rows = []
        for bias, u, d in itertools.product(args.biases, args.utilities, args.drifts):
            coef = GateCoefficients(bias=bias, utility=u, drift=d)
            t0 = time.perf_counter()
            r = evaluate(runner, sampler, coef, args.cadence, args.warmup, u > 0, d > 0)
            r |= {"stage": "gate", "k_shot": k_shot, "lr": best_lr,
                  "bias": bias, "utility": u, "drift": d,
                  "seconds": time.perf_counter() - t0}
            rows.append(r); gate_rows.append(r)
            print(f"  bias={bias:<4g} aU={u:<4g} bD={d:<4g} "
                  f"acc {r['accuracy']*100:5.2f} +/- {r['ci95']*100:4.2f}"
                  f"  gate {r['gate_mean']:.3f}  params {r['fraction_modified']*100:5.1f}%")
            (out_dir / f"gate_sweep_{args.dataset}_raw.json").write_text(json.dumps(rows, indent=2))

        top = max(gate_rows, key=lambda r: r["accuracy"])
        best[str(k_shot)] = {
            "lr": best_lr, "inner_steps": args.inner_steps,
            "bias": top["bias"], "utility": top["utility"], "drift": top["drift"],
            "val_accuracy": top["accuracy"],
        }
        (out_dir / f"best_gate_{args.dataset}.json").write_text(json.dumps(best, indent=2))

        # The comparison that matters: does feedback beat the same gate held constant?
        flat = [r for r in gate_rows if r["utility"] == 0 and r["drift"] == 0]
        if flat:
            ref = max(flat, key=lambda r: r["accuracy"])
            delta = (top["accuracy"] - ref["accuracy"]) * 100
            print(f"\n  best with signals : {top['accuracy']*100:5.2f} "
                  f"(bias={top['bias']:g} aU={top['utility']:g} bD={top['drift']:g})")
            print(f"  best without      : {ref['accuracy']*100:5.2f} (bias={ref['bias']:g})")
            print(f"  signals are worth : {delta:+.2f} points on val")

    print(f"\nwrote {out_dir / f'best_gate_{args.dataset}.json'}")


if __name__ == "__main__":
    main()
