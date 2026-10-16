#!/usr/bin/env python3
"""Do the gate's signals predict where adapting a unit actually helps?

This is the question the Level-1 null leaves unanswered. The gate is a map from
per-unit signals to per-unit adaptation intensity, and it does not help -- on EuroSAT
the marginal effects are -0.05 / -0.10 and on ISIC -1.43 / -3.33. But "the controller
is wrong" and "the signals carry nothing to control on" are different failures with
different fixes, and nothing distinguishes them yet.

So measure the thing a signal would have to predict. For each unit, by direct
intervention:

    benefit_only[l]  = accuracy adapting ONLY unit l      - accuracy adapting nothing
    benefit_loo[l]   = accuracy adapting everything       - accuracy adapting all but l

`only` is the standalone value of a unit; `loo` is its marginal value in context. They
differ when units are redundant, and a gate acts in context, so `loo` is the closer
match to what a gate would need -- both are reported because the gap between them is
itself informative about redundancy.

Then correlate each signal against each benefit, per episode and pooled. If no signal
correlates with realised benefit, the state vector is the problem rather than the
controller, and no amount of controller engineering over U and D can work. That is a
sharper and more useful negative than "our gate did not help", and it is the precise
condition any *replacement* signal has to satisfy.

Also recorded properly, with episode-to-episode variance this time: the depth
correlation of each signal. A one-episode profile in docs/EXPERIMENT_LOG.md put drift
at Spearman rho = +0.989 with layer index, which would make `-beta*D` a stage-4
detector rather than a state signal -- a fixed depth schedule wearing a feedback
controller's costume. n=17 units and one episode is not enough to assert that; this
run settles it.

Signals are computed after `--warmup` adaptation steps rather than at step 0, because
at step 0 the adapted and frozen branches are the same network, drift is identically
zero everywhere, and the measurement would be vacuous. That matches what AdaptiveGate
does with its own warmup.

Cost is 2 + 2*n_units adaptation runs per episode (36 for ResNet-50), so this is
deliberately run at tens of episodes, not hundreds.

    python experiments/diagnose_unit_benefit.py --dataset isic --k-shot 5 --episodes 30
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arc.backbones.tapped import TappedBackbone  # noqa: E402
from arc.data.base import EpisodicSampler, make_splits  # noqa: E402
from arc.data.medical import chestx, isic  # noqa: E402
from arc.data.sources import ImageFolderSource, build_transform, eurosat  # noqa: E402
from arc.heads.protonet import ProtoHead  # noqa: E402
from arc.signals.drift import drift_signal  # noqa: E402
from arc.signals.utility import utility_signal  # noqa: E402
from arc.train.episodic import (  # noqa: E402
    EpisodeRunner,
    StaticController,
    select_all,
    select_none,
    select_units,
)
from arc.train.objectives import inner_loss  # noqa: E402

log = logging.getLogger("arc.diagnose_unit_benefit")


def rho(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Spearman correlation, or (nan, nan) when an input has no variance.

    A constant input is a real outcome here, not an error: if every unit scores the
    same realised benefit in an episode, there is nothing for a signal to rank. Guarded
    so it reports that instead of emitting a warning and a bare nan.
    """
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    if a.size < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan"), float("nan")
    r = spearmanr(a, b)
    return float(r.statistic), float(r.pvalue)

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


def measure_signals(runner: EpisodeRunner, episode, warmup: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-unit (utility, drift) after `warmup` steps of full adaptation."""
    backbone = runner.backbone
    runner.reset()
    runner.apply_selector(select_all)
    runner.set_train_mode()

    def support_loss():
        return inner_loss(
            lambda b: backbone(b).embedding,
            episode.support_x, episode.support_y, episode.n_way, runner.head,
            objective=runner.objective,
        )

    if warmup > 0:
        params = [p for p in backbone.adapted.parameters() if p.requires_grad]
        opt = torch.optim.Adam(params + list(runner.head.parameters()), lr=runner.inner_lr)
        for _ in range(warmup):
            opt.zero_grad(set_to_none=True)
            support_loss().backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

    backbone.eval()
    with torch.no_grad():
        ref = backbone.forward_reference(episode.support_x).activations
        adp = backbone.forward_adapted(episode.support_x).activations
    drift = drift_signal(ref, adp, backbone.unit_names).detach().float().cpu().numpy()

    backbone.train()
    runner.set_train_mode()
    util = utility_signal(backbone, support_loss).detach().float().cpu().numpy()
    backbone.zero_grad(set_to_none=True)
    return util, drift


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="isic",
                   choices=["eurosat", "isic", "chestx", "imagefolder"])
    p.add_argument("--data-root", default=None)
    p.add_argument("--group-pattern", default=None)
    p.add_argument("--fractions", type=float, nargs=3, default=[0.1, 0.4, 0.5])
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--arch", default="resnet50")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--n-way", type=int, default=5)
    p.add_argument("--k-shot", type=int, default=5)
    p.add_argument("--n-query", type=int, default=15)
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--inner-steps", type=int, default=20)
    p.add_argument("--inner-lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=5,
                   help="adaptation steps before signals are read; 0 makes drift vacuous")
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-pretrained", action="store_true",
                   help="smoke-testing only; never for a reported run")
    p.add_argument("--grad-checkpoint", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="results/unit_benefit")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s | %(message)s")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    source = build_source(args)
    splits = make_splits(source, "group_disjoint", tuple(args.fractions), seed=args.split_seed)
    split = splits[args.split]
    log.info("probing on the %r split (%d images)", args.split, len(split.indices))

    backbone = TappedBackbone(arch=args.arch, pretrained=not args.no_pretrained)
    backbone.grad_checkpoint = args.grad_checkpoint
    runner = EpisodeRunner(
        backbone, head=ProtoHead(), device=args.device,
        inner_steps=args.inner_steps, inner_lr=args.inner_lr, amp=not args.no_amp,
    )
    sampler = EpisodicSampler(
        source, split, n_way=args.n_way, k_shot=args.k_shot, n_query=args.n_query,
        n_episodes=args.episodes, seed=args.seed + 10_000,
    )

    units = backbone.unit_names
    n = len(units)
    ones = torch.ones(n)
    rows: list[dict] = []
    started = time.perf_counter()

    for i, episode in enumerate(sampler):
        episode = episode.to(runner.device)
        util, drift = measure_signals(runner, episode, args.warmup)

        def acc(select, steps=None, ep=episode):
            # ep is bound explicitly: a closure over the loop variable would be a
            # latent bug the moment any of these calls became deferred.
            runner.inner_steps = args.inner_steps if steps is None else steps
            return runner.run_episode(
                ep, StaticController(ones.clone()), select=select
            ).metrics.accuracy

        a_none = acc(select_none, steps=0)
        a_all = acc(select_all)
        only = np.array([acc(select_units(u)) for u in units])
        loo = np.array([acc(select_units(*[v for v in units if v != u])) for u in units])

        rows.append({
            "episode": int(episode.episode_id),
            "acc_none": a_none, "acc_all": a_all,
            "utility": util.tolist(), "drift": drift.tolist(),
            "benefit_only": (only - a_none).tolist(),
            "benefit_loo": (a_all - loo).tolist(),
        })
        rate = (time.perf_counter() - started) / (i + 1)
        log.info("%d/%d  none %.3f  all %.3f  (%.0f s/ep, %.0f min left)",
                 i + 1, args.episodes, a_none, a_all, rate,
                 rate * (args.episodes - i - 1) / 60)

    # -- report --------------------------------------------------------------
    U = np.array([r["utility"] for r in rows])
    D = np.array([r["drift"] for r in rows])
    B_only = np.array([r["benefit_only"] for r in rows])
    B_loo = np.array([r["benefit_loo"] for r in rows])
    depth = np.arange(n)

    print(f"\n=== {args.dataset} {args.k_shot}-shot, {len(rows)} episodes, "
          f"{args.arch} ===")
    print(f"adapt nothing  {100*np.mean([r['acc_none'] for r in rows]):.2f}")
    print(f"adapt all      {100*np.mean([r['acc_all'] for r in rows]):.2f}\n")

    print(f"{'unit':<12}{'utility':>10}{'drift':>9}{'benefit_only':>14}{'benefit_loo':>13}")
    for j, u in enumerate(units):
        print(f"{u:<12}{U[:,j].mean():>10.3f}{D[:,j].mean():>9.3f}"
              f"{100*B_only[:,j].mean():>13.2f}%{100*B_loo[:,j].mean():>12.2f}%")

    print("\n--- does either signal predict realised benefit? (Spearman, pooled "
          "over episodes x units) ---")
    corr: dict[str, dict[str, float]] = {}
    for sname, S in (("utility", U), ("drift", D)):
        corr[sname] = {}
        for bname, B in (("benefit_only", B_only), ("benefit_loo", B_loo)):
            r, pv = rho(S, B)
            corr[sname][bname] = r
            corr[sname][bname + "_p"] = pv
            shown = f"{r:+.3f}  (p = {pv:.2e})" if np.isfinite(r) else "undefined (no variance)"
            print(f"  {sname:<8} vs {bname:<14} rho = {shown}")
        # within-episode correlation, averaged: removes any cross-episode offset
        within = [rho(S[e], B_loo[e])[0] for e in range(len(rows))]
        corr[sname]["within_episode_loo_mean"] = float(np.nanmean(within))
        print(f"  {sname:<8} vs benefit_loo, within-episode mean rho = "
              f"{np.nanmean(within):+.3f} (sd {np.nanstd(within):.3f})")

    print("\n--- is either signal just a depth index? ---")
    for sname, S in (("utility", U), ("drift", D)):
        per = [rho(S[e], depth)[0] for e in range(len(rows))]
        corr[sname]["depth_rho_mean"] = float(np.nanmean(per))
        corr[sname]["depth_rho_sd"] = float(np.nanstd(per))
        cv = np.nanmean(np.nanstd(S, axis=0) / (np.abs(np.nanmean(S, axis=0)) + 1e-9))
        corr[sname]["episode_cv"] = float(cv)
        print(f"  {sname:<8} depth rho = {np.nanmean(per):+.3f} +/- {np.nanstd(per):.3f}"
              f"   episode-to-episode CV per unit = {cv:.3f}")
    print("\nA signal with depth rho near 1 is a depth schedule, not state feedback.")
    print("A signal with CV near 0 does not vary by task, so it cannot be task-adaptive.")

    path = out_dir / f"unit_benefit_{args.dataset}_{args.k_shot}shot.json"
    path.write_text(json.dumps(
        {"config": vars(args), "units": units, "correlations": corr, "episodes": rows}, indent=2))
    print(f"\nwrote {path}", flush=True)


if __name__ == "__main__":
    main()
