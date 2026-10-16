#!/usr/bin/env python3
"""Can the adaptation be precomputed offline and applied with no gradient steps?

The probes in docs/RESULTS_SUBSPACE.md found that ~90% of fine-tuning's
representational displacement is a per-channel affine, that it is near rank-1 across
200 tasks at the top of the network, and that no per-unit signal predicts within an
episode which unit is worth adapting. Together those say few-shot adaptation on a
domain-shifted target may be mostly a *task-independent domain correction* rather than
task-specific learning.

If that is right, the correction can be estimated ONCE from the train split and applied
at test time with zero optimisation. This tests exactly that.

**Extract** (train split only). For each of N episodes: adapt with full fine-tuning,
then compare adapted and pretrained activations of a probe batch, per unit, per
channel. Fit the fixed affine that best maps pretrained to adapted,

    gamma = s_adp / s_ref                  (per-channel scale)
    beta  = m_adp - m_ref * gamma          (per-channel shift)

and average gamma and beta over episodes. Note this is a *batch-statistic-free* form:
unlike the decomposition in diagnose_subspace.py, which used the probe batch's own
m_ref to measure energy, gamma and beta here are constants that can be applied to any
future input. That is the whole point -- a correction that needed the test batch's
statistics would be transductive, not precomputed.

**Evaluate** (val split). Run the *frozen* backbone, applying the fixed affine after
each unit, and classify with the prototype head. No optimiser is constructed; no
parameter of the network is touched. The comparison points, all ISIC 5-shot val:
frozen 38.2, bn_only_ft 49.4 (0.22% of parameters, 20 gradient steps per episode),
full_ft 49.6 (100%, 20 steps).

**Variants.** `shift` applies h -> h + (m_adp - m_ref); `scale` applies h -> h * gamma;
`both` applies h -> h * gamma + beta. `identity` is a control that must reproduce
frozen_proto exactly -- if it does not, the corrected forward path is wrong and every
other number here is meaningless.

**Unit subsets** test a specific open question. E1 found bn_shift_ft losing to
bn_scale_ft (48.32 vs 48.79) even though shift carries ~85% of the displacement
energy, so energy share does not translate into accuracy share. The candidate
explanation is that the prototype head's cosine metric normalises embeddings and so
divides out a large common-mode shift at the top -- precisely the component measured as
dominant. If that holds, `--units layer4` should buy very little while
`--units no_layer4` should buy most of it. If instead layer4 carries the gain, the
explanation is wrong and the energy/accuracy mismatch needs another one.

Extraction uses `--n-query 3` by default because the train split is thin: at
0.1/0.4/0.5 its smallest class (VASC) holds 11 images, so 5-shot with 15 queries cannot
be sampled there at all.

    python experiments/precomputed_shift.py --dataset isic --extract-episodes 100 \
        --eval-episodes 200
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
from arc.data.base import EpisodicSampler, make_splits  # noqa: E402
from arc.data.medical import chestx, isic  # noqa: E402
from arc.data.sources import ImageFolderSource, build_transform, eurosat  # noqa: E402
from arc.eval.metrics import episode_metrics, mean_ci95  # noqa: E402
from arc.heads.protonet import ProtoHead  # noqa: E402
from arc.train.baselines import BASELINES  # noqa: E402
from arc.train.episodic import EpisodeRunner  # noqa: E402

log = logging.getLogger("arc.precomputed_shift")

DEFAULT_ROOTS = {"eurosat": "data/eurosat", "isic": "data/isic2018", "chestx": "data/chestx"}
EPS = 1e-6


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


def channel_stats(activation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-channel mean and std over samples and space, as (C,) doubles."""
    a = activation.double()
    if a.ndim == 4:
        a = a.permute(1, 0, 2, 3).reshape(a.shape[1], -1)
    else:
        a = a.T
    return a.mean(dim=1), a.std(dim=1, unbiased=False)


def apply_correction(
    backbone: TappedBackbone,
    x: torch.Tensor,
    gamma: dict[str, torch.Tensor] | None,
    beta: dict[str, torch.Tensor] | None,
) -> torch.Tensor:
    """Frozen forward with a fixed per-channel affine after each unit.

    Written here rather than in TappedBackbone because nothing is being trained and
    nothing is being gated: this is the pretrained network plus a constant, which is a
    different object from the gated interpolation the backbone exists to provide.
    """
    h = x
    for name in backbone.unit_names:
        h = backbone.frozen[name](h)
        g = None if gamma is None else gamma.get(name)
        b = None if beta is None else beta.get(name)
        if g is not None:
            h = h * g.to(h.dtype).view(1, -1, *([1] * (h.ndim - 2)))
        if b is not None:
            h = h + b.to(h.dtype).view(1, -1, *([1] * (h.ndim - 2)))
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
    p.add_argument("--n-query", type=int, default=3,
                   help="for EXTRACTION on the thin train split")
    p.add_argument("--eval-n-query", type=int, default=15,
                   help="for evaluation on val; 15 matches every other reported number")
    p.add_argument("--extract-episodes", type=int, default=100)
    p.add_argument("--eval-episodes", type=int, default=200)
    p.add_argument("--extract-baseline", default="full_ft", choices=sorted(BASELINES))
    p.add_argument("--inner-steps", type=int, default=20)
    p.add_argument("--inner-lr", type=float, default=1e-4)
    p.add_argument("--variants", nargs="+",
                   default=["identity", "shift", "scale", "both"])
    p.add_argument("--unit-subsets", nargs="+",
                   default=["all", "layer4", "no_layer4", "stem_layer1"])
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-pretrained", action="store_true", help="smoke-testing only")
    p.add_argument("--grad-checkpoint", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="results/precomputed")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s | %(message)s")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    source = build_source(args)
    splits = make_splits(source, "group_disjoint", tuple(args.fractions), seed=args.split_seed)
    log.info("extract on train (%d images), evaluate on val (%d images)",
             len(splits["train"].indices), len(splits["val"].indices))

    backbone = TappedBackbone(arch=args.arch, pretrained=not args.no_pretrained)
    backbone.grad_checkpoint = args.grad_checkpoint
    head = ProtoHead().to(args.device)
    runner = EpisodeRunner(
        backbone, head=ProtoHead(), device=args.device,
        inner_steps=args.inner_steps, inner_lr=args.inner_lr, amp=not args.no_amp,
    )
    units = backbone.unit_names

    # -- extraction, train split only ---------------------------------------
    spec = BASELINES[args.extract_baseline]
    controller, select, steps = spec.build(backbone, args.inner_steps)
    runner.inner_steps = steps
    train_sampler = EpisodicSampler(
        source, splits["train"], n_way=args.n_way, k_shot=args.k_shot,
        n_query=args.n_query, n_episodes=args.extract_episodes, seed=args.seed + 20_000,
    )

    acc_g: dict[str, list] = {u: [] for u in units}
    acc_b: dict[str, list] = {u: [] for u in units}
    acc_s: dict[str, list] = {u: [] for u in units}
    started = time.perf_counter()
    for i, episode in enumerate(train_sampler):
        episode = episode.to(runner.device)
        probe = torch.cat([episode.support_x, episode.query_x])

        backbone.eval()
        with torch.no_grad():
            ref = backbone.forward_reference(probe).activations
            ref = {k: channel_stats(v.float().cpu()) for k, v in ref.items()}
        runner.run_episode(episode, controller, select=select)
        backbone.eval()
        with torch.no_grad():
            adp = backbone.forward_adapted(probe).activations
            adp = {k: channel_stats(v.float().cpu()) for k, v in adp.items()}

        for u in units:
            if u not in ref or u not in adp:
                continue
            m_ref, s_ref = ref[u]
            m_adp, s_adp = adp[u]
            g = s_adp / s_ref.clamp_min(EPS)
            acc_g[u].append(g)
            acc_b[u].append(m_adp - m_ref * g)      # pairs with gamma, for 'both'
            acc_s[u].append(m_adp - m_ref)          # shift alone, gamma = 1
        if (i + 1) % 25 == 0:
            rate = (time.perf_counter() - started) / (i + 1)
            log.info("extract %d/%d (%.1f s/ep, %.1f min left)", i + 1,
                     args.extract_episodes, rate,
                     rate * (args.extract_episodes - i - 1) / 60)

    gamma = {u: torch.stack(v).mean(0) for u, v in acc_g.items() if v}
    beta_both = {u: torch.stack(v).mean(0) for u, v in acc_b.items() if v}
    beta_shift = {u: torch.stack(v).mean(0) for u, v in acc_s.items() if v}
    log.info("extracted corrections for %d units", len(gamma))

    # -- evaluation, val split, zero gradient steps -------------------------
    val_sampler = EpisodicSampler(
        source, splits["val"], n_way=args.n_way, k_shot=args.k_shot,
        n_query=args.eval_n_query, n_episodes=args.eval_episodes, seed=args.seed + 10_000,
    )
    episodes = [e.to(runner.device) for e in val_sampler] if args.eval_episodes <= 50 else None

    def subset(d: dict[str, torch.Tensor] | None, which: str):
        if d is None:
            return None
        if which == "all":
            keep = units
        elif which == "layer4":
            keep = [u for u in units if u.startswith("layer4")]
        elif which == "no_layer4":
            keep = [u for u in units if not u.startswith("layer4")]
        elif which == "stem_layer1":
            keep = [u for u in units if u == "stem" or u.startswith("layer1")]
        else:
            raise SystemExit(f"unknown unit subset {which!r}")
        return {u: v for u, v in d.items() if u in keep}

    # Control, computed rather than quoted: run the real frozen_proto baseline through
    # EpisodeRunner on the same episodes. The 'identity' variant goes through the
    # corrected forward path with no correction, so the two must agree to within
    # floating point. If they do not, apply_correction is wrong -- different pooling,
    # a missed unit, a transposed channel axis -- and every number below is noise
    # dressed as a result. A hardcoded reference could not catch that on any corpus
    # other than the one it was written for.
    fc, fsel, _ = BASELINES["frozen_proto"].build(backbone, 0)
    runner.inner_steps = 0
    frozen_accs = []
    for episode in val_sampler:
        ep = episode.to(runner.device)
        frozen_accs.append(runner.run_episode(ep, fc, select=fsel).metrics.accuracy)
    frozen_ref, frozen_ci = mean_ci95(frozen_accs)
    log.info("control: frozen_proto baseline %.2f +/- %.2f", 100 * frozen_ref, 100 * frozen_ci)

    backbone.eval()
    rows = []
    for variant in args.variants:
        for which in (["all"] if variant == "identity" else args.unit_subsets):
            g = {"identity": None, "shift": None, "scale": gamma, "both": gamma}[variant]
            b = {"identity": None, "shift": beta_shift, "scale": None,
                 "both": beta_both}[variant]
            g, b = subset(g, which), subset(b, which)

            accs = []
            src = episodes if episodes is not None else val_sampler
            for episode in src:
                ep = episode if episodes is not None else episode.to(runner.device)
                with torch.no_grad():
                    se = apply_correction(backbone, ep.support_x, g, b)
                    qe = apply_correction(backbone, ep.query_x, g, b)
                    logits = head(se, ep.support_y, qe, ep.n_way)
                accs.append(episode_metrics(logits.float(), ep.query_y, ep.n_way).accuracy)
            mean, ci = mean_ci95(accs)
            rows.append({"variant": variant, "units": which,
                         "accuracy": mean, "ci95": ci, "n": len(accs)})
            log.info("%-9s %-12s acc %.2f +/- %.2f", variant, which, 100 * mean, 100 * ci)

    # -- report --------------------------------------------------------------
    print(f"\n=== precomputed correction, {args.dataset} {args.k_shot}-shot, "
          f"val, ZERO gradient steps ===")
    print(f"extracted from {args.extract_episodes} train episodes of "
          f"{args.extract_baseline}; evaluated on {args.eval_episodes} val episodes\n")
    print(f"{'variant':<10}{'units':<14}{'accuracy':>12}")
    for r in rows:
        print(f"{r['variant']:<10}{r['units']:<14}{100*r['accuracy']:>8.2f} "
              f"+/-{100*r['ci95']:>5.2f}")
    print("\nreference points (ISIC 5-shot val, 20 gradient steps per episode):")
    print("  frozen_proto   38.0    0.00% params, 0 steps")
    print("  bn_only_ft     49.4    0.23% params, 20 steps")
    print("  full_ft        49.6  100.00% params, 20 steps")
    ident = next((r for r in rows if r["variant"] == "identity"), None)
    if ident is not None:
        delta = 100 * abs(ident["accuracy"] - frozen_ref)
        verdict = "PASS" if delta < 0.5 else "FAIL -- corrected forward path is wrong"
        print(f"\ncontrol: identity {100*ident['accuracy']:.2f} vs frozen_proto baseline "
              f"{100*frozen_ref:.2f} on the same episodes, delta {delta:.2f}  [{verdict}]")
        if delta >= 0.5:
            print("Nothing above this line is interpretable. Fix apply_correction first.")

    path = out_dir / f"precomputed_{args.dataset}_{args.k_shot}shot.json"
    path.write_text(json.dumps(
        {"config": vars(args), "frozen_control": frozen_ref, "results": rows}, indent=2))
    print(f"\nwrote {path}", flush=True)


if __name__ == "__main__":
    main()
