#!/usr/bin/env python3
"""Is few-shot adaptation low-dimensional in activation space, and is it diagonal?

Motivation. `bn_only_ft` adapts 0.22% of ResNet-50 and ties full fine-tuning on ISIC
(49.44 vs 49.55). BatchNorm affine is a per-channel scale and shift -- a *diagonal*
affine map on each unit's activations -- so BatchNorm-only adaptation is already a
restricted-subspace method, and its strength suggests the binding constraint in
few-shot adaptation is the dimensionality of the update rather than its location.
That also explains the Level-1 null: the gate modulated the magnitude of a 23M-
parameter update estimated from 5-25 images, when the problem was the 23M.

This measures the premise directly, using the frozen reference twin that
`backbones/tapped.py` already maintains. For each episode: adapt the backbone, then
compare the adapted and pretrained activations of the *same* probe batch, per unit.

Two questions, two outputs.

**1. Is the displacement diagonal?** Write the per-unit map h_ref -> h_adp as the
best per-channel affine plus a residual:

    A(h) = (h - m_ref) * (s_adp / s_ref) + m_adp
    A(h_ref) - h_ref = (m_adp - m_ref)            <- shift, constant over samples
                     + (h_ref - m_ref)*(s_adp/s_ref - 1)   <- scale, centred
    R = h_adp - A(h_ref)                          <- residual, what a diagonal
                                                     affine cannot express

The shift and scale terms are orthogonal in the sample average (one is constant, the
other centred), so the energy split is meaningful. `shift` and `scale` together are
exactly what BatchNorm affine can represent. A small residual means BatchNorm-only is
near-optimal *in principle* and the ceiling of any diagonal method is already reached
-- an important negative. A large residual means a non-diagonal basis has headroom.

**2. Is the subspace shared across episodes?** Stack each unit's per-episode
displacement vectors into an (episodes x channels) matrix and take its singular
spectrum. Fast decay means episodes adapt along a *common* low-dimensional subspace,
which can therefore be extracted once, offline, and adapted within at test time using
only its coefficients. Flat means every episode moves somewhere new and no
pre-computed basis can help.

Effective rank is reported as the participation ratio (sum s^2)^2 / sum s^4, which is
smooth and does not need a threshold, alongside the energy in the top 1/2/4/8
components.

Measurement notes. The probe batch is the episode's *query* set, under `no_grad` --
larger than support (75 vs 25 images at 5-way 5-shot) and never touched by the
optimiser, so this reads the displacement that generalises rather than the one driven
directly. Activations are spatially average-pooled to (samples, channels) before any
of this, matching what `signals/drift.py` does for CKA, so "channel" is the unit of
analysis throughout. Nothing here trains anything or writes to the reported results.

    python experiments/diagnose_subspace.py --dataset isic --k-shot 5 --episodes 200
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arc.backbones.tapped import TappedBackbone  # noqa: E402
from arc.data.base import EpisodicSampler, make_splits  # noqa: E402
from arc.data.medical import chestx, isic  # noqa: E402
from arc.data.sources import ImageFolderSource, build_transform, eurosat  # noqa: E402
from arc.heads.protonet import ProtoHead  # noqa: E402
from arc.train.baselines import BASELINES  # noqa: E402
from arc.train.episodic import EpisodeRunner  # noqa: E402

log = logging.getLogger("arc.diagnose_subspace")

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


def _pool(activation: torch.Tensor) -> torch.Tensor:
    """(N,C,H,W) -> (N,C). Same reduction signals/drift.py uses, so the unit of
    analysis is the channel in both places."""
    if activation.ndim == 4:
        return activation.mean(dim=(2, 3))
    return activation.flatten(1)


def decompose(ref: torch.Tensor, adp: torch.Tensor, eps: float = 1e-8) -> dict[str, float]:
    """Energy split of the displacement into shift / scale / non-diagonal residual."""
    ref, adp = _pool(ref).double(), _pool(adp).double()
    m_ref, m_adp = ref.mean(0, keepdim=True), adp.mean(0, keepdim=True)
    s_ref = ref.std(0, keepdim=True, unbiased=False).clamp_min(eps)
    s_adp = adp.std(0, keepdim=True, unbiased=False)

    delta = adp - ref
    shift = (m_adp - m_ref).expand_as(delta)
    scale = (ref - m_ref) * (s_adp / s_ref - 1.0)
    approx = ref + shift + scale               # == the best per-channel affine of ref
    resid = adp - approx

    sq = lambda t: float(t.pow(2).sum())       # noqa: E731
    total = sq(delta)
    if total < eps:
        return {"total": 0.0, "shift": 0.0, "scale": 0.0, "residual": 0.0, "norm": 0.0}
    return {
        "total": 1.0,
        "shift": sq(shift) / total,
        "scale": sq(scale) / total,
        "residual": sq(resid) / total,
        "norm": float(delta.pow(2).sum().sqrt()),
    }


def spectrum(matrix: np.ndarray) -> dict[str, float]:
    """Singular spectrum of an (episodes x channels) displacement matrix."""
    if matrix.shape[0] < 2:
        return {}
    s = np.linalg.svd(matrix, compute_uv=False)
    e = s**2
    if e.sum() <= 0:
        return {}
    frac = e / e.sum()
    # Participation ratio: smooth effective rank, no threshold to choose.
    pr = float(e.sum() ** 2 / (e**2).sum())
    out = {"effective_rank": pr, "n_components": len(s)}
    for k in (1, 2, 4, 8):
        if k <= len(frac):
            out[f"energy_top{k}"] = float(frac[:k].sum())
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="isic",
                   choices=["eurosat", "isic", "chestx", "imagefolder"])
    p.add_argument("--data-root", default=None)
    p.add_argument("--group-pattern", default=None)
    p.add_argument("--fractions", type=float, nargs=3, default=[0.1, 0.4, 0.5])
    p.add_argument("--split", default="val", choices=["train", "val", "test"],
                   help="val by default: this is a diagnostic, test stays unread")
    p.add_argument("--arch", default="resnet50")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--n-way", type=int, default=5)
    p.add_argument("--k-shot", type=int, default=5)
    p.add_argument("--n-query", type=int, default=15)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--baseline", default="full_ft", choices=sorted(BASELINES),
                   help="which adaptation to measure the displacement of")
    p.add_argument("--inner-steps", type=int, default=20)
    p.add_argument("--inner-lr", type=float, default=1e-4)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-pretrained", action="store_true",
                   help="smoke-testing only; never for a reported run")
    p.add_argument("--grad-checkpoint", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="results/subspace")
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
    spec = BASELINES[args.baseline]
    controller, select, steps = spec.build(backbone, args.inner_steps)
    runner.inner_steps = steps

    units = backbone.unit_names
    energy: dict[str, list[dict[str, float]]] = {u: [] for u in units}
    shifts: dict[str, list[np.ndarray]] = {u: [] for u in units}
    accs: list[float] = []
    started = time.perf_counter()

    for i, episode in enumerate(sampler):
        episode = episode.to(runner.device)

        # Pretrained reference on the probe batch, before anything is adapted.
        backbone.eval()
        with torch.no_grad():
            ref = {k: v.float().cpu() for k, v in
                   backbone.forward_reference(episode.query_x).activations.items()}

        result = runner.run_episode(episode, controller, select=select)
        accs.append(result.metrics.accuracy)

        # Adapted activations on the same probe batch. forward_adapted, not forward():
        # the gate is all-ones for every baseline here, and this keeps the comparison
        # to the pure pretrained branch rather than to a fused stream.
        backbone.eval()
        with torch.no_grad():
            adp = {k: v.float().cpu() for k, v in
                   backbone.forward_adapted(episode.query_x).activations.items()}

        for u in units:
            if u not in ref or u not in adp:
                continue
            energy[u].append(decompose(ref[u], adp[u]))
            d = (_pool(adp[u].double()) - _pool(ref[u].double())).mean(0)
            shifts[u].append(d.numpy())

        if (i + 1) % 25 == 0:
            rate = (time.perf_counter() - started) / (i + 1)
            log.info("%d/%d episodes  (%.2f s/ep, %.1f min left)",
                     i + 1, args.episodes, rate, rate * (args.episodes - i - 1) / 60)

    # -- report --------------------------------------------------------------
    print(f"\n=== displacement of {args.baseline} on {args.dataset}, "
          f"{args.k_shot}-shot, {args.episodes} episodes ===")
    print(f"query accuracy {100*np.mean(accs):.2f} +/- {100*1.96*np.std(accs, ddof=1)/np.sqrt(len(accs)):.2f}\n")

    print("Can a per-channel diagonal affine express the displacement?")
    print("(fractions of squared displacement; shift+scale = what BatchNorm affine can do)\n")
    print(f"{'unit':<12}{'shift':>9}{'scale':>9}{'residual':>10}{'||delta||':>12}")
    summary: dict[str, dict] = {}
    for u in units:
        if not energy[u]:
            continue
        m = {k: float(np.mean([e[k] for e in energy[u]])) for k in ("shift", "scale", "residual", "norm")}
        summary[u] = {"energy": m}
        print(f"{u:<12}{m['shift']:>9.3f}{m['scale']:>9.3f}{m['residual']:>10.3f}{m['norm']:>12.3f}")
    diag = np.mean([summary[u]["energy"]["shift"] + summary[u]["energy"]["scale"] for u in summary])
    print(f"\n  mean diagonal-expressible fraction across units: {diag:.3f}")
    print(f"  mean non-diagonal residual:                      {1-diag:.3f}")

    print("\nIs the displacement subspace shared across episodes?")
    print("(SVD of the episodes x channels mean-shift matrix, per unit)\n")
    # ncomp is min(episodes, channels): the SVD of an (E x C) matrix has at most
    # min(E, C) singular values, so with 200 episodes the deeper units (C up to 2048)
    # are episode-limited rather than channel-limited. Effective rank is bounded by it.
    print(f"{'unit':<12}{'ncomp':>7}{'eff.rank':>10}{'top1':>8}{'top2':>8}{'top4':>8}{'top8':>8}")
    for u in units:
        if len(shifts[u]) < 2:
            continue
        sp = spectrum(np.stack(shifts[u]))
        if not sp:
            continue
        summary.setdefault(u, {})["spectrum"] = sp
        cell = lambda k: (f"{sp[k]:>8.3f}" if k in sp else f"{'-':>8}")  # noqa: E731
        print(f"{u:<12}{sp['n_components']:>7}{sp['effective_rank']:>10.2f}"
              f"{cell('energy_top1')}{cell('energy_top2')}{cell('energy_top4')}{cell('energy_top8')}")

    ranks = [summary[u]["spectrum"]["effective_rank"] for u in summary if "spectrum" in summary[u]]
    top8 = [summary[u]["spectrum"]["energy_top8"] for u in summary
            if "spectrum" in summary[u] and "energy_top8" in summary[u]["spectrum"]]
    print(f"\n  mean effective rank across units: {np.mean(ranks):.2f} "
          f"(of {args.episodes} episodes x channels)")
    if top8:
        print(f"  mean energy in top 8 components:  {np.mean(top8):.3f}")
    print("\nRead: a high diagonal fraction means BatchNorm-only is near the ceiling of")
    print("any diagonal method. A low effective rank means a basis extracted offline")
    print("can carry adaptation, so only its coefficients need fitting per episode.")

    payload = {
        "config": vars(args),
        "accuracy": float(np.mean(accs)),
        "mean_diagonal_fraction": float(diag),
        "mean_effective_rank": float(np.mean(ranks)) if ranks else None,
        "units": summary,
    }
    path = out_dir / f"subspace_{args.dataset}_{args.baseline}_{args.k_shot}shot.json"
    path.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {path}", flush=True)


if __name__ == "__main__":
    main()
