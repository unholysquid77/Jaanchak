#!/usr/bin/env python
"""Instrument one inner adaptation loop, step by step.

Written to answer a specific question: the LR sweep produced accuracies identical to
four decimals across a 300x learning-rate range, and ``lr=0`` reproduced them exactly.
Something in the inner loop is not learning. This prints, per step, the support loss,
the true (unscaled) gradient norm, how far parameters actually moved, and how far the
BatchNorm running statistics moved -- which separates "the optimiser did nothing" from
"the optimiser did something but BatchNorm drowned it".

    python experiments/diagnose_inner_loop.py --k-shot 1 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arc.backbones.tapped import TappedBackbone  # noqa: E402
from arc.data.base import EpisodicSampler, make_splits  # noqa: E402
from arc.data.sources import eurosat  # noqa: E402
from arc.heads.protonet import ProtoHead  # noqa: E402
from arc.train.episodic import select_units  # noqa: E402
from arc.train.objectives import inner_loss  # noqa: E402


def bn_modules(module: torch.nn.Module):
    return [m for m in module.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]


def bn_state(module: torch.nn.Module) -> torch.Tensor:
    """Flattened running statistics, for measuring drift."""
    parts = []
    for m in bn_modules(module):
        if m.running_mean is not None:
            parts.append(m.running_mean.flatten())
            parts.append(m.running_var.flatten())
    return torch.cat(parts) if parts else torch.zeros(1)


def run(backbone, head, episode, lr, steps, amp, freeze_bn, device):
    backbone.adapted.load_state_dict(backbone._diag_pristine)
    select_units(*[n for n in backbone.unit_names if n.startswith("layer4")])(backbone)

    params = [p for p in backbone.adapted.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    backbone.train()
    if freeze_bn:
        for m in bn_modules(backbone.adapted):
            m.eval()

    before = torch.cat([p.detach().flatten() for p in params]).clone()
    bn_before = bn_state(backbone.adapted).clone()

    n_way = episode.n_way
    rows = []
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp):
            gates = torch.ones(backbone.n_units, device=device)
            loss = inner_loss(
                lambda b: backbone(b, gates=gates).embedding,
                episode.support_x, episode.support_y, n_way, head,
                objective=OBJECTIVE,
            )
        scaler.scale(loss).backward()

        # Unscale so the reported gradient norm is the true one, not the AMP-scaled one.
        scaler.unscale_(opt)
        gnorm = torch.nn.utils.clip_grad_norm_(params, float("inf")).item()

        snapshot = torch.cat([p.detach().flatten() for p in params]).clone()
        scaler.step(opt)
        scaler.update()
        moved = (torch.cat([p.detach().flatten() for p in params]) - snapshot).norm().item()

        rows.append((step, float(loss.detach()), gnorm, moved, scaler.get_scale()))

    total_moved = (torch.cat([p.detach().flatten() for p in params]) - before).norm().item()
    bn_moved = (bn_state(backbone.adapted) - bn_before).norm().item()
    return rows, total_moved, bn_moved


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default="data/eurosat")
    p.add_argument("--k-shot", type=int, nargs="+", default=[1, 5])
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--objective", default="two_view")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    global OBJECTIVE
    OBJECTIVE = args.objective
    device = torch.device(args.device)
    source = eurosat(args.data_root)
    splits = make_splits(source, mode="group_disjoint", fractions=(0.2, 0.1, 0.7), seed=0)

    backbone = TappedBackbone(arch="resnet50", pretrained=True).to(device)
    backbone._diag_pristine = {k: v.detach().clone() for k, v in backbone.adapted.state_dict().items()}
    head = ProtoHead().to(device)

    for k_shot in args.k_shot:
        sampler = EpisodicSampler(
            source, splits["val"], n_way=5, k_shot=k_shot, n_query=15,
            n_episodes=1, seed=12345,
        )
        episode = sampler.episode(0).to(device)

        for amp in (True, False):
            for freeze_bn in (False, True):
                rows, moved, bn_moved = run(
                    backbone, head, episode, args.lr, args.steps, amp, freeze_bn, device
                )
                tag = f"k={k_shot} amp={int(amp)} freeze_bn={int(freeze_bn)}"
                print(f"\n--- {tag}  lr={args.lr:g} ---")
                print(f"{'step':>4} {'loss':>12} {'grad_norm':>12} {'param_delta':>12} {'scale':>10}")
                for step, loss, gnorm, mv, scale in rows:
                    print(f"{step:>4} {loss:>12.3e} {gnorm:>12.3e} {mv:>12.3e} {scale:>10.0f}")
                print(f"  total param drift : {moved:.4e}")
                print(f"  BN running-stat drift: {bn_moved:.4e}")


if __name__ == "__main__":
    main()
