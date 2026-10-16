#!/usr/bin/env python3
"""Diagonal versus non-diagonal activation bases at matched coefficient budgets.

This is the comparison a Consensus review of the literature names as the field's most
direct missing ablation: *does a non-diagonal activation basis learned from measured
fine-tuning displacements outperform BatchNorm-affine or SSF under matched coefficient
budgets?* Nothing in that corpus appears to run it.

Every method here adapts a **frozen** ResNet-50 by inserting a correction after each of
the 17 units, and differs only in what that correction is allowed to be and how many
trainable coefficients it has. The backbone never moves. Identical episodes throughout,
so the accuracy-versus-budget curve is a controlled comparison rather than a table of
numbers from different papers.

    variant        correction after unit l               trainable / unit
    identity       none                                  0        (control)
    shift          h + beta_l        (precomputed)       0
    fitted         h + a_l * beta_l  (a_l scalar)        1
    rank{k}        h + sum_j a_j V_l[j]                  k
    ssf            h * gamma_l + beta_l                  2 * C_l

`shift` and `rank{k}` use a basis extracted offline from TRAIN-split displacements:
`beta_l` is the mean per-channel displacement, `V_l` its top-k right singular vectors.
`ssf` is the diagonal reference: per-channel scale and shift, learned per episode,
initialised to the identity. **It is SSF-style, not SSF.** Lian et al. (2022) apply
scale-and-shift inside ViT blocks and merge the parameters into the weights for
inference; this inserts the same per-channel affine after each of a ResNet-50's 17
units, on a frozen backbone, and never merges. The adaptation *form* is theirs; the
placement, architecture and protocol are not, so their published numbers are not
comparable to these and must not be quoted alongside them. `fitted` is the
one-parameter degenerate case of rank-k along the mean direction.

A note on backbone, since it decides who this is comparable to. Four of the five
nearest methods -- DIPA, SSF, LN-Tune and BOLT -- are vision transformers; only
Yazdanpanah et al. (2022) is a CNN, because BatchNorm affines only exist in one. The
diagonal-special-case claim is intrinsically a normalisation-layer claim, so a BN
network is the right place to make it, and the ViT analogue is LayerNorm-only tuning
(LN-Tune). That analogue is untested here and is the obvious extension.

The point of the budget axis: `ssf` has 2*C_l coefficients per unit, which is 30,336
in total for ResNet-50, while rank-8 has 8 per unit, 136 in total. If a non-diagonal
basis at k=8 matches `ssf`, the diagonal form is an unnecessary restriction costing
200x the coefficients. If it does not, BatchNorm-style diagonal adaptation is doing
something a low-rank empirical basis cannot, which is the more interesting answer and
the one the plateau in docs/RESULTS_SUBSPACE.md section 3 predicts.

**A warning the basis construction cannot fix.** `results/precomputed` showed that the
highest-energy component of the displacement contributes the *least* accuracy: the
top-layer shift carries most of the norm but buys +1.47 against +7.44 for everything
below it, consistent with the cosine prototype head normalising a common-mode shift
away. An SVD orders directions by energy, so rank-k here is ordered by precisely the
quantity shown not to predict accuracy. That is deliberate -- it is what BOLT-style
spectral basis selection does (Park et al., 2025) -- and if rank-k underperforms, that
mechanism is the explanation rather than an excuse. `--exclude-layer4` tests it
directly.

    python experiments/subspace_frontier.py --dataset isic --ranks 1 2 4 8 16
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
from arc.train.objectives import inner_loss  # noqa: E402

log = logging.getLogger("arc.subspace_frontier")
DEFAULT_ROOTS = {"eurosat": "data/eurosat", "isic": "data/isic2018", "chestx": "data/chestx"}
EPS = 1e-8


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


def pooled_mean(activation: torch.Tensor) -> torch.Tensor:
    """(N,C,H,W) -> (C,) mean over samples and space, in float64."""
    a = activation.double()
    if a.ndim == 4:
        return a.mean(dim=(0, 2, 3))
    return a.mean(dim=0)


class Correction(torch.nn.Module):
    """Per-unit correction inserted after a frozen unit's output.

    Holds only the trainable coefficients; any basis is a non-trainable buffer, so
    `n_trainable` is the number this method is actually charged for on the budget axis.
    """

    def __init__(self, kind: str, units: list[str], channels: dict[str, int],
                 basis: dict[str, torch.Tensor] | None = None,
                 mean_dir: dict[str, torch.Tensor] | None = None,
                 rank: int = 0, device="cpu") -> None:
        super().__init__()
        self.kind, self.units, self.rank = kind, units, rank
        self.coef = torch.nn.ParameterDict()
        self._buf: dict[str, torch.Tensor] = {}
        for u in units:
            c = channels[u]
            if kind == "ssf":
                self.coef[u + "_g"] = torch.nn.Parameter(torch.ones(c, device=device))
                self.coef[u + "_b"] = torch.nn.Parameter(torch.zeros(c, device=device))
            elif kind == "fitted":
                # Initialised at 1.0, not 0: the correction then STARTS at the
                # precomputed `shift` and learns to rescale it, so `fitted` is a strict
                # refinement of `shift` rather than a search that has to rediscover it.
                self.coef[u] = torch.nn.Parameter(torch.ones(1, device=device))
                self._buf[u] = mean_dir[u].to(device=device, dtype=torch.float32)
            elif kind == "rank":
                # Basis vectors are unit-norm from the SVD while real displacements have
                # norms of 11-35, so zero-initialised coefficients at a shared learning
                # rate cannot reach the right magnitude in 20 steps -- the correction
                # stays a no-op and the variant silently reports the frozen baseline.
                # Fix both ends: scale each basis vector to the unit's typical
                # displacement norm so coefficients are O(1), and warm-start them at the
                # projection of the mean displacement onto the basis.
                v = basis[u][:rank].to(device=device, dtype=torch.float32)
                scale = mean_dir[u].to(torch.float32).norm().clamp_min(1e-6)
                init = (v @ mean_dir[u].to(device=device, dtype=torch.float32)) / scale
                self.coef[u] = torch.nn.Parameter(init.clone())
                self._buf[u] = v * scale

    @property
    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def apply_to(self, h: torch.Tensor, unit: str) -> torch.Tensor:
        shape = (1, -1, *([1] * (h.ndim - 2)))
        if self.kind == "ssf":
            return h * self.coef[unit + "_g"].view(shape) + self.coef[unit + "_b"].view(shape)
        if self.kind == "fitted":
            return h + self.coef[unit] * self._buf[unit].view(shape)
        if self.kind == "rank":
            delta = (self.coef[unit].unsqueeze(1) * self._buf[unit]).sum(0)
            return h + delta.view(shape)
        return h


def embed(backbone: TappedBackbone, x: torch.Tensor,
          corr: Correction | None, fixed: dict[str, torch.Tensor] | None) -> torch.Tensor:
    """Frozen forward with an optional trainable correction and/or a fixed shift."""
    h = x
    for u in backbone.unit_names:
        h = backbone.frozen[u](h)
        if fixed is not None and u in fixed:
            h = h + fixed[u].to(device=h.device, dtype=h.dtype).view(
                1, -1, *([1] * (h.ndim - 2)))
        if corr is not None and u in corr.units:
            h = corr.apply_to(h, u)
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
    p.add_argument("--n-query", type=int, default=3, help="extraction, on the thin train split")
    p.add_argument("--eval-n-query", type=int, default=15)
    p.add_argument("--extract-episodes", type=int, default=100)
    p.add_argument("--eval-episodes", type=int, default=200)
    p.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    p.add_argument("--variants", nargs="+",
                   default=["identity", "shift", "fitted", "rank", "ssf"])
    p.add_argument("--exclude-layer4", action="store_true",
                   help="drop layer4 corrections; the top-layer shift was measured "
                        "actively harmful (42.27 with it, 45.47 without)")
    p.add_argument("--inner-steps", type=int, default=20)
    p.add_argument("--inner-lr", type=float, default=1e-4)
    p.add_argument("--fit-steps", type=int, default=20)
    p.add_argument("--fit-lr", type=float, default=0.05)
    p.add_argument("--fit-lrs", type=float, nargs="+", default=None,
                   help="sweep the fit learning rate; a correction that cannot reach "
                        "the right magnitude reports the frozen baseline and looks "
                        "like a null, so this is tuned rather than assumed")
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-pretrained", action="store_true", help="smoke-testing only")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="results/frontier")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s | %(message)s")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    source = build_source(args)
    splits = make_splits(source, "group_disjoint", tuple(args.fractions), seed=args.split_seed)
    backbone = TappedBackbone(arch=args.arch, pretrained=not args.no_pretrained)
    head = ProtoHead().to(args.device)
    runner = EpisodeRunner(backbone, head=ProtoHead(), device=args.device,
                           inner_steps=args.inner_steps, inner_lr=args.inner_lr,
                           amp=not args.no_amp)

    units = [u for u in backbone.unit_names
             if not (args.exclude_layer4 and u.startswith("layer4"))]
    log.info("correcting %d of %d units%s", len(units), backbone.n_units,
             " (layer4 excluded)" if args.exclude_layer4 else "")

    # -- extraction: train split only ---------------------------------------
    spec = BASELINES["full_ft"]
    controller, select, steps = spec.build(backbone, args.inner_steps)
    runner.inner_steps = steps
    train_sampler = EpisodicSampler(
        source, splits["train"], n_way=args.n_way, k_shot=args.k_shot,
        n_query=args.n_query, n_episodes=args.extract_episodes, seed=args.seed + 20_000)

    disp: dict[str, list[torch.Tensor]] = {u: [] for u in units}
    started = time.perf_counter()
    for i, episode in enumerate(train_sampler):
        episode = episode.to(runner.device)
        probe = torch.cat([episode.support_x, episode.query_x])
        backbone.eval()
        with torch.no_grad():
            ref = {k: pooled_mean(v.float().cpu())
                   for k, v in backbone.forward_reference(probe).activations.items()}
        runner.run_episode(episode, controller, select=select)
        backbone.eval()
        with torch.no_grad():
            adp = {k: pooled_mean(v.float().cpu())
                   for k, v in backbone.forward_adapted(probe).activations.items()}
        for u in units:
            if u in ref and u in adp:
                disp[u].append(adp[u] - ref[u])
        if (i + 1) % 25 == 0:
            rate = (time.perf_counter() - started) / (i + 1)
            log.info("extract %d/%d (%.1f s/ep, %.1f min left)", i + 1,
                     args.extract_episodes, rate,
                     rate * (args.extract_episodes - i - 1) / 60)

    mean_dir = {u: torch.stack(v).mean(0) for u, v in disp.items() if v}
    basis: dict[str, torch.Tensor] = {}
    spectra: dict[str, list[float]] = {}
    for u, v in disp.items():
        if not v:
            continue
        m = torch.stack(v)                               # (episodes, channels)
        # Unit-norm rows so the SVD is not dominated by a few large-displacement
        # episodes, then take right singular vectors as the activation-space basis.
        m = m / m.norm(dim=1, keepdim=True).clamp_min(EPS)
        _, s, vh = torch.linalg.svd(m, full_matrices=False)
        basis[u] = vh
        spectra[u] = (s[:8] ** 2 / (s**2).sum()).tolist()
    channels = {u: mean_dir[u].numel() for u in mean_dir}
    log.info("extracted basis for %d units", len(basis))

    # Log the magnitude of the extracted shift per unit. This exists because the first
    # frontier run reported `shift` at 38.55 while experiments/precomputed_shift.py
    # reported 45.21 for what should be the identical quantity -- the same mean
    # displacement over the same train episodes, applied to the same frozen units --
    # with `identity` agreeing exactly at 38.04 in both. The per-unit statistics and
    # the application arithmetic were checked and are bit-identical between the two
    # scripts, so the divergence is in the extracted VECTOR, and a near-zero shift
    # would explain a +0.51 gain where the other script sees +4.30. Without this line
    # the two scripts disagree by 6.7 points and neither can be audited.
    norms = {u: float(mean_dir[u].norm()) for u in mean_dir}
    log.info("mean |shift| per unit: %s",
             " ".join(f"{u}={norms[u]:.3f}" for u in backbone.unit_names if u in norms))
    log.info("total |shift| across units: %.3f", sum(norms.values()))

    # -- evaluation: val split, identical episodes across variants ----------
    val_sampler = EpisodicSampler(
        source, splits["val"], n_way=args.n_way, k_shot=args.k_shot,
        n_query=args.eval_n_query, n_episodes=args.eval_episodes, seed=args.seed + 10_000)

    fc, fsel, _ = BASELINES["frozen_proto"].build(backbone, 0)
    runner.inner_steps = 0
    frozen_ref, _ = mean_ci95([
        runner.run_episode(e.to(runner.device), fc, select=fsel).metrics.accuracy
        for e in val_sampler])
    log.info("control: frozen_proto %.2f", 100 * frozen_ref)

    backbone.eval()
    backbone.requires_grad_(False)

    jobs = []
    for v in args.variants:
        if v == "rank":
            jobs += [("rank", k) for k in args.ranks]
        else:
            jobs.append((v, 0))

    fit_lrs = args.fit_lrs or [args.fit_lr]
    rows = []
    for kind, rank in jobs:
      for fit_lr in (fit_lrs if kind in ("ssf", "fitted", "rank") else [0.0]):
        accs, n_train = [], 0
        for episode in val_sampler:
            ep = episode.to(runner.device)
            corr = None
            if kind in ("ssf", "fitted", "rank"):
                corr = Correction(kind, units, channels, basis, mean_dir,
                                  rank, args.device).to(args.device)
                n_train = corr.n_trainable
                opt = torch.optim.Adam(corr.parameters(), lr=fit_lr)
                for _ in range(args.fit_steps):
                    opt.zero_grad(set_to_none=True)
                    inner_loss(lambda b, _c=corr: embed(backbone, b, _c, None),
                               ep.support_x, ep.support_y, ep.n_way, head,
                               objective=runner.objective).backward()
                    opt.step()
            fixed = mean_dir if kind == "shift" else None
            with torch.no_grad():
                se = embed(backbone, ep.support_x, corr, fixed)
                qe = embed(backbone, ep.query_x, corr, fixed)
                logits = head(se, ep.support_y, qe, ep.n_way)
            accs.append(episode_metrics(logits.float(), ep.query_y, ep.n_way).accuracy)
        mean, ci = mean_ci95(accs)
        name = f"rank{rank}" if kind == "rank" else kind
        rows.append({"variant": name, "trainable": n_train, "fit_lr": fit_lr,
                     "accuracy": mean, "ci95": ci, "n": len(accs)})
        log.info("%-8s lr %-6g trainable %-7d acc %.2f +/- %.2f", name, fit_lr,
                 n_train, 100 * mean, 100 * ci)

    print(f"\n=== diagonal vs non-diagonal activation bases, {args.dataset} "
          f"{args.k_shot}-shot, val ===")
    print(f"frozen backbone throughout; basis from {args.extract_episodes} train "
          f"episodes; {args.eval_episodes} val episodes"
          f"{'; layer4 excluded' if args.exclude_layer4 else ''}\n")
    print(f"{'variant':<10}{'trainable':>11}{'accuracy':>14}{'gain':>8}   fit_lr")
    best = {}
    for r in rows:
        if r["variant"] not in best or r["accuracy"] > best[r["variant"]]["accuracy"]:
            best[r["variant"]] = r
    for r in best.values():
        print(f"{r['variant']:<10}{r['trainable']:>11}{100*r['accuracy']:>9.2f} "
              f"+/-{100*r['ci95']:>4.2f}{100*(r['accuracy']-frozen_ref):>+8.2f}   "
              f"{r['fit_lr']:g}")
    print("\nreference (20 gradient steps on the BACKBONE, not a frozen correction):")
    print("  bn_only_ft  49.4   53,120 params      full_ft  49.6   23.5M params")
    print("note: 'ssf' is SSF-style ported to ResNet-50, not SSF as published (ViT,")
    print("in-block, merged at inference). Do not quote their numbers against these.")
    ident = best.get("identity")
    if ident:
        d = 100 * abs(ident["accuracy"] - frozen_ref)
        print(f"\ncontrol: identity {100*ident['accuracy']:.2f} vs frozen {100*frozen_ref:.2f}, "
              f"delta {d:.2f}  [{'PASS' if d < 0.5 else 'FAIL -- forward path is wrong'}]")

    path = out_dir / f"frontier_{args.dataset}_{args.k_shot}shot.json"
    path.write_text(json.dumps({"config": vars(args), "frozen_control": frozen_ref,
                                "results": rows, "spectra": spectra,
                                "shift_norms": norms}, indent=2))
    print(f"\nwrote {path}", flush=True)


if __name__ == "__main__":
    main()
