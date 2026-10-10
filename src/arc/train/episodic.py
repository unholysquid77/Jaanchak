"""Per-episode test-time adaptation.

Every method in the paper -- from "frozen backbone" to the FBSDE controller -- runs
through :func:`run_episode`. They differ only in the gate vector supplied by a
controller and in which units carry trainable parameters. Sharing one code path is what
makes "matched experimental conditions" true by construction rather than by discipline.

The protocol, fixed here once:

1. Reset the adapted backbone to pretrained weights. **Every episode starts clean** --
   otherwise episode 400 is evaluating a model that has seen 399 other tasks, which is
   a different (and much easier) problem than few-shot adaptation.
2. Run ``inner_steps`` gradient steps on the *support* set only.
3. Evaluate the query set under ``no_grad``.

Keeping query out of the adaptation loop is both the honest protocol and the thing that
keeps 10 GiB of VRAM sufficient: gradients are only ever held for ``n_way * k_shot``
images.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Mapping, Protocol

import torch
import torch.nn.functional as F

from arc.backbones.tapped import TappedBackbone
from arc.data.base import Episode
from arc.eval.metrics import EpisodeMetrics, episode_metrics
from arc.heads.protonet import ProtoHead
from arc.train.objectives import ObjectiveName, inner_loss


class Controller(Protocol):
    """Produces per-unit adaptation intensities.

    ``step`` is called before every inner update, so a controller may be static (a
    fixed vector), state-feedback (the gate), or temporal (FBSDE).
    """

    def reset(self, backbone: TappedBackbone, episode: Episode) -> None: ...

    def step(self, step_index: int) -> torch.Tensor | None:
        """Return a ``(n_units,)`` gate vector, or ``None`` for "fully adapted"."""


ParamSelector = Callable[[TappedBackbone], None]


def select_all(backbone: TappedBackbone) -> None:
    """Every backbone parameter is trainable."""
    backbone.adapted.requires_grad_(True)


def select_none(backbone: TappedBackbone) -> None:
    """No backbone parameter is trainable (the head may still train)."""
    backbone.adapted.requires_grad_(False)


def select_units(*names: str) -> ParamSelector:
    """Only the named units are trainable.

    Note the empty case is meaningful and distinct from "all": ``select_units()``
    freezes the backbone entirely.
    """

    def selector(backbone: TappedBackbone) -> None:
        unknown = set(names) - set(backbone.unit_names)
        if unknown:
            raise ValueError(f"unknown units {sorted(unknown)}")
        backbone.adapted.requires_grad_(False)
        for name in names:
            backbone.adapted[name].requires_grad_(True)

    return selector


def select_batchnorm(backbone: TappedBackbone) -> None:
    """Only BatchNorm affine parameters, across every unit.

    A strong and very cheap domain-shift baseline that is often omitted and often
    competitive -- worth beating explicitly rather than ignoring.
    """
    backbone.adapted.requires_grad_(False)
    for module in backbone.adapted.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            if module.weight is not None:
                module.weight.requires_grad_(True)
            if module.bias is not None:
                module.bias.requires_grad_(True)


class StaticController:
    """Fixed gate vector. Expresses every Level-0 baseline.

    ``frozen`` -> zeros, ``full fine-tuning`` -> ones, ``last-layer`` -> one-hot on the
    final block, ``progressive unfreezing`` -> a schedule that opens units from the top
    down as the inner loop proceeds.
    """

    def __init__(self, gates: torch.Tensor, schedule: list[torch.Tensor] | None = None) -> None:
        self.gates = gates
        self.schedule = schedule

    def reset(self, backbone: TappedBackbone, episode: Episode) -> None:
        self.gates = self.gates.to(episode.support_x.device)
        if self.schedule is not None:
            self.schedule = [g.to(episode.support_x.device) for g in self.schedule]

    def step(self, step_index: int) -> torch.Tensor:
        if self.schedule is None:
            return self.gates
        return self.schedule[min(step_index, len(self.schedule) - 1)]


@dataclass
class EpisodeResult:
    metrics: EpisodeMetrics
    episode_id: int
    loss: float
    seconds: float
    fraction_modified: float
    gate_profile: list[float] = field(default_factory=list)
    extra: dict[str, float] = field(default_factory=dict)


class EpisodeRunner:
    """Holds the backbone and the pristine weight snapshot used to reset each episode."""

    def __init__(
        self,
        backbone: TappedBackbone,
        head: ProtoHead | None = None,
        device: str | torch.device = "cuda",
        inner_steps: int = 20,
        inner_lr: float = 1e-4,
        amp: bool = True,
        objective: ObjectiveName = "two_view",
        bn_mode: str = "eval",
        grad_clip: float | None = 1.0,
    ) -> None:
        self.device = torch.device(device)
        self.backbone = backbone.to(self.device)
        self.head = (head or ProtoHead()).to(self.device)
        self.inner_steps = inner_steps
        self.inner_lr = inner_lr
        self.amp = amp and self.device.type == "cuda"
        self.objective = objective
        self.grad_clip = grad_clip

        if bn_mode not in {"eval", "train"}:
            raise ValueError(f"bn_mode must be 'eval' or 'train', got {bn_mode!r}")
        # 'eval' keeps the pretrained running statistics fixed during adaptation.
        # Without it, BatchNorm recomputes its statistics from the support batch on the
        # *forward* pass -- which happens regardless of learning rate, and regardless of
        # whether the optimiser does anything. Measured on a 1-shot episode with lr=0:
        # parameter drift 0.0, BatchNorm running-stat drift 162.1, and accuracy falling
        # from 61.2% to 51.2% purely as a function of step count. 'train' is retained
        # because transductive BatchNorm is a legitimate (and strong) cross-domain
        # technique, but it must be an explicit, separately-reported choice.
        self.bn_mode = bn_mode

        if bn_mode == "train" and getattr(backbone, "grad_checkpoint", False):
            raise ValueError(
                "grad_checkpoint with bn_mode='train' double-counts BatchNorm running "
                "statistics: the backward recomputation re-runs the forward pass and "
                "updates them a second time. Use bn_mode='eval', or disable "
                "checkpointing."
            )

        # Snapshot on the device so resetting is a device-to-device copy, not a
        # host round-trip. ~100 MB for ResNet-50; cheap insurance against drift
        # leaking across episodes.
        self._pristine = {k: v.detach().clone() for k, v in self.backbone.adapted.state_dict().items()}
        self._head_pristine = {k: v.detach().clone() for k, v in self.head.state_dict().items()}

    def reset(self) -> None:
        self.backbone.adapted.load_state_dict(self._pristine)
        self.head.load_state_dict(self._head_pristine)

    def apply_selector(self, select: ParamSelector | None) -> None:
        """Choose which parameters carry gradients this episode.

        Distinct from gating: a unit may be gated open but frozen (it then contributes
        its pretrained output), which is how the "static gate" ablation is expressed.

        Takes a callable rather than a list of unit names because some baselines select
        parameters by *kind* rather than by unit -- BatchNorm-only adaptation spans
        every unit. An earlier list-based API made that inexpressible, and the
        BatchNorm baseline silently ran as full fine-tuning.
        """
        (select or select_all)(self.backbone)

    def set_train_mode(self) -> None:
        """Put the backbone in training mode, honouring ``bn_mode``.

        The frozen twin is always in eval mode (``TappedBackbone.train`` enforces it);
        this governs the *adapted* branch only.
        """
        self.backbone.train()
        if self.bn_mode == "eval":
            for module in self.backbone.adapted.modules():
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                    module.eval()

    def run_episode(
        self,
        episode: Episode,
        controller: Controller,
        select: ParamSelector | None = None,
    ) -> EpisodeResult:
        started = time.perf_counter()
        self.reset()
        self.apply_selector(select)
        episode = episode.to(self.device)
        controller.reset(self.backbone, episode)

        params = [p for p in self.backbone.adapted.parameters() if p.requires_grad]
        params += list(self.head.parameters())
        opt = torch.optim.Adam(params, lr=self.inner_lr) if params else None
        scaler = torch.amp.GradScaler("cuda", enabled=self.amp)

        n_way = episode.n_way
        last_loss = float("nan")
        gates: torch.Tensor | None = None

        self.set_train_mode()
        skipped = 0
        for step in range(self.inner_steps if opt is not None else 0):
            gates = controller.step(step)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=self.amp):
                loss = inner_loss(
                    lambda batch: self.backbone(batch, gates=gates).embedding,
                    episode.support_x,
                    episode.support_y,
                    n_way,
                    self.head,
                    objective=self.objective,
                )
            scaler.scale(loss).backward()

            if self.grad_clip is not None:
                # unscale_ before clipping, or the clip threshold applies to
                # AMP-scaled gradients and means nothing.
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, self.grad_clip)

            before_scale = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            # A decreasing scale means the step was skipped for inf/nan gradients.
            skipped += int(scaler.get_scale() < before_scale)
            last_loss = float(loss.detach())

        self.backbone.eval()
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=self.amp):
            gates = controller.step(max(self.inner_steps - 1, 0))
            support_emb = self.backbone(episode.support_x, gates=gates).embedding
            query_emb = self.backbone(episode.query_x, gates=gates).embedding
            logits = self.head(support_emb, episode.support_y, query_emb, n_way)

        metrics = episode_metrics(logits.float(), episode.query_y, n_way)
        extra = {"skipped_steps": float(skipped)}
        return EpisodeResult(
            metrics=metrics,
            episode_id=episode.episode_id,
            loss=last_loss,
            seconds=time.perf_counter() - started,
            fraction_modified=self.backbone.fraction_modified(),
            gate_profile=[] if gates is None else gates.detach().float().cpu().tolist(),
            extra=extra,
        )
