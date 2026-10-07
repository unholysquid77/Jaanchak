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
from typing import Mapping, Protocol

import torch
import torch.nn.functional as F

from arc.backbones.tapped import TappedBackbone
from arc.data.base import Episode
from arc.eval.metrics import EpisodeMetrics, episode_metrics
from arc.heads.protonet import ProtoHead


class Controller(Protocol):
    """Produces per-unit adaptation intensities.

    ``step`` is called before every inner update, so a controller may be static (a
    fixed vector), state-feedback (the gate), or temporal (FBSDE).
    """

    def reset(self, backbone: TappedBackbone, episode: Episode) -> None: ...

    def step(self, step_index: int) -> torch.Tensor | None:
        """Return a ``(n_units,)`` gate vector, or ``None`` for "fully adapted"."""


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
        inner_lr: float = 1e-3,
        amp: bool = True,
    ) -> None:
        self.device = torch.device(device)
        self.backbone = backbone.to(self.device)
        self.head = (head or ProtoHead()).to(self.device)
        self.inner_steps = inner_steps
        self.inner_lr = inner_lr
        self.amp = amp and self.device.type == "cuda"

        # Snapshot on the device so resetting is a device-to-device copy, not a
        # host round-trip. ~100 MB for ResNet-50; cheap insurance against drift
        # leaking across episodes.
        self._pristine = {k: v.detach().clone() for k, v in self.backbone.adapted.state_dict().items()}
        self._head_pristine = {k: v.detach().clone() for k, v in self.head.state_dict().items()}

    def reset(self) -> None:
        self.backbone.adapted.load_state_dict(self._pristine)
        self.head.load_state_dict(self._head_pristine)

    def trainable_units(self, units: list[str] | None) -> None:
        """Restrict which units carry gradients.

        Distinct from gating: a unit may be gated open but frozen (it then contributes
        its pretrained output), which is how the "static gate" ablation is expressed.
        """
        self.backbone.adapted.requires_grad_(False)
        for name in units or self.backbone.unit_names:
            self.backbone.adapted[name].requires_grad_(True)

    def run_episode(
        self,
        episode: Episode,
        controller: Controller,
        trainable: list[str] | None = None,
        collect: Mapping[str, object] | None = None,
    ) -> EpisodeResult:
        started = time.perf_counter()
        self.reset()
        self.trainable_units(trainable)
        episode = episode.to(self.device)
        controller.reset(self.backbone, episode)

        params = [p for p in self.backbone.adapted.parameters() if p.requires_grad]
        params += list(self.head.parameters())
        opt = torch.optim.Adam(params, lr=self.inner_lr) if params else None
        scaler = torch.amp.GradScaler("cuda", enabled=self.amp)

        n_way = episode.n_way
        last_loss = float("nan")
        gates: torch.Tensor | None = None

        self.backbone.train()
        for step in range(self.inner_steps if opt is not None else 0):
            gates = controller.step(step)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=self.amp):
                emb = self.backbone(episode.support_x, gates=gates).embedding
                # Support set serves as both prototypes and targets. Leave-one-out
                # would be cleaner but costs n_way*k_shot forward passes; the standard
                # ProtoNet fine-tuning recipe uses the plain form and we match it.
                logits = self.head(emb, episode.support_y, emb, n_way)
                loss = F.cross_entropy(logits, episode.support_y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            last_loss = float(loss.detach())

        self.backbone.eval()
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=self.amp):
            gates = controller.step(max(self.inner_steps - 1, 0))
            support_emb = self.backbone(episode.support_x, gates=gates).embedding
            query_emb = self.backbone(episode.query_x, gates=gates).embedding
            logits = self.head(support_emb, episode.support_y, query_emb, n_way)

        metrics = episode_metrics(logits.float(), episode.query_y, n_way)
        return EpisodeResult(
            metrics=metrics,
            episode_id=episode.episode_id,
            loss=last_loss,
            seconds=time.perf_counter() - started,
            fraction_modified=self.backbone.fraction_modified(),
            gate_profile=[] if gates is None else gates.detach().float().cpu().tolist(),
        )
