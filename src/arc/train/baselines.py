"""Level-0 baselines.

All of them run through :class:`~arc.train.episodic.EpisodeRunner`, differing only in
which units are trainable and what gate vector the controller emits. Two consequences
worth stating, because they are what make the comparison fair:

* Every method sees *identical episodes* (the sampler is seeded per episode index).
* Every method pays the same per-episode reset, so none of them benefits from having
  seen earlier tasks.

A note on progressive unfreezing: because ``lerp(h_ref, h_adp, 0) == h_ref`` has zero
derivative with respect to ``h_adp``, a unit gated to 0 is frozen both functionally and
in its gradient. Progressive unfreezing is therefore expressible exactly as a gate
schedule that opens units from the top down, rather than needing a separate mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from arc.backbones.tapped import TappedBackbone
from arc.train.episodic import StaticController


@dataclass(frozen=True)
class BaselineSpec:
    name: str
    description: str
    build: Callable[[TappedBackbone, int], tuple[StaticController, list[str] | None, int]]


def _ones(backbone: TappedBackbone) -> torch.Tensor:
    return torch.ones(backbone.n_units)


def _frozen_proto(backbone: TappedBackbone, inner_steps: int):
    """Pretrained features, prototype classifier, no adaptation whatsoever."""
    return StaticController(torch.zeros(backbone.n_units)), [], 0


def _last_layer_ft(backbone: TappedBackbone, inner_steps: int):
    return StaticController(_ones(backbone)), [backbone.unit_names[-1]], inner_steps


def _stage4_ft(backbone: TappedBackbone, inner_steps: int):
    units = [n for n in backbone.unit_names if n.startswith("layer4")]
    return StaticController(_ones(backbone)), units, inner_steps


def _full_ft(backbone: TappedBackbone, inner_steps: int):
    return StaticController(_ones(backbone)), None, inner_steps


def _progressive_unfreeze(backbone: TappedBackbone, inner_steps: int):
    """Open units from the top down over the course of the inner loop."""
    n = backbone.n_units
    schedule: list[torch.Tensor] = []
    for step in range(max(inner_steps, 1)):
        # Fraction of the inner loop elapsed -> how many top units are open.
        opened = max(1, round(n * (step + 1) / max(inner_steps, 1)))
        g = torch.zeros(n)
        g[n - opened :] = 1.0
        schedule.append(g)
    return StaticController(schedule[-1], schedule=schedule), None, inner_steps


def _bn_only_ft(backbone: TappedBackbone, inner_steps: int):
    """Adapt BatchNorm affine parameters only -- a strong, very cheap domain-shift
    baseline that is frequently omitted and frequently competitive."""
    return StaticController(_ones(backbone)), None, inner_steps


BASELINES: dict[str, BaselineSpec] = {
    "frozen_proto": BaselineSpec(
        "frozen_proto", "Pretrained features + prototype head, no adaptation", _frozen_proto
    ),
    "last_layer_ft": BaselineSpec(
        "last_layer_ft", "Fine-tune the final bottleneck only", _last_layer_ft
    ),
    "stage4_ft": BaselineSpec("stage4_ft", "Fine-tune all of stage 4", _stage4_ft),
    "progressive_unfreeze": BaselineSpec(
        "progressive_unfreeze", "Open units top-down across the inner loop", _progressive_unfreeze
    ),
    "bn_only_ft": BaselineSpec(
        "bn_only_ft", "Adapt BatchNorm affine parameters only", _bn_only_ft
    ),
    "full_ft": BaselineSpec("full_ft", "Fine-tune the entire backbone", _full_ft),
}


def select_bn_parameters(backbone: TappedBackbone) -> None:
    """Restrict gradients to BatchNorm affine parameters across every unit."""
    backbone.adapted.requires_grad_(False)
    for module in backbone.adapted.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            if module.weight is not None:
                module.weight.requires_grad_(True)
            if module.bias is not None:
                module.bias.requires_grad_(True)
