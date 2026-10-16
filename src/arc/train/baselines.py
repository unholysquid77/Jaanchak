"""Level-0 baselines.

All of them run through :class:`~arc.train.episodic.EpisodeRunner`, differing only in
which parameters are trainable and what gate vector the controller emits. Two
consequences worth stating, because they are what make the comparison fair:

* Every method sees *identical episodes* (the sampler is seeded per episode index).
* Every method pays the same per-episode reset, so none of them benefits from having
  seen earlier tasks.

A note on progressive unfreezing: because ``lerp(h_ref, h_adp, 0) == h_ref`` has zero
derivative with respect to ``h_adp``, a unit gated to 0 is frozen both functionally and
in its gradient. Progressive unfreezing is therefore expressible exactly as a gate
schedule that opens units from the top down, rather than needing a separate mechanism.

Each baseline declares ``tunable``: whether a learning rate means anything for it.
``frozen_proto`` performs no gradient steps, so sweeping its LR would be wasted compute.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from arc.backbones.tapped import TappedBackbone
from arc.train.episodic import (
    ParamSelector,
    StaticController,
    select_all,
    select_batchnorm,
    select_bn_scale,
    select_bn_shift,
    select_conv,
    select_downsample,
    select_none,
    select_units,
)


@dataclass(frozen=True)
class BaselineSpec:
    name: str
    description: str
    build: Callable[[TappedBackbone, int], tuple[StaticController, ParamSelector, int]]
    tunable: bool = True


def _ones(backbone: TappedBackbone) -> torch.Tensor:
    return torch.ones(backbone.n_units)


def _frozen_proto(backbone: TappedBackbone, inner_steps: int):
    """Pretrained features, prototype classifier, no adaptation whatsoever."""
    return StaticController(torch.zeros(backbone.n_units)), select_none, 0


def _last_layer_ft(backbone: TappedBackbone, inner_steps: int):
    return StaticController(_ones(backbone)), select_units(backbone.unit_names[-1]), inner_steps


def _stage4_ft(backbone: TappedBackbone, inner_steps: int):
    units = [n for n in backbone.unit_names if n.startswith("layer4")]
    return StaticController(_ones(backbone)), select_units(*units), inner_steps


def _full_ft(backbone: TappedBackbone, inner_steps: int):
    return StaticController(_ones(backbone)), select_all, inner_steps


def _bn_only_ft(backbone: TappedBackbone, inner_steps: int):
    return StaticController(_ones(backbone)), select_batchnorm, inner_steps


def _bn_scale_ft(backbone: TappedBackbone, inner_steps: int):
    return StaticController(_ones(backbone)), select_bn_scale, inner_steps


def _bn_shift_ft(backbone: TappedBackbone, inner_steps: int):
    return StaticController(_ones(backbone)), select_bn_shift, inner_steps


def _conv_only_ft(backbone: TappedBackbone, inner_steps: int):
    return StaticController(_ones(backbone)), select_conv, inner_steps


def _downsample_ft(backbone: TappedBackbone, inner_steps: int):
    return StaticController(_ones(backbone)), select_downsample, inner_steps


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
    return StaticController(schedule[-1], schedule=schedule), select_all, inner_steps


BASELINES: dict[str, BaselineSpec] = {
    "frozen_proto": BaselineSpec(
        "frozen_proto",
        "Pretrained features + prototype head, no adaptation",
        _frozen_proto,
        tunable=False,
    ),
    "last_layer_ft": BaselineSpec(
        "last_layer_ft", "Fine-tune the final bottleneck only", _last_layer_ft
    ),
    "stage4_ft": BaselineSpec("stage4_ft", "Fine-tune all of stage 4", _stage4_ft),
    "bn_only_ft": BaselineSpec(
        "bn_only_ft", "Adapt BatchNorm affine parameters only", _bn_only_ft
    ),
    "progressive_unfreeze": BaselineSpec(
        "progressive_unfreeze", "Open units top-down across the inner loop", _progressive_unfreeze
    ),
    "full_ft": BaselineSpec("full_ft", "Fine-tune the entire backbone", _full_ft),
    # Parameter-type partition. The ISIC result -- bn_only_ft at 0.22% of parameters
    # tying full fine-tuning -- says the axis that pays under domain shift may be
    # parameter *type* rather than layer depth. These four decompose that claim:
    # BatchNorm's two halves separately, the convolutional complement, and the
    # shortcut projections. Accuracy is reported against %params as a frontier rather
    # than at matched budget, because convolution weights are 88% of the backbone and
    # cannot be budget-matched to BatchNorm's 0.2%.
    "bn_scale_ft": BaselineSpec(
        "bn_scale_ft", "Adapt BatchNorm scale (weight) only, ~0.11%", _bn_scale_ft
    ),
    "bn_shift_ft": BaselineSpec(
        "bn_shift_ft", "Adapt BatchNorm shift (bias) only, ~0.11%", _bn_shift_ft
    ),
    "conv_only_ft": BaselineSpec(
        "conv_only_ft", "Adapt convolution weights only, no shortcuts, ~88%", _conv_only_ft
    ),
    "downsample_ft": BaselineSpec(
        "downsample_ft", "Adapt shortcut projections only, ~11.8%", _downsample_ft
    ),
}
