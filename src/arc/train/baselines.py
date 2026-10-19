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
    select_attn_block,
    select_attn_last_n,
    ParamSelector,
    StaticController,
    select_all,
    select_batchnorm,
    select_bn_scale,
    select_bn_shift,
    select_attn,
    select_conv,
    select_downsample,
    select_layernorm,
    select_ln_scale,
    select_ln_shift,
    select_mlp,
    select_none,
    select_patch_embed,
    select_units,
)


@dataclass(frozen=True)
class BaselineSpec:
    name: str
    description: str
    build: Callable[[TappedBackbone, int], tuple[StaticController, ParamSelector, int]]
    tunable: bool = True
    # Which backbone family this baseline is defined for. "resnet" and "vit" baselines
    # are not interchangeable -- a ViT parameter-type baseline selects nothing on a
    # ResNet and would run as a silent frozen baseline -- so callers that sweep
    # BASELINES must filter on this. None means architecture-agnostic.
    arch_family: str | None = None


def _ones(backbone: TappedBackbone) -> torch.Tensor:
    return torch.ones(backbone.n_units)


def _frozen_proto(backbone: TappedBackbone, inner_steps: int):
    """Pretrained features, prototype classifier, no adaptation whatsoever."""
    return StaticController(torch.zeros(backbone.n_units)), select_none, 0


def _last_layer_ft(backbone: TappedBackbone, inner_steps: int):
    # The last *computational* unit, not literally unit_names[-1]. On a ResNet those
    # coincide (layer4_2, a full bottleneck). On a ViT unit_names[-1] is the pre-head
    # LayerNorm -- 1536 parameters -- so taking it literally would make
    # "fine-tune the last layer" a near-frozen baseline that still reports a number.
    # That is the mis-baseline class select_* guards against, so it is handled here
    # rather than left to the caller.
    names = [n for n in backbone.unit_names if n != "norm"]
    return StaticController(_ones(backbone)), select_units(names[-1]), inner_steps


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


def _ln_only_ft(backbone: TappedBackbone, inner_steps: int):
    return StaticController(_ones(backbone)), select_layernorm, inner_steps


def _ln_scale_ft(backbone: TappedBackbone, inner_steps: int):
    return StaticController(_ones(backbone)), select_ln_scale, inner_steps


def _ln_shift_ft(backbone: TappedBackbone, inner_steps: int):
    return StaticController(_ones(backbone)), select_ln_shift, inner_steps


def _attn_only_ft(backbone: TappedBackbone, inner_steps: int):
    return StaticController(_ones(backbone)), select_attn, inner_steps


# The capacity sweep. RESULTS_VIT.md section 3.3.1 orders four parameter TYPES by
# size and finds an inverted U peaking around a third of the backbone, but those four
# differ in more than size, so the ordering cannot separate "how many parameters" from
# "which parameters". These vary capacity within ONE type -- attention, in a growing
# suffix of the block stack -- so attn_last12_ft IS attn_only_ft and the sweep inherits
# the existing measurement as its endpoint instead of re-running it.
def _attn_last_n_ft(n: int):
    def build(backbone: TappedBackbone, inner_steps: int):
        return StaticController(_ones(backbone)), select_attn_last_n(n), inner_steps

    return build


# The positional sweep. Section 3.3.2's suffix sweep found one block is enough, but
# every point in it included block 11, so the established claim is "the LAST block is
# enough". These separate position from count: same 2.75% of the backbone, one block,
# at five depths. attn_block11_ft is deliberately NOT registered -- it is
# attn_last1_ft, and two names for one experiment is how a table quotes a single
# measurement twice (see the n=12 case above).
def _attn_block_ft(index: int):
    def build(backbone: TappedBackbone, inner_steps: int):
        return StaticController(_ones(backbone)), select_attn_block(index), inner_steps

    return build


def _mlp_only_ft(backbone: TappedBackbone, inner_steps: int):
    return StaticController(_ones(backbone)), select_mlp, inner_steps


def _patch_embed_ft(backbone: TappedBackbone, inner_steps: int):
    return StaticController(_ones(backbone)), select_patch_embed, inner_steps


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
    "stage4_ft": BaselineSpec("stage4_ft", "Fine-tune all of stage 4", _stage4_ft, arch_family="resnet"),
    "bn_only_ft": BaselineSpec(
        "bn_only_ft", "Adapt BatchNorm affine parameters only", _bn_only_ft, arch_family="resnet"
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
        "bn_scale_ft", "Adapt BatchNorm scale (weight) only, ~0.11%", _bn_scale_ft, arch_family="resnet"
    ),
    "bn_shift_ft": BaselineSpec(
        "bn_shift_ft", "Adapt BatchNorm shift (bias) only, ~0.11%", _bn_shift_ft, arch_family="resnet"
    ),
    "conv_only_ft": BaselineSpec(
        "conv_only_ft", "Adapt convolution weights only, no shortcuts, ~88%", _conv_only_ft, arch_family="resnet"
    ),
    "downsample_ft": BaselineSpec(
        "downsample_ft", "Adapt shortcut projections only, ~11.8%", _downsample_ft, arch_family="resnet"
    ),
    # ViT parameter-type partition. Only meaningful on a ViT backbone -- on a ResNet
    # every one of these selects nothing, which is why experiments assert the arch
    # rather than silently reporting a frozen number. The partition is
    # {LayerNorm affine, attention, MLP, patch embedding} and covers every backbone
    # parameter exactly once; ln_scale/ln_shift subdivide the first group.
    #
    # ln_only_ft is LN-Tune (Basu et al. 2023) and attn/mlp is the split a ResNet
    # cannot express at all, so this is the half of the parameter-type question that
    # is genuinely new rather than a port.
    "ln_only_ft": BaselineSpec(
        "ln_only_ft", "Adapt LayerNorm affine parameters only (LN-Tune)", _ln_only_ft, arch_family="vit"
    ),
    "ln_scale_ft": BaselineSpec(
        "ln_scale_ft", "Adapt LayerNorm scale (weight) only", _ln_scale_ft, arch_family="vit"
    ),
    "ln_shift_ft": BaselineSpec(
        "ln_shift_ft", "Adapt LayerNorm shift (bias) only", _ln_shift_ft, arch_family="vit"
    ),
    "attn_only_ft": BaselineSpec(
        "attn_only_ft", "Adapt attention qkv and output projections only", _attn_only_ft, arch_family="vit"
    ),
    "mlp_only_ft": BaselineSpec(
        "mlp_only_ft", "Adapt block MLP (fc1, fc2) only", _mlp_only_ft, arch_family="vit"
    ),
    "patch_embed_ft": BaselineSpec(
        "patch_embed_ft", "Adapt patch embedding, cls_token and pos_embed only",
        _patch_embed_ft, arch_family="vit"
    ),
}

# Registered in a loop rather than written out: the only thing that varies is n, and a
# hand-written block invites one entry whose description says 4 while its selector
# opens 6.
#
# n = 12 is deliberately NOT registered. It would select exactly what attn_only_ft
# selects, and test_every_vit_baseline_is_a_distinct_experiment caught it on the first
# run -- correctly, since two names for one experiment is how a table ends up quoting
# the same measurement twice as if it were a replication. The sweep uses attn_only_ft
# itself as its right-hand endpoint, which is the same experiment under the name it
# already has. select_attn_last_n(12) still works and is still asserted equal to
# select_attn, because that is a property of the selector worth pinning down.
for _n in (1, 2, 3, 4, 6, 8):
    BASELINES[f"attn_last{_n}_ft"] = BaselineSpec(
        f"attn_last{_n}_ft",
        f"Adapt attention in the last {_n} transformer block{'s' if _n > 1 else ''} only",
        _attn_last_n_ft(_n),
        arch_family="vit",
    )
del _n

for _i in (0, 2, 5, 8):
    BASELINES[f"attn_block{_i}_ft"] = BaselineSpec(
        f"attn_block{_i}_ft",
        f"Adapt attention in block {_i} only",
        _attn_block_ft(_i),
        arch_family="vit",
    )
del _i
