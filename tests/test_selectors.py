"""Parameter-selection tests.

These exist because the BatchNorm-only baseline previously ran as full fine-tuning: the
runner re-applied a list-based unit selection after the caller had narrowed to BN
parameters, and nothing caught it. A baseline that is secretly a different baseline is
worse than no baseline, so each selector now asserts its own footprint.
"""

from __future__ import annotations

import torch

from arc.backbones.tapped import TappedBackbone
from arc.train.baselines import BASELINES
from arc.train.episodic import (
    select_all,
    select_batchnorm,
    select_bn_scale,
    select_bn_shift,
    select_conv,
    select_downsample,
    select_none,
    select_units,
)


def _backbone() -> TappedBackbone:
    return TappedBackbone(arch="resnet18", pretrained=False)


def _trainable(backbone: TappedBackbone) -> int:
    return sum(p.numel() for p in backbone.adapted.parameters() if p.requires_grad)


def _selected_names(backbone: TappedBackbone) -> tuple[str, ...]:
    """Which parameters a selector actually opened, by name.

    The identity of an experiment is the set it selects, not how large that set is.
    """
    return tuple(sorted(
        n for n, p in backbone.adapted.named_parameters() if p.requires_grad
    ))


def test_select_none_freezes_everything() -> None:
    net = _backbone()
    select_none(net)
    assert _trainable(net) == 0


def test_select_all_unfreezes_everything() -> None:
    net = _backbone()
    select_none(net)
    select_all(net)
    assert _trainable(net) == sum(p.numel() for p in net.adapted.parameters())


def test_select_units_with_no_arguments_freezes_rather_than_unfreezes() -> None:
    """The old `units or all_units` idiom turned an empty selection into a full one."""
    net = _backbone()
    select_units()(net)
    assert _trainable(net) == 0


def test_select_units_is_exact() -> None:
    net = _backbone()
    target = net.unit_names[-1]
    select_units(target)(net)
    expected = sum(p.numel() for p in net.adapted[target].parameters())
    assert _trainable(net) == expected
    assert 0 < expected < sum(p.numel() for p in net.adapted.parameters())


def test_select_batchnorm_is_a_small_slice_of_the_network() -> None:
    net = _backbone()
    select_batchnorm(net)
    trainable = _trainable(net)
    total = sum(p.numel() for p in net.adapted.parameters())
    assert 0 < trainable < 0.02 * total, (
        f"BatchNorm affine params should be a tiny fraction of the net, got "
        f"{trainable}/{total}. If this equals `total`, BN-only is running as full FT."
    )
    for module in net.adapted.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            assert module.weight.requires_grad and module.bias.requires_grad


def test_every_baseline_is_a_distinct_experiment() -> None:
    """Two baselines with the same trainable parameters and the same gate trajectory
    are one experiment wearing two names.

    The comparison is over the *whole* schedule, not the final gate: progressive
    unfreezing ends fully open, exactly like full fine-tuning, and differs only in how
    it got there.
    """
    net = _backbone()
    steps = 10
    footprints: dict[str, tuple[tuple, tuple]] = {}
    for key, spec in BASELINES.items():
        # ViT-only baselines select nothing on a ResNet, so on this backbone they
        # would all share the empty footprint and collide. They are not skipped
        # silently anywhere that matters: the selectors themselves raise on the
        # wrong family (see test_vit_selectors_refuse_a_resnet below), and the ViT
        # mirror of this test lives in tests/test_selectors_vit.py.
        if spec.arch_family not in (None, "resnet"):
            continue
        controller, select, spec_steps = spec.build(net, steps)
        select(net)
        trajectory = tuple(
            tuple(controller.step(i).tolist()) for i in range(max(spec_steps, 1))
        )
        # The footprint is the *set of parameter names*, not their count. A count does
        # not identify a set: bn_scale_ft and bn_shift_ft select exactly as many
        # parameters as each other (every BatchNorm weight vs every BatchNorm bias)
        # while being different experiments, and a count-based footprint called them
        # duplicates. Names are an exact identity, and would have caught the original
        # BatchNorm-as-full-fine-tuning defect more directly than a count did.
        footprints[key] = (_selected_names(net), trajectory)

    # frozen_proto takes zero steps, so its footprint is irrelevant; the rest must differ.
    stepping = {k: v for k, v in footprints.items() if k != "frozen_proto"}
    assert len(set(stepping.values())) == len(stepping), f"duplicate baselines: {stepping}"


def test_frozen_proto_takes_no_gradient_steps() -> None:
    net = _backbone()
    _, _, steps = BASELINES["frozen_proto"].build(net, 20)
    assert steps == 0
    assert BASELINES["frozen_proto"].tunable is False


# -- the parameter-type partition -------------------------------------------

def test_bn_scale_and_bn_shift_partition_bn_only_exactly() -> None:
    """The two halves must be disjoint and together reconstruct select_batchnorm.

    This is the claim the ISIC result rests on -- that BatchNorm-only adaptation is a
    per-channel affine, scale plus shift -- so the code had better implement exactly
    that split and nothing else.
    """
    net = _backbone()
    select_bn_scale(net)
    scale = set(_selected_names(net))
    select_bn_shift(net)
    shift = set(_selected_names(net))
    select_batchnorm(net)
    both = set(_selected_names(net))

    assert scale and shift
    assert not (scale & shift), "scale and shift overlap"
    assert scale | shift == both, "the halves do not reconstruct select_batchnorm"
    assert all(n.endswith(".weight") for n in scale)
    assert all(n.endswith(".bias") for n in shift)


def test_conv_selection_excludes_shortcut_projections() -> None:
    """select_conv and select_downsample must not overlap, or the parameter-type
    frontier double-counts the 1x1 projections."""
    net = _backbone()
    select_conv(net)
    conv = set(_selected_names(net))
    select_downsample(net)
    down = set(_selected_names(net))

    assert conv and down
    assert not (conv & down)
    assert all("downsample" not in n for n in conv)
    assert all("downsample" in n for n in down)


def test_no_group_selector_touches_a_batchnorm_running_statistic() -> None:
    """Running statistics are buffers, not parameters, and must never become
    trainable -- that is landmine 3/4 in a new costume."""
    net = _backbone()
    for select in (select_bn_scale, select_bn_shift, select_conv, select_downsample):
        select(net)
        assert all(
            "running_mean" not in n and "running_var" not in n
            for n in _selected_names(net)
        )


def test_vit_selectors_refuse_a_resnet() -> None:
    """A ViT selector on a ResNet must raise, not quietly select nothing.

    This is the guard that keeps the arch_family filter above honest. Without it,
    filtering baselines by family would merely hide the collision in this test
    while a real sweep could still pass a ViT baseline a ResNet and report the
    resulting frozen number as a result.
    """
    import pytest

    from arc.train.episodic import (
        select_attn,
        select_layernorm,
        select_ln_scale,
        select_ln_shift,
        select_mlp,
        select_patch_embed,
    )

    net = _backbone()
    for sel in (
        select_layernorm,
        select_ln_scale,
        select_ln_shift,
        select_attn,
        select_mlp,
        select_patch_embed,
    ):
        with pytest.raises(ValueError, match="vit selector"):
            sel(net)


def test_resnet_selectors_declare_their_family() -> None:
    """Every parameter-type baseline must be tagged, or the filter above lets an
    untagged ViT baseline through as architecture-agnostic."""
    for key in ("bn_only_ft", "bn_scale_ft", "bn_shift_ft", "conv_only_ft", "downsample_ft"):
        assert BASELINES[key].arch_family == "resnet", key
    for key in ("ln_only_ft", "ln_scale_ft", "ln_shift_ft", "attn_only_ft", "mlp_only_ft"):
        assert BASELINES[key].arch_family == "vit", key
