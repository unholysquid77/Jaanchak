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
    select_none,
    select_units,
)


def _backbone() -> TappedBackbone:
    return TappedBackbone(arch="resnet18", pretrained=False)


def _trainable(backbone: TappedBackbone) -> int:
    return sum(p.numel() for p in backbone.adapted.parameters() if p.requires_grad)


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
    footprints: dict[str, tuple[int, tuple]] = {}
    for key, spec in BASELINES.items():
        controller, select, spec_steps = spec.build(net, steps)
        select(net)
        trajectory = tuple(
            tuple(controller.step(i).tolist()) for i in range(max(spec_steps, 1))
        )
        footprints[key] = (_trainable(net), trajectory)

    # frozen_proto takes zero steps, so its footprint is irrelevant; the rest must differ.
    stepping = {k: v for k, v in footprints.items() if k != "frozen_proto"}
    assert len(set(stepping.values())) == len(stepping), f"duplicate baselines: {stepping}"


def test_frozen_proto_takes_no_gradient_steps() -> None:
    net = _backbone()
    _, _, steps = BASELINES["frozen_proto"].build(net, 20)
    assert steps == 0
    assert BASELINES["frozen_proto"].tunable is False
