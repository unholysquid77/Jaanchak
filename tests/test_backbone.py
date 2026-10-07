"""Correctness anchors for the gated backbone.

The two exactness properties below are what justify calling the gate "representation
control" rather than a heuristic blend. If either breaks, every downstream number is
meaningless, so these run first and fast (resnet18, random weights -- the invariants
are independent of the weights).
"""

from __future__ import annotations

import torch

from arc.backbones.tapped import TappedBackbone


def _perturbed_backbone(seed: int = 0) -> TappedBackbone:
    """Adapted and frozen twins start identical, which would make the invariants
    vacuous. Perturb the adapted side so the two endpoints are genuinely different."""
    torch.manual_seed(seed)
    net = TappedBackbone(arch="resnet18", pretrained=False)
    net.eval()  # freeze BatchNorm running stats; otherwise forwards are not comparable
    with torch.no_grad():
        for p in net.adapted.parameters():
            p.add_(torch.randn_like(p) * 0.02)
    return net


def test_gate_zero_recovers_pretrained_network_exactly() -> None:
    net = _perturbed_backbone()
    x = torch.randn(2, 3, 64, 64)
    gated = net(x, gates=torch.zeros(net.n_units)).embedding
    reference = net.forward_reference(x).embedding
    assert torch.allclose(gated, reference, atol=1e-5), (
        (gated - reference).abs().max().item()
    )


def test_gate_one_recovers_adapted_network_exactly() -> None:
    net = _perturbed_backbone()
    x = torch.randn(2, 3, 64, 64)
    gated = net(x, gates=torch.ones(net.n_units)).embedding
    adapted = net.forward_adapted(x).embedding
    assert torch.allclose(gated, adapted, atol=1e-5), (gated - adapted).abs().max().item()


def test_intermediate_gate_is_strictly_between_the_endpoints() -> None:
    net = _perturbed_backbone()
    x = torch.randn(2, 3, 64, 64)
    ref = net.forward_reference(x).embedding
    adp = net.forward_adapted(x).embedding
    mid = net(x, gates=torch.full((net.n_units,), 0.5)).embedding
    assert not torch.allclose(mid, ref, atol=1e-4)
    assert not torch.allclose(mid, adp, atol=1e-4)


def test_gate_receives_gradient_from_task_loss() -> None:
    """The point of forward interpolation: the gate must be trainable.

    A gradient-masking implementation would leave this gradient at exactly zero, which
    is the failure mode documented in docs/ARC_SPEC.md section 5.
    """
    net = _perturbed_backbone()
    gates = torch.full((net.n_units,), 0.5, requires_grad=True)
    out = net(torch.randn(2, 3, 64, 64), gates=gates)
    out.embedding.sum().backward()
    assert gates.grad is not None
    assert gates.grad.abs().sum() > 0, "gate got no gradient -- interpolation is broken"


def test_frozen_twin_never_accumulates_gradient() -> None:
    net = _perturbed_backbone()
    out = net(torch.randn(2, 3, 64, 64), gates=torch.full((net.n_units,), 0.5))
    out.embedding.sum().backward()
    assert all(p.grad is None for p in net.frozen.parameters())


def test_frozen_twin_stays_in_eval_mode() -> None:
    net = _perturbed_backbone()
    net.train()
    assert not net.frozen.training, "frozen BatchNorm stats would drift in train mode"
    assert net.adapted.training


def test_unit_names_cover_stem_and_every_block() -> None:
    net = TappedBackbone(arch="resnet18", pretrained=False)
    assert net.unit_names[0] == "stem"
    assert net.n_units == 1 + 8  # resnet18: stem + 2 blocks per stage
    net50 = TappedBackbone(arch="resnet50", pretrained=False)
    assert net50.n_units == 1 + 16  # the 17 controllable units the spec assumes
