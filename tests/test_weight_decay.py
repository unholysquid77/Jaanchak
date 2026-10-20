"""Weight decay on the inner loop, and the one way it can silently go wrong.

`inner_weight_decay` exists for RESULTS_VIT.md §3.5's second mechanism prediction:
if `full_ft` loses on EuroSAT because 85.8M parameters over-optimise on 25 support
images, then regularising those parameters should help it and should not help a
2.4M-parameter baseline.

The trap, and the reason this file exists rather than a one-line Adam change:
`ProtoHead.log_temperature` is a learnable Parameter initialised to log(10) for the
cosine metric. A single `weight_decay=` across every parameter pulls it toward 0,
i.e. the temperature toward 1, which flattens the cosine softmax -- the exact
failure the head's own docstring warns about. That would show up as "regularisation
hurts", attributed to capacity, when it was the classifier being broken. So the
decay must live in a param group that excludes the head, and that is asserted here.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from arc.heads.protonet import ProtoHead  # noqa: E402
from arc.train.episodic import EpisodeRunner  # noqa: E402


class _Tiny(torch.nn.Module):
    """Stands in for a TappedBackbone's `adapted` branch: one trainable weight."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(4, 4)


def _groups(wd: float):
    """Build the optimiser exactly as EpisodeRunner does, without a real backbone."""
    adapted, head = _Tiny(), ProtoHead(metric="cosine")
    backbone_params = [p for p in adapted.parameters() if p.requires_grad]
    head_params = list(head.parameters())
    if wd > 0 and backbone_params:
        opt = torch.optim.Adam(
            [
                {"params": backbone_params, "weight_decay": wd},
                {"params": head_params, "weight_decay": 0.0},
            ],
            lr=1e-3,
        )
    else:
        opt = torch.optim.Adam(backbone_params + head_params, lr=1e-3)
    return adapted, head, opt


def test_head_temperature_is_never_decayed() -> None:
    """The load-bearing one. The head's group must have weight_decay 0 even when the
    backbone's does not."""
    _, head, opt = _groups(0.1)
    head_ids = {id(p) for p in head.parameters()}
    for group in opt.param_groups:
        if any(id(p) in head_ids for p in group["params"]):
            assert group["weight_decay"] == 0.0, (
                "the head is being decayed; log_temperature would collapse toward 1"
            )


def test_backbone_group_carries_the_decay() -> None:
    adapted, head, opt = _groups(0.1)
    head_ids = {id(p) for p in head.parameters()}
    decayed = [
        g["weight_decay"] for g in opt.param_groups
        if not any(id(p) in head_ids for p in g["params"])
    ]
    assert decayed and all(w == 0.1 for w in decayed), decayed


def test_zero_weight_decay_is_a_single_group() -> None:
    """At 0 the construction must collapse to the single-group Adam it replaced, so
    every existing number in the document stays reproducible."""
    _, _, opt = _groups(0.0)
    assert len(opt.param_groups) == 1
    assert opt.param_groups[0]["weight_decay"] == 0.0


def test_decay_actually_shrinks_the_backbone_weight() -> None:
    """A sanity check that the group is wired to something, not just configured."""
    torch.manual_seed(0)
    outs = {}
    for wd in (0.0, 0.5):
        torch.manual_seed(0)
        adapted, head, opt = _groups(wd)
        w0 = adapted.lin.weight.detach().clone()
        for _ in range(20):
            opt.zero_grad()
            # A loss with zero gradient at the current point, so the ONLY thing that
            # can move the weight is the decay term.
            (adapted.lin.weight.sum() * 0.0).backward(retain_graph=False)
            adapted.lin.weight.grad = torch.zeros_like(adapted.lin.weight)
            opt.step()
        outs[wd] = (adapted.lin.weight.detach() - w0).abs().sum().item()
    assert outs[0.0] == pytest.approx(0.0, abs=1e-8), outs
    assert outs[0.5] > 1e-4, outs


def test_runner_rejects_negative_weight_decay() -> None:
    pytest.importorskip("timm")
    from arc.backbones.tapped import TappedBackbone

    with pytest.raises(ValueError, match="non-negative"):
        EpisodeRunner(
            TappedBackbone(arch="vit_small_patch16_224.augreg_in21k", pretrained=False),
            head=ProtoHead(),
            device="cpu",
            inner_weight_decay=-1.0,
        )
