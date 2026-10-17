"""ViT parameter-selection and unit-split tests.

The ResNet selectors earned `tests/test_selectors.py` by having a baseline silently
run as full fine-tuning. The ViT port carries two *new* ways to fail the same way,
so both are asserted here rather than trusted:

1. The unit split transfers ownership of ``blocks`` and ``norm`` out of the host
   ``VisionTransformer``. If that transfer is incomplete, a parameter is owned by
   two units, ``select_*`` opens it twice, and the a=0 endpoint stops being
   bit-exact against the pretrained network -- the framework's correctness anchor.
2. ``cls_token`` and ``pos_embed`` are bare parameters, not modules, so every
   module-typed selector misses them. A partition that quietly omits them looks
   fine until the per-type percentages do not sum to 100.

Skipped entirely when timm is absent, so the ResNet suite still runs on an
environment without it.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

from arc.backbones.tapped import TappedBackbone  # noqa: E402
from arc.train.episodic import (  # noqa: E402
    select_all,
    select_attn,
    select_layernorm,
    select_ln_scale,
    select_ln_shift,
    select_mlp,
    select_none,
    select_patch_embed,
)

ARCH = "vit_small_patch16_224.augreg_in21k"


@pytest.fixture(scope="module")
def backbone() -> TappedBackbone:
    # pretrained=False: these tests are about structure, and downloading 21k weights
    # to assert a parameter-name partition would be a waste of bandwidth.
    return TappedBackbone(arch=ARCH, pretrained=False)


def _names(backbone: TappedBackbone) -> set[str]:
    return {n for n, p in backbone.adapted.named_parameters() if p.requires_grad}


def _count(backbone: TappedBackbone) -> int:
    return sum(p.numel() for p in backbone.adapted.parameters() if p.requires_grad)


# -- unit split ---------------------------------------------------------------


def test_unit_names(backbone: TappedBackbone) -> None:
    """ViT-S/16 is the stem, 12 blocks and the pre-head norm, in that order."""
    assert backbone.unit_names == (
        ["stem"] + [f"block_{i}" for i in range(12)] + ["norm"]
    )
    assert backbone.n_units == 14


def test_is_vit_and_embed_dim(backbone: TappedBackbone) -> None:
    assert backbone.is_vit is True
    assert backbone.embed_dim == 384
    assert backbone.n_prefix_tokens == 1
    assert backbone.global_pool == "token"


def test_every_parameter_owned_exactly_once(backbone: TappedBackbone) -> None:
    """The split must not leave blocks reachable through the stem's host model.

    This is the test that catches an incomplete ownership transfer. Counting
    ``adapted.parameters()`` would not: PyTorch de-duplicates shared parameters in
    ``parameters()``, so a double-owned block is invisible there. Counting per unit
    and comparing to the de-duplicated total is what exposes it.
    """
    per_unit = sum(
        sum(p.numel() for p in backbone.adapted[u].parameters())
        for u in backbone.unit_names
    )
    total = sum(p.numel() for p in backbone.adapted.parameters())
    assert per_unit == total

    stem_names = {n for n, _ in backbone.adapted["stem"].named_parameters()}
    assert not any("blocks." in n for n in stem_names), (
        f"stem still reaches transformer blocks: {sorted(stem_names)[:5]}"
    )


# -- gate endpoints -----------------------------------------------------------


def test_gate_endpoints_are_bit_exact(backbone: TappedBackbone) -> None:
    """a=0 is the pretrained net and a=1 is the adapted net, exactly.

    Same anchor the ResNet backbone asserts. Run under eval() so no stochastic
    depth or dropout path can make the comparison flaky for reasons unrelated to
    the gate.
    """
    backbone.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        ref = backbone.forward_reference(x).embedding
        adp = backbone.forward_adapted(x).embedding
        zeros = backbone(x, gates=torch.zeros(backbone.n_units)).embedding
        ones = backbone(x, gates=torch.ones(backbone.n_units)).embedding
    assert torch.equal(zeros, ref)
    assert torch.equal(ones, adp)


def test_embedding_is_cls_token(backbone: TappedBackbone) -> None:
    """A CLS-trained checkpoint must be read at the CLS token, not mean-pooled."""
    backbone.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = backbone.forward_adapted(x)
        emb = out.embedding
        last = out.activations["norm"]
    assert emb.shape == (2, 384)
    assert last.dim() == 3  # (batch, tokens, channels)
    assert torch.equal(emb, last[:, 0])


# -- the partition ------------------------------------------------------------


def test_type_partition_is_exact(backbone: TappedBackbone) -> None:
    """LayerNorm + attention + MLP + patch-embedding = every parameter, once.

    Asserted on parameter *names*, not counts. Two selectors can open equal-sized
    disjoint-looking sets and still overlap; only the name sets settle it.
    """
    select_all(backbone)
    everything = _names(backbone)

    groups = {}
    for label, sel in (
        ("ln", select_layernorm),
        ("attn", select_attn),
        ("mlp", select_mlp),
        ("patch", select_patch_embed),
    ):
        sel(backbone)
        groups[label] = _names(backbone)
        assert groups[label], f"{label} selected nothing"

    labels = list(groups)
    for i, a in enumerate(labels):
        for b in labels[i + 1 :]:
            overlap = groups[a] & groups[b]
            assert not overlap, f"{a} and {b} both open {sorted(overlap)[:4]}"

    union = set().union(*groups.values())
    missing = everything - union
    assert not missing, f"partition misses {sorted(missing)}"
    assert union == everything


def test_cls_token_and_pos_embed_are_covered(backbone: TappedBackbone) -> None:
    """The two bare parameters no module-typed selector can see."""
    select_patch_embed(backbone)
    opened = _names(backbone)
    assert any(n.endswith("cls_token") for n in opened), sorted(opened)
    assert any(n.endswith("pos_embed") for n in opened), sorted(opened)


def test_ln_halves_are_disjoint_and_reconstruct(backbone: TappedBackbone) -> None:
    select_ln_scale(backbone)
    scale = _names(backbone)
    select_ln_shift(backbone)
    shift = _names(backbone)
    select_layernorm(backbone)
    both = _names(backbone)

    assert not (scale & shift)
    assert scale | shift == both
    assert all(n.endswith(".weight") for n in scale)
    assert all(n.endswith(".bias") for n in shift)


def test_ln_is_a_tiny_fraction(backbone: TappedBackbone) -> None:
    """LN-Tune's whole point is that it is nearly free; if this grows, something
    has started selecting Linear layers by accident."""
    select_all(backbone)
    total = _count(backbone)
    select_layernorm(backbone)
    frac = _count(backbone) / total
    assert 0.0005 < frac < 0.01, frac


def test_attn_is_smaller_than_mlp(backbone: TappedBackbone) -> None:
    """A ViT block's MLP holds more parameters than its attention (4x hidden
    against 3+1 projections). If this inverts, the two selectors are crossed."""
    select_attn(backbone)
    attn = _count(backbone)
    select_mlp(backbone)
    mlp = _count(backbone)
    assert 0 < attn < mlp


# -- no selector touches a buffer --------------------------------------------


def test_no_selector_opens_a_buffer(backbone: TappedBackbone) -> None:
    """LayerNorm has no running statistics, but the patch embedding is a Conv2d and
    future backbones may reintroduce buffers. Cheap to assert, expensive to miss."""
    for sel in (
        select_all,
        select_layernorm,
        select_ln_scale,
        select_ln_shift,
        select_attn,
        select_mlp,
        select_patch_embed,
    ):
        sel(backbone)
        for name, buf in backbone.adapted.named_buffers():
            assert not buf.requires_grad, f"{sel.__name__} opened buffer {name}"


def test_select_none_closes_everything(backbone: TappedBackbone) -> None:
    select_all(backbone)
    assert _count(backbone) > 0
    select_none(backbone)
    assert _count(backbone) == 0


def test_every_vit_baseline_is_a_distinct_experiment(backbone: TappedBackbone) -> None:
    """Mirror of the ResNet distinctness test, on a ViT.

    Same reasoning: footprints are parameter-name sets, not counts, because
    ln_scale_ft and ln_shift_ft select exactly as many parameters as each other
    while being different experiments.
    """
    from arc.train.baselines import BASELINES

    steps = 10
    footprints: dict[str, tuple[tuple, tuple]] = {}
    for key, spec in BASELINES.items():
        if spec.arch_family not in (None, "vit"):
            continue
        controller, select, spec_steps = spec.build(backbone, steps)
        select(backbone)
        trajectory = tuple(
            tuple(controller.step(i).tolist()) for i in range(max(spec_steps, 1))
        )
        footprints[key] = (tuple(sorted(_names(backbone))), trajectory)

    stepping = {k: v for k, v in footprints.items() if k != "frozen_proto"}
    assert len(set(stepping.values())) == len(stepping), (
        f"duplicate ViT baselines: {sorted(stepping)}"
    )
    for key, (names, _) in stepping.items():
        assert names, f"{key} selected nothing on a ViT"
