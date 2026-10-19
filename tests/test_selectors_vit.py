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


def test_last_layer_ft_is_the_last_block_not_the_trailing_norm(
    backbone: TappedBackbone,
) -> None:
    """``unit_names[-1]`` is the pre-head LayerNorm on a ViT, which has ~1.5k
    parameters. If last_layer_ft resolved to it, the baseline would be effectively
    frozen while still reporting an accuracy."""
    from arc.train.baselines import BASELINES

    _, select, _ = BASELINES["last_layer_ft"].build(backbone, 10)
    select(backbone)
    opened = _names(backbone)
    assert opened, "last_layer_ft selected nothing"
    assert all(n.startswith("block_11.") for n in opened), sorted(opened)
    # A whole transformer block, not just a norm: attention and MLP must be in there.
    assert any(".attn." in n for n in opened)
    assert any(".mlp." in n for n in opened)


# -- the capacity sweep -------------------------------------------------------
#
# These exist because the sweep's whole point is that n is proportional to the
# adapted fraction. A selector that opened the wrong blocks, or silently opened all
# of them, would still produce a smooth-looking curve -- and a smooth curve is
# exactly what the sweep is looking for, so the failure would confirm the
# hypothesis rather than announce itself.


def test_attn_last_n_opens_exactly_the_last_n_blocks(backbone: TappedBackbone) -> None:
    from arc.train.episodic import select_attn_last_n

    for n in (1, 2, 3, 4, 6, 12):
        select_attn_last_n(n)(backbone)
        opened = _names(backbone)
        assert opened, f"n={n} selected nothing"
        blocks = {int(name.split(".")[0].removeprefix("block_")) for name in opened}
        assert blocks == set(range(12 - n, 12)), f"n={n} opened blocks {sorted(blocks)}"
        assert all(".attn." in f".{name}" for name in opened), sorted(opened)[:4]


def test_attn_last_n_is_linear_in_n(backbone: TappedBackbone) -> None:
    """Every block holds the same number of attention parameters, so the count must
    be exactly proportional. If it is not, the sweep's x-axis is wrong."""
    from arc.train.episodic import select_attn_last_n

    select_attn_last_n(1)(backbone)
    per_block = _count(backbone)
    assert per_block > 0
    for n in (2, 3, 4, 6, 8, 12):
        select_attn_last_n(n)(backbone)
        assert _count(backbone) == n * per_block, n


def test_attn_last_12_is_exactly_select_attn(backbone: TappedBackbone) -> None:
    """The sweep's right-hand endpoint must be the measurement it is extending, not
    a near-miss. Compared on names, not counts."""
    from arc.train.episodic import select_attn_last_n

    select_attn(backbone)
    whole = _names(backbone)
    select_attn_last_n(12)(backbone)
    assert _names(backbone) == whole


def test_attn_last_n_nests(backbone: TappedBackbone) -> None:
    """Each n must be a superset of the one below; a suffix sweep that is not nested
    is varying which parameters as well as how many."""
    from arc.train.episodic import select_attn_last_n

    previous: set[str] = set()
    for n in (1, 2, 3, 4, 6, 8, 12):
        select_attn_last_n(n)(backbone)
        current = _names(backbone)
        assert previous < current, f"n={n} is not a strict superset of the previous"
        previous = current


def test_attn_last_n_refuses_more_blocks_than_exist(backbone: TappedBackbone) -> None:
    """Asking for 24 of 12 must raise, not quietly return all 12 -- that would report
    a 24-block number that is really the 12-block one."""
    from arc.train.episodic import select_attn_last_n

    with pytest.raises(ValueError, match="at least 24 blocks"):
        select_attn_last_n(24)(backbone)
    with pytest.raises(ValueError, match="at least 1"):
        select_attn_last_n(0)


def test_attn_last_n_baselines_are_registered_and_distinct(backbone: TappedBackbone) -> None:
    from arc.train.baselines import BASELINES

    footprints = {}
    for n in (1, 2, 3, 4, 6, 8):
        key = f"attn_last{n}_ft"
        assert key in BASELINES, key
        _, select, _ = BASELINES[key].build(backbone, 10)
        select(backbone)
        footprints[key] = tuple(sorted(_names(backbone)))
    assert len(set(footprints.values())) == len(footprints)

    # n=12 is not registered: it is attn_only_ft under another name, and registering
    # both is how a table quotes one measurement twice as a replication.
    assert "attn_last12_ft" not in BASELINES

    # The description must agree with what the selector opens. A loop-registered
    # family makes a mismatch unlikely; asserting it makes it impossible.
    for n in (1, 2, 3, 4, 6, 8):
        spec = BASELINES[f"attn_last{n}_ft"]
        assert f"last {n} transformer block" in spec.description, spec.description


# -- the positional sweep -----------------------------------------------------


def test_attn_block_opens_exactly_that_block(backbone: TappedBackbone) -> None:
    from arc.train.episodic import select_attn_block

    for i in (0, 2, 5, 8, 11):
        select_attn_block(i)(backbone)
        opened = _names(backbone)
        assert opened, f"block {i} selected nothing"
        blocks = {int(n.split(".")[0].removeprefix("block_")) for n in opened}
        assert blocks == {i}, f"index {i} opened blocks {sorted(blocks)}"
        assert all(".attn." in f".{n}" for n in opened), sorted(opened)[:4]


def test_attn_block_is_the_same_size_at_every_depth(backbone: TappedBackbone) -> None:
    """The point of a positional sweep is that only position varies. If the counts
    differed, a difference in accuracy could be about size instead."""
    from arc.train.episodic import select_attn_block

    counts = set()
    for i in range(12):
        select_attn_block(i)(backbone)
        counts.add(_count(backbone))
    assert len(counts) == 1, counts


def test_attn_block_11_is_attn_last_1(backbone: TappedBackbone) -> None:
    """The positional sweep's right-hand end must be the suffix sweep's left-hand
    end, exactly -- that shared point is what makes the two curves comparable."""
    from arc.train.episodic import select_attn_block, select_attn_last_n

    select_attn_last_n(1)(backbone)
    suffix = _names(backbone)
    select_attn_block(11)(backbone)
    assert _names(backbone) == suffix


def test_attn_block_refuses_an_index_past_the_end(backbone: TappedBackbone) -> None:
    from arc.train.episodic import select_attn_block

    with pytest.raises(ValueError, match="indices 0..11"):
        select_attn_block(12)(backbone)
    with pytest.raises(ValueError, match="non-negative"):
        select_attn_block(-1)


def test_attn_block_baselines_registered_without_duplicating_block_11(
    backbone: TappedBackbone,
) -> None:
    from arc.train.baselines import BASELINES

    for i in (0, 2, 5, 8):
        key = f"attn_block{i}_ft"
        assert key in BASELINES, key
        _, select, _ = BASELINES[key].build(backbone, 10)
        select(backbone)
        assert _names(backbone)
    # block 11 is attn_last1_ft; registering both would quote one measurement twice.
    assert "attn_block11_ft" not in BASELINES


# -- the MLP positional sweep --------------------------------------------------
#
# The attention positional sweep (RESULTS_VIT.md §3.3.2) is the largest effect on
# this arm, and it leaves one question open that decides what the effect is ABOUT:
# whether a well-placed block of any type works, or specifically attention. These
# assert the counterpart selector is a clean instrument for that -- same depths,
# same single-block shape, only the sublayer differs.


def test_mlp_block_opens_exactly_that_block(backbone: TappedBackbone) -> None:
    from arc.train.episodic import select_mlp_block

    for i in (0, 2, 5, 8, 11):
        select_mlp_block(i)(backbone)
        opened = _names(backbone)
        assert opened, f"block {i} selected nothing"
        blocks = {int(n.split(".")[0].removeprefix("block_")) for n in opened}
        assert blocks == {i}, f"index {i} opened blocks {sorted(blocks)}"
        assert all(".mlp." in f".{n}" for n in opened), sorted(opened)[:4]


def test_mlp_block_is_the_same_size_at_every_depth(backbone: TappedBackbone) -> None:
    """Same reason as the attention version: only position may vary within the
    sweep, or a difference across depth could be about size."""
    from arc.train.episodic import select_mlp_block

    counts = set()
    for i in range(12):
        select_mlp_block(i)(backbone)
        counts.add(_count(backbone))
    assert len(counts) == 1, counts


def test_mlp_block_opens_no_attention_and_vice_versa(
    backbone: TappedBackbone,
) -> None:
    """The two positional sweeps must be disjoint at every depth. If either leaked
    into the other, the comparison that separates 'about attention' from 'about
    depth' would be comparing two things that overlap."""
    from arc.train.episodic import select_attn_block, select_mlp_block

    for i in (0, 5, 11):
        select_mlp_block(i)(backbone)
        mlp = _names(backbone)
        select_attn_block(i)(backbone)
        attn = _names(backbone)
        assert mlp and attn
        assert not (mlp & attn), sorted(mlp & attn)[:4]


def test_mlp_block_is_twice_attn_block(backbone: TappedBackbone) -> None:
    """An MLP block has exactly twice a block's attention parameters, matching the
    whole-type ratio asserted by test_attn_is_smaller_than_mlp.

    This is the one thing the MLP sweep does NOT hold fixed against the attention
    sweep, and it is recorded as an assertion rather than a comment because the
    interpretation in select_mlp_block's docstring depends on the direction: a flat
    MLP curve matters precisely because the intervention is the larger one.
    """
    from arc.train.episodic import select_attn_block, select_mlp_block

    select_attn_block(5)(backbone)
    attn = _count(backbone)
    select_mlp_block(5)(backbone)
    mlp = _count(backbone)
    # Not exactly 2x: attention carries a square qkv/proj pair plus biases, the MLP
    # a 4x-wide pair plus biases, so assert the ratio rather than equality of double.
    assert 1.9 < mlp / attn < 2.1, (attn, mlp, mlp / attn)


def test_mlp_blocks_reconstruct_select_mlp(backbone: TappedBackbone) -> None:
    """Every block's MLP, unioned, is exactly select_mlp -- so the positional sweep
    partitions the whole-type baseline rather than sampling some subset of it."""
    from arc.train.episodic import select_mlp, select_mlp_block

    select_mlp(backbone)
    whole = _names(backbone)
    union: set[str] = set()
    for i in range(12):
        select_mlp_block(i)(backbone)
        union |= _names(backbone)
    assert union == whole, sorted(union ^ whole)[:6]


def test_mlp_block_refuses_an_index_past_the_end(backbone: TappedBackbone) -> None:
    from arc.train.episodic import select_mlp_block

    with pytest.raises(ValueError, match="indices 0..11"):
        select_mlp_block(12)(backbone)
    with pytest.raises(ValueError, match="non-negative"):
        select_mlp_block(-1)


def test_mlp_block_baselines_are_registered_at_all_five_depths(
    backbone: TappedBackbone,
) -> None:
    from arc.train.baselines import BASELINES

    for i in (0, 2, 5, 8, 11):
        key = f"mlp_block{i}_ft"
        assert key in BASELINES, key
        spec = BASELINES[key]
        assert spec.arch_family == "vit"
        assert f"block {i}" in spec.description, spec.description
        _, select, _ = spec.build(backbone, 10)
        select(backbone)
        opened = _names(backbone)
        assert opened
        assert all(".mlp." in f".{n}" for n in opened), sorted(opened)[:4]


def test_mlp_block_11_is_not_a_duplicate_of_anything(
    backbone: TappedBackbone,
) -> None:
    """block 11 IS registered here, unlike attn_block11_ft, because there is no MLP
    suffix family for it to duplicate. Asserted so that adding one later fails this
    test rather than silently creating two names for one experiment."""
    from arc.train.baselines import BASELINES

    assert "mlp_block11_ft" in BASELINES
    assert "mlp_last1_ft" not in BASELINES
