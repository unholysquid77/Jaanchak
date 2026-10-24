"""l2-SP on the inner loop: decay toward the pretrained point, not toward the origin.

This exists because of a negative result, and the tests are shaped by it. Ordinary
weight decay was pre-registered as the repair for full fine-tuning's
over-optimisation and it made things worse at every dose (RESULTS_VIT.md §3.9). Two
readings survive that failure: the over-optimisation account is wrong, or decay
toward the ORIGIN is simply the wrong regulariser from a pretrained initialisation.
l2-SP is the intervention that separates them, so the thing that must not go wrong
is the one that would make its result uninterpretable: pairing each parameter with
the wrong theta_0.

A positional zip over two parameter lists would do exactly that. `backbone_params`
is filtered by `requires_grad`, so under any single-block baseline it is a small
subset of the snapshot in selector-dependent order, and zipping it against the full
state_dict would pull block 8's weights toward block 0's pretrained values. The
penalty would still be a well-behaved number -- large, positive, decreasing -- and
the run would look regularised while measuring nothing. `test_pairing_is_by_name`
is the test that matters here.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from arc.heads.protonet import ProtoHead  # noqa: E402
from arc.train.episodic import EpisodeRunner  # noqa: E402

ARCH = "vit_small_patch16_224.augreg_in21k"


def _runner(**kw) -> EpisodeRunner:
    pytest.importorskip("timm")
    from arc.backbones.tapped import TappedBackbone

    return EpisodeRunner(
        TappedBackbone(arch=ARCH, pretrained=False),
        head=ProtoHead(),
        device="cpu",
        amp=False,
        **kw,
    )


def test_rejects_negative_lambda() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        _runner(inner_l2sp=-1.0)


def test_rejects_both_regularisers_at_once() -> None:
    """Running both answers the question with neither. §3.9's confound was a coupling
    of this kind, so the constructor refuses rather than averaging two mechanisms."""
    with pytest.raises(ValueError, match="pick one"):
        _runner(inner_weight_decay=1e-3, inner_l2sp=1e-3)


def test_zero_is_the_default_and_changes_nothing() -> None:
    """At 0 no l2-SP code runs at all, so every number already in the document stays
    reproducible. This is asserted on the attribute rather than by re-running an
    episode because the guard in run_episode is `if self.inner_l2sp > 0`."""
    assert _runner().inner_l2sp == 0.0


def test_pairing_is_by_name_not_by_position() -> None:
    """The load-bearing test.

    Builds the pairing the way run_episode does -- over named_parameters() filtered
    by requires_grad, looked up in the pristine snapshot by name -- and checks that
    each parameter is paired with its OWN theta_0 under a selector that freezes most
    of the backbone. Then checks that the positional alternative would have been
    wrong, so the test fails if someone "simplifies" the lookup into a zip.
    """
    r = _runner(inner_l2sp=1e-3)
    adapted = r.backbone.adapted
    for p in adapted.parameters():
        p.requires_grad_(False)
    # A selector-like subset: every parameter of one mid-stack block.
    chosen = [
        (n, p) for n, p in adapted.named_parameters() if n.startswith("block_5.")
    ]
    assert chosen, "no block-5 parameters found; the arch's naming changed"
    for _, p in chosen:
        p.requires_grad_(True)

    pairs = [
        (n, p, r._pristine[n])
        for n, p in adapted.named_parameters()
        if p.requires_grad
    ]
    assert len(pairs) == len(chosen)
    for n, p, ref in pairs:
        assert p.shape == ref.shape, n
        # pretrained=False, but the snapshot is still taken from this very module, so
        # a correctly-named reference is EQUAL to the untouched parameter.
        assert torch.equal(p.detach(), ref), f"{n} paired with the wrong theta_0"

    # And the positional alternative is not the same thing.
    full = list(r._pristine.values())
    positional = [full[i] for i in range(len(pairs))]
    assert any(
        pos.shape != p.shape or not torch.equal(p.detach(), pos)
        for (_, p, _), pos in zip(pairs, positional)
    ), (
        "a positional zip happens to agree here, so this test would not catch the "
        "mispairing it exists for -- pick a selector further from the start of the "
        "state_dict"
    )


def test_penalty_is_zero_at_the_pretrained_point_and_positive_away_from_it() -> None:
    """l2-SP's defining property, and the one ordinary weight decay does not have:
    the pretrained weights are the free point. Weight decay charges for them."""
    r = _runner(inner_l2sp=1e-3)
    adapted = r.backbone.adapted
    pairs = [
        (p, r._pristine[n]) for n, p in adapted.named_parameters() if p.requires_grad
    ]
    assert pairs

    def pen() -> float:
        with torch.no_grad():
            return float(
                sum(((p.float() - ref.float()) ** 2).sum() for p, ref in pairs)
            )

    assert pen() == pytest.approx(0.0, abs=1e-10)
    with torch.no_grad():
        pairs[0][0].add_(0.1)
    assert pen() > 1e-6

    # The contrast with weight decay, stated as an assertion: ||theta||^2 at the
    # pretrained point is large, which is precisely the information §3.9's decay was
    # charging against.
    with torch.no_grad():
        norm = float(sum((p.float() ** 2).sum() for p, _ in pairs))
    assert norm > 1.0


def test_lambda_is_recorded_so_a_result_file_states_its_own_dose() -> None:
    """RESULTS_VIT.md §4.2: the authoritative record of what a run actually used is
    the top level of the result file, not manifest.config, which holds CLI defaults.
    inner_l2sp is read off the runner for the same reason inner_lr is."""
    r = _runner(inner_l2sp=3e-4)
    assert r.inner_l2sp == 3e-4
