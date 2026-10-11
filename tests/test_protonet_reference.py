"""Cross-validate our prototypical head against a published reference.

Our head is correct as far as our own tests go, but it is unreviewed code
implementing a standard method -- and we deliberately departed from the original
paper by defaulting to cosine similarity and by dividing the squared-Euclidean metric
by the embedding dimension (see ProtoHead's docstring: the unnormalised form saturates
to exactly zero loss on 2048-d ImageNet features). Those are defensible changes, but
"trust us" is a weak position. Agreeing with an independent implementation is stronger.

Reference: easyfsl 1.5.0, `PrototypicalNetworks` / `FewShotClassifier`.

The logits deliberately differ in scale:

    easyfsl   euclidean:  -cdist(q, p)                  (unsquared, no temperature)
    ours      sqeuclidean: -cdist(q, p)^2 / dim * temp
    easyfsl   cosine:      cos(q, p)
    ours      cosine:      cos(q, p) * temp

Both of ours are strictly monotone transforms of the reference (temperature is
positive, and -d^2 is strictly decreasing in d for d >= 0), so *predictions* and
therefore accuracy must match exactly even though the raw logits do not. That is the
property worth pinning: a disagreement here would mean a genuine difference in method,
not a difference in scaling.

easyfsl is a development dependency, used only by this file.
"""

from __future__ import annotations

import pytest
import torch

from arc.heads.protonet import ProtoHead, prototypes

easyfsl = pytest.importorskip("easyfsl", reason="dev-only dependency")
from easyfsl.methods import PrototypicalNetworks  # noqa: E402


def _episode(
    n_way: int = 5,
    k_shot: int = 5,
    n_query: int = 15,
    dim: int = 2048,
    seed: int = 0,
    noise: float = 0.2,
):
    """Embeddings at a realistic ImageNet-ResNet-50 scale (2048-d, norm ~14).

    ``noise`` is deliberately small. In 2048 dimensions the within-class scatter is
    ``dim * noise^2``, so a seemingly modest noise of 3.0 gives ~18000 against a
    between-centre separation of only ~390 -- the classes overlap completely and the
    episode sits at chance. Comparing two implementations on an unsolvable task still
    passes, but proves much less. At 0.2 the scatter is ~82 and the task is separable,
    which is both realistic and a far stronger test.
    """
    g = torch.Generator().manual_seed(seed)
    centres = torch.randn(n_way, dim, generator=g)
    centres = centres / centres.norm(dim=1, keepdim=True) * 14.0

    support_y = torch.arange(n_way).repeat_interleave(k_shot)
    query_y = torch.arange(n_way).repeat_interleave(n_query)
    support = centres[support_y] + torch.randn(len(support_y), dim, generator=g) * noise
    query = centres[query_y] + torch.randn(len(query_y), dim, generator=g) * noise
    return support, support_y, query, query_y


def test_fixture_is_actually_solvable() -> None:
    """Guards the comparison above: if this episode were at chance, every agreement
    test would be vacuous."""
    support, support_y, query, query_y = _episode()
    logits = ProtoHead(metric="cosine")(support, support_y, query, n_way=5)
    acc = (logits.argmax(1) == query_y).float().mean().item()
    assert acc > 0.9, f"fixture is not separable (accuracy {acc:.2f}); tests would be weak"


def _reference(support, support_y, query, metric: str) -> torch.Tensor:
    """easyfsl logits, with the backbone bypassed so only the head logic is compared."""
    model = PrototypicalNetworks(backbone=torch.nn.Identity())
    model.process_support_set(support, support_y)
    if metric == "cosine":
        return model.cosine_distance_to_prototypes(query)
    return model.l2_distance_to_prototypes(query)


@pytest.mark.parametrize("k_shot", [1, 5, 20])
def test_prototypes_match_the_reference_exactly(k_shot: int) -> None:
    support, support_y, _, _ = _episode(k_shot=k_shot)
    ours = prototypes(support, support_y, n_way=5)

    model = PrototypicalNetworks(backbone=torch.nn.Identity())
    model.process_support_set(support, support_y)

    assert torch.allclose(ours, model.prototypes, atol=1e-4), (
        (ours - model.prototypes).abs().max().item()
    )


@pytest.mark.parametrize("metric", ["sqeuclidean", "cosine"])
@pytest.mark.parametrize("k_shot", [1, 5, 20])
def test_predictions_match_the_reference(metric: str, k_shot: int) -> None:
    support, support_y, query, _ = _episode(k_shot=k_shot)

    ours = ProtoHead(metric=metric)(support, support_y, query, n_way=5)
    theirs = _reference(support, support_y, query, metric)

    assert torch.equal(ours.argmax(1), theirs.argmax(1)), (
        f"{metric} k={k_shot}: predictions disagree on "
        f"{(ours.argmax(1) != theirs.argmax(1)).sum().item()} of {len(query)} queries"
    )


@pytest.mark.parametrize("metric", ["sqeuclidean", "cosine"])
def test_accuracy_matches_the_reference_across_many_episodes(metric: str) -> None:
    """One episode could agree by luck; 50 independent ones could not."""
    for seed in range(50):
        support, support_y, query, query_y = _episode(k_shot=5, seed=seed)
        ours = ProtoHead(metric=metric)(support, support_y, query, n_way=5)
        theirs = _reference(support, support_y, query, metric)
        a = (ours.argmax(1) == query_y).float().mean().item()
        b = (theirs.argmax(1) == query_y).float().mean().item()
        assert a == b, f"{metric} seed {seed}: ours {a:.4f} vs reference {b:.4f}"


@pytest.mark.parametrize("metric", ["sqeuclidean", "cosine"])
def test_our_logits_are_a_monotone_transform_of_the_reference(metric: str) -> None:
    """Stronger than matching argmax: the whole ranking over classes must agree, which
    is what makes the temperature and dimension scaling provably harmless."""
    support, support_y, query, _ = _episode(k_shot=5)
    ours = ProtoHead(metric=metric)(support, support_y, query, n_way=5)
    theirs = _reference(support, support_y, query, metric)

    for row_ours, row_theirs in zip(ours, theirs):
        assert torch.equal(row_ours.argsort(), row_theirs.argsort()), (
            f"{metric}: class ordering differs within a query row"
        )


def test_reference_euclidean_would_saturate_where_ours_does_not() -> None:
    """Documents why we diverged from the reference's unnormalised metric.

    easyfsl returns raw -cdist, which is fine for its own use (it reports accuracy, not
    a training loss). Feeding squared distances at this scale into cross-entropy is
    what underflowed to exactly zero gradient and silently broke our first LR sweep.
    """
    import torch.nn.functional as F

    support, support_y, query, query_y = _episode(k_shot=5)
    protos = prototypes(support, support_y, 5)

    naive = -torch.cdist(query, protos).pow(2)  # the original ProtoNet formulation
    assert F.cross_entropy(naive, query_y).item() == 0.0

    ours = ProtoHead(metric="sqeuclidean")(support, support_y, query, n_way=5)
    assert F.cross_entropy(ours, query_y).item() > 1e-4
