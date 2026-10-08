"""Regression tests for the inner adaptation loop.

Every test here corresponds to a defect that silently produced plausible-looking but
meaningless numbers. None of them would have been caught by checking that the code ran.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from arc.backbones.tapped import TappedBackbone
from arc.data.base import Episode
from arc.heads.protonet import ProtoHead, prototypes
from arc.train.augment import random_resized_crop_flip
from arc.train.objectives import inner_loss, leave_one_out_prototypes


def _episode(n_way: int = 5, k_shot: int = 1, n_query: int = 4, size: int = 64) -> Episode:
    torch.manual_seed(0)
    return Episode(
        support_x=torch.randn(n_way * k_shot, 3, size, size),
        support_y=torch.arange(n_way).repeat_interleave(k_shot),
        query_x=torch.randn(n_way * n_query, 3, size, size),
        query_y=torch.arange(n_way).repeat_interleave(n_query),
        global_classes=torch.arange(n_way),
        episode_id=0,
    )


def _backbone() -> TappedBackbone:
    torch.manual_seed(0)
    net = TappedBackbone(arch="resnet18", pretrained=False)
    net.eval()
    return net


# -- the degenerate-objective defect ----------------------------------------

def _loss_and_grad(net: TappedBackbone, objective: str, k_shot: int) -> tuple[float, float]:
    head, ep = ProtoHead(), _episode(k_shot=k_shot)
    net.zero_grad(set_to_none=True)
    net.adapted.requires_grad_(True)
    torch.manual_seed(1)  # fix the augmentation draw so the comparison is like-for-like
    loss = inner_loss(
        lambda b: net(b).embedding, ep.support_x, ep.support_y, 5, head, objective=objective
    )
    loss.backward()
    gnorm = sum(
        p.grad.pow(2).sum().item() for p in net.adapted.parameters() if p.grad is not None
    ) ** 0.5
    return loss.item(), gnorm


def test_self_objective_is_exactly_degenerate_at_one_shot() -> None:
    """The mathematical core of the defect, independent of weights or metric.

    With prototypes built from the support set itself and k=1, every point *is* its own
    prototype. Its correct-class logit is therefore pinned at the metric's maximum for
    all inputs, no matter what the network does -- so no gradient can ever improve it.
    """
    head = ProtoHead(metric="cosine")
    emb = torch.randn(5, 64) * 14.0  # arbitrary scale; the property is scale-free
    labels = torch.arange(5)
    logits = head.score(emb, prototypes(emb, labels, 5))
    correct = logits.gather(1, labels[:, None]).squeeze(1)

    assert torch.allclose(correct, correct[0].expand(5), atol=1e-4), (
        "correct-class logits should be pinned to a constant -- that is the defect"
    )
    assert torch.allclose(correct, logits.max(dim=1).values, atol=1e-4)


def test_head_does_not_saturate_on_realistic_pretrained_feature_scale() -> None:
    """The bug that survived the first fix.

    ResNet-50 penultimate features are 2048-d with norms around 14, giving squared
    distances from ~6 to ~419 on EuroSAT. Under the original unnormalised
    squared-Euclidean metric at temperature 1, the softmax saturated to exactly 1.0 and
    cross-entropy underflowed to exactly 0.0 -- zero gradient regardless of objective or
    learning rate. Any metric we ship must stay in a trainable range at this scale.
    """
    torch.manual_seed(0)
    dim, n_way = 2048, 5
    protos = torch.randn(n_way, dim)
    protos = protos / protos.norm(dim=1, keepdim=True) * 14.0
    query = protos[torch.arange(n_way)] + torch.randn(n_way, dim) * 2.0
    labels = torch.arange(n_way)

    for metric in ("cosine", "sqeuclidean"):
        head = ProtoHead(metric=metric)
        loss = F.cross_entropy(head.score(query, protos), labels)
        assert loss.item() > 1e-4, f"{metric} saturated: loss {loss.item():.3e}"
        assert torch.isfinite(loss), f"{metric} produced {loss.item()}"


def test_unnormalised_squared_distance_would_still_saturate() -> None:
    """Pins the reason sqeuclidean divides by the embedding dimension."""
    torch.manual_seed(0)
    dim, n_way = 2048, 5
    protos = torch.randn(n_way, dim)
    protos = protos / protos.norm(dim=1, keepdim=True) * 14.0
    query = protos + torch.randn(n_way, dim) * 2.0
    naive = -torch.cdist(query, protos).pow(2)  # the original formulation
    assert F.cross_entropy(naive, torch.arange(n_way)).item() == 0.0


@pytest.mark.parametrize("k_shot", [1, 2, 5])
def test_two_view_objective_has_real_gradient_at_every_k(k_shot: int) -> None:
    """The fix: a second augmented view makes the task non-trivial, including at k=1."""
    net, head, ep = _backbone(), ProtoHead(), _episode(k_shot=k_shot)
    net.adapted.requires_grad_(True)
    loss = inner_loss(
        lambda b: net(b).embedding, ep.support_x, ep.support_y, 5, head, objective="two_view"
    )
    assert loss.item() > 1e-3, f"two_view loss collapsed at k={k_shot}: {loss.item():.3e}"

    loss.backward()
    gnorm = sum(
        p.grad.pow(2).sum().item() for p in net.adapted.parameters() if p.grad is not None
    ) ** 0.5
    assert gnorm > 0, f"zero gradient at k={k_shot} -- nothing would learn"


def test_leave_one_out_excludes_the_point_from_its_own_prototype() -> None:
    emb = torch.tensor([[0.0, 0.0], [2.0, 0.0], [0.0, 10.0], [0.0, 20.0]])
    labels = torch.tensor([0, 0, 1, 1])
    protos = leave_one_out_prototypes(emb, labels, n_way=2)
    # Point 0's own-class prototype must be point 1 alone.
    assert torch.allclose(protos[0, 0], torch.tensor([2.0, 0.0]))
    assert torch.allclose(protos[1, 0], torch.tensor([0.0, 0.0]))
    # The other class is unaffected and keeps its full mean.
    assert torch.allclose(protos[0, 1], torch.tensor([0.0, 15.0]))


def test_leave_one_out_rejects_one_shot() -> None:
    emb, labels = torch.randn(2, 4), torch.tensor([0, 1])
    with pytest.raises(ValueError, match="k_shot >= 2"):
        leave_one_out_prototypes(emb, labels, n_way=2)


# -- the BatchNorm-pollution defect ------------------------------------------

def _bn_stats(module: torch.nn.Module) -> torch.Tensor:
    parts = [
        t.flatten()
        for m in module.modules()
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)
        for t in (m.running_mean, m.running_var)
        if t is not None
    ]
    return torch.cat(parts)


def test_bn_mode_eval_leaves_running_statistics_untouched() -> None:
    """With lr=0 this previously still moved BN statistics by an L2 of 162, which was
    the entire content of a 120-config learning-rate sweep."""
    from arc.train.episodic import EpisodeRunner, StaticController

    net = _backbone()
    runner = EpisodeRunner(net, device="cpu", inner_steps=5, inner_lr=0.0,
                           amp=False, bn_mode="eval")
    before = _bn_stats(net.adapted).clone()
    runner.run_episode(_episode(k_shot=5), StaticController(torch.ones(net.n_units)))
    assert torch.equal(_bn_stats(net.adapted), before)


def test_bn_mode_train_does_move_running_statistics() -> None:
    """The transductive variant must still be reachable -- it is a real technique, just
    one that has to be chosen explicitly."""
    from arc.train.episodic import EpisodeRunner, StaticController

    net = _backbone()
    runner = EpisodeRunner(net, device="cpu", inner_steps=5, inner_lr=0.0,
                           amp=False, bn_mode="train")
    before = _bn_stats(net.adapted).clone()
    runner.run_episode(_episode(k_shot=5), StaticController(torch.ones(net.n_units)))
    assert not torch.equal(_bn_stats(net.adapted), before)


def test_learning_rate_actually_changes_the_outcome() -> None:
    """The headline symptom: a 300x LR range produced bit-identical accuracy.

    Parameter drift must be strictly monotone in the learning rate, and exactly zero
    when lr=0.
    """
    from arc.train.episodic import EpisodeRunner, StaticController, select_all

    drifts = []
    for lr in (0.0, 1e-5, 1e-3):
        net = _backbone()
        runner = EpisodeRunner(net, device="cpu", inner_steps=3, inner_lr=lr,
                               amp=False, bn_mode="eval")
        runner.run_episode(
            _episode(k_shot=5), StaticController(torch.ones(net.n_units)), select=select_all
        )
        drifts.append(sum(net.parameter_drift().values()))

    assert drifts[0] == 0.0, "lr=0 must not move parameters"
    assert drifts[1] > 0.0, "lr=1e-5 moved nothing -- the loop is not learning"
    assert drifts[2] > drifts[1], f"drift not monotone in lr: {drifts}"


# -- augmentation -------------------------------------------------------------

def test_augmentation_changes_images_but_preserves_shape_and_scale() -> None:
    x = torch.randn(8, 3, 32, 32)
    a = random_resized_crop_flip(x)
    assert a.shape == x.shape
    assert not torch.allclose(a, x)
    # Reflection padding must not introduce wild values outside the input's range.
    assert a.abs().max() <= x.abs().max() + 1e-4


def test_augmentation_is_independent_across_the_batch() -> None:
    """A single shared crop per batch would make the two views trivially alignable."""
    x = torch.randn(1, 3, 32, 32).repeat(16, 1, 1, 1)
    a = random_resized_crop_flip(x)
    first = a[0:1]
    assert not torch.allclose(a, first.expand_as(a)), "all samples got the same crop"
