"""Inner-loop adaptation objectives.

A diagnostic (``experiments/diagnose_inner_loop.py``) showed the obvious objective --
embed the support set, build prototypes from it, and classify that same support set --
is **degenerate**, catastrophically so at 1-shot:

    k=1  loss = 0.000e+00   grad_norm = 0.000e+00   param_delta = 0.000e+00
    k=5  loss = 1.562e-05   grad_norm = 7.101e-03

At ``k=1`` each support embedding *is* its own prototype, so its distance to the correct
class is exactly zero, the softmax saturates, the loss underflows to zero and the
gradient is identically zero. Nothing learns, at any learning rate. At ``k=5`` the point
still contributes 1/k of its own prototype, leaving a near-zero loss and a vanishing
gradient. This is why an entire LR sweep returned accuracies identical to four decimals
across a 300x learning-rate range.

Two non-degenerate objectives are provided.

``two_view`` (default)
    Build prototypes from one augmented view of the support set and classify a
    *different* augmented view. Well-defined at every k including k=1, and it is what
    the cross-domain few-shot fine-tuning literature actually does.

``loo``
    Leave-one-out prototypes: classify support point i against a prototype of its own
    class computed from the other k-1 members. Exact and augmentation-free, but
    undefined at k=1, so it is available for k>=2 as an ablation that isolates how much
    of the gain comes from augmentation.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F

from arc.heads.protonet import ProtoHead, prototypes
from arc.train.augment import random_resized_crop_flip

ObjectiveName = Literal["two_view", "loo", "self"]


def leave_one_out_prototypes(
    embeddings: torch.Tensor, labels: torch.Tensor, n_way: int
) -> torch.Tensor:
    """Prototypes with each point excluded from its own class mean.

    Returns ``(n, n_way, dim)``: for point ``i``, row ``i`` holds the prototype set it
    should be scored against. Only its own class entry differs from the global
    prototype, so the whole thing is one gather plus one correction rather than ``n``
    recomputations.
    """
    n, dim = embeddings.shape
    sums = embeddings.new_zeros(n_way, dim)
    counts = embeddings.new_zeros(n_way)
    sums.index_add_(0, labels, embeddings)
    counts.index_add_(0, labels, torch.ones_like(labels, dtype=embeddings.dtype))

    if (counts < 2).any():
        thin = torch.nonzero(counts < 2).flatten().tolist()
        raise ValueError(
            f"leave-one-out needs k_shot >= 2; classes {thin} have fewer than 2 support "
            f"examples. Use objective='two_view' for 1-shot."
        )

    protos = (sums / counts.unsqueeze(1)).unsqueeze(0).expand(n, n_way, dim).clone()
    own = labels
    own_counts = counts[own].unsqueeze(1)
    own_sums = sums[own]
    protos[torch.arange(n, device=embeddings.device), own] = (
        own_sums - embeddings
    ) / (own_counts - 1)
    return protos


def inner_loss(
    forward,
    support_x: torch.Tensor,
    support_y: torch.Tensor,
    n_way: int,
    head: ProtoHead,
    objective: ObjectiveName = "two_view",
    augment_scale: tuple[float, float] = (0.6, 1.0),
) -> torch.Tensor:
    """Compute the support-set adaptation loss.

    ``forward`` maps a batch of images to embeddings; the caller supplies it already
    bound to the current gate vector.
    """
    if objective == "two_view":
        view_a = random_resized_crop_flip(support_x, scale=augment_scale)
        view_b = random_resized_crop_flip(support_x, scale=augment_scale)
        # One forward over the concatenation keeps BatchNorm (when it is in train mode
        # for the transductive variant) seeing a single consistent batch.
        both = forward(torch.cat([view_a, view_b], dim=0))
        emb_a, emb_b = both.chunk(2, dim=0)
        protos = prototypes(emb_a, support_y, n_way)
        logits = head.score(emb_b, protos)
        return F.cross_entropy(logits, support_y)

    if objective == "loo":
        emb = forward(support_x)
        protos = leave_one_out_prototypes(emb, support_y, n_way)
        logits = head.score(emb, protos)
        return F.cross_entropy(logits, support_y)

    if objective == "self":
        # Retained only so the degenerate baseline can be reproduced on demand; it is
        # not a valid training objective. See the module docstring.
        emb = forward(support_x)
        protos = prototypes(emb, support_y, n_way)
        logits = head.score(emb, protos)
        return F.cross_entropy(logits, support_y)

    raise ValueError(f"unknown objective {objective!r}")
