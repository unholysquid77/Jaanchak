"""Few-shot classification heads.

The backbone is the contribution; these are deliberately standard so that any
improvement is attributable to representation control rather than head engineering.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def prototypes(embeddings: torch.Tensor, labels: torch.Tensor, n_way: int) -> torch.Tensor:
    """Class means of the support embeddings. Shape ``(n_way, dim)``.

    Uses an index_add rather than a Python loop so this stays cheap inside the inner
    adaptation loop, where it runs once per step per episode.
    """
    dim = embeddings.size(1)
    sums = embeddings.new_zeros(n_way, dim)
    counts = embeddings.new_zeros(n_way)
    sums.index_add_(0, labels, embeddings)
    counts.index_add_(0, labels, torch.ones_like(labels, dtype=embeddings.dtype))
    if (counts == 0).any():
        missing = torch.nonzero(counts == 0).flatten().tolist()
        raise ValueError(f"no support examples for classes {missing}")
    return sums / counts.unsqueeze(1)


class ProtoHead(nn.Module):
    """Prototypical network head (Snell et al., 2017).

    Parameters
    ----------
    metric
        ``"cosine"`` is the default, and that is a deliberate departure from the
        original paper. ProtoNet used squared Euclidean distance over a 64-dimensional
        embedding trained from scratch, where distances are O(1). Over the 2048-d
        penultimate features of an ImageNet ResNet-50 they are not: measured on EuroSAT,
        squared distances span 5.6 to 419 with a correct-class value of ~11. At
        temperature 1 that drives the softmax to *exactly* 1.0 and cross-entropy
        underflows to *exactly* 0.0 in fp32 -- zero loss, zero gradient, no learning at
        any learning rate, whatever the objective. Cosine similarity is bounded in
        [-1, 1] and does not have this failure mode.

        ``"sqeuclidean"`` remains available as an ablation and divides by the embedding
        dimension to keep its logits O(1); it still needs a larger temperature than
        cosine to be competitive.
    temperature
        Learnable logit scale. Defaults to 10 for cosine (whose raw range is too narrow
        to produce confident predictions) and 10 for the dimension-normalised
        squared-Euclidean metric.
    """

    def __init__(self, metric: str = "cosine", temperature: float | None = None) -> None:
        super().__init__()
        if metric not in {"sqeuclidean", "cosine"}:
            raise ValueError(f"unknown metric {metric!r}")
        self.metric = metric
        init = 10.0 if temperature is None else temperature
        self.log_temperature = nn.Parameter(torch.tensor(float(init)).log())

    def score(self, query: torch.Tensor, protos: torch.Tensor) -> torch.Tensor:
        """Logits of ``query`` against ``protos``.

        Accepts shared prototypes ``(n_way, dim)`` or per-point prototypes
        ``(n, n_way, dim)`` as produced by leave-one-out. Both the head and the inner
        objectives route through here: the metric previously lived in two places, and
        two implementations of one formula is a defect waiting to happen.
        """
        temperature = self.log_temperature.exp()
        per_point = protos.ndim == 3

        if self.metric == "cosine":
            q = F.normalize(query, dim=-1)
            p = F.normalize(protos, dim=-1)
            logits = torch.einsum("nd,nkd->nk", q, p) if per_point else q @ p.t()
        else:
            if per_point:
                sq = (query.unsqueeze(1) - protos).pow(2).sum(dim=2)
            else:
                # Avoids materialising the (n, n_way, dim) difference tensor.
                sq = torch.cdist(query, protos).pow(2)
            logits = -sq / query.size(-1)

        return logits * temperature

    def forward(
        self,
        support: torch.Tensor,
        support_y: torch.Tensor,
        query: torch.Tensor,
        n_way: int,
    ) -> torch.Tensor:
        return self.score(query, prototypes(support, support_y, n_way))


class LinearProbe(nn.Module):
    """Logistic regression on frozen features, refit per episode.

    The honest lower bound for "the pretrained representation is already good enough".
    Fit with L-BFGS on the support set, which converges in a few dozen iterations at
    these sizes and avoids an arbitrary SGD schedule becoming a confound.
    """

    def __init__(self, dim: int, n_way: int, weight_decay: float = 1e-2) -> None:
        super().__init__()
        self.fc = nn.Linear(dim, n_way)
        self.weight_decay = weight_decay

    def fit(self, support: torch.Tensor, support_y: torch.Tensor, steps: int = 100) -> None:
        self.to(support.device)
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        opt = torch.optim.LBFGS(self.parameters(), lr=1.0, max_iter=steps)
        support = support.detach()

        def closure() -> torch.Tensor:
            opt.zero_grad()
            loss = F.cross_entropy(self.fc(support), support_y)
            loss = loss + self.weight_decay * self.fc.weight.pow(2).sum()
            loss.backward()
            return loss

        opt.step(closure)

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        return self.fc(query)
