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
        missing = torch.flatnonzero(counts == 0).tolist()
        raise ValueError(f"no support examples for classes {missing}")
    return sums / counts.unsqueeze(1)


class ProtoHead(nn.Module):
    """Prototypical network head (Snell et al., 2017).

    Parameters
    ----------
    metric
        ``"sqeuclidean"`` reproduces the original paper and is the default.
        ``"cosine"`` is scale-invariant, which matters when adaptation changes
        embedding norms -- worth reporting as a robustness check, since a method that
        only "wins" by inflating norms has not learned anything.
    temperature
        Logit scale. Learnable, initialised at 1 (10 for cosine, whose logits are
        bounded in [-1, 1] and otherwise produce near-uniform softmaxes).
    """

    def __init__(self, metric: str = "sqeuclidean", temperature: float | None = None) -> None:
        super().__init__()
        if metric not in {"sqeuclidean", "cosine"}:
            raise ValueError(f"unknown metric {metric!r}")
        self.metric = metric
        init = temperature if temperature is not None else (10.0 if metric == "cosine" else 1.0)
        self.log_temperature = nn.Parameter(torch.tensor(init).log())

    def forward(
        self,
        support: torch.Tensor,
        support_y: torch.Tensor,
        query: torch.Tensor,
        n_way: int,
    ) -> torch.Tensor:
        protos = prototypes(support, support_y, n_way)
        if self.metric == "cosine":
            logits = F.normalize(query, dim=1) @ F.normalize(protos, dim=1).t()
        else:
            # -||q - p||^2, computed without materialising the (nq, n_way, dim) tensor.
            logits = -torch.cdist(query, protos).pow(2)
        return logits * self.log_temperature.exp()


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
