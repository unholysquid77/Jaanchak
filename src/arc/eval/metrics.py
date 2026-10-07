"""Episode-level metrics and their confidence intervals.

Few-shot results are noisy enough that a point estimate is close to meaningless -- a
2-point "improvement" over 100 episodes is routinely within noise. Everything here
reports a 95% CI over episodes, and the protocol fixes 600 episodes x 3 seeds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class EpisodeMetrics:
    accuracy: float
    balanced_accuracy: float
    macro_f1: float

    def as_dict(self) -> dict[str, float]:
        return {
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "macro_f1": self.macro_f1,
        }


def episode_metrics(logits: torch.Tensor, targets: torch.Tensor, n_way: int) -> EpisodeMetrics:
    """Accuracy, balanced accuracy and macro-F1 for one episode.

    Computed with a confusion matrix rather than sklearn so this stays on-device and
    adds nothing measurable to the inner loop.
    """
    pred = logits.argmax(dim=1)
    cm = torch.zeros(n_way, n_way, dtype=torch.long, device=pred.device)
    cm.index_put_((targets, pred), torch.ones_like(pred), accumulate=True)

    tp = cm.diag().to(torch.float64)
    support = cm.sum(dim=1).to(torch.float64)
    predicted = cm.sum(dim=0).to(torch.float64)

    accuracy = (tp.sum() / support.sum().clamp_min(1)).item()
    recall = tp / support.clamp_min(1)
    precision = tp / predicted.clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)

    # Classes absent from an episode's query set would otherwise drag the macro down.
    present = support > 0
    return EpisodeMetrics(
        accuracy=accuracy,
        balanced_accuracy=recall[present].mean().item(),
        macro_f1=f1[present].mean().item(),
    )


def mean_ci95(values: list[float] | np.ndarray) -> tuple[float, float]:
    """Mean and half-width of the 95% confidence interval.

    Normal approximation, which is appropriate at n >= 600 episodes.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.size < 2:
        return (float(arr.mean()) if arr.size else float("nan"), float("nan"))
    return float(arr.mean()), float(1.96 * arr.std(ddof=1) / np.sqrt(arr.size))


def summarise(records: list[dict[str, float]]) -> dict[str, float]:
    """Collapse per-episode records into ``{metric: mean, metric_ci95: halfwidth}``."""
    if not records:
        return {}
    out: dict[str, float] = {"n_episodes": float(len(records))}
    for key in records[0]:
        mean, ci = mean_ci95([r[key] for r in records if key in r])
        out[key] = mean
        out[f"{key}_ci95"] = ci
    return out


def format_summary(name: str, summary: dict[str, float]) -> str:
    acc = summary.get("accuracy", float("nan")) * 100
    ci = summary.get("accuracy_ci95", float("nan")) * 100
    f1 = summary.get("macro_f1", float("nan")) * 100
    n = int(summary.get("n_episodes", 0))
    return f"{name:<34} acc {acc:5.2f} +/- {ci:4.2f}   macro-F1 {f1:5.2f}   (n={n})"
