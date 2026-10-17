"""D_l -- how far a layer's representation has moved from its pretrained form.

Measured functionally rather than by parameter distance. Two networks can have very
different weights and compute nearly the same function, and conversely a small weight
change in an early layer can reshape everything downstream; what the controller needs
to know is whether the *representation* has changed, so that is what we measure.

Linear CKA (Kornblith et al., 2019) is the standard tool: invariant to orthogonal
transformation and to isotropic scaling of the features, which matters because
fine-tuning routinely rotates and rescales a representation without destroying the
information in it. A metric that flagged those as "drift" would throttle adaptation for
no reason.

Parameter drift is still recorded, as a secondary diagnostic, by
``TappedBackbone.parameter_drift``.
"""

from __future__ import annotations

import torch


def _pool(activation: torch.Tensor) -> torch.Tensor:
    """Reduce an activation to ``(batch, channels)``.

    Convolutional activations are globally average-pooled over space and transformer
    tokens are averaged over position. CKA needs a
    2-D matrix of (samples x features), and pooling keeps the sample axis equal to the
    batch, which is what makes the comparison between the frozen and adapted branches
    sample-aligned -- the property the whole measure rests on.
    """
    if activation.ndim == 4:
        return activation.mean(dim=(2, 3))
    if activation.ndim == 3:
        # Transformer tokens, (batch, tokens, channels). Averaged over tokens, which
        # is the direct analogue of the spatial pooling above: both collapse the
        # positional axis and keep (sample x channel).
        #
        # NOT flatten(1), which was the previous fallback here. Flattening is a
        # legitimate feature space but not a comparable one -- it yields
        # tokens*channels position-specific features (75k for ViT-S/16 at 224px
        # against ResNet-50's 2048) and folds position into the feature axis, so
        # drift measured that way is not on the same scale as the ResNet arm's and
        # the two cannot be put in one table.
        return activation.mean(dim=1)
    if activation.ndim == 2:
        return activation
    return activation.flatten(1)


def linear_cka(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Linear CKA between two sample-aligned feature matrices.

    ``x``: ``(n, p)``, ``y``: ``(n, q)``. Returns a scalar in ``[0, 1]``.

    Uses the feature-space form ``||Y^T X||_F^2 / (||X^T X||_F ||Y^T Y||_F)``, which is
    equivalent to the Gram-matrix form but costs ``O(n p q)`` instead of ``O(n^2 (p+q))``
    -- the right choice here, where the batch is small (5-100 support images) and the
    feature dimension is large (up to 2048).
    """
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"sample count mismatch: {x.shape[0]} vs {y.shape[0]}")
    if x.shape[0] < 2:
        # Centering a single sample yields zeros and CKA is undefined. Report "no
        # measurable drift" rather than a NaN that would silently poison the gate.
        return torch.ones((), device=x.device, dtype=x.dtype)

    x = x.float()
    y = y.float()
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)

    xty = y.T @ x
    hsic = xty.pow(2).sum()
    norm_x = (x.T @ x).pow(2).sum().sqrt()
    norm_y = (y.T @ y).pow(2).sum().sqrt()
    denom = norm_x * norm_y
    if denom < eps:
        # A layer whose activations are constant across the batch (dead ReLU, or a
        # batch of near-identical images) carries no information to compare.
        return torch.ones((), device=x.device, dtype=x.dtype)
    return (hsic / denom).clamp(0.0, 1.0)


@torch.no_grad()
def drift_signal(
    reference: dict[str, torch.Tensor],
    adapted: dict[str, torch.Tensor],
    unit_names: list[str],
    device: torch.device | None = None,
) -> torch.Tensor:
    """``1 - CKA`` per unit, in ``unit_names`` order, each in ``[0, 1]``.

    Units with no recorded activation score 0 (no observed drift), so an experiment
    that taps only some layers still produces a well-formed state vector.
    """
    values = []
    for name in unit_names:
        ref, adp = reference.get(name), adapted.get(name)
        if ref is None or adp is None:
            values.append(torch.zeros((), device=device))
            continue
        cka = linear_cka(_pool(ref), _pool(adp))
        values.append((1.0 - cka).to(device) if device is not None else 1.0 - cka)
    return torch.stack(values)
