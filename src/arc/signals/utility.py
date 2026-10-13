"""U_l -- whether adapting a layer is doing anything reliable.

**This departs from docs/ARC_SPEC.md section 4, deliberately.** The spec defines utility
as the alignment between the support gradient and the *query* gradient -- "does a
support step reduce query loss?". That is a clean definition and it is what a
meta-learner would use during meta-training, but it is not computable at test time:
query labels are precisely what the model is being asked to predict. Using them to
steer adaptation is label leakage, and it would inflate every ARC number in a way that
nothing in the output would reveal.

What is available at test time is the support set and the augmentation distribution. So
utility is measured as **gradient agreement between two independent augmented draws of
the support set**:

    U_l = cos( grad_l L(draw 1), grad_l L(draw 2) )

Read it as a per-layer estimate of gradient signal-to-noise. If two independent views of
the same five images imply the same direction of change for layer l, that direction
reflects the task rather than the sampling of the augmentation, and adapting there is
justified. If they disagree -- cosine near zero or negative -- the gradient is mostly
noise, and with five images that is the common case for layers with millions of
parameters. Throttling those is exactly the behaviour ARC is supposed to produce.

Properties that make this usable where the spec's version is not: it needs no labels
beyond the support set's own, it is defined at k=1, and it costs two backward passes
rather than requiring a held-out split.

The label-using form remains available as ``oracle_utility`` for meta-training
experiments and as an upper bound on how much a perfect utility signal could buy. It
must never be used in a reported test-time number.
"""

from __future__ import annotations

from typing import Callable, Iterable

import torch

from arc.signals.base import frozen_batchnorm


def _flat_grad(params: Iterable[torch.nn.Parameter]) -> torch.Tensor | None:
    grads = [p.grad.detach().flatten() for p in params if p.grad is not None]
    return torch.cat(grads) if grads else None


def _grads_per_unit(
    backbone,
    loss_fn: Callable[[], torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Run one backward pass and collect a flattened gradient per controllable unit."""
    backbone.zero_grad(set_to_none=True)
    # Measurement, not training: these passes must not move BatchNorm statistics.
    with frozen_batchnorm(backbone.adapted):
        loss_fn().backward()
    out: dict[str, torch.Tensor] = {}
    for name in backbone.unit_names:
        g = _flat_grad(backbone.adapted[name].parameters())
        if g is not None:
            out[name] = g.clone()
    backbone.zero_grad(set_to_none=True)
    return out


def utility_signal(
    backbone,
    loss_fn: Callable[[], torch.Tensor],
    device: torch.device | None = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Cosine agreement between two independent draws, per unit, in ``[-1, 1]``.

    ``loss_fn`` must be stochastic across calls -- it is called twice and the two
    results are compared, so a deterministic loss would return a cosine of exactly 1
    everywhere and the signal would be vacuous. With the two-view objective the
    randomness comes from the augmentation.

    Units with no trainable parameters score 0: no gradient, so no evidence either way.
    """
    first = _grads_per_unit(backbone, loss_fn)
    second = _grads_per_unit(backbone, loss_fn)

    values = []
    for name in backbone.unit_names:
        a, b = first.get(name), second.get(name)
        if a is None or b is None:
            values.append(torch.zeros((), device=device))
            continue
        denom = a.norm() * b.norm()
        cos = (a @ b) / denom if denom > eps else torch.zeros((), device=a.device)
        values.append(cos.to(device) if device is not None else cos)
    return torch.stack(values).detach()


def oracle_utility(
    backbone,
    support_loss_fn: Callable[[], torch.Tensor],
    query_loss_fn: Callable[[], torch.Tensor],
    device: torch.device | None = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """The spec's label-using definition: support-gradient vs query-gradient alignment.

    **Uses query labels.** Valid only for meta-training, or for measuring how much a
    perfect utility signal would be worth as an upper bound. Never valid in a reported
    test-time result -- see this module's docstring.
    """
    sup = _grads_per_unit(backbone, support_loss_fn)
    qry = _grads_per_unit(backbone, query_loss_fn)

    values = []
    for name in backbone.unit_names:
        a, b = sup.get(name), qry.get(name)
        if a is None or b is None:
            values.append(torch.zeros((), device=device))
            continue
        denom = a.norm() * b.norm()
        cos = (a @ b) / denom if denom > eps else torch.zeros((), device=a.device)
        values.append(cos.to(device) if device is not None else cos)
    return torch.stack(values).detach()
