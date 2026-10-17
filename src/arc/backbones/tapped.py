"""Backbone with per-block taps and a frozen reference twin.

The whole framework rests on one structural idea: carry a *single* activation stream
through the network, and at every controllable block decide how much of the adapted
block's output to admit versus the pretrained block's output:

    h_l = (1 - a_l) * frozen_l(h_{l-1}) + a_l * adapted_l(h_{l-1})

This has two exact endpoints, which are the framework's correctness anchors and are
asserted in ``tests/test_backbone.py``:

* all ``a_l = 0``  ->  the network is *bit-for-bit* the pretrained network
* all ``a_l = 1``  ->  the network is *bit-for-bit* the fully fine-tuned network

Everything in between is a genuine interpolation of representations rather than a
heuristic blend of gradients. Crucially ``a_l`` sits on the forward path, so it receives
gradient from the task loss -- see ``docs/ARC_SPEC.md`` section 5 for why the
gradient-masking reading of the figure cannot train the gate at all.

Two distinct reference quantities exist and must not be confused:

``local reference``
    ``frozen_l(h_{l-1})`` -- the frozen block applied to the *current fused stream*.
    Used for the interpolation above. Depends on adapted weights, so not cacheable.

``pure reference``
    The activations of the pretrained network run on its own from the input. Used as
    the comparison point for drift and topology signals. Independent of adapted
    weights, therefore computed **once per episode** and cached -- this is what makes
    persistent homology affordable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torchvision import models


@dataclass
class ForwardResult:
    """Output of a tapped forward pass."""

    embedding: torch.Tensor
    activations: dict[str, torch.Tensor] = field(default_factory=dict)


def _resnet_units(net: models.ResNet) -> list[tuple[str, nn.Module]]:
    """Split a ResNet into its controllable units: the stem plus every bottleneck."""
    stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
    units: list[tuple[str, nn.Module]] = [("stem", stem)]
    for stage in (1, 2, 3, 4):
        layer = getattr(net, f"layer{stage}")
        for i, block in enumerate(layer):
            units.append((f"layer{stage}_{i}", block))
    return units


def _is_vit(arch: str) -> bool:
    """A ViT is anything timm names as one and torchvision does not provide here.

    Deliberately a name test rather than an isinstance test: the decision is made
    before any model is built, because building the wrong family wastes a download
    and a GPU allocation.
    """
    return arch.startswith("vit_") or arch.startswith("deit") or arch.startswith("eva")


def _build_vit(arch: str, pretrained: bool) -> nn.Module:
    """Build a timm ViT with no classifier head.

    ``timm`` is imported here rather than at module scope so that the ResNet path
    -- and every existing test and result -- keeps working on an environment that
    has no timm installed.
    """
    try:
        import timm
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            f"arch {arch!r} needs timm; install it (uv pip install timm) or use a "
            "torchvision ResNet name"
        ) from exc
    return timm.create_model(arch, pretrained=pretrained, num_classes=0)


class _ViTStem(nn.Module):
    """Everything a ViT does before its first transformer block.

    Holds the ``VisionTransformer`` itself and calls timm's own ``_pos_embed``
    rather than reimplementing it. That matters: ``_pos_embed`` branches on
    ``no_embed_class``, dynamic image size and register tokens, and a
    hand-rolled "prepend CLS then add pos_embed" is silently wrong for several
    of timm's checkpoints. The host model's ``blocks`` and ``norm`` are emptied
    by :func:`_vit_units` after the blocks are extracted, so no parameter is
    owned twice.
    """

    def __init__(self, vit: nn.Module) -> None:
        super().__init__()
        self.vit = vit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = self.vit
        x = v.patch_embed(x)
        x = v._pos_embed(x)
        x = v.patch_drop(x)
        return v.norm_pre(x)


def _vit_units(vit: nn.Module) -> list[tuple[str, nn.Module]]:
    """Split a timm ``VisionTransformer`` into controllable units.

    ViT-S/16 gives 14: the stem, 12 blocks, and the pre-head LayerNorm. The
    trailing ``norm`` is its own unit rather than being folded into block_11 or
    attached to the pooling step, for one reason: every parameter must sit in
    exactly one gated unit, or the ``a_l = 0`` endpoint stops being bit-exact
    against the pretrained network and the framework's correctness anchor is
    gone. ``cls_token`` and ``pos_embed`` are parameters too, which is why the
    stem owns the model object instead of a copy of its pieces.
    """
    blocks = list(vit.blocks)
    head_norm = vit.norm

    # Transfer ownership out of the host model so the ModuleDict is the sole
    # owner. Done after the references are taken, never before.
    vit.blocks = nn.ModuleList()
    vit.norm = nn.Identity()

    units: list[tuple[str, nn.Module]] = [("stem", _ViTStem(vit))]
    units += [(f"block_{i}", b) for i, b in enumerate(blocks)]
    units.append(("norm", head_norm))
    return units


class TappedBackbone(nn.Module):
    """A pretrained backbone paired with a frozen twin, gated per block.

    Parameters
    ----------
    arch
        Any ``torchvision.models`` ResNet constructor name. ResNet-50 gives 17 units
        (stem + 16 bottlenecks), which is the granularity the spec settles on.
    tap
        Which units to record activations for. ``None`` records all of them. Recording
        is cheap, but holding 17 activation maps for a 175-image episode is not -- the
        signal modules pool them immediately.
    """

    def __init__(
        self,
        arch: str = "resnet50",
        pretrained: bool = True,
        tap: Sequence[str] | None = None,
    ) -> None:
        super().__init__()
        self.arch = arch
        self.is_vit = _is_vit(arch)

        if self.is_vit:
            adapted_net = _build_vit(arch, pretrained)
            frozen_net = _build_vit(arch, pretrained)
            frozen_net.load_state_dict(adapted_net.state_dict())
            # Read embed_dim and the prefix-token count BEFORE splitting: the split
            # empties the host model, so these attributes must be captured first.
            self.embed_dim: int = adapted_net.embed_dim
            self.n_prefix_tokens: int = int(getattr(adapted_net, "num_prefix_tokens", 1))
            self.global_pool: str = str(getattr(adapted_net, "global_pool", "token"))
            split = _vit_units
        else:
            weights = "DEFAULT" if pretrained else None
            adapted_net = getattr(models, arch)(weights=weights)
            frozen_net = getattr(models, arch)(weights=weights)
            frozen_net.load_state_dict(adapted_net.state_dict())
            self.embed_dim = adapted_net.fc.in_features
            self.n_prefix_tokens = 0
            self.global_pool = "avg"
            split = _resnet_units

        self.adapted = nn.ModuleDict(dict(split(adapted_net)))
        self.frozen = nn.ModuleDict(dict(split(frozen_net)))
        self.unit_names: list[str] = list(self.adapted.keys())
        self.tap: list[str] = list(tap) if tap is not None else list(self.unit_names)

        unknown = set(self.tap) - set(self.unit_names)
        if unknown:
            raise ValueError(f"unknown tap names {sorted(unknown)}; have {self.unit_names}")

        # The frozen twin is a fixed reference, never trained and never in train mode
        # (its BatchNorm running statistics must stay at their pretrained values --
        # letting them drift would silently move the baseline we measure against).
        self.frozen.requires_grad_(False)
        self.frozen.eval()

        self.pool = nn.AdaptiveAvgPool2d(1)

        # Recompute each unit's activations during the backward pass instead of
        # storing them. Costs roughly 30% more compute and makes 20-shot episodes
        # fit: the two-view objective forwards 2 * n_way * k_shot images, which at
        # 20-shot is 200 images and overflows 10 GiB without this.
        self.grad_checkpoint = False

    # -- bookkeeping ---------------------------------------------------------

    @property
    def n_units(self) -> int:
        return len(self.unit_names)

    def unit_parameters(self, name: str) -> Iterable[nn.Parameter]:
        return self.adapted[name].parameters()

    def unit_param_counts(self) -> dict[str, int]:
        return {n: sum(p.numel() for p in self.adapted[n].parameters()) for n in self.unit_names}

    def train(self, mode: bool = True) -> "TappedBackbone":
        super().train(mode)
        self.frozen.eval()  # never follows the module's train/eval state
        return self

    def _embed(self, h: torch.Tensor) -> torch.Tensor:
        """Collapse a final-unit activation into an embedding vector.

        ResNet hands back ``(B, C, H, W)`` and gets global average pooling. A ViT
        hands back ``(B, N, D)`` tokens and gets whatever its checkpoint was
        trained to read: the CLS token for ``global_pool="token"``, the mean over
        patch tokens (excluding prefix tokens) otherwise. Using the wrong one is
        not a small error -- a CLS-trained checkpoint read by mean-pooling loses
        several accuracy points for no visible reason.
        """
        if h.dim() == 4:
            return torch.flatten(self.pool(h), 1)
        if h.dim() != 3:
            raise ValueError(f"cannot embed activation of shape {tuple(h.shape)}")
        if self.global_pool == "token":
            return h[:, 0]
        return h[:, self.n_prefix_tokens :].mean(dim=1)

    # -- forward paths -------------------------------------------------------

    @torch.no_grad()
    def forward_reference(self, x: torch.Tensor) -> ForwardResult:
        """Pure pretrained forward. Cacheable per episode; the comparison point for
        drift and topology."""
        acts: dict[str, torch.Tensor] = {}
        h = x
        for name in self.unit_names:
            h = self.frozen[name](h)
            if name in self.tap:
                acts[name] = h
        return ForwardResult(embedding=self._embed(h), activations=acts)

    def forward_adapted(self, x: torch.Tensor) -> ForwardResult:
        """Pure adapted forward -- equivalent to ``forward`` with all gates at 1."""
        acts: dict[str, torch.Tensor] = {}
        h = x
        for name in self.unit_names:
            h = self._run_adapted(name, h)
            if name in self.tap:
                acts[name] = h
        return ForwardResult(embedding=self._embed(h), activations=acts)

    def _run_adapted(self, name: str, h: torch.Tensor) -> torch.Tensor:
        """Apply one adapted unit, optionally with gradient checkpointing.

        Checkpointing is skipped when no grad is being tracked (it would only add the
        recompute cost) and is incompatible with BatchNorm in train mode, where the
        backward recomputation would update the running statistics a second time.
        """
        # Deliberately not gated on h.requires_grad: the stem's input is the raw
        # image, which never requires grad, but the stem's own parameters do.
        if not (self.grad_checkpoint and torch.is_grad_enabled()):
            return self.adapted[name](h)
        return checkpoint(self.adapted[name], h, use_reentrant=False)

    def forward(
        self,
        x: torch.Tensor,
        gates: Mapping[str, torch.Tensor] | torch.Tensor | None = None,
    ) -> ForwardResult:
        """Gated forward.

        ``gates`` may be a mapping ``unit -> scalar`` or a tensor of shape
        ``(n_units,)`` ordered as :attr:`unit_names`. ``None`` means fully adapted.

        Gate values are expected in ``[0, 1]`` but are not clamped: the controller is
        responsible for producing them through a sigmoid, and silently clamping here
        would mask a controller bug.
        """
        if gates is None:
            return self.forward_adapted(x)
        if isinstance(gates, torch.Tensor):
            if gates.numel() != self.n_units:
                raise ValueError(
                    f"gate tensor has {gates.numel()} entries, expected {self.n_units}"
                )
            gates = {n: gates[i] for i, n in enumerate(self.unit_names)}

        acts: dict[str, torch.Tensor] = {}
        h = x
        for name in self.unit_names:
            a = gates.get(name)
            if a is None:
                h = self._run_adapted(name, h)
            else:
                h_adp = self._run_adapted(name, h)
                with torch.no_grad():
                    h_ref = self.frozen[name](h)
                # a broadcasts over (B, C, H, W) for a ResNet and (B, N, D) for a
                # ViT when scalar. A per-channel gate is 4D-only: on tokens it would
                # broadcast along D, which is channels, so it still lines up -- but a
                # per-*token* gate would not and is not supported.
                h = torch.lerp(h_ref, h_adp, a)
            if name in self.tap:
                acts[name] = h
        return ForwardResult(embedding=self._embed(h), activations=acts)

    # -- adaptation accounting ----------------------------------------------

    @torch.no_grad()
    def parameter_drift(self) -> dict[str, float]:
        """Relative L2 drift of each unit's parameters from pretrained.

        Reported as a secondary metric; the primary drift signal is functional (CKA).
        """
        out: dict[str, float] = {}
        for name in self.unit_names:
            num = sum(
                (pa - pf).pow(2).sum().item()
                for pa, pf in zip(self.adapted[name].parameters(), self.frozen[name].parameters())
            )
            den = sum(pf.pow(2).sum().item() for pf in self.frozen[name].parameters())
            out[name] = (num / den) ** 0.5 if den > 0 else 0.0
        return out

    @torch.no_grad()
    def fraction_modified(self, rtol: float = 1e-3) -> float:
        """Fraction of backbone parameters moved more than ``rtol`` relative to their
        pretrained scale. This is the efficiency number the paper reports against
        full fine-tuning."""
        moved = total = 0
        for name in self.unit_names:
            for pa, pf in zip(self.adapted[name].parameters(), self.frozen[name].parameters()):
                scale = pf.abs().mean().clamp_min(1e-12)
                moved += ((pa - pf).abs() > rtol * scale).sum().item()
                total += pa.numel()
        return moved / max(total, 1)
