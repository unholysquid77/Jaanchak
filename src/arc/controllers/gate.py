"""Controller A -- the lightweight adaptive gate.

Maps the layer-wise representation state to an adaptation intensity:

    a_l = sigmoid( w0 + alpha*U_l - beta*D_l - gamma*T_l + delta*S_l + eta*N_l )

At Level 1 only ``U`` and ``D`` are populated; the Level-2 signals contribute zero
until they exist, so this class does not need rewriting as they land.

Two things make this a *controller* rather than a fixed schedule, even with the
coefficients held fixed (mode (iii) of docs/ARC_SPEC.md section 5):

* The signals are recomputed during the inner loop, so the gate responds to what
  adaptation is actually doing. At step 0 the adapted branch equals the pretrained one,
  so ``D_l = 0`` everywhere and the gate is driven purely by utility; as the
  representation moves, drift rises and throttles the layers that have moved most. That
  negative feedback is the homeostatic behaviour the methodology asks for.
* Because ``lerp(h_ref, h_adp, 0)`` has zero derivative with respect to ``h_adp``,
  a gate near zero freezes a unit both functionally and in its gradient. The gate does
  not merely scale an update; it decides whether the layer participates at all.

Signals are detached. Differentiating through their computation would require
second-order derivatives, and at Level 1 the coefficients are hyperparameters anyway.
Mode (ii) -- learning them through the forward path -- becomes available for free once
this is working, because the gate already sits on the forward path.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from arc.backbones.tapped import TappedBackbone
from arc.data.base import Episode
from arc.signals.base import LayerSignals, frozen_batchnorm
from arc.signals.drift import drift_signal
from arc.signals.utility import utility_signal


@dataclass
class GateCoefficients:
    """Fixed coefficients for the gate.

    Signs follow the methodology: utility, stability and novelty argue *for* adapting a
    layer; drift and topological change argue *against*. The caller supplies magnitudes
    only, so a negative coefficient here would be a sign error rather than a choice.
    """

    bias: float = 2.0       # sigmoid(2) ~ 0.88: start near-adapted, throttle downward
    utility: float = 2.0
    drift: float = 4.0
    topology: float = 0.0   # Level 2
    stability: float = 0.0  # Level 2
    novelty: float = 0.0    # Level 2

    def __post_init__(self) -> None:
        negative = [k for k, v in self.as_dict().items() if v < 0]
        if negative:
            raise ValueError(
                f"coefficients are magnitudes; signs are fixed by the gate equation. "
                f"Negative values given for {negative}."
            )

    def as_dict(self) -> dict[str, float]:
        return {
            "utility": self.utility,
            "drift": self.drift,
            "topology": self.topology,
            "stability": self.stability,
            "novelty": self.novelty,
        }


class AdaptiveGate:
    """State-feedback gate controller.

    Parameters
    ----------
    cadence
        Recompute signals every ``cadence`` inner steps, reusing the previous gate in
        between. Utility costs two extra backward passes, so recomputing every step
        would roughly triple the cost of adaptation for a signal that moves slowly.
    warmup
        Steps at the start of the episode during which the gate is held fully open.
        At step 0 drift is identically zero (the branches are the same network), so the
        drift term carries no information until some adaptation has happened; opening
        the gate first gives it something to measure.
    """

    def __init__(
        self,
        coefficients: GateCoefficients | None = None,
        cadence: int = 5,
        warmup: int = 1,
        use_utility: bool = True,
        use_drift: bool = True,
    ) -> None:
        self.coef = coefficients or GateCoefficients()
        self.cadence = max(1, cadence)
        self.warmup = warmup
        self.use_utility = use_utility
        self.use_drift = use_drift

        self.backbone: TappedBackbone | None = None
        self.episode: Episode | None = None
        self._loss_fn = None
        self._reference: dict[str, torch.Tensor] = {}
        self._gates: torch.Tensor | None = None
        self.history: list[dict[str, float]] = []

    # -- plumbing ------------------------------------------------------------

    def bind_loss(self, loss_fn) -> None:
        """Supply the stochastic support loss used for the utility signal.

        The runner provides this because the objective (two-view, leave-one-out) is the
        runner's choice, not the controller's.
        """
        self._loss_fn = loss_fn

    def reset(self, backbone: TappedBackbone, episode: Episode) -> None:
        self.backbone = backbone
        self.episode = episode
        self.history = []
        self._gates = torch.ones(backbone.n_units, device=episode.support_x.device)

        # The pretrained reference is fixed for the episode, so its activations are
        # computed once here rather than at every signal update.
        self._reference = (
            backbone.forward_reference(episode.support_x).activations
            if self.use_drift
            else {}
        )

    # -- control -------------------------------------------------------------

    def compute_signals(self) -> LayerSignals:
        assert self.backbone is not None and self.episode is not None
        device = self.episode.support_x.device
        signals = LayerSignals(n_units=self.backbone.n_units)

        if self.use_drift:
            # Measure drift without disturbing the mode the runner established.
            #
            # An earlier version did `backbone.eval()` then `backbone.train()` to
            # restore. That looks symmetric but is not: `.train()` puts BatchNorm back
            # into training mode unconditionally, overriding the runner's
            # bn_mode='eval' and letting the support batch pollute the running
            # statistics from the first signal update onward. Because it only triggers
            # when drift is enabled, it made drift look actively harmful in the gate
            # sweep -- roughly 9 accuracy points at 1-shot -- when the damage was
            # BatchNorm, not the signal.
            #
            # Only BatchNorm needs suppressing (ResNet has no dropout), and each
            # module's own flag is saved and restored, so this is correct under both
            # bn_mode='eval' and the transductive bn_mode='train'.
            with frozen_batchnorm(self.backbone.adapted), torch.no_grad():
                adapted = self.backbone(
                    self.episode.support_x, gates=self._gates
                ).activations

            signals.drift = drift_signal(
                self._reference, adapted, self.backbone.unit_names, device
            )

        if self.use_utility and self._loss_fn is not None:
            signals.utility = utility_signal(self.backbone, self._loss_fn, device)

        return signals

    def gate_from(self, signals: LayerSignals, device: torch.device) -> torch.Tensor:
        c = self.coef
        logit = torch.full((signals.n_units,), c.bias, device=device)
        logit = logit + c.utility * signals.get("utility", device)
        logit = logit - c.drift * signals.get("drift", device)
        logit = logit - c.topology * signals.get("topology", device)
        logit = logit + c.stability * signals.get("stability", device)
        logit = logit + c.novelty * signals.get("novelty", device)
        return torch.sigmoid(logit)

    def step(self, step_index: int) -> torch.Tensor:
        assert self.backbone is not None and self.episode is not None
        device = self.episode.support_x.device

        if step_index < self.warmup:
            self._gates = torch.ones(self.backbone.n_units, device=device)
            return self._gates

        if (step_index - self.warmup) % self.cadence == 0:
            signals = self.compute_signals()
            self._gates = self.gate_from(signals, device).detach()
            self.history.append(
                {"step": float(step_index), **signals.summary(),
                 "gate_mean": float(self._gates.mean())}
            )

        return self._gates
