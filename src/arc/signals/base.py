"""Layer-wise representation state.

Each signal maps the current state of a layer to a scalar in a bounded range, so the
controller's coefficients are comparable across signals and a sigmoid over their
weighted sum is well-scaled without per-dataset tuning.

The state vector from the methodology is ``s_l = [U_l, D_l, T_l, S_l, N_l]``. Level 1
implements ``U`` (utility) and ``D`` (drift); ``T`` (topology), ``S`` (stability) and
``N`` (novelty) arrive at Level 2. :class:`LayerSignals` carries all five so the
controller does not need rewriting as they land -- absent signals are simply zero and
contribute nothing to the gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class LayerSignals:
    """Per-unit signal values, each a 1-D tensor indexed like ``backbone.unit_names``.

    Ranges are deliberately bounded and sign-conventioned so the controller reads
    naturally -- positive means "adapting here is justified":

    ``utility``  in [-1, 1]  gradient agreement; higher = more consistent descent
    ``drift``    in [ 0, 1]  1 - CKA to the pretrained representation
    ``topology`` in [ 0, 1]  normalised structural change          (Level 2)
    ``stability`` in [ 0, 1] higher = better-conditioned dynamics  (Level 2)
    ``novelty``  in [ 0, 1]  target-vs-pretrained statistic gap    (Level 2)
    """

    n_units: int
    utility: torch.Tensor | None = None
    drift: torch.Tensor | None = None
    topology: torch.Tensor | None = None
    stability: torch.Tensor | None = None
    novelty: torch.Tensor | None = None
    extra: dict[str, float] = field(default_factory=dict)

    NAMES = ("utility", "drift", "topology", "stability", "novelty")

    def get(self, name: str, device: torch.device | None = None) -> torch.Tensor:
        """Signal values, or zeros if that signal is not being computed.

        Returning zeros rather than raising is what lets a Level-1 controller and a
        Level-2 controller share one implementation: an unimplemented signal simply
        has no influence on the gate.
        """
        if name not in self.NAMES:
            raise ValueError(f"unknown signal {name!r}; expected one of {self.NAMES}")
        value = getattr(self, name)
        if value is None:
            return torch.zeros(self.n_units, device=device)
        return value if device is None else value.to(device)

    def as_matrix(self, device: torch.device | None = None) -> torch.Tensor:
        """``(n_units, 5)`` state matrix, in :attr:`NAMES` order.

        This is the ``X_t`` the FBSDE controller will consume at Level 3.
        """
        return torch.stack([self.get(n, device) for n in self.NAMES], dim=1)

    def summary(self) -> dict[str, float]:
        """Mean of each present signal, for logging and for the paper's figures."""
        out: dict[str, float] = {}
        for name in self.NAMES:
            value = getattr(self, name)
            if value is not None:
                out[name] = float(value.mean())
        return out
