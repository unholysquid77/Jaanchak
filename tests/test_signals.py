"""Tests for the Level-1 signals and the adaptive gate.

The risk these guard against is a controller that runs, emits plausible gate values,
and is actually responding to nothing -- which is how the first LR sweep failed. So the
tests check that each signal *moves in the right direction for the right reason*, not
merely that it returns a tensor of the right shape.
"""

from __future__ import annotations

import pytest
import torch

from arc.backbones.tapped import TappedBackbone
from arc.controllers.gate import AdaptiveGate, GateCoefficients
from arc.data.base import Episode
from arc.signals.base import LayerSignals
from arc.signals.drift import drift_signal, linear_cka
from arc.signals.utility import utility_signal


def _episode(n_way: int = 5, k_shot: int = 2, size: int = 64) -> Episode:
    torch.manual_seed(0)
    n = n_way * k_shot
    return Episode(
        support_x=torch.randn(n, 3, size, size),
        support_y=torch.arange(n_way).repeat_interleave(k_shot),
        query_x=torch.randn(n_way * 3, 3, size, size),
        query_y=torch.arange(n_way).repeat_interleave(3),
        global_classes=torch.arange(n_way),
        episode_id=0,
    )


def _backbone() -> TappedBackbone:
    torch.manual_seed(0)
    net = TappedBackbone(arch="resnet18", pretrained=False)
    net.eval()
    return net


# -- CKA ---------------------------------------------------------------------

def test_cka_of_a_representation_with_itself_is_one() -> None:
    x = torch.randn(16, 32)
    assert linear_cka(x, x).item() == pytest.approx(1.0, abs=1e-5)


def test_cka_is_invariant_to_rotation_and_scaling() -> None:
    """The reason for choosing CKA over a distance. Fine-tuning routinely rotates and
    rescales a representation without destroying its content; a measure that called
    that 'drift' would throttle adaptation for no reason."""
    torch.manual_seed(0)
    x = torch.randn(32, 16)
    q, _ = torch.linalg.qr(torch.randn(16, 16))
    assert linear_cka(x, x @ q).item() == pytest.approx(1.0, abs=1e-4)
    assert linear_cka(x, x * 7.5).item() == pytest.approx(1.0, abs=1e-5)


def test_cka_falls_for_an_unrelated_representation() -> None:
    torch.manual_seed(0)
    x, y = torch.randn(64, 16), torch.randn(64, 16)
    assert linear_cka(x, y).item() < 0.5


def test_cka_degrades_monotonically_as_noise_is_added() -> None:
    torch.manual_seed(0)
    x = torch.randn(64, 16)
    values = [linear_cka(x, x + torch.randn_like(x) * s).item() for s in (0.1, 1.0, 5.0)]
    assert values[0] > values[1] > values[2], values


def test_cka_handles_degenerate_input_without_nan() -> None:
    """A dead ReLU gives constant activations; returning NaN here would poison the gate
    with no visible cause."""
    const = torch.ones(8, 4)
    assert torch.isfinite(linear_cka(const, const))
    assert torch.isfinite(linear_cka(torch.randn(1, 4), torch.randn(1, 4)))


# -- drift signal ------------------------------------------------------------

def test_drift_is_zero_before_any_adaptation() -> None:
    """At step 0 the adapted branch *is* the pretrained branch, so drift must be
    exactly zero. If it is not, the reference is being computed wrongly."""
    net, ep = _backbone(), _episode()
    ref = net.forward_reference(ep.support_x).activations
    with torch.no_grad():
        adapted = net.forward_adapted(ep.support_x).activations
    d = drift_signal(ref, adapted, net.unit_names)
    assert d.shape == (net.n_units,)
    assert d.abs().max().item() < 1e-5, d


def test_drift_rises_when_the_adapted_branch_is_perturbed() -> None:
    net, ep = _backbone(), _episode()
    ref = net.forward_reference(ep.support_x).activations
    with torch.no_grad():
        for p in net.adapted.parameters():
            p.add_(torch.randn_like(p) * 0.1)
        adapted = net.forward_adapted(ep.support_x).activations
    d = drift_signal(ref, adapted, net.unit_names)
    assert d.max().item() > 0.01, f"perturbation produced no drift: {d}"
    assert (d >= 0).all() and (d <= 1).all()


def test_drift_increases_with_perturbation_size() -> None:
    def measure(scale: float) -> float:
        net, ep = _backbone(), _episode()
        ref = net.forward_reference(ep.support_x).activations
        with torch.no_grad():
            for p in net.adapted.parameters():
                p.add_(torch.randn_like(p) * scale)
            adapted = net.forward_adapted(ep.support_x).activations
        return drift_signal(ref, adapted, net.unit_names).mean().item()

    small, large = measure(0.02), measure(0.2)
    assert large > small, f"drift did not grow with perturbation: {small} -> {large}"


# -- utility signal ----------------------------------------------------------

def test_utility_is_one_for_a_deterministic_loss() -> None:
    """Sanity check on the mechanism: two identical draws must agree perfectly.

    It is also why the real objective has to be stochastic -- a deterministic loss
    makes this signal vacuously 1 everywhere.
    """
    net, ep = _backbone(), _episode()
    net.adapted.requires_grad_(True)

    def loss_fn():
        return net(ep.support_x).embedding.pow(2).mean()

    u = utility_signal(net, loss_fn)
    assert u.shape == (net.n_units,)
    assert u.min().item() > 0.99, u


def test_utility_is_low_when_the_gradient_direction_is_random() -> None:
    """The case the gate exists to detect: a gradient that reflects sampling noise
    rather than task structure.

    Note what does *not* work as a negative control. Feeding random images into
    ``embedding.pow(2).mean()`` still yields high agreement, because "shrink the
    embedding" is a consistent direction whatever the input -- the randomness is in the
    data, but the objective's pull is not. To make the gradient itself random, the
    *objective* has to change per draw, which is what projecting onto a fresh random
    direction does.
    """
    net, ep = _backbone(), _episode()
    net.adapted.requires_grad_(True)

    def loss_fn():
        direction = torch.randn(net.embed_dim)
        return (net(ep.support_x).embedding @ direction).mean()

    u = utility_signal(net, loss_fn)
    assert u.mean().item() < 0.5, f"random objectives agreed too well: {u.mean():.3f}"


def test_utility_stays_in_range() -> None:
    net, ep = _backbone(), _episode()
    net.adapted.requires_grad_(True)

    def loss_fn():
        return net(ep.support_x + torch.randn_like(ep.support_x) * 0.5).embedding.mean()

    u = utility_signal(net, loss_fn)
    assert (u >= -1.001).all() and (u <= 1.001).all(), u


def test_utility_leaves_no_gradients_behind() -> None:
    """It runs two backward passes of its own; leaking those into the optimiser would
    corrupt the very step it is meant to inform."""
    net, ep = _backbone(), _episode()
    net.adapted.requires_grad_(True)

    def loss_fn():
        return net(ep.support_x).embedding.pow(2).mean()

    utility_signal(net, loss_fn)
    assert all(p.grad is None for p in net.adapted.parameters())


# -- the gate ----------------------------------------------------------------

def test_signals_default_to_zero_so_unimplemented_ones_are_inert() -> None:
    s = LayerSignals(n_units=4)
    assert torch.equal(s.get("topology"), torch.zeros(4))
    assert s.as_matrix().shape == (4, 5)


def test_coefficients_reject_negative_magnitudes() -> None:
    with pytest.raises(ValueError, match="magnitudes"):
        GateCoefficients(drift=-1.0)


def test_gate_closes_as_drift_rises() -> None:
    """The core control behaviour: a layer that has moved far from pretrained should
    be throttled."""
    gate = AdaptiveGate(GateCoefficients(bias=2.0, utility=0.0, drift=8.0))
    device = torch.device("cpu")

    low = LayerSignals(n_units=3, drift=torch.zeros(3))
    high = LayerSignals(n_units=3, drift=torch.ones(3))
    assert gate.gate_from(low, device).mean() > 0.8
    assert gate.gate_from(high, device).mean() < 0.1


def test_gate_opens_as_utility_rises() -> None:
    gate = AdaptiveGate(GateCoefficients(bias=0.0, utility=6.0, drift=0.0))
    device = torch.device("cpu")
    disagree = LayerSignals(n_units=3, utility=-torch.ones(3))
    agree = LayerSignals(n_units=3, utility=torch.ones(3))
    assert gate.gate_from(disagree, device).mean() < 0.1
    assert gate.gate_from(agree, device).mean() > 0.9


def test_gate_discriminates_between_units() -> None:
    """A gate that emits the same value everywhere is a global learning-rate
    multiplier wearing a costume -- the whole premise is per-layer control."""
    # bias=4 centres the sigmoid over the drift range being tested; at bias=0 the whole
    # response is squashed into the lower half and the spread is capped at 0.5 by
    # construction rather than by any property of the controller.
    gate = AdaptiveGate(GateCoefficients(bias=4.0, utility=0.0, drift=8.0))
    signals = LayerSignals(n_units=4, drift=torch.tensor([0.0, 0.3, 0.6, 1.0]))
    gates = gate.gate_from(signals, torch.device("cpu"))
    assert gates[0] > gates[1] > gates[2] > gates[3]
    assert (gates.max() - gates.min()) > 0.8, gates


def test_gate_is_held_open_during_warmup() -> None:
    net, ep = _backbone(), _episode()
    gate = AdaptiveGate(warmup=2, cadence=1)
    gate.reset(net, ep)
    assert torch.equal(gate.step(0), torch.ones(net.n_units))
    assert torch.equal(gate.step(1), torch.ones(net.n_units))


def test_gate_recomputes_only_on_cadence() -> None:
    net, ep = _backbone(), _episode()
    net.adapted.requires_grad_(True)
    gate = AdaptiveGate(cadence=3, warmup=0, use_utility=False)
    gate.reset(net, ep)
    for i in range(7):
        gate.step(i)
    # Steps 0, 3 and 6 trigger a recomputation; the rest reuse the cached gate.
    assert len(gate.history) == 3, gate.history


def test_gate_runs_end_to_end_and_produces_varied_values() -> None:
    net, ep = _backbone(), _episode()
    net.adapted.requires_grad_(True)

    def loss_fn():
        x = ep.support_x + torch.randn_like(ep.support_x) * 0.3
        return net(x, gates=None).embedding.pow(2).mean()

    gate = AdaptiveGate(cadence=1, warmup=1)
    gate.reset(net, ep)
    gate.bind_loss(loss_fn)

    with torch.no_grad():
        for p in net.adapted.parameters():
            p.add_(torch.randn_like(p) * 0.05)

    gates = gate.step(2)
    assert gates.shape == (net.n_units,)
    assert torch.isfinite(gates).all()
    assert (gates >= 0).all() and (gates <= 1).all()
    assert gate.history and "drift" in gate.history[0] and "utility" in gate.history[0]


# -- train-mode discipline ---------------------------------------------------

def _bn_stats(module: torch.nn.Module) -> torch.Tensor:
    parts = [
        t.flatten()
        for m in module.modules()
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)
        for t in (m.running_mean, m.running_var)
        if t is not None
    ]
    return torch.cat(parts)


def test_computing_signals_does_not_resurrect_batchnorm_training() -> None:
    """The controller runs its own forward pass to measure drift. Restoring the
    backbone with a plain .train() afterwards puts BatchNorm back into training mode,
    silently overriding the runner's bn_mode='eval' and reintroducing running-statistic
    pollution from the support batch -- the defect fixed in the inner loop, re-entering
    through the controller. It fires only when drift is enabled, which is exactly the
    configuration the gate sweep found 'harmful'.
    """
    net, ep = _backbone(), _episode()
    net.train()
    for m in net.adapted.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()

    gate = AdaptiveGate(cadence=1, warmup=0, use_utility=False, use_drift=True)
    gate.reset(net, ep)
    before = _bn_stats(net.adapted).clone()
    gate.compute_signals()

    still_eval = [
        m.training for m in net.adapted.modules()
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)
    ]
    assert not any(still_eval), "BatchNorm was switched back to training mode"
    assert torch.equal(_bn_stats(net.adapted), before), "BatchNorm statistics moved"


def test_signal_computation_preserves_module_training_flags() -> None:
    """Whatever mode the runner established must survive a signal update, in either
    bn_mode -- the controller does not own that policy."""
    net, ep = _backbone(), _episode()
    net.train()
    before = {name: m.training for name, m in net.adapted.named_modules()}

    gate = AdaptiveGate(cadence=1, warmup=0, use_utility=False, use_drift=True)
    gate.reset(net, ep)
    gate.compute_signals()

    after = {name: m.training for name, m in net.adapted.named_modules()}
    assert before == after, {k: (before[k], after[k]) for k in before if before[k] != after[k]}
