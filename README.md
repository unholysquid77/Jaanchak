# ARC — Adaptive Representation Control

Few-shot recognition under domain shift, where a pretrained vision backbone is adapted
*selectively* rather than uniformly: every layer is scored by a representation-state
vector and a controller decides how strongly it may change.

## Start here

**→ [`docs/HANDOFF.md`](docs/HANDOFF.md)** — current state, how to run things, the list
of bugs that produced plausible-but-wrong numbers, and the open decisions. Read it
before touching anything.

| | |
|---|---|
| [`docs/ARC_SPEC.md`](docs/ARC_SPEC.md) | engineering design and the experimental ladder |
| [`docs/RESULTS_LEVEL0.md`](docs/RESULTS_LEVEL0.md) | tuned baseline table (complete) |
| [`docs/RESULTS_LEVEL1.md`](docs/RESULTS_LEVEL1.md) | adaptive gate (null result) |
| [`docs/DATA_REQUEST.md`](docs/DATA_REQUEST.md) | what is needed from the forensic datasets |

## State

Level 0 (baselines) complete and reportable. Level 1 (drift + utility signals, adaptive
gate) built and tuned, currently a null result on EuroSAT. Levels 2 and 3 not started.

## Quickstart

```bash
pip install -e ".[dev]"
PYTHONPATH=src python -m pytest tests -q        # 82 tests, CPU, ~3 min
```

GPU experiments run on a separate machine; see `docs/HANDOFF.md` section 1, which also
explains why a cloud session cannot reach it.

## One thing to know about the design

The core mechanism is in `src/arc/backbones/tapped.py`: a single activation stream is
carried through the network, and at each controllable unit the gate interpolates between
the pretrained and adapted outputs,

```
h_l = (1 - a_l)·frozen_l(h_{l-1}) + a_l·adapted_l(h_{l-1})
```

This has two exact endpoints, both asserted in tests — all gates at 0 reproduces the
pretrained network bit-for-bit, all gates at 1 reproduces full fine-tuning — and it puts
`a_l` on the forward path so it can receive gradient. A gate at 0 freezes a unit both
functionally and in its gradient, so every baseline is expressible as a gate schedule
and shares one code path with ARC.
