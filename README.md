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
| [`docs/RESULTS_LEVEL1.md`](docs/RESULTS_LEVEL1.md) | adaptive gate on EuroSAT (null result) |
| [`docs/RESULTS_ISIC.md`](docs/RESULTS_ISIC.md) | ISIC: tuned baselines and the gate (null replicates) |
| [`docs/RESULTS_SUBSPACE.md`](docs/RESULTS_SUBSPACE.md) | what adaptation actually does: four probes, and the pivot they imply |
| [`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md) | every run performed, including the void ones, with exact numbers |
| [`docs/DATA_REQUEST.md`](docs/DATA_REQUEST.md) | what is needed from the forensic datasets |

## State

Level 0 (baselines) complete and reportable on EuroSAT, tuned on ISIC. Level 1 (drift +
utility signals, adaptive gate) built and tuned, and a null result on **both** datasets —
on ISIC the signals are actively harmful and no parameter-efficiency win is produced.
BatchNorm-only adaptation matches full fine-tuning there at 0.22% of the parameters.
Levels 2 and 3 not started.

## Quickstart

```bash
pip install -e ".[dev]"
PYTHONPATH=src python -m pytest tests -q        # 110 tests, CPU, ~35 s
```

GPU experiments run on a separate machine; see `docs/HANDOFF.md` section 1, which also
explains why a cloud session cannot reach it.

`src/arc/data/` was missing from every commit until 2026-10-15, because `.gitignore`
matched it — recovered and now tested; see `docs/HANDOFF.md` section 6, landmine 16.

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
