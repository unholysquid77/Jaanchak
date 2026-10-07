# ARC — Adaptive Representation Control
### Few-Shot Forensic Evidence Recognition — Engineering Spec v0.1

Source of truth for the science: `Adaptive_Representation_Control_Latest_Methodology.docx`.
This document is the *engineering* translation: what we build, in what order, on what hardware,
and which claims we are allowed to make at the end.

---

## 0. One-paragraph summary

A pretrained vision backbone is adapted to a data-scarce, domain-shifted forensic target.
Instead of a single global fine-tuning strategy, every layer `l` is given a **representation
state** `s_l = [U_l, D_l, T_l, S_l, N_l]` (utility / drift / topological mismatch / stability /
novelty). A **controller** maps `s_l` to an adaptation intensity `a_l` in `[0,1]`. Controller A
is a lightweight differentiable gate; Controller B is an FBSDE stochastic-optimal-control policy.
The central empirical question is not "does ARC beat fine-tuning" alone — it is **"does the
expensive controller buy anything over the cheap one?"**

---

## 1. Hardware and environment (verified 2026-10-08)

| | |
|---|---|
| Host | `prashant@100.103.26.120` (`JPD-2AB-114-D02`), key-based SSH works |
| GPU | 1x RTX 3080, 10 GiB, driver 595.91.07 |
| CPU / RAM | i9-10850K, 20 threads / 61 GiB |
| Disk | 330 GiB free on `/` |
| Present | `git`, `python3.14` |
| **Absent** | `conda`, `mamba`, `uv`, `nvcc`, `docker`, `tmux`, `torch` |

Consequences:

- Python 3.14 has no stable PyTorch wheels, so install **`uv`** (userspace, no sudo) and pin
  **Python 3.12** plus `torch` cu126 wheels inside a project venv.
- No `tmux` and possibly no sudo, so long runs go through
  `setsid nohup ... > logs/run.log 2>&1 &` plus a run-registry file.
  Not negotiable: **every run must survive SSH disconnect.**
- 10 GiB VRAM is the real constraint. ResNet-50 at 224x224 with AMP and episode-sized batches
  (5-way 20-shot = 175 images) fits comfortably. FBSDE meta-training is the only thing at risk.

---

## 2. THE BLOCKING RISK — datasets

The dataset paragraph circulated on WhatsApp (FFM: 12 fiber classes / ~1,200 images;
GFA: 8 glass classes / ~800; PCF: 10 paint classes / ~1,000) describes datasets I can find
**no public evidence for**. Round numbers with no source, no licence and no acquisition protocol
are the signature of a drafted-ahead paper section rather than data in hand. Section 18 of the
methodology doc says exactly this itself: *"verify provenance, class definitions, image counts,
acquisition conditions, licensing, and leakage risks."*

**Nothing downstream is real until this resolves.** Three possible worlds:

- **(A) The lab actually holds the images.** Best case. Week 1 = ingest, document acquisition,
  define the class taxonomy, build a leakage-safe split (split by *source object*, not by image —
  multiple crops of one fiber must never straddle train/test).
- **(B) No such data exists.** Then the forensic framing survives only if we substitute a real,
  citeable microscopy benchmark. Strongest candidate: **LUWA** (lithic use-wear analysis,
  CVPR 2024) — microscopic, genuinely few-shot, and already reference [2] in the professor's own
  SOS list, so it is defensible as "the dataset we said we would use."
- **(C) Hybrid.** Whatever the lab has, plus a public anchor.

**Independent of A/B/C, a second purely methodological track is strongly recommended:**
validate ARC on the standard cross-domain few-shot benchmark (**BSCD-FSL**: CropDisease,
EuroSAT, ISIC, ChestX). ARC *is* a cross-domain few-shot method. On BSCD-FSL the numbers are
directly comparable to published work, which is what makes the method credible; the forensic data
then carries the *application* contribution. A paper with only an in-house, unreleasable dataset
and a novel controller is hard to defend — it asks reviewers to trust the method and the data at
the same time.

---

## 3. Architecture

```
image ─┬─► frozen ResNet-50 (no-grad, reference) ──► h_l^ref ──┐
       │                                                       ├─► signals U,D,T,S,N
       └─► adapted ResNet-50 (trainable)         ──► h_l^adp ──┘        │
                                                                        ▼
                                                                   controller
                                                                 (gate | FBSDE)
                                                                        │ a_l
                          h_l = (1-a_l)·h_l^ref + a_l·h_l^adp  ◄────────┘
                                         │
                                         ▼
                             embedding ─► ProtoNet head ─► logits
```

Keeping a **frozen reference copy** of the backbone (~100 MB fp32) is what makes drift,
topological mismatch and the forward gate all computable. It is the single most important
structural decision in the codebase.

### Layer granularity

ResNet-50 gives **16 bottleneck blocks** (plus stem) as the controllable units, optionally
coarsened to the 5 stage groups. 16 is the right default: fine enough to show interesting
layer-wise patterns in the paper's figures, coarse enough that the state tensor stays at
`16 x 5 = 80` dims.

---

## 4. The five signals

| Signal | Definition | Cost |
|---|---|---|
| `U_l` utility | `<grad_{θ_l} L_support , grad_{θ_l} L_query>` normalised — does a support step actually reduce query loss? A one-step meta-gradient. | 1 extra backward |
| `D_l` drift | functional: `1 - CKA(h_l^ref, h_l^adp)` on the same batch. Parameter drift `‖θ_l-θ_l⁰‖/‖θ_l⁰‖` logged as secondary. | negligible |
| `T_l` topology | activations → sample N=128 points → PCA to d'=16 → Vietoris–Rips (H0,H1, `ripser`) → persistence landscapes → **sliced Wasserstein** between ref and adapted. | **expensive, see below** |
| `S_l` stability | Hessian-vector product (Pearlmutter) + 2–3 power iterations → `λ_max`; `S_l = σ(-λ_max·lr)`. Grad-norm variance as cheap fallback. | 1 extra backward per power iter |
| `N_l` novelty | `‖μ_target - μ_BN‖ / σ_BN` using the pretrained BatchNorm running statistics. | free |

### Topology cost control (the main engineering risk)

Persistent homology per layer per step is infeasible. Mitigations, all mandatory:

1. **Cadence** — recompute `T_l` every K inner steps (K ≈ 5), hold constant between.
2. **Cache the reference** — the frozen net never changes, so `h_l^ref` diagrams for a given
   episode are computed **once** and reused across every inner step and every controller variant.
3. **Subsample hard** — N ≤ 128 points, PCA to ≤ 16 dims before Rips. Both are cited choices
   ([18] PCA, [20] Rips approximation), not hacks.
4. **Parallelise on CPU** — 20 threads sit idle while the GPU trains; a worker pool computes
   diagrams off the critical path.

If `T_l` still dominates wall-clock it becomes a cadence hyperparameter we report honestly
rather than something we hide.

---

## 5. Controller A — the gate, and a correctness trap

`a_l = σ(α·U_l - β·D_l - γ·T_l + δ·S_l + η·N_l)`

**The trap:** Figure 1 shows the gate feeding a *Gradient Masking Unit* that scales the
classification-loss gradient. If `a_l` only multiplies gradients, then `∂L/∂α = 0` through that
path — **the coefficients α…η receive no gradient and never learn.** Any implementation that
"trains the gate" this way is silently training nothing. Three legitimate resolutions, built in
this order:

- **(iii) Fixed coefficients.** α…η are hyperparameters tuned on a validation split. Honest,
  trivial, and the correct first baseline. *Ships first.*
- **(ii) Forward interpolation.** `h_l = (1-a_l)·h_l^ref + a_l·h_l^adp`. Now `a_l` sits on the
  forward path, is differentiable end-to-end, and the gate learns from task loss directly.
  This is also the most literal reading of "representation *control*." *Primary method.*
- **(i) Meta-gradient.** Learn α…η by backpropagating query loss through the inner loop
  (MAML-style) — exactly the meta-learning loop in Figure 3. *Secondary, expensive.*

Binary freeze/unfreeze ablation uses Gumbel-softmax ([19]).

---

## 6. Controller B — FBSDE

State `X_t` in `R^80` (16 layers x 5 signals), with "time" = inner adaptation step.
`dX_t = b(X_t,a_t,t)dt + σ(X_t,a_t,t)dW_t`, value `Y_t = V(t,X_t)`, `Z_t = grad V · σ`.
Deep-BSDE solver ([11]) with a **transformer value network over layer tokens** (per Figure 2) —
attention across layers is the right inductive bias, since whether to adapt layer 7 depends on
what layers 1–6 did. Terminal condition `Y_T = L_task(X_T)`. Optimal-stopping reading: keep
adapting while expected task benefit exceeds expected representational cost.

**Stated plainly: this may not beat the gate, and the paper is fine either way.** The docx
already frames it as a comparison, not an assumption. If the gate matches FBSDE at a fraction of
the cost, that is a publishable and honest negative result.

---

## 7. Experimental ladder

| Level | What | Gate mode | Must finish by |
|---|---|---|---|
| 0 | Baselines: frozen+linear, frozen+proto, last-layer FT, progressive unfreeze, full FT, LoRA/adapter | — | Week 1 |
| 1 | Gate with `U`,`D` only | (iii) | Week 2 |
| 2 | Gate + `T` + `S` + `N`, forward interpolation, ablations | (ii) | Week 3 |
| 3 | FBSDE controller, matched conditions | — | Week 4 / stretch |

Level 3 is explicitly a **stretch goal**. Levels 0–2 alone constitute a complete, submittable
paper. Scoping it that way now prevents a half-finished FBSDE from holding the whole thing
hostage.

### Protocol

- 5-way, K in {1,5,10,20}-shot, 15 queries per class.
- **600 test episodes x 3 seeds**, report mean ± 95% CI. Less than that is noise at these sizes.
- Per-episode test-time adaptation: T inner steps on the support set under controller
  supervision, then classify the query set. Backbone pretrained on ImageNet; no meta-training
  needed for Levels 0–2 with fixed coefficients.
- **Leakage discipline:** split by source object/specimen, never by image.

### Metrics

Accuracy / balanced accuracy / macro-F1 · representation drift at convergence · topological
descriptor distance · **fraction of parameters substantially modified** · wall-clock and control
overhead · the layer-wise `a_l` profile (this is the paper's money figure).

---

## 8. Repository layout

```
arc/
├── configs/              # hydra; one yaml per ladder level
├── src/arc/
│   ├── data/             # datasets, episodic sampler, leakage-safe splits
│   ├── backbones/        # resnet50 + activation taps + frozen reference
│   ├── signals/          # utility, drift, topology, stability, novelty
│   ├── controllers/      # static_gate, adaptive_gate, fbsde
│   ├── heads/            # protonet, linear probe
│   ├── train/            # episodic loop, baselines, meta loop
│   └── eval/             # metrics, CIs, protocol runners
├── experiments/          # launch scripts per level
├── scripts/              # server bootstrap, run registry, sync
├── tests/                # signal unit tests with analytic fixtures
└── paper/                # figures, tables, generated from results
```

Non-negotiables: every run writes a JSON manifest (git SHA, config hash, seed, full env);
results are append-only; figures regenerate from results with one command.

---

## 9. Four-week plan (revised from docx section 20)

| Week | Work | Gate to next week |
|---|---|---|
| 1 | Resolve datasets (section 2). Bootstrap server env. Backbone + taps + frozen ref. Episodic sampler. **All Level-0 baselines running end-to-end.** | Baseline numbers exist and are stable across seeds |
| 2 | Drift + utility signals. Gate modes (iii) and (ii). Level-1 results. Run infrastructure hardened. | Gate at least matches baselines on one dataset |
| 3 | Topology + stability signals (+ CPU worker pool). Level-2 and full ablation grid. **FBSDE skeleton in parallel.** | Ablations show which signals actually carry weight |
| 4 | FBSDE if alive; matched comparison; seed repeats; figures/tables; write-up. | — |

---

## 10. Citation hygiene (flagged, not urgent)

The reference lists contain at least one definite error: SOS ref [16] attributes *"Overcoming
catastrophic forgetting in neural networks"* to McCloskey and Cohen (1989). That title is
Kirkpatrick et al. (2017); McCloskey and Cohen wrote *"Catastrophic Interference in Connectionist
Networks: The Sequential Learning Problem."* Several 2026-dated entries should also be verified
against the actual arXiv/DOI records before submission. Full reference audit scheduled for
Week 4, noted now so a reviewer is not the one to find it.

---

## 11. What we will be able to claim

Earned by Levels 0–2:

> Layer-wise, representation-aware control of adaptation intensity improves few-shot
> classification under domain shift relative to uniform fine-tuning strategies, while modifying
> a smaller fraction of pretrained parameters.

Earned only with Level 3:

> Explicit temporal stochastic control of the adaptation trajectory further improves on
> state-feedback gating.

**Not** earned unless the optimisation dynamics genuinely support it (docx section 10 says this
too):

> any formal Lyapunov-stability *guarantee*.

We use Lyapunov exponents as a measured signal. That is a measurement, not a theorem, and the
manuscript must say so.
