# Real-Time Driver Drowsiness Detection via Signal Reliability Gating

A real-time, webcam-based driver drowsiness detector built on MediaPipe
FaceMesh, classical facial geometry, and temporal fusion. The **research
contribution is not a new classifier** — it is a **signal-reliability gate**
that estimates how *trustworthy* the fatigue evidence is on each frame (given
lighting, landmark stability, and cue agreement) and **multiplicatively
attenuates that evidence before it accumulates over time**.

> **Status (2026-07-29): IMPLEMENTATION + FIRST EXPERIMENT CYCLE COMPLETE
> (EXP-000 … EXP-004), plus an independent scientific audit of EXP-004.**
> The pipeline, LOSO harness, CNN training, quantization, and the full V0–V4
> ablation have all been run, measured, and logged.
>
> **Headline result — an honest negative.** Under strict subject-disjoint LOSO
> on NTHU-DDD, the signal-reliability gate **does not** reduce frame-level
> FPR at matched TPR=0.80 (V2 0.624 vs V0 0.624, flat), the speech-jitter
> filter *raises* FPR (V1/V3/V4 ≈0.669), and ROC-AUC never beats baseline
> (V0 0.629 highest). See [EXP-004](reports/EXP-004_REPORT.md) and the
> [audit](reports/EXP-004_AUDIT/). EXP-005, the **event-level alarm
> evaluation**, is now **complete** (66,521 frames, audited ACCEPT): it
> confirms the negative — event recall 0.122 at 6.5–9.7 false alarms/hour,
> only 2 of 4 subjects fire, and all three observability gates fail. See
> [EXP-005](reports/EXP-005_REPORT.md) and its [audit](reports/EXP-005_AUDIT.md).
>
> Measured latency remains **host-only** (EXP-001: 3.205 ms/frame on a
> Darwin-arm64 host — **NOT** a Raspberry Pi 4).

---

## Two claimed novelties

1. **Decomposed signal-reliability gate** — three interpretable sub-scores
   (landmark-stability, brightness-quality, cue-consistency) combined by a
   **weighted geometric mean** into a reliability index `r ∈ [0,1]`, weights
   `(0.45, 0.30, 0.25)`. The gate attenuates the fatigue **score** for all
   states (`fatigue_fusion.py`, `raw_score *= reliability`, unconditional); a
   separate **state-level** guard governs exit from SEVERE (`state_manager.py`).
2. **Speech-jitter MAR filter** — suppresses talking-induced
   *yawn* false positives by gating on the **mean absolute per-frame change in
   MAR** (mean |ΔMAR|, threshold 0.05) of the mouth-aspect ratio.

## Pipeline

```
frame → FaceMesh → geometry (EAR, MAR 2D, head pose solvePnP)
      → SignalQuality (3 sub-scores)
      → RobustnessGuard → reliability r ∈ [0,1]
      → TemporalAnalyzer (speech-jitter filter; injectable clock)
      → FatigueFusionEngine (weighted sum × agreement × r; applied to all states)
      → StateManager (5-state hysteresis machine; state-level SEVERE guard)
      → alert / UI
```

`src/frame_processor.py` is the single per-frame code path shared by both the
live app (`src/main.py`) and the offline evaluation harness — so the benchmark
and deployment cannot diverge.

## Repository layout

| Path | Role |
|---|---|
| `src/` | Detection pipeline (geometry, pose, robustness gate, temporal, fusion, state machine, frame processor, config, live app). |
| `evaluation/` | LOSO harness, NTHU ground-truth mapping, latency benchmark, figure generator, integrity verifier. |
| `tools/` | CNN trainer, subject-disjoint split generator, dataset integrity audit. |
| `tests/` | 17 unit tests + 3 smoke tests + 65 event-metric tests (`test_event_metrics.py`). |
| `Data/` | NTHU-DDD, MRL Eye, YawDD (and the quarantined `drowsiness_detection`). |
| `results/` | `measured_results.json` + canonical paper figures. |
| `experiments/` | Per-experiment artifact folders (`EXP-002_*`, `EXP-003_*`, `EXP-004_*`): metrics JSON, CSVs, plots, logs. |
| `reports/` | Current experiment reports + the **frozen** spec + the EXP-004 audit. See [reports/README.md](reports/README.md); historical/process docs are in [reports/archive/](reports/archive/). |
| `paper/` | Manuscript draft (results populated only from committed artifacts). |

## Datasets (facts on disk)

| Dataset | Contents | Role |
|---|---|---|
| NTHU-DDD | 66,521 JPG, 4 subjects, 36,030 drowsy / 30,491 notdrowsy | Primary temporal eval |
| MRL Eye | 84,898 PNG, 37 subjects (subject-disjoint split) | CNN-ablation training only |
| YawDD | 348 AVI | Video/yawn eval |
| drowsiness_detection | 100% byte-duplicate of MRL Eye | **QUARANTINED** — loader raises `RuntimeError` |

## Getting started

```bash
pip install -r requirements.txt      # protobuf>=4.25.3,<5, tensorflow==2.17.1

python3 src/main.py                   # live webcam demo

# verification gates (keep all green)
python3 evaluation/verify_integrity.py     # 6/6 integrity invariants
python3 -m unittest tests.test_suite       # 17/17 unit tests
python3 tests/smoke_test.py                # 3/3 smoke tests
```

## Evaluation & experiment discipline

The primary metric is **FPR at matched TPR**: FPR is read at each variant's
**nearest achievable TPR** to the `target_tpr = 0.80` set on the **V0 baseline**
ROC (realized TPRs 0.80 / 0.7989 / 0.8006 across variants; see
`evaluation/loso_harness.py`), for the frozen ablation variants **V0–V4**
(toggles: speech filter, reliability gate, CNN). Splitting is subject-disjoint
**LOSO**, seed **42**.

**Hard rule:** no performance number is citable without an `EXP-###` row in
`EXPERIMENT_REGISTRY.md` **and** a committed artifact in `results/` or
`experiments/`. This regime exists because earlier fabricated claims (EXP-000's
"0 leakage / <0.5 ms") were retracted. Figures regenerate only from measured
artifacts.

**Experiments run so far** (see [EXPERIMENT_REGISTRY.md](EXPERIMENT_REGISTRY.md)
for the full ledger and [reports/](reports/) for the per-experiment reports):

| Exp | What | Headline |
|---|---|---|
| EXP-001 | Host latency benchmark | 3.205 ms/frame (Darwin-arm64 host, **not** Pi 4) |
| EXP-002 | MicroEyeNet training (subject-disjoint MRL) | TEST F1 0.9623; 19,745 params |
| EXP-003 | INT8 / FP16 quantization | INT8 25.55 KB, −0.026% F1 |
| EXP-004 | LOSO ablation V0–V4 on NTHU-DDD | **Negative:** gate does not lower FPR@TPR; see report + audit |

**EXP-005** Event-Level Alarm Evaluation is **complete** (see above). The
remaining experiments follow the **official roadmap** in
[EXPERIMENT_REGISTRY.md](EXPERIMENT_REGISTRY.md) §4: **EXP-006** Gate Redesign
Evaluation, **EXP-007** Raspberry Pi Deployment Evaluation, and the optional
**EXP-008** Second Dataset Validation.
Every new run must land an `EXP-###` row + committed artifact before it is cited.

## Documentation (read in this order)

1. **[AGENT_MEMORY.md](AGENT_MEMORY.md)** — fast start for a new engineer/agent.
2. **[PROJECT_CONTEXT.md](PROJECT_CONTEXT.md)** — the single source of truth.
3. **[HANDOVER.md](HANDOVER.md)** — supervisor/thesis handover.
4. **[reports/IMPLEMENTATION_SPECIFICATION_FROZEN.md](reports/IMPLEMENTATION_SPECIFICATION_FROZEN.md)** — the frozen engineering contract.
5. **[EXPERIMENT_REGISTRY.md](EXPERIMENT_REGISTRY.md)** — what has actually been measured.
6. **[IMPLEMENTATION_LOG.md](IMPLEMENTATION_LOG.md)** — why the repo looks like it does.

## Frozen decisions (do not change without design sign-off)

- Reliability gate = exactly **3** components, weighted geometric mean, weights
  `(0.45, 0.30, 0.25)`.
- The gate attenuates the fatigue score for all states; a separate state-level
  guard governs exit from SEVERE (`state_manager.py`).
- Subject-disjoint LOSO everywhere; seed 42.
- MAR stays 2D; CNN is ablation-only and OFF by default.
- `drowsiness_detection` stays quarantined.
- No performance number without an `EXP-###` row + committed artifact.
