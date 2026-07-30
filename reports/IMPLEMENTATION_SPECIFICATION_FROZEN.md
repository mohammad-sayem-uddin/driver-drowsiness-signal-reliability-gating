# FROZEN IMPLEMENTATION SPECIFICATION (ENGINEERING BLUEPRINT)

**Project:** Real-Time Driver Drowsiness Detection via Signal Reliability Gating
**Status:** 🔒 **FROZEN** — 2026-07-28
**Target venue:** IEEE IV / IEEE ITSC (T-ITS as extension)
**Preconditions:** All five freeze-report integrity preconditions CLEARED (see §0).

This document is the single authoritative engineering contract for the
implementation phase. The **research design is frozen** and MUST NOT be
revisited unless implementation uncovers a *concrete* flaw (in which case:
stop, document the flaw here, and get design sign-off before proceeding).
This spec governs *how* we build and measure — not *what* we research.

---

## 0. Precondition clearance record (entry gate)

| # | Precondition | Resolution | Evidence |
|---|---|---|---|
| 1 | Purge fabricated results | Paper results→PLACEHOLDER; figure generator refuses without measured JSON; benchmark dry-run returns empty metrics + `NotImplementedError`; unmeasured latency claims neutralized in `cnn_validator.py`; EXP-000 retraction logged | `paper/main.tex`, `evaluation/plot_paper_figures.py`, `evaluation/benchmark_nthan_yawdd.py`, `EXPERIMENT_REGISTRY.md` |
| 2 | Restore executability | `main.py:455` badge block fixed; standalone headless smoke test added (3/3 pass) | `src/main.py`, `tests/smoke_test.py` |
| 3 | Remove data leakage | Subject-disjoint MRL split (seed 42, asserted 0 overlap); `drowsiness_detection` loader raises on use | `tools/build_subject_disjoint_splits.py`, `Data/mrl_eye/splits_subject_disjoint/`, `src/data_loaders.py` |
| 4 | Honest reliability gate | Phantom `tracking` component removed; 3 real components (stability, brightness, consistency), weights (0.45,0.30,0.25) | `src/robustness.py`, `src/config.py`, `tests/test_suite.py` |
| 5 | Remove false model asset | Byte-identical `micro_eyenet_int8.tflite` deleted; trainer docstring corrected | `models/`, `tools/train_cnn.py` |

**Verification at freeze:** `python3 -m unittest tests.test_suite` → 17/17 OK;
`python3 tests/smoke_test.py` → 3/3 OK.

---

## 1. Research question & contributions (frozen — restated for traceability)

**RQ.** Does a decomposed signal-reliability gate combined with a speech-jitter
MAR filter reduce the false-positive rate (FPR) at a matched true-positive rate
(TPR), relative to a weighted-fusion baseline, while remaining real-time on a
Raspberry Pi 4 CPU?

**Core contributions (claimed):**
1. **Decomposed signal-reliability gate** — 3 component sub-scores
   (landmark-stability, brightness, cue-consistency) → geometric-mean
   reliability index → multiplicative attenuation of fused evidence *before*
   temporal accumulation; **safety-asymmetric**: SEVERE states are never
   suppressed.
2. **Variance-based speech-jitter MAR filter** — suppresses talking-induced
   yawn false positives via temporal σ²(MAR) gating.

**Supporting (not a claimed contribution):** 2D/3D metric separation (MAR in
image plane, EAR in 3D). **Ablation-only prior art:** selective MicroEyeNet
CNN eye validation.

---

## 2. Frozen system architecture (module contract)

Pipeline order is fixed:

```
frame → FaceMesh → geometry (EAR 3D, MAR 2D, head pose solvePnP)
      → SignalQuality (3 sub-scores)
      → RobustnessGuard → reliability r ∈ [0,1]
      → TemporalAnalyzer (wall-clock; speech-jitter filter)
      → FatigueFusionEngine (weighted; attenuated by r; SEVERE-exempt)
      → StateManager (5-state hysteresis machine)
      → alert / UI
```

| Module | File | Frozen responsibility | Do NOT |
|---|---|---|---|
| Detector | `src/detector.py` | EAR (3D), MAR (2D-only), calibration | inflate MAR with z |
| Robustness | `src/robustness.py` | 3-component reliability, geometric mean, SEVERE exemption | add a 4th/phantom component |
| Temporal | `src/temporal_analyzer.py` | monotonic-clock durations, σ²(MAR) speech gate | frame-count timing |
| Fusion | `src/fatigue_fusion.py` | weighted sum × agreement × reliability | suppress SEVERE |
| State | `src/state_manager.py` | 5-state hysteresis, face-loss escalation | skip escalation |
| CNN (ablation) | `src/cnn_validator.py` | optional selective eye validation | be on the default path |

Config is centralized in `src/config.py`; **no magic numbers** in logic modules.

---

## 3. Frozen experimental protocol

**Datasets.** NTHU-DDD + YawDD (primary evaluation, video/temporal). MRL Eye
(CNN-ablation training only, via the subject-disjoint split). `drowsiness_detection` is **banned**.

**Splitting.** Leave-One-Subject-Out (LOSO) / subject-disjoint GroupKFold.
Every reported number is cross-subject. Seed = 42, deterministic.

**Primary metric.** FPR at matched TPR (operating point fixed on the baseline
ROC, then held constant across variants). **Secondary:** ROC-AUC, PR-AUC,
FPR/hour. **Feasibility:** Raspberry Pi 4 per-stage latency, throughput (FPS),
memory, thermal — all *measured on-device*.

**Ablation variants (EXP-005):**
`V0` baseline weighted fusion → `V1` + speech-jitter filter → `V2` +
reliability gate → `V3` full (both) → `V4` full + CNN arm (prior-art ablation).

**Result flow (hard rule).** Every run gets an `EXP-###` row in
`EXPERIMENT_REGISTRY.md` **before** citation. Figures/tables regenerate ONLY
from `results/measured_results.json`. No number reaches `paper/main.tex` that
is not backed by a committed results artifact.

---

## 4. Implementation work-list (execution order)

1. **Wire the real per-frame pipeline into the benchmark** — replace the
   `NotImplementedError` path in `evaluation/benchmark_nthan_yawdd.py` with the
   actual MediaPipe→geometry→gate→fusion→state loop over NTHU/YawDD frames;
   emit per-frame scores + latencies.
2. **Results artifact writer** — produce `results/measured_results.json` in the
   schema `plot_paper_figures.py` expects (`roc`, `latency_ms`).
3. **LOSO harness** — drive `SubjectGroupKFoldSplitter` over the eval set;
   aggregate FPR@matched-TPR per fold; log to registry.
4. **Ablation runner** — execute V0–V4; one registry row each.
5. **On-device profiling** — run `evaluation/latency_memory_profiler.py` on the
   Pi 4; record per-stage latency/memory/thermal.
6. **CNN ablation (optional)** — retrain via `tools/train_cnn.py` pointed at the
   subject-disjoint split; single artifact `eye_state_model.tflite`.
7. **Populate paper** — fill `paper/main.tex` results ONLY from committed JSON;
   regenerate figures.

Each step: land code → run → log `EXP-###` → commit artifact. Never batch
numbers ahead of measurement.

---

## 5. Integrity invariants (must hold at every commit)

- `python3 tests/smoke_test.py` and `python3 -m unittest tests.test_suite` pass.
- No performance number exists in code/paper/reports without a matching
  `EXP-###` registry row and a committed artifact.
- Reliability gate has exactly 3 components; SEVERE is never suppressed.
- All reported metrics are subject-disjoint (LOSO).
- `drowsiness_detection` is never loaded.

**End of frozen specification.**

---

## Appendix A. Implementation-phase change log (enabling changes only)

These are engineering enablers discovered during implementation. None alters
the frozen research design; each is a bug/infra fix required to *run* it.

- **A1 — Dependency alignment (infra bug).** The environment shipped protobuf
  7.35 + TensorFlow 2.21, which crashes mediapipe's legacy FaceMesh
  (`FieldDescriptor has no attribute 'label'`) — i.e. the core pipeline could
  not start at all. Fixed by pinning `protobuf>=4.25.3,<5`, `tensorflow==2.17.1`,
  and removing a mismatched `jax/jaxlib`. Recorded in `requirements.txt`.
  Verified: FrameProcessor runs end-to-end on real NTHU frames (8/8 faces).
- **A2 — Injectable timestamps (enabling change).** `TemporalAnalyzer.update`
  and `StateManager.update` gained an optional `timestamp` param (defaults to
  `time.monotonic()`, so the live path is unchanged). Offline video evaluation
  passes the video clock (`frame_index / fps`) so temporal integration reflects
  the recording, not processing speed. Required by the frozen LOSO protocol.
- **A3 — Shared headless pipeline core (`src/frame_processor.py`).** Extracts
  the exact per-frame flow used by the live app into a camera-free
  `FrameProcessor`, so the benchmark and deployment cannot diverge.

- **A4 — Post-freeze reconciliation notes (2026-07-29; documentation only, no
  design change).** Recorded here to keep this frozen contract honest against
  what was actually built and measured:
  - **Parameter count.** MicroEyeNet's measured parameter count is **19,745**
    (EXP-002, `reports/EXP-002_PARAMETER_AUDIT.md`). Earlier prose in working
    notes said "~9.5K"; that estimate is superseded — cite 19,745.
  - **§4 work-list file names are aspirational, not current.** The work-list
    above names `evaluation/latency_memory_profiler.py` (step 5) and an
    "Ablation runner" (step 4); both were later removed as broken/contradictory
    and **superseded** by `evaluation/benchmark_nthan_yawdd.py`,
    `evaluation/loso_harness.py`, and the additive `evaluation/exp004_report.py`
    orchestrator (see `IMPLEMENTATION_LOG.md`, cleanup 2026-07-28). `tools/
    train_eye_cnn.py` was likewise removed; EXP-002 used
    `tools/train_exp002_microeyenet.py`. The *steps* were all executed; only the
    file names differ.
  - **EXP-004 outcome pointer.** The ablation (work-list steps 3–4) was run as
    **EXP-004** and produced an **honest negative result**: at matched TPR=0.80
    the reliability gate does not reduce FPR and the speech-jitter filter raises
    it; V4≡V3 byte-identical (the CNN verdict routes only to the `should_alarm`
    boolean, not the swept `fatigue_score`). See `reports/EXP-004_REPORT.md`, the
    independent audit `reports/EXP-004_AUDIT/`, and `EXPERIMENT_REGISTRY.md` §3.
    This does not alter the frozen design; it constrains the claims and motivates
    the event-level EXP-005.
  - **Experiment-ID reconciliation (2026-07-29).** The §3 label "**Ablation
    variants (EXP-005)**" is a *provisional* pre-freeze number: that ablation was
    actually executed as **EXP-004** (V0–V4, complete). Future experiment IDs are
    now fixed by the official roadmap in `EXPERIMENT_REGISTRY.md §4` — **EXP-005**
    Event-Level Alarm Evaluation, **EXP-006** Gate Redesign Evaluation, **EXP-007**
    Raspberry Pi Deployment Evaluation, **EXP-008** Second Dataset Validation
    (optional). Treat that registry section (not the provisional label in §3) as
    authoritative; the §3 design text is left unchanged as a frozen historical
    record.

- **A5 — Additive alarm-decision exposure on `FrameResult` (2026-07-29; enabling
  change for EXP-005, no design change).** `FrameResult` (the `src/frame_processor.py`
  per-frame return artifact declared infra in A3) gained six additive, default-valued
  fields so the event-level EXP-005 evaluation can read the alarm decision the
  pipeline already computes, without re-running or altering it:
  - `should_alarm: bool = False`, `alarm_level: int = 0` — the alarm the
    `StateManager` already emitted this frame.
  - `cnn_override_active: bool = False` — whether the CNN false-positive
    suppression fired (V4 arm).
  - `alarm_suppressed_actual: bool = False` — the reliability-gate suppression
    that *actually* flipped `should_alarm` (populated from `state.alert_suppressed`;
    deliberately distinct from the pre-existing `alert_suppressed` field, which
    mirrors the guard's *recommendation* `snap.alert_suppressed`).
  - `face_visible: bool = True`, `seconds_since_face_lost: float = 0.0` —
    face-loss context for event segmentation.
  All fields are populated verbatim from the already-computed `SystemState`; the
  swept `fatigue_score` is untouched, and no algorithm, threshold, weight, or
  decision path changed. Backward compatibility: the sole existing reader
  (`evaluation/loso_harness.py:110`) reads only `.fatigue_score`; the new fields
  carry defaults, so no existing reader breaks. Verified green: `tests/test_suite.py`
  (17/17), `tests/smoke_test.py` (3/3), `evaluation/verify_integrity.py` (I1–I6, 6/6).
