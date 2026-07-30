# PROJECT HANDOVER

**Project:** Real-Time Driver Drowsiness Detection via Signal Reliability Gating
**Audience:** Thesis supervisor, external examiner, and the next research engineer
**Status:** First full experiment cycle complete (EXP-000 … EXP-004) + independent audit — **EXPERIMENT CYCLE 1 COMPLETE**
**Date:** 2026-07-29
**Target venues:** IEEE Intelligent Vehicles Symposium (IV) / IEEE ITSC (T-ITS as extension)

> This is a truthful handover. Every number quoted here traces to a logged
> experiment (`EXPERIMENT_REGISTRY.md`) or a committed artifact
> (`results/measured_results.json`). No performance result has been fabricated,
> estimated, or projected. Where a claim is *not yet measured*, it is marked
> **NOT MEASURED**.

---

## 1. Project Overview

The project builds a real-time, camera-based driver drowsiness detector whose
research contribution is **not** a new classifier but a **signal-reliability
gate**: a mechanism that decides *how much to trust* the fatigue evidence on
each frame before it is accumulated over time. The thesis argument is that
false positives in facial-landmark drowsiness systems come mostly from
*unreliable measurement conditions* (poor lighting, unstable landmarks,
talking) rather than from a weak classifier — so gating the evidence by its
measured reliability should reduce false-positive rate (FPR) at a matched
true-positive rate (TPR) without hurting detection of real drowsiness.

**Two claimed novelties:**

1. **Decomposed signal-reliability gate** — three interpretable component
   sub-scores (landmark-stability, brightness-quality, cue-consistency)
   combined by a weighted **geometric mean** into a reliability index
   `r ∈ [0,1]`, which **multiplicatively attenuates fused fatigue evidence
   before temporal accumulation**. It is **safety-asymmetric**: a SEVERE
   fatigue state is never suppressed by the gate.
2. **Variance-based speech-jitter MAR filter** — suppresses talking-induced
   *yawn* false positives by gating on the temporal variance σ²(MAR) of the
   mouth-aspect ratio.

Everything else (EAR/MAR geometry, head pose, PERCLOS, the 5-state hysteresis
machine, the optional CNN eye-validator) is standard and serves the two
novelties or the ablation.

---

## 2. System Overview

Fixed pipeline order (see `reports/IMPLEMENTATION_SPECIFICATION_FROZEN.md §2`):

```
frame → FaceMesh → geometry (EAR, MAR 2D, head pose solvePnP)
      → SignalQuality (3 sub-scores)
      → RobustnessGuard → reliability r ∈ [0,1]
      → TemporalAnalyzer (speech-jitter filter; injectable clock)
      → FatigueFusionEngine (weighted sum × agreement × r; SEVERE-exempt)
      → StateManager (5-state hysteresis machine)
      → alert / UI
```

| Module | File | Responsibility |
|---|---|---|
| Detector | `src/detector.py` | EAR, MAR (2D image-plane only), calibration |
| Pose | `src/pose_estimator.py` | head pose via `solvePnP` |
| Robustness | `src/robustness.py` | 3-component reliability, geometric mean, SEVERE exemption |
| Temporal | `src/temporal_analyzer.py` | monotonic/injected-clock durations, σ²(MAR) speech gate, PERCLOS |
| Fusion | `src/fatigue_fusion.py` | weighted evidence × agreement × reliability |
| State | `src/state_manager.py` | 5-state hysteresis machine, face-loss escalation |
| CNN (ablation) | `src/cnn_validator.py` | optional selective eye validation — **OFF by default** |
| Headless core | `src/frame_processor.py` | camera-free per-frame pipeline shared by app + benchmark |
| Config | `src/config.py` | all thresholds/weights centralized — no magic numbers in logic |

**Key design fact:** `src/frame_processor.py` is the single per-frame code path
used by *both* the live application (`src/main.py`) and the offline evaluation
harness — the benchmark and deployment cannot diverge.

---

## 3. Dataset Documentation

All datasets live under `Data/`. Counts below are **facts about the corpus on
disk** (not results), reproducible via the enumeration tools.

| Dataset | Role | Contents (measured) | Status |
|---|---|---|---|
| **NTHU-DDD** | Primary video/temporal evaluation | 66,521 labelled JPG frames, 4 subjects (`001, 002, 005, 006`); 36,030 drowsy / 30,491 notdrowsy; conditions: nonsleepyCombination, sleepyCombination, slowBlinkWithNodding, yawning | Active |
| **MRL Eye** | CNN-ablation training ONLY (subject-disjoint split) | 84,898 PNG eye crops, 37 subjects | Active (ablation) |
| **YawDD** | Video/yawn evaluation | 348 AVI clips | Active |
| **drowsiness_detection** | — | 100% byte-duplicate of MRL Eye | **QUARANTINED** — loader raises `RuntimeError` |

**NTHU ground truth** is not invented: labels are embedded in the filename
grammar `<subject>_<glasses>_<condition>_<frameindex>_<label>.jpg`, parsed by
`evaluation/nthu_ground_truth.py`. Mapping is `notdrowsy → 0`, `drowsy → 1`.

**Honest limitation (must appear in the paper):** NTHU labels are
clip-condition derived, so an open-eyed frame inside a "yawning" clip is still
labelled drowsy. This makes *frame-level* FPR a **conservative** (pessimistic)
estimator of the system's true *event-level* FPR. Temporal integration is what
converts these noisy per-frame labels into stable event decisions.

**Data integrity:** MRL subject partitions are regenerated leak-free by
`tools/build_subject_disjoint_splits.py` (seed 42, asserted zero pairwise
subject overlap). The earlier shipped MRL split had 100% subject leakage and is
no longer used — see the EXP-000 retraction (§6).

---

## 4. Experimental Design (frozen)

- **Splitting:** Leave-One-Subject-Out (LOSO) / subject-disjoint GroupKFold.
  Every reported number is cross-subject. Seed = 42, deterministic.
- **Primary metric:** FPR at matched TPR — the operating point is fixed on the
  **V0 baseline** ROC (`target_tpr = 0.80`) and then held constant across all
  variants.
- **Secondary metrics:** ROC-AUC (trapezoidal), FPR/hour.
- **Feasibility:** per-stage latency, throughput (FPS), memory — to be
  *measured on-device* (Raspberry Pi 4) for the final paper claim.
- **Threshold variable:** `fatigue_score ∈ [0,1]` (continuous, exposed on
  `FrameResult`) is the ROC sweep variable.

**Ablation variants (frozen V0–V4)** — toggles are
`(enable_speech_filter, enable_reliability_gate, enable_cnn)`:

| Variant | Speech filter | Reliability gate | CNN | Meaning |
|---|---|---|---|---|
| V0 | ✗ | ✗ | ✗ | baseline weighted fusion |
| V1 | ✓ | ✗ | ✗ | + speech-jitter filter |
| V2 | ✗ | ✓ | ✗ | + reliability gate |
| V3 | ✓ | ✓ | ✗ | full proposed system |
| V4 | ✓ | ✓ | ✓ | full + CNN arm (prior-art ablation) |

**Result flow (hard rule):** every run gets an `EXP-###` row in
`EXPERIMENT_REGISTRY.md` **before** citation; figures/tables regenerate ONLY
from `results/measured_results.json`; no number reaches `paper/main.tex`
without a committed artifact.

Harness: `evaluation/loso_harness.py` implements V0–V4, the ROC/AUC math, and
the fixed-operating-point logic. It **does not persist** results unless run with
`--write`, and it prints a reminder to log an `EXP-###` row before citing.

---

## 5. Current Progress

**Done and verified:**
- Full per-frame pipeline runs end-to-end on real NTHU frames (headless).
- NTHU ground-truth mapping (`evaluation/nthu_ground_truth.py`).
- LOSO evaluation harness with V0–V4 ablation toggles, ROC/AUC, fixed
  operating point (`evaluation/loso_harness.py`).
- Latency benchmark harness (`evaluation/benchmark_nthan_yawdd.py`).
- Experiment registry + measured-results schema + figure generator that
  refuses to run without measured JSON.
- Integrity verifier (`evaluation/verify_integrity.py`) — **6/6 invariants
  pass**.
- Test suite: **17/17 unit tests pass**, **3/3 smoke tests pass** (verified
  2026-07-29).

**Measured results to date** (each with an `EXP-###` row + committed artifact):
- **EXP-001 — host latency:** mean **3.205 ms/frame** (p50 3.080, p95 3.316,
  max 29.27 [FaceMesh cold-start], ≈ 312 FPS), 300 NTHU frames, **Darwin-arm64
  host, NOT Raspberry Pi 4**, seed 42.
- **EXP-002 — MicroEyeNet training** (subject-disjoint MRL, seed 42): VAL acc
  0.9402 / F1 0.9262; TEST acc 0.9362 / F1 **0.9623**; **19,745 params measured**
  (see `reports/EXP-002_PARAMETER_AUDIT.md`). Early-stopped epoch 13 (best 8).
- **EXP-003 — quantization:** INT8 **25.55 KB** (3.18× smaller, −0.026% F1);
  FP16 43.46 KB (0.0% F1 loss). No retraining.
- **EXP-004 — V0–V4 LOSO ablation on NTHU-DDD (66,521 frames):** **honest
  NEGATIVE result.** At matched TPR=0.80 the reliability gate (V2) does not
  reduce FPR vs V0 (0.6244 vs 0.6241, flat); the speech filter (V1/V3/V4) *raises*
  FPR to ≈0.669; ROC-AUC never beats baseline (V0 0.629 highest, V3/V4 0.613
  lowest). **V4 ≡ V3 byte-identical** — the CNN verdict feeds only the
  `should_alarm` boolean, never the swept `fatigue_score`. Independently
  re-audited (`reports/EXP-004_AUDIT/`).

**Not yet done — the next phase (official roadmap, `EXPERIMENT_REGISTRY.md §4`):**
- **EXP-005 — Event-Level Alarm Evaluation** (FPR/hour + episode latency over
  V0–V4), where the CNN arm and the gate's spurious-alarm protection can be
  observed.
- **EXP-006 — Gate Redesign Evaluation** — re-architect the gate as an additive
  decision-layer term (the EXP-004 audit's recommended follow-up; the audit
  labels it "EXP-005", official ID is EXP-006).
- **EXP-007 — Raspberry Pi Deployment Evaluation** (on-device Pi 4
  latency/memory/thermal).
- **EXP-008 — Second Dataset Validation** (optional; another subject-disjoint
  dataset).
- Populating `paper/main.tex` results from the committed artifacts (writing
  task; no `EXP-###` of its own).

---

## 6. Known Risks & Mitigations

| Risk | Mitigation (in place) |
|---|---|
| Fabricated/optimistic numbers re-entering the record | Integrity verifier gates every commit; `plot_paper_figures.py` refuses without measured JSON; EXP-000's false claims are formally retracted in the registry. |
| Latency claim mistaken for Pi 4 | EXP-001 and `results/measured_results.json` both explicitly state "Darwin-arm64 host, NOT a Raspberry Pi 4." |
| Data leakage inflating accuracy | Subject-disjoint LOSO everywhere; leaky MRL split replaced; `drowsiness_detection` duplicate loader hard-raises. |
| NTHU frame-level labels being over-interpreted | Limitation documented in code + here; framed as conservative FPR; temporal integration handles per-frame noise. |
| Novelty scope creep (a phantom 4th gate component) | Gate is asserted to have exactly 3 components by test + integrity verifier; a 4th component fails CI. |
| Registry numbering collision | **Resolved.** The stale "planned EXP-001…005" block was removed; §2/§3 (EXP-000…004, measured) are authoritative for completed work. Future experiments now follow the **official roadmap** in `EXPERIMENT_REGISTRY.md §4` (EXP-005 event-level, EXP-006 gate redesign, EXP-007 Pi deployment, EXP-008 second dataset). A reconciliation table there maps every earlier provisional ID (audit + CNN spec) onto the official one, so no live conflict remains. |

---

## 7. Thesis Writing Notes

- Frame the contribution as **reliability-aware evidence gating**, not "a better
  drowsiness classifier." The baseline (V0) is a competent weighted-fusion
  detector; the story is the *delta* V0 → V3.
- **Report EXP-004 honestly as a negative result at the frame level.** At matched
  TPR=0.80 the V0→V3 delta is not an improvement (gate flat, speech filter worse,
  AUC never above baseline). The scientific value is the *diagnosis* — the
  frame-level FPR@TPR metric, on NTHU's conservative clip-derived labels, cannot
  observe episode-level spurious-alarm behaviour (and V4≡V3 shows the CNN never
  touches the swept score). This motivates the EXP-005 event-level re-evaluation.
  Do not retune thresholds until the number turns positive — that would be the
  integrity failure this whole regime exists to prevent.
- Lead the results with **FPR@matched-TPR** (the metric the design optimizes),
  then ROC-AUC and FPR/hour as support.
- Report the NTHU frame-level-label limitation up front — reviewers will raise
  it; owning it (as a conservative estimator) is stronger than hiding it.
- The **geometric mean** is the right thing to explain carefully: it means any
  single collapsing component (e.g. brightness → 0) drives reliability toward 0,
  which is the intended "one bad signal poisons trust" behaviour, unlike an
  arithmetic mean.
- The **SEVERE exemption** is a safety argument, not an accuracy trick — state
  it as such.
- Latency: only the host number (3.205 ms) is real today. Do **not** write a Pi
  4 number until on-device profiling is logged.

---

## 8. Final Recommendation

The pipeline, evaluation infrastructure, data mapping, and integrity tooling are
complete and verified. The frozen research design and implementation spec are
intact. The first full experiment cycle (EXP-000 … EXP-004) has been run,
measured, logged, and — for EXP-004 — independently re-audited. No fabricated
results survive anywhere citable.

**The headline outcome is an honest negative result:** at the frame level, under
strict subject-disjoint LOSO on NTHU-DDD, the signal-reliability gate does not
reduce FPR at matched TPR and the speech-jitter filter worsens it. This is a
legitimate scientific finding, not a blocker — and the EXP-004 audit explains
*why* the frame-level protocol cannot see the mechanisms the design targets.

**The project is EXPERIMENT-CYCLE-1 COMPLETE.** The next engineer should follow
the **official roadmap** (`EXPERIMENT_REGISTRY.md §4`): **EXP-005** Event-Level
Alarm Evaluation, then **EXP-006** Gate Redesign Evaluation, **EXP-007**
Raspberry Pi Deployment Evaluation, and the optional **EXP-008** Second Dataset
Validation — then paper population. Log every run as an `EXP-###` row before
citing it. The earlier EXP-005 naming ambiguity is resolved by that roadmap and
its reconciliation table.
