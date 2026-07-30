# MASTER EXPERIMENT REGISTRY

**Project**: Driver Drowsiness Signal Reliability Gating  
**Maintained By**: AI Research Team & Reproducibility Engineers  
**Target Venues**: IEEE Intelligent Vehicles Symposium (IV) / IEEE T-ITS  
**Date Initialized**: July 2026

---

## 1. Registry Standard & Logging Instructions

Every neural network training run, ablation experiment, and benchmark evaluation MUST be assigned a sequential Experiment ID (e.g., `EXP-001`, `EXP-002`) and logged in this registry before the results can be cited in the research manuscript.

### Required Logging Fields:
- **Exp ID**: Unique identifier (e.g., `EXP-001`)
- **Date**: ISO Date (`YYYY-MM-DD`)
- **Dataset & Split**: Training dataset and split version (`BASELINE_v1.0`)
- **Random Seed**: Integer seed (default `42`)
- **Hyperparameters**: Epochs, Batch Size, Optimizer, Initial Learning Rate
- **Model Architecture**: Layer specs & parameter count
- **Validation Metrics**: Loss, Accuracy %, F1-Score, ROC-AUC
- **Test Metrics**: Accuracy %, F1-Score, FPR/hr
- **Export Specs**: TFLite size (KB) & CPU latency (ms)
- **Scientific Takeaway**: Core takeaway / action item

---

## 2. Master Experiment Log Table

| Exp ID | Date | Dataset / Split | Model Arch | Seed | Epochs | Batch | Optimizer | LR | Val Acc | F1-Score | TFLite Size | Latency | Key Takeaway / Notes |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| **EXP-000** | 2026-07-24 | Ingested Base | MicroEyeNet | 42 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A (unmeasured) | Data foundation initialized. NOTE: earlier "0 leakage / <0.5ms" claims RETRACTED — see below. |
| **EXP-001** | 2026-07-28 | NTHU-DDD (300 frames, no split) | Full pipeline (no CNN) | 42 | N/A | N/A | N/A | N/A | N/A | N/A | 25.9 KB | 3.205 ms mean (Darwin-arm64) | First MEASURED latency. Host-only, NOT Pi 4. Latency feasibility harness verified. |
| **EXP-002** | 2026-07-28 | MRL subject-disjoint (train 70,551 / val 4,970 / test 9,377) | MicroEyeNet (19,745 params) | 42 | 13 | 64 | Adam | 1e-3 | 0.9402 | 0.9262 | NOT MEASURED | NOT MEASURED | First trained MicroEyeNet baseline. VAL acc 0.9402 / F1 0.9262 @0.5; TEST acc 0.9362 / F1 0.9623 @0.5. Early-stopped at epoch 13 (best epoch 8). TFLite export = EXP-003 scope; no on-device latency measured. |
| **EXP-003** | 2026-07-28 | MRL subject-disjoint TEST split (9,377 samples) | MicroEyeNet (19,745 params) Float16 & INT8 TFLite | 42 | N/A | N/A | N/A | N/A | N/A | 0.9623 (FP16) / 0.9620 (INT8) | FP16: 43.46 KB / INT8: 25.55 KB | NOT MEASURED (Pi 4 profile pending) | Quantization completed without retraining. Float16: 43.46 KB (1.87x comp, 0.0% F1 degradation). INT8: 25.55 KB (3.18x comp, -0.026% F1 degradation). Verification passed. |
| **EXP-004** | 2026-07-28 | NTHU-DDD LOSO (subject-disjoint, 66,521 frames, 4 subjects) | Full pipeline ablation V0–V4 (heuristic; CNN = trained MicroEyeNet arm in V4) | 42 | N/A | N/A | N/A | N/A | N/A | 0.687 (V0) @ fixed op pt | N/A (no export) | N/A (accuracy exp) | First MEASURED accuracy/ROC. **Negative result:** reliability gate (V2) does not reduce FPR@matched-TPR=0.80 vs V0 (0.6244 vs 0.6241, flat); speech-filter variants (V1/V3/V4) raise FPR (0.669). ROC-AUC 0.613–0.629 (V0 highest). **V4 ≡ V3 byte-identical** — CNN verdict routes only to alarm-suppression boolean, not to the ROC's fatigue_score (frozen code). Frame-level labels ⇒ conservative FPR. |

---

## 3. Detailed Experiment Logs

### EXP-000: Data Infrastructure Verification & Baseline Setup
- **Date**: July 24, 2026
- **Status**: Superseded / corrected 2026-07-28 (freeze-report precondition 1).
- **Objective**: Establish pre-flight dataset validation and subject-independent partitioning.
- **RETRACTION**: The original EXP-000 entry claimed "0 subject leakage across
  Train/Val/Test" and "<0.5 ms" latency. Both are **false and retracted**.
  Direct verification (see `Data/FINAL_PRETRAINING_VERIFICATION_REPORT.md`)
  found 100% subject leakage in the shipped MRL partitions and no measured
  latency existed. No latency number has been measured to date.
- **Actual state**: Leak-free partitioning is now provided by
  `Data/mrl_eye/splits_subject_disjoint/` (generator
  `tools/build_subject_disjoint_splits.py`, seed 42, asserted zero pairwise
  subject overlap). `Data/drowsiness_detection` (a 100% MRL duplicate) is
  quarantined in code.
- **Scientific Takeaway**: Data foundation was NOT certified previously; it is
  only now leak-free. No performance metric may be cited until `EXP-001+` are
  actually run and logged here.

---

### EXP-001: First Measured Pipeline Latency (host feasibility)
- **Date**: July 28, 2026
- **Status**: Completed (measured, logged artifact).
- **Objective**: Prove the full per-frame pipeline runs end-to-end on real data
  and obtain the FIRST honestly-measured latency (replacing the retracted
  "<0.5ms" claim).
- **Setup**: `evaluation/benchmark_nthan_yawdd.py --max-frames 300 --write`
  driving `src/frame_processor.FrameProcessor` (no CNN arm) over 300 real
  NTHU-DDD frames at video clock 30 fps. Seed 42, deterministic frame order.
- **Measured (Darwin-arm64, this host)**: mean 3.205 ms/frame, p50 3.080 ms,
  p95 3.316 ms, max 29.27 ms, throughput ≈ 312 FPS. Artifact:
  `results/measured_results.json`.
- **Caveats / NOT claimed**: This is HOST latency, explicitly not Raspberry
  Pi 4. No accuracy/ROC/FPR here — those require the LOSO harness + fixed
  operating point (EXP-004/005). The one-frame max (29 ms) is FaceMesh
  cold-start.
- **Scientific Takeaway**: Real-time feasibility on a laptop-class ARM CPU is
  confirmed; on-device Pi 4 profiling (EXP planned) is the remaining latency
  claim needed for the paper.

---

### EXP-002: First MicroEyeNet Baseline Training (subject-disjoint MRL)
- **Date**: July 28, 2026
- **Status**: Completed (measured, logged artifacts).
- **Objective**: Train the frozen MicroEyeNet CNN to convergence on the
  leak-free subject-disjoint MRL split and record the FIRST honestly-measured
  classification metrics for the eye-state model.
- **Setup**: `tools/train_exp002_microeyenet.py` over
  `Data/mrl_eye/splits_subject_disjoint/{train,val,test}.csv`
  (train 70,551 / val 4,970 / test 9,377 samples; 26 / 6 / 5 disjoint
  subjects). Frozen spec: 24×24×1 grayscale input, Adam LR=1e-3, clipnorm=1.0,
  batch 64, max 30 epochs, BCE loss, EarlyStopping(val_loss, patience 5,
  restore best), ReduceLROnPlateau(0.5, patience 3), dropout 0.3, mixed
  precision OFF, class_weights null, seed 42, TF 2.17.1. Measured parameter
  count **19,745**.
- **Training outcome**: Ran 13 epochs, early-stopped; best epoch 8 (val_loss
  0.17927). Wall clock: data load 26.244 s, training 102.777 s. LR schedule:
  1e-3 (ep 0–5) → 5e-4 (ep 6–10) → 2.5e-4 (ep 11–12).
- **Measured VALIDATION @0.5**: acc 0.9402, F1 0.9262, ROC-AUC 0.9824,
  PR-AUC 0.9701, precision 0.9316, recall 0.9210, specificity 0.9535,
  balanced-acc 0.9372, Brier 0.0480. CM {tp 1865, tn 2808, fp 137, fn 160}.
- **Measured TEST @0.5**: acc 0.9362, F1 0.9623, ROC-AUC 0.9692, PR-AUC 0.9937.
  CM {tp 7632, tn 1147, fp 202, fn 396}.
- **Operating point**: Best-F1 threshold on VAL = 0.3700 (val F1 0.9286). At
  that threshold, TEST acc 0.9433 / F1 0.9668 (CM {tp 7748, tn 1097, fp 252,
  fn 280}). Reported for reference; 0.5 remains the default frozen threshold.
- **Caveats / NOT claimed**: NO TFLite export was produced here (that is EXP-003
  scope) — the existing `models/eye_state_model.tflite` is untouched. NO
  on-device or host latency was measured for this model; **no Raspberry Pi 4
  number exists**. TFLite Size and Latency are logged as NOT MEASURED.
- **Artifacts**: `experiments/EXP-002_microeyenet_baseline/`
  (exp002_metrics.json, training_config.json, model_summary.txt,
  learning_curves.png, roc_curve_test.png, pr_curve_test.png,
  confusion_matrix_test.png, reliability_diagram_val.png),
  `logs/EXP-002_training_log.csv`, `reports/EXP-002_REPORT.md`.
- **Scientific Takeaway**: The frozen MicroEyeNet reaches ~94% validation
  accuracy / 0.926 F1 on a genuinely subject-disjoint split at 19,745
  parameters, confirming the architecture is viable for the deployment target.
  Quantized export and on-device latency remain the outstanding items
  (EXP-003+).

---

### EXP-003: MicroEyeNet Float16 & Full INT8 Quantization Verification
- **Date**: July 28, 2026
- **Status**: Completed (measured, logged artifacts).
- **Objective**: Generate deployment-ready TensorFlow Lite models (Float16 and Full INT8) from the trained FP32 MicroEyeNet checkpoint (`microeyenet_epoch08_valloss0.1793.keras`), measure quantization accuracy degradation on the frozen TEST split, measure model size compression, and verify functional input/output contracts.
- **Setup**: `tools/export_and_evaluate_quantization.py` over frozen TEST split (`Data/mrl_eye/splits_subject_disjoint/test.csv`, 9,377 samples). FP16 conversion via `TFLiteConverter` (`Optimize.DEFAULT`, `target_types=[float16]`). INT8 conversion via `TFLiteConverter` (`Optimize.DEFAULT`, 500 representative training image crops, `supported_ops=[TFLITE_BUILTINS_INT8]`, float32 input/output interface). Zero retraining or hyperparameter changes.
- **Measured Model Sizes**:
  - FP32 TFLite Baseline: 83,060 bytes (81.11 KB)
  - Float16 TFLite: **44,500 bytes (43.46 KB)** — Compression Ratio **1.87x** (46.42% size reduction)
  - Full INT8 TFLite: **26,160 bytes (25.55 KB)** — Compression Ratio **3.18x** (68.50% size reduction)
- **Measured Accuracy & Metrics (TEST Split @ 0.50)**:
  - FP32 Keras Baseline: Accuracy 0.936227, F1 0.962300, ROC-AUC 0.969219, PR-AUC 0.993701, Brier 0.047995
  - Float16 TFLite: Accuracy 0.936227, F1 **0.962300** (0.000% F1 degradation), ROC-AUC 0.969219, PR-AUC 0.993702, Brier 0.047976. CM {tp 7632, tn 1147, fp 202, fn 396}
  - Full INT8 TFLite: Accuracy 0.935694, F1 **0.962044** (-0.026% F1 degradation), ROC-AUC 0.968339, PR-AUC 0.993865, Brier 0.048386. CM {tp 7642, tn 1132, fp 217, fn 386}
- **Functional Verification**: Verified input tensor contract `[1, 24, 24, 1]` float32 $[0, 1]$ and output sigmoid tensor contract `[1, 1]` float32 $[0, 1]$ for both models. All integrity verifications (I1–I6), unit tests (17/17), and smoke tests (3/3) passed.
- **Artifacts**: `models/eye_state_model_fp16.tflite`, `models/eye_state_model_int8.tflite`, `experiments/EXP-003_quantization/` (metrics.json, quantization_report.json, fp32_vs_fp16_vs_int8.csv, conversion.log, verification_report.json), `reports/EXP-003_REPORT.md`.
- **Scientific Takeaway**: MicroEyeNet quantizes to INT8 with a 3.18x reduction in model size (down to 25.55 KB) while preserving >99.97% of its baseline classification F1-Score (0.9620 vs 0.9623), establishing a highly compact, deployment-ready asset for real-time edge drowsiness monitoring.

---

### EXP-004: Leave-One-Subject-Out Ablation Evaluation on NTHU-DDD (V0–V4)
- **Date**: July 28, 2026
- **Status**: Completed (measured, logged artifacts). Process exit code 0.
- **Objective**: Run the frozen LOSO ablation (V0–V4) over the full NTHU-DDD
  dataset and record the FIRST measured accuracy/ROC results, testing the
  primary hypothesis (does the reliability gate + speech filter reduce FPR at a
  matched TPR vs weighted-fusion baseline?).
- **Setup**: `evaluation/exp004_report.py --write` — an additive orchestrator
  that calls the **unmodified** frozen `evaluation/loso_harness.py` helpers
  (`_roc_curve`, `_auc` trapezoid, `_fix_operating_point(target_tpr=0.80)`,
  `_fpr_at_tpr`) and `FrameProcessor`. NTHU-DDD `Data/nthu_ddd/`, 66,521 frames,
  4 subjects (001/002/005/006), subject-disjoint LOSO, seed 42, deterministic
  order, video clock 30 fps. Variants
  `(speech_filter, reliability_gate, cnn)`: V0(F,F,F), V1(T,F,F), V2(F,T,F),
  V3(T,T,F), V4(T,T,T). Sweep variable `FrameResult.fatigue_score`. Operating
  point fixed on V0 at TPR=0.80 (fixed score threshold ≈2.6e-08) and held
  constant. `sklearn` absent ⇒ hand-rolled numpy metrics (same methodology as
  EXP-002/003). Wall clock 24.8 min. Frame/drowsy counts match ground truth
  exactly (no decode drops).
- **Measured OVERALL (N=66,521, 4 subjects)** — ROC-AUC / PR-AUC / FPR@matched-TPR=0.80:
  - V0 V0_baseline: AUC 0.628786, PR-AUC 0.637354, Acc 0.605598, Prec 0.602332,
    Recall 0.800000, Spec 0.375881, F1 0.687235, **FPR@mTPR 0.624119**.
    CM {tp 28824, fp 19030, tn 11461, fn 7206}.
  - V1 V1_speech_filter: AUC 0.617005, PR-AUC 0.628353, Acc 0.597450,
    Recall 0.767166, Spec 0.396904, F1 0.673678, **FPR@mTPR 0.669411**.
    CM {tp 27641, fp 18389, tn 12102, fn 8389}.
  - V2 V2_reliability_gate: AUC 0.624539, PR-AUC 0.621659, Acc 0.606560,
    Recall 0.797835, Spec 0.380539, F1 0.687177, **FPR@mTPR 0.624447**.
    CM {tp 28746, fp 18888, tn 11603, fn 7284}.
  - V3 V3_full: AUC 0.613299, PR-AUC 0.614859, Acc 0.598322, Recall 0.764530,
    Spec 0.401922, F1 0.673398, **FPR@mTPR 0.669017**.
    CM {tp 27546, fp 18236, tn 12255, fn 8484}.
  - V4 V4_full_cnn: **byte-identical to V3** (AUC 0.613299, FPR@mTPR 0.669017,
    CM {tp 27546, fp 18236, tn 12255, fn 8484}).
- **Primary-hypothesis outcome — NEGATIVE / NULL.** At matched TPR=0.80 the
  reliability gate in isolation (V2) leaves FPR essentially unchanged vs V0
  (0.624447 vs 0.624119, +0.000328, flat within noise); every variant that adds
  the speech-jitter filter (V1/V3/V4) *raises* FPR to ≈0.669. ROC-AUC does not
  improve over baseline (V0=0.6288 highest; V3/V4=0.6133 lowest). The claimed
  FPR-at-matched-TPR reduction is not observed at frame level under these
  conditions.
- **V4 ≡ V3 — verified root cause (structural, not a bug).** Per-variant score
  CSVs are byte-for-byte identical (md5 f8c298c8a7f521011ad9317da0b9c9b5). The
  ROC variable is `fusion.fatigue_score`;
  `FatigueFusionEngine.update(ts, reliability)` (`src/fatigue_fusion.py:154`)
  takes no CNN input, and the CNN verdict is consumed only by
  `StateManager.update` (`src/state_manager.py:374–388`) to toggle the
  `should_alarm` boolean — it never modifies `fatigue_score`
  (`src/state_manager.py:425`). Hence enabling the CNN cannot change any
  frame-level score. The CNN is nonetheless active (instrumented: 89
  invocations over 6,000 subject-005 frames; EAR entered the [0.17,0.27]
  uncertainty zone on 2,825 frames). Its effect is confined to alarm suppression,
  which this frame-level ROC does not measure.
- **Subject-independence note**: subject 006 is a below-chance outlier across all
  variants (per-subject ROC-AUC ≈0.36–0.37; inverted label balance), depressing
  the aggregate.
- **Caveats / NOT claimed**: NTHU frame labels are clip-condition-derived ⇒
  frame-level FPR is conservative (documented known limitation). No TFLite export
  and NO on-device/Pi 4 latency measured here (none exists). This is an accuracy
  experiment only.
- **Artifacts**: `experiments/EXP-004_loso/` (exp004_metrics.json [10.8 MB full
  roc curves + extended metrics], per_variant_metrics.csv, per_subject_metrics.csv,
  scores/V0–V4_*.csv, plots/{roc_overlay,auc_prauc_bars,fpr_at_matched_tpr,
  confusion_matrices}.png), canonical `roc` block merged into
  `results/measured_results.json` by the frozen writer,
  `results/fig_roc_curves.png`, `reports/EXP-004_REPORT.md`.
- **Verification**: `verify_integrity.py` 6/6 PASS; `tests.test_suite` 17/17 OK;
  `smoke_test.py` 3/3 OK.
- **Scientific Takeaway**: On frame-level NTHU-DDD under strict subject-disjoint
  LOSO, the frozen heuristic fatigue score is a weak discriminator (AUC ≈0.61–0.63)
  and the signal-reliability gate does **not** reduce FPR at matched TPR=0.80.
  This is an honest negative result that constrains the paper's claims and
  motivates an event-/episode-level ablation (EXP-005) where the CNN
  alarm-suppression arm — and the gate's protection against spurious alarms —
  can actually be observed.

---

## 4. Planned / Next Experiments — Official Roadmap

> The authoritative numbering is Section 2 above: **EXP-000 … EXP-004 are done
> and measured.** This section defines the **official IDs for all future
> experiments.** These IDs are canonical and supersede every earlier provisional
> numbering (see the reconciliation note below). Assign the next sequential ID,
> and add a Section 2 row + Section 3 detailed log the moment a run completes —
> never cite before logging.

| Official ID | Title | Status |
|:---|:---|:---|
| **EXP-005** | Event-Level Alarm Evaluation | Planned (next) |
| **EXP-006** | Gate Redesign Evaluation | Planned |
| **EXP-007** | Raspberry Pi Deployment Evaluation | Planned |
| **EXP-008** | Second Dataset Validation (Optional) | Planned (optional) |

- **EXP-005 — Event-Level Alarm Evaluation (next).** Motivated directly by the
  EXP-004 negative result. Replace the frame-level `fatigue_score` ROC with an
  **alarm-event metric** (FPR/hour + episode-detection latency) evaluated over
  the same V0–V4 variants on NTHU-DDD (and, where applicable, YawDD). Rationale:
  EXP-004 showed V4≡V3 at frame level because the CNN verdict routes only to the
  `should_alarm` boolean (not to `fatigue_score`), and the gate's protection
  against spurious alarms is an event-level, not a frame-level, effect. An
  event-level protocol is where V4≠V3 and any gate benefit could become
  observable.
- **EXP-006 — Gate Redesign Evaluation.** Re-architect the reliability gate as an
  **additive decision-layer term** rather than a multiplicative pre-accumulation
  attenuator, and re-evaluate against V0. This is the follow-up the EXP-004
  scientific audit recommends (the audit calls it "EXP-005"; under this official
  roadmap it is **EXP-006** — see the reconciliation note).
- **EXP-007 — Raspberry Pi Deployment Evaluation.** On-device Raspberry Pi 4
  per-stage latency / throughput / memory / thermal profiling. The only latency
  measured to date (EXP-001) is a Darwin-arm64 host number; **no Pi 4 number
  exists or may be cited** until this experiment logs one.
- **EXP-008 — Second Dataset Validation (Optional).** Reproduce the primary
  evaluation on an additional subject-disjoint dataset to test whether the
  EXP-004 finding generalizes beyond NTHU-DDD. Optional; run only if a suitable
  dataset is secured.

**Not a numbered experiment:** paper population — fill `paper/main.tex` results
*only* from committed artifacts (`results/measured_results.json`, the
`experiments/EXP-00X_*/` folders), including the EXP-004 negative result stated
honestly. This is a writing task, not a measurement, so it carries no `EXP-###`.

### Numbering reconciliation (supersedes all earlier provisional schemes)

Before this roadmap was fixed, three documents used conflicting provisional
future-numbering. The mapping below records how each maps onto the official IDs.
Historical documents are **not** rewritten; consult this table when reading them.

| Source (provisional) | Provisional ID | Meaning there | Official ID |
|:---|:---|:---|:---|
| EXP-004 scientific audit | "EXP-005" | Gate redesign (decision-layer term) | **EXP-006** |
| EXP-004 scientific audit | "EXP-006" | Subject-006 front-end failure diagnosis | *(audit-internal follow-up; fold into EXP-005/EXP-006 as diagnostics — not a standalone official ID)* |
| EXP-004 scientific audit | "EXP-007" | Direct evaluation of CNN alarm-suppression path | **EXP-005** (subsumed by the event-level evaluation) |
| EXP-004 scientific audit | "EXP-008" | Full-precision score serialization + re-freeze | *(audit-internal engineering follow-up; not a standalone official ID)* |
| `CNN_IMPLEMENTATION_SPECIFICATION.md` Part 10 | "EXP-005" | Full ablation adding V4 (CNN arm) | *(already done — subsumed into the completed EXP-004, which ran V0–V4)* |
| `CNN_IMPLEMENTATION_SPECIFICATION.md` Part 10 | "EXP-006" | CNN-invocation behaviour analysis | *(fold into EXP-005 event-level as a diagnostic; not a standalone official ID)* |
| `CNN_IMPLEMENTATION_SPECIFICATION.md` Part 10 | "EXP-007" | On-device Raspberry Pi 4 profiling | **EXP-007** (same meaning) |

The completed rows EXP-000 … EXP-004 (Sections 2–3) are final and unaffected.
