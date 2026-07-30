# CNN_IMPLEMENTATION_SPECIFICATION.md — The Frozen CNN Engineering Contract

**Project:** Real-Time Driver Drowsiness Detection via Signal Reliability Gating
**Document status:** FROZEN pre-implementation contract (2026-07-28)
**Scope:** the optional, ablation-only CNN eye-state validator (`MicroEyeNet`).
**Authority:** this document is the engineering contract for every CNN
experiment. After approval, implementation begins; no further CNN redesign is
permitted. If a future need conflicts with this contract, it must be logged as a
change request in Appendix A, not silently edited into the code.

> **Truth policy (inherited, binding):** no performance number appears in this
> document unless it traces to a logged `EXP-###` row and a committed artifact
> in `results/`. Every accuracy / ROC / Raspberry-Pi number below is written as
> **NOT MEASURED** or as a *success criterion / target*, never as a result. The
> only real measured number in the project is **EXP-001 latency (3.205 ms/frame,
> Darwin-arm64 host — NOT a Raspberry Pi 4)**.

> **Design-freeze boundary (binding):** this contract does **not** redesign the
> research. The thesis contribution remains the **signal-reliability gate**; the
> CNN is prior-art baseline for the **V4** ablation only. Nothing here changes
> the 3-component geometric-mean gate, the SEVERE exemption, the LOSO seed, the
> V0–V4 definitions, or the FPR@matched-TPR protocol.

---

## Part 1 — CNN Purpose

**Why the CNN exists.** The frozen research question asks whether a *reliability
gate* + a *speech-jitter filter* reduce FPR at matched TPR versus a
weighted-fusion baseline. To make that claim defensible against the reviewer
question *"would a learned eye classifier have solved this anyway?"*, the study
needs a **prior-art CNN arm** in the ablation. The CNN is therefore an
**instrument of the ablation**, not a component of the contribution. It exists to
occupy the **V4** slot `(speech=T, gate=T, cnn=T)` and to answer: *does adding a
learned eye-state validator on top of the full proposed system move the
operating point?*

**What it is.** `MicroEyeNet` — a tiny binary eye-state classifier
(open vs. closed) applied to a single cropped eye ROI. It is a *selective
uncertainty resolver*, not a frame classifier.

**When invoked (all conditions must hold — see `CNNValidator.should_invoke`):**
1. `enable_cnn=True` was passed to `FrameProcessor` **and** the model file loaded
   (`is_available`).
2. A face is detected this frame.
3. Smoothed EAR is inside the uncertainty zone `[uncertainty_zone_low,
   uncertainty_zone_high] = [0.17, 0.27]`.
4. Smoothed system reliability `≥ 0.3`.
5. The rate-limiter allows it (`max_invocations_per_second = 5`).

**When bypassed / never invoked:**
- Whenever `enable_cnn=False` (the **default** — the CNN object is not even
  constructed).
- EAR clearly open (`> 0.27`) or clearly closed (`< 0.17`) — the heuristic is
  already confident, so the CNN adds cost without information.
- Reliability `< 0.3`, or the per-second budget is exhausted.
- **Never** in `SEVERE_FATIGUE` or `FACE_LOST_CRITICAL` — a learned validator may
  not soften a genuine severe alert (safety asymmetry, consistent with the gate's
  SEVERE exemption). The CNN operates only in the `ALERT ↔ SLIGHT_FATIGUE`
  boundary.
- If the model file is absent, the validator degrades gracefully to
  heuristic-only (no crash, `invoked=False`).

**Why only an ablation / OFF by default.** The contribution is measurement-trust
gating, not classification accuracy. Putting the CNN on the default path would
(a) confound the gate's effect with the classifier's, and (b) contradict the
frozen decision "CNN is ablation-only, OFF by default." The default detection
path (V0–V3) must remain CNN-free so the V0→V3 delta isolates the novelties.

---

## Part 2 — CNN Architecture

### Selected architecture: **MicroEyeNet** (FROZEN)

Chosen because it is (a) already coded and frozen in `src/cnn_validator.py` and
`tools/train_cnn.py`, (b) small enough (19,745 params, measured) to be an honest
"lightweight prior-art eye validator" on a Pi-4-class CPU, and (c)
interpretable and reproducible. Its only job is binary eye-state on a 24×24
grayscale crop; a larger model would be scope creep for an ablation arm.

### Rejected alternatives

| Candidate | Rejected because |
|---|---|
| **TinyCNN (generic 3–4 block CNN)** | No advantage over MicroEyeNet for a 24×24 binary problem; more params, more latency, no accuracy mandate that justifies it. Would require re-freezing the coded architecture. |
| **MobileNet / MobileNetV2 (depthwise-separable, ImageNet-scale)** | Massively over-parameterized for one 24×24 eye crop; pretrained RGB ImageNet features are irrelevant to grayscale eye-state; TFLite INT8 would still dwarf the 5/sec selective budget. Contradicts the "micro, selective" role. |
| **Custom deeper CNN (≥4 conv blocks, BN, >100K params)** | Encourages the CNN to become the story; the thesis contribution is the gate. Larger capacity invites overfitting on 37 subjects and undermines the "prior-art baseline" framing. |
| **Non-CNN learned validator (SVM/MLP on EAR features)** | Would blur the line between "learned eye classifier" (the reviewer's alternative hypothesis) and the geometric heuristic; a CNN on pixels is the cleaner prior-art foil. |

### Frozen architecture table (matches `src/cnn_validator.py` docstring and `tools/train_cnn.py` Keras build)

| # | Layer | Config | Output shape | Notes |
|---|---|---|---|---|
| 0 | Input | 24×24×1 | (24, 24, 1) | **Grayscale** (not RGB). `input_size=24`. |
| 1 | Conv2D | 8 filters, 3×3, `padding='same'`, ReLU | (24, 24, 8) | First feature block. |
| 2 | MaxPool2D | 2×2 | (12, 12, 8) | Downsample. |
| 3 | Conv2D | 16 filters, 3×3, `padding='same'`, ReLU | (12, 12, 16) | Second feature block. |
| 4 | MaxPool2D | 2×2 | (6, 6, 16) | Downsample. |
| 5 | Flatten | — | (576,) | 6·6·16 = 576. |
| 6 | Dense | 32 units, ReLU | (32,) | Bottleneck. |
| 7 | Dropout | rate = 0.3 | (32,) | Regularization (train only). |
| 8 | Dense | 1 unit, Sigmoid | (1,) | P(eye closed) ∈ [0,1]. |

- **Input size / color:** 24×24×1 grayscale — FROZEN. Do **not** switch to RGB
  or a different resolution.
- **Conv blocks:** exactly 2 (8 → 16 filters), 3×3 kernels, `padding='same'`,
  ReLU — FROZEN.
- **Normalization:** input normalized by `/255.0` at preprocessing time
  (`extract_eye_roi` and the trainer both do this). **No BatchNorm layer** —
  FROZEN (keeps the model tiny and INT8-quantization-friendly).
- **Pooling:** 2×2 max-pool after each conv block — FROZEN.
- **Dropout:** 0.3 on the dense bottleneck — FROZEN (see Part 4/6 for the
  0.2-vs-0.4 sensitivity check, which is a *hyperparameter experiment*, not a
  redesign).
- **Dense / output:** Dense(32, ReLU) → Dense(1, Sigmoid) — FROZEN. Sigmoid
  output = probability the eye is **closed**.
- **Parameter budget:** **19,745 params** (measured, logged from the built model
  in EXP-002; dominated by the 576→32 dense bottleneck under `padding='same'`).
  Note: the earlier "~9.5K" estimate corresponded to a `padding='valid'` variant
  (256-wide flatten); the frozen implementation uses `padding='same'`, which is
  not that variant.

**Output semantics (frozen in `CNNVerdict`):** `probability_closed` is the raw
sigmoid; `cnn_says_closed = probability_closed ≥ 0.5`; `confidence =
abs(probability_closed − 0.5) ∈ [0, 0.5]`; `cnn_agrees_with_heuristic` compares
`cnn_says_closed` against the heuristic `smoothed_ear < ear_threshold`.

---

## Part 3 — Dataset Specification

### Training / validation / test corpus (FROZEN)

- **Dataset:** **MRL Eye** only (`Data/mrl_eye/`). 84,898 PNG eye crops, 37
  subjects. This is the **only** dataset the CNN trains/validates/tests on. NTHU
  and YawDD are for the *system-level* LOSO ablation, not for CNN training.
- **`drowsiness_detection` is BANNED** — it is a 100% byte-duplicate of MRL Eye;
  its loader raises `RuntimeError` by design. Using it would be silent
  double-counting.

### Subject split (FROZEN — the single most important integrity rule here)

- **MUST use `Data/mrl_eye/splits_subject_disjoint/`** (generated by
  `tools/build_subject_disjoint_splits.py`, seed 42, asserted **0 pairwise
  subject overlap**).
- **MUST NOT use the shipped subject-leaky partitions.** The trainer
  (`tools/train_cnn.py`) currently loads via `MRLEyeDataLoader.get_partition_files`
  and its docstring warns this path reads the leaky partitions. **Before EXP-002
  is run, the loader must be repointed to the subject-disjoint split** (this is a
  precondition, tracked in Part 10 / Part 11, not a redesign — the split already
  exists on disk).
- **Rationale:** the retracted EXP-000 shipped 100% subject leakage; any CNN
  number produced on a leaky split is non-citable.

### Folder structure (expected on disk)

```
Data/mrl_eye/
  splits_subject_disjoint/     # ← the ONLY split the CNN may use (seed 42)
    train/   val/   test/      # subject-disjoint partitions
  <shipped partitions>         # ← BANNED for CNN training (subject-leaky)
```

### Label format (FROZEN — from `MRLEyeDataLoader`)

- `class_map = {"awake": 0, "sleepy": 1}` where the task-level meaning is
  **open eye = 0, closed eye = 1**. The sigmoid output is P(closed).
- Subject ID is parsed by regex `^(s\d{4})_` (used to enforce subject-disjointness).
- `get_partition_files(split)` yields `(file_path, label, subject_id)`.

### Preprocessing / normalization / resize (FROZEN — identical in train and inference)

- **Grayscale**, single channel.
- **Resize to 24×24** with `cv2.INTER_AREA` (both `extract_eye_roi` at inference
  and `load_mrl_dataset_tensors` at train time).
- **Normalize:** divide by `255.0` → float32 in `[0,1]`, final tensor shape
  `(24, 24, 1)`.
- **Inference ROI extraction:** eye bounding box from FaceMesh landmark indices
  `_LEFT_EYE_BBOX_IDX = [33,133,160,159,158,144,145,153]`,
  `_RIGHT_EYE_BBOX_IDX = [362,263,387,386,385,373,374,380]`, with `margin=5`.
  **This exact preprocessing must be reused at training time** so train and
  deploy cannot diverge.

### Class balancing (FROZEN policy)

- Report the on-disk open/closed class counts in EXP-002 (measured, logged).
- If imbalance is material, use **class weights in the loss** (not resampling
  that could leak subjects across the split). Do not oversample across the
  subject-disjoint boundary.

### Data leakage prevention (FROZEN)

1. Subject-disjoint split only (above).
2. No subject appears in more than one of train/val/test (assert at load time).
3. Same preprocessing pipeline for train and inference (no train-only pixel ops
   that shift the distribution the deployed model never sees).
4. Test partition touched **once**, at the end, for the final logged number.

---

## Part 4 — Data Augmentation

**Guiding rule:** an augmentation is allowed only if it **cannot flip or blur the
open/closed eye-state label**. The label is defined by eyelid geometry, which is
fragile under geometric distortion.

### Frozen augmentation set (applied to training only, in this order)

| Order | Augmentation | Probability | Parameters | Purpose |
|---|---|---|---|---|
| 1 | Horizontal flip | 0.5 | left↔right mirror | Left/right eye symmetry; label-preserving. |
| 2 | Brightness jitter | 0.5 | ±20% multiplicative | Robustness to lighting (aligns with the brightness-quality gate rationale). |
| 3 | Contrast jitter | 0.3 | ±15% | Camera/exposure variation. |
| 4 | Small rotation | 0.3 | ±8° | Minor head-tilt tolerance. Bounded so eyelid state is preserved. |
| 5 | Additive Gaussian noise | 0.2 | σ ≤ 0.02 (on [0,1] scale) | Sensor-noise robustness. |

- Augmentation runs on the **training split only**; validation and test are
  **never** augmented.
- All augmentation happens **after** grayscale + resize + normalize so the tensor
  contract (24×24×1, [0,1]) is unchanged.

### Rejected augmentations (would invalidate the eye-state label)

| Augmentation | Rejected because |
|---|---|
| **Vertical flip** | Inverts eyelid geometry; a closed eye can read as a different shape — label integrity risk. |
| **Large rotation (>~10°) / shear** | Can push a partially-open eye across the open/closed boundary. |
| **Random erasing / cutout over the eye region** | Removes the exact pixels that define the label. |
| **Heavy blur / large downscale-upscale** | Destroys the eyelid edge that distinguishes open from closed. |
| **Elastic / grid distortion** | Warps eyelid aperture non-rigidly → ambiguous label. |
| **Color/hue jitter** | Meaningless on grayscale, and would imply an RGB pipeline (contradicts Part 2). |

---

## Part 5 — Training Specification (FROZEN)

Baseline (matches `tools/train_cnn.py`; explicit values frozen here):

| Item | Frozen value | Notes |
|---|---|---|
| Framework | TensorFlow / Keras (`tensorflow==2.17.1`) | Pinned; MediaPipe compat (protobuf ≥4.25.3,<5). |
| Optimizer | Adam | As coded. |
| Learning rate | 1e-3 | Baseline (planned registry row). |
| LR scheduler | ReduceLROnPlateau (factor 0.5, patience 3, on val_loss) | Frozen for baseline; alternative schedules are a *hyperparameter experiment*, not a redesign. |
| Weight decay | None (baseline) | Dropout is the primary regularizer. |
| Batch size | 64 | As coded. |
| Epochs | 30 (max) | Planned baseline; early stopping may end sooner. |
| Loss | Binary cross-entropy | As coded (`loss='binary_crossentropy'`). |
| Metrics | accuracy (train) | Full metric set logged in validation (Part 6). |
| Early stopping | monitor `val_loss`, patience 5, restore best weights | Frozen. |
| Gradient clipping | clipnorm = 1.0 | Frozen (cheap safety against rare spikes). |
| Mixed precision | OFF | Model is tiny; FP32 keeps quantization behavior predictable. |
| Random seed | **42** | Matches the project-wide seed; set for Python/NumPy/TF. |
| Checkpoint frequency | every epoch, keep best-by-val_loss | Frozen. |
| Checkpoint naming | `checkpoints/microeyenet_epoch{epoch:02d}_valloss{val_loss:.4f}.keras` | See Part 12. |
| Best-model export name | `models/eye_state_model.tflite` | The **single** committed model artifact. |
| TensorBoard | `tensorboard/EXP-002_microeyenet_baseline/` | Frozen. |
| CSV logging | `logs/EXP-002_training_log.csv` (per-epoch metrics) | Frozen. |
| Reproducibility | seed 42 + deterministic ops where available; record TF version, split hash, class counts in the EXP row | Frozen. |

- **`execute_training` guard:** the trainer only trains when explicitly invoked
  with `execute_training=True`; the default is a dry build. This is intentional —
  training is a *logged experiment*, not a side effect.
- **No number produced here is citable** until EXP-002 is logged with its
  artifact.

---

## Part 6 — Validation Protocol (FROZEN)

- **Validation frequency:** once per epoch on the subject-disjoint `val` split.
- **Best-model selection:** lowest `val_loss` (ties broken by higher `val`
  balanced accuracy).
- **Metrics logged (on val, then once on test):** accuracy, **balanced
  accuracy**, precision, recall, F1, confusion matrix, ROC curve + **ROC-AUC
  (trapezoidal)**, PR-AUC.
- **Operating-threshold selection:** default decision threshold **0.5** on
  P(closed) for the reported confusion matrix; additionally report the
  threshold that maximizes F1 and the threshold at a fixed recall — all logged,
  none silently tuned on test.
- **Calibration:** report a reliability diagram + Brier score on val. (The gate
  and the CNN are separate; the CNN's calibration is informational, it does not
  feed the reliability gate.)
- **Misclassification analysis:** log a sample of false-positive and
  false-negative crops with subject IDs and EAR values, to check whether errors
  concentrate in the `[0.17, 0.27]` uncertainty zone (the only zone where the CNN
  is ever invoked at deploy time).
- **Test discipline:** the `test` split is evaluated **exactly once**, for the
  final EXP-002/EXP-003 numbers; no iteration on test.

---

## Part 7 — Quantization (FROZEN)

Matches `tools/train_cnn.py` (TFLite converter, `Optimize.DEFAULT`,
representative dataset), frozen explicitly:

| Item | Frozen value |
|---|---|
| Baseline precision | FP32 Keras model (best checkpoint). |
| Target deploy precision | **INT8** via `tf.lite.TFLiteConverter` with `optimizations = [tf.lite.Optimize.DEFAULT]`. |
| Representative dataset | `representative_data_gen` yielding **100** real samples from the subject-disjoint `train` split, same 24×24×1 `/255.0` preprocessing. |
| Calibration procedure | Converter runs the representative generator to calibrate activation ranges; input/output kept as documented in the converter config; no synthetic data. |
| Export path | `models/eye_state_model.tflite` (the single committed model artifact). |
| Runtime | `tflite_runtime` preferred at inference; falls back to full `tensorflow` (see `CNNValidator`). |

**Quantization verification steps (all logged in EXP-003):**
1. Load the exported `.tflite` and confirm it runs on a 24×24×1 input.
2. Re-run the **test** split through the INT8 model and record accuracy /
   balanced accuracy / F1.
3. **Accuracy-drop check:** report FP32 → INT8 delta. Success criterion:
   INT8 within a pre-declared tolerance of FP32 (tolerance recorded in the EXP-003
   row *before* running). If it fails, Part 11 governs.
4. Confirm no byte-duplicate `.tflite` is introduced (integrity invariant I4 —
   there must be exactly one model artifact).
5. Record file size (measured) in the EXP row.

**No accuracy or size number is written until EXP-003 is logged with its
artifact.** (The 25.9 KB figure that appears in EXP-001's registry note is a
prior measured file-size datapoint, not an accuracy claim.)

---

## Part 8 — Raspberry Pi Deployment (FROZEN)

> **NO Raspberry Pi 4 latency/memory/thermal number exists.** All values below
> are **procedure and success criteria**, not results. Any Pi number must come
> from a logged on-device EXP row (Part 10, EXP-007).

- **Model loading:** load `models/eye_state_model.tflite` via `tflite_runtime`
  Interpreter (preferred) or `tensorflow.lite` fallback; allocate tensors once at
  construction (as in `CNNValidator`), never per frame.
- **Inference pipeline:** `extract_eye_roi` (grayscale, 24×24, /255.0) →
  set_tensor → invoke → get_tensor → `CNNVerdict`. Single ROI per invocation.
- **Memory constraints:** the interpreter is built once; no per-frame
  allocations; the 19,745-param INT8 model has a small footprint (exact RSS **NOT
  MEASURED** — to be logged in EXP-007).
- **Latency measurement:** per-invocation wall-clock around `invoke()`, plus
  end-to-end frame latency with the CNN arm enabled, reported as mean/p50/p95/max
  over ≥300 real frames — mirroring the EXP-001 methodology, but **on the Pi**
  and logged as a distinct EXP row.
- **Fallback behaviour:** if the model file is missing or the runtime import
  fails, `CNNValidator.is_available = False` → `should_invoke` returns False →
  the system runs **heuristic-only** with no crash (graceful degradation is
  frozen behaviour).
- **Error handling:** any exception during ROI extraction or invoke is caught and
  treated as "CNN not invoked this frame" (heuristic verdict stands); logged, not
  fatal.
- **Warm-up strategy:** run **N warm-up invocations** (default 5) at startup on a
  dummy 24×24×1 tensor to absorb the cold-start cost (EXP-001 saw a 29.27 ms
  FaceMesh cold-start max; the CNN warm-up is the analogous guard). Warm-up frames
  are excluded from latency statistics.

---

## Part 9 — CNN Integration (exact execution flow, FROZEN)

The CNN plugs into the **single** per-frame path in `src/frame_processor.py`. It
is constructed only when `enable_cnn=True`:

```
FrameProcessor.__init__(cfg, enable_cnn=False)
  self.cnn = CNNValidator(cfg) if enable_cnn else None      # OFF by default
```

Per-frame execution flow inside `FrameProcessor.process(frame_bgr, timestamp)`:

```
1. FaceMesh(frame) ────────────────► landmarks / face_detected
2. Detector ───────────────────────► EAR, MAR (2D, idx [78,13,308,14])
3. PoseEstimator ──────────────────► pitch/yaw/roll (solvePnP)
4. SignalQuality ──────────────────► 3 sub-scores
5. RobustnessGuard.update ─────────► snap.system_reliability  r∈[0,1]
6. TemporalAnalyzer.update(ts=…) ──► smoothed_ear, PERCLOS, σ²(MAR) speech gate
7. Reliability-gate bypass:
      if cfg.ablation.reliability_gate_enabled:
          gate_reliability = snap.system_reliability
      else:
          gate_reliability = 1.0
8. CNN block (ONLY if enabled + invocable):
      cnn_verdict = None
      if self.enable_cnn and face_detected \
         and self.cnn.should_invoke(ts.smoothed_ear, snap.system_reliability):
              roi         = extract_eye_roi(frame, landmarks, w, h)
              cnn_verdict = self.cnn.validate_eye_state(roi, ts.smoothed_ear,
                                                        ear_threshold)
9. FatigueFusionEngine ────────────► weighted sum × agreement × gate_reliability
                                       (SEVERE bypasses attenuation)
10. StateManager.update(…, cnn_verdict=cnn_verdict, …) ► 5-state machine
11. return FrameResult(fatigue_score, cnn_invoked, …)
```

**Contact points (and what each does with the CNN):**
- **Detector:** supplies EAR/MAR; EAR's smoothed value is the gate that decides
  whether the CNN is even invoked (`should_invoke`).
- **Reliability gate:** supplies `system_reliability`; the CNN is invoked only
  when reliability `≥ 0.3`. The CNN does **not** modify the gate or its
  components (the gate stays exactly 3 components).
- **Temporal Analyzer:** supplies `smoothed_ear` used by `should_invoke`; the
  CNN does not alter temporal state.
- **Fusion:** unchanged by the CNN in signal terms — the fatigue score still
  comes from weighted geometry × agreement × reliability. The CNN's influence is
  routed through the **State Manager**, not by rewriting the fused score.
- **State Manager:** receives `cnn_verdict` and may use it only to resolve the
  `ALERT ↔ SLIGHT_FATIGUE` boundary; it must **never** let the CNN soften
  `SEVERE_FATIGUE` or `FACE_LOST_CRITICAL`.
- **FrameProcessor / FrameResult:** `fatigue_score ∈ [0,1]` remains the ROC sweep
  variable; `cnn_invoked` records whether the CNN fired (for ablation
  bookkeeping).

**Invariant:** with `enable_cnn=False`, the flow is byte-for-byte the V0–V3 path;
the CNN can only ever be the V4 arm.

---

## Part 10 — CNN Experiment Plan (FROZEN, resolves the registry collision)

> ⚠️ **SUPERSEDED NUMBERING (note added 2026-07-29).** This Part 10 plan was
> written on 2026-07-28, *before* the experiments were run. History diverged
> from it: **EXP-002, EXP-003, and EXP-004 have since been executed**, and
> EXP-004 ran the **full V0–V4 ablation in one experiment** (including the CNN
> V4 arm). So the provisional "EXP-005 (V4 ablation)" and "EXP-006 (CNN-invocation
> analysis)" rows below are **obsolete** — that work is already folded into the
> completed EXP-004. **The authoritative forward numbering is the official
> roadmap in `EXPERIMENT_REGISTRY.md §4`:** EXP-005 Event-Level Alarm Evaluation,
> EXP-006 Gate Redesign Evaluation, EXP-007 Raspberry Pi Deployment Evaluation,
> EXP-008 Second Dataset Validation (optional). A reconciliation table in that
> section maps every ID below onto the official one (note that this file's
> "EXP-007 — Pi 4" keeps the same meaning under the official roadmap). The table
> below — **and every other place in this file that mentions the provisional
> "EXP-005" or "EXP-006"** (Part 11's failure-recovery tree, Part 13's
> expected-outputs list and instructions) — is retained unchanged as a historical
> planning record; **do not use any of them as the current numbering.** Wherever
> this file says "EXP-005 (V4 ablation)" or "EXP-006 (CNN-invocation analysis)",
> read it as *work already completed inside EXP-004*; the only live forward IDs
> are those in `EXPERIMENT_REGISTRY.md §4`.

> **Numbering fix:** the real **EXP-001** is the measured latency. The planned
> rows in `EXPERIMENT_REGISTRY.md §4` that reuse `EXP-001..EXP-005` are
> renumbered here to **EXP-002+**. Register these rows *before* running.

| ID | Objective | Preconditions | Expected artifact | Success criterion | Output files | Registry entry |
|---|---|---|---|---|---|---|
| **EXP-002** | Train MicroEyeNet baseline on subject-disjoint MRL split | Trainer repointed to `splits_subject_disjoint/` (Part 3); seed 42 | Trained FP32 model + training log | Converges; val balanced-accuracy logged (no target invented); 0 subject overlap asserted | `checkpoints/…`, `logs/EXP-002_training_log.csv`, `tensorboard/EXP-002…/` | dataset/split/seed/hyperparams + measured val metrics |
| **EXP-003** | Quantize FP32 → INT8 TFLite + verify | EXP-002 done | `models/eye_state_model.tflite` (single) | INT8 test metrics within pre-declared tolerance of FP32; exactly one `.tflite` (I4) | `models/eye_state_model.tflite`, quant report in `results/` | FP32→INT8 delta (measured), file size (measured) |
| **EXP-004** | System LOSO ablation V0–V3 on NTHU (no CNN) | `loso_harness.py --write`; seed 42 | `roc` block in `measured_results.json` | FPR@matched-TPR + AUC per variant recorded | `results/measured_results.json` | V0–V3 numbers (measured) |
| **EXP-005** | Full ablation adding **V4** (CNN arm) | EXP-003 + EXP-004 done; `enable_cnn=True` in harness V4 | Updated `roc` block incl. V4 | V4 vs V3 delta recorded (whatever the sign) | `results/measured_results.json` | V4 numbers (measured) |
| **EXP-006** | CNN-invocation behaviour analysis (how often the CNN fires, agreement rate) | EXP-005 done | Invocation-rate / agreement artifact | `cnn_invoked` rate + heuristic-agreement logged | `results/…` | measured invocation stats |
| **EXP-007** | On-device Raspberry Pi 4 latency/memory/thermal | Real Pi 4 hardware | Pi latency artifact | mean/p50/p95/max on Pi (first real Pi number) | `results/…` (Pi-labelled) | measured Pi profile (NOT the host number) |

**Hard rule (unchanged):** land code → run → log `EXP-###` row → commit artifact
→ *then* cite. Never batch numbers ahead of measurement.

---

## Part 11 — Failure Recovery (FROZEN decision tree)

**If training fails / does not converge (EXP-002):**
1. First check the **split**: confirm the loader points at
   `splits_subject_disjoint/` and 0 subject overlap is asserted (the single most
   likely misconfiguration).
2. Verify label mapping (`awake=0/sleepy=1` → open/closed) and class counts.
3. Verify preprocessing parity (grayscale, 24×24 INTER_AREA, /255.0) between
   train and `extract_eye_roi`.
4. Only then touch LR / batch size — and log any change.

**If overfitting (train ≫ val):**
- Increase dropout (0.3 → run the frozen 0.2-vs-0.4 sensitivity check as a logged
  hyperparameter experiment), lean harder on the frozen augmentation set, rely on
  early stopping (restore best weights). Do **not** grow the architecture (that
  would be a redesign, forbidden by this contract).

**If quantization hurts performance (EXP-003):**
- If INT8 exceeds the pre-declared tolerance vs FP32: (a) improve the
  representative dataset coverage (still real MRL samples, subject-disjoint),
  (b) if still failing, the deploy model **may** fall back to a documented
  higher-precision TFLite export — but this must be logged in EXP-003 and stated
  in the paper; it does not silently change the artifact contract.

**If the CNN performs worse than the heuristic (EXP-005/EXP-006):**
- **It stays in the paper.** A V4 that does not beat V3 is a *legitimate,
  reportable* ablation result and directly strengthens the thesis: it shows the
  gain comes from the **reliability gate**, not from bolting on a learned
  classifier. A null/negative CNN result is not a failure to hide — it is
  evidence for the contribution. Report it honestly with its EXP row.

---

## Part 12 — Repository Organization (FROZEN)

```
models/                         # deployed artifacts — exactly ONE .tflite
  eye_state_model.tflite        #   the single committed model (I4: no duplicates)
checkpoints/                    # per-epoch Keras checkpoints (best-by-val_loss kept)
  microeyenet_epoch{NN}_valloss{X.XXXX}.keras
tensorboard/                    # TB event logs, one dir per experiment
  EXP-002_microeyenet_baseline/
logs/                           # CSV training/validation logs
  EXP-002_training_log.csv
exports/                        # intermediate export staging (pre-commit)
results/                        # committed measured artifacts (JSON) — source of all figures
  measured_results.json
datasets/                       # (pointer/manifest only; raw data stays under Data/)
experiments/                    # per-EXP notes/configs (optional, human-readable)
```

**Naming conventions (FROZEN):**
- Model artifact: **exactly** `models/eye_state_model.tflite` (no variants
  committed; duplicates violate integrity invariant I4).
- Every experiment output is prefixed with its resolved `EXP-###` (e.g.
  `EXP-002_…`), matching the registry row.
- Raw datasets remain under `Data/` (capital D) — `datasets/` holds only
  manifests/pointers, never a second copy.
- Checkpoints are working files (not necessarily committed); the committed
  deliverable is the single `.tflite` + the `results/*.json`.

---

## Part 13 — Documentation

This file **is** the Part 13 deliverable (`CNN_IMPLEMENTATION_SPECIFICATION.md`).
It covers: architecture (Part 2), training (Part 5), validation (Part 6), export/
quantization (Part 7), deployment (Part 8), integration/execution flow (Part 9),
the experiment workflow (Part 10), expected outputs and success criteria
(Parts 6/7/10), and future-AI instructions (below).

### Instructions for a future AI/engineer implementing the CNN

1. **Read the freeze contract first:** this file, then
   `reports/IMPLEMENTATION_SPECIFICATION_FROZEN.md`, then `EXPERIMENT_REGISTRY.md`.
2. **Repoint the loader to the subject-disjoint split before EXP-002.** This is
   the #1 leakage hazard. Nothing citable comes from the leaky partitions.
3. **Do not redesign the architecture.** MicroEyeNet is frozen (Part 2). Dropout
   0.2/0.4 and LR schedule are the only sanctioned *hyperparameter* knobs, and
   each variation is its own logged experiment.
4. **Keep the CNN OFF by default.** It exists for V4 only. `enable_cnn=False`
   must reproduce the V0–V3 path exactly.
5. **Never soften SEVERE with the CNN.** The State Manager must ignore the CNN in
   `SEVERE_FATIGUE` / `FACE_LOST_CRITICAL`.
6. **Log before you cite.** Register the `EXP-###` row and commit the artifact
   before any number reaches a report or `paper/main.tex`. Figures come only from
   `results/measured_results.json`.
7. **Write no Raspberry Pi number until EXP-007 measures it on real hardware.**
   The 3.205 ms figure is a Darwin-arm64 host, not a Pi.
8. **Keep the three gates green** after every change:
   `evaluation/verify_integrity.py` (6/6), `python3 -m unittest tests.test_suite`
   (17/17), `python3 tests/smoke_test.py` (3/3).

### Expected outputs (once the plan is executed)

- `models/eye_state_model.tflite` (single INT8 artifact, from EXP-003).
- `results/measured_results.json` extended with the CNN-arm `roc` block
  (EXP-005) and Pi profile (EXP-007).
- `EXPERIMENT_REGISTRY.md` rows EXP-002 … EXP-007, each with a committed artifact.

---

## Final Decision

**READY FOR CNN IMPLEMENTATION.**

Justification: the architecture (MicroEyeNet), configuration, preprocessing,
integration path, dataset, and experiment protocol are all already present in the
frozen codebase and are now captured as an explicit, unambiguous contract. The
only precondition to executing EXP-002 — repointing the trainer's loader from the
shipped subject-leaky partitions to the existing `Data/mrl_eye/splits_subject_disjoint/`
split — is a known, mechanical, one-line fix (the leak-free split already exists
on disk), governed by Part 3 and Part 11. No research redesign is required, no
result has been invented, and the CNN remains ablation-only and OFF by default.

Implementation may begin, starting with EXP-002, under this contract.

---

## Appendix A — Change log

- **2026-07-28** — Contract created and frozen. Renumbered planned CNN experiments
  to EXP-002…EXP-007 to resolve the `EXPERIMENT_REGISTRY.md §4` collision with the
  real measured EXP-001 (latency). No code changed; no experiment run.
