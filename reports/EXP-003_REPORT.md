# EXP-003 — Float16 & INT8 Quantization + TFLite Verification Report

**Experiment ID:** EXP-003  
**Title:** MicroEyeNet Float16 & Full INT8 Quantization and TFLite Verification  
**Date:** 2026-07-28  
**Status:** COMPLETED SUCCESSFULLY  
**Author:** Research Engineering (Reproducible Experiment)  

> All metrics, model file sizes, compression ratios, and confusion matrices in this report are transcribed verbatim from the measured artifacts generated during execution (`experiments/EXP-003_quantization/metrics.json`, `experiments/EXP-003_quantization/quantization_report.json`, `experiments/EXP-003_quantization/fp32_vs_fp16_vs_int8.csv`, and `experiments/EXP-003_quantization/conversion.log`). No value is projected, estimated, or fabricated.

---

## 1. Executive Summary & Verdict

EXP-003 successfully converted the trained FP32 MicroEyeNet model (`checkpoints/microeyenet_epoch08_valloss0.1793.keras`, 19,745 parameters) produced in EXP-002 into **Float16** and **Full INT8** TensorFlow Lite deployment models.

- **Float16 TFLite model** (`models/eye_state_model_fp16.tflite`): Size **44,500 bytes** (43.46 KB), achieving a **1.87x compression ratio** (46.42% reduction) with **0.000% F1 degradation** relative to FP32 Keras baseline.
- **Full INT8 TFLite model** (`models/eye_state_model_int8.tflite`): Size **26,160 bytes** (25.55 KB), achieving a **3.18x compression ratio** (68.50% reduction) with minimal F1 degradation of **-0.026%** (F1 0.962044 vs 0.962299).

**Final Verdict:** `EXP-003 COMPLETED SUCCESSFULLY`

---

## 2. Objective & Scope

The objective of EXP-003 is to export deployment-ready TensorFlow Lite models for edge inference (e.g., Raspberry Pi 4), measure quantization trade-offs, verify functional input/output contracts, and validate evaluation metrics on the frozen subject-disjoint TEST split (`Data/mrl_eye/splits_subject_disjoint/test.csv`, 9,377 samples).

### Frozen Constraints:
- Architecture unchanged: MicroEyeNet (19,745 total parameters).
- No retraining or fine-tuning performed.
- Hyperparameters frozen.
- TEST dataset split frozen (9,377 images: 1,349 awake / 8,028 sleepy).
- Operating point threshold: $0.50$.

---

## 3. Quantization Methodology & Conversion Pipeline

Conversion runner: `tools/export_and_evaluate_quantization.py`

### 3.1 Float16 Quantization Spec
- **Source Checkpoint:** `checkpoints/microeyenet_epoch08_valloss0.1793.keras`
- **Converter Engine:** `tf.lite.TFLiteConverter.from_keras_model`
- **Optimizations:** `[tf.lite.Optimize.DEFAULT]`
- **Target Types:** `[tf.float16]`
- **Output Asset:** `models/eye_state_model_fp16.tflite`

### 3.2 Full INT8 Quantization Spec
- **Source Checkpoint:** `checkpoints/microeyenet_epoch08_valloss0.1793.keras`
- **Converter Engine:** `tf.lite.TFLiteConverter.from_keras_model`
- **Optimizations:** `[tf.lite.Optimize.DEFAULT]`
- **Representative Dataset:** 500 normalized $[0, 1]$ grayscale image tensors $(1, 24, 24, 1)$ sampled from the MRL subject-disjoint training set (`train.csv`).
- **Supported Ops:** `[tf.lite.OpsSet.TFLITE_BUILTINS_INT8]`
- **Interface Contract:** `inference_input_type = tf.float32`, `inference_output_type = tf.float32` (maintains drop-in compatibility with `FrameProcessor` and `cnn_validator.py`).
- **Output Asset:** `models/eye_state_model_int8.tflite`

---

## 4. Measured Results

### 4.1 Model Size & Compression Comparison

| Model Variant | Format | Storage Path | Size (Bytes) | Size (KB) | Compression Ratio | Size Reduction (%) |
|---|---|---|---|---|---|---|
| **FP32 Keras Baseline** | `.keras` | `checkpoints/microeyenet_epoch08...` | 276,666 | 270.18 | 1.00x | 0.00% |
| **FP32 TFLite Baseline** | `.tflite` | `experiments/EXP-003_quantization/...` | 83,060 | 81.11 | 1.00x | 0.00% |
| **Float16 TFLite** | `.tflite` | `models/eye_state_model_fp16.tflite` | **44,500** | **43.46** | **1.87x** | **46.42%** |
| **Full INT8 TFLite** | `.tflite` | `models/eye_state_model_int8.tflite` | **26,160** | **25.55** | **3.18x** | **68.50%** |

---

### 4.2 Accuracy & Evaluation Metrics Comparison (TEST Split, $N=9,377$)

Evaluated on 9,377 images from `Data/mrl_eye/splits_subject_disjoint/test.csv` at threshold $\tau = 0.50$:

| Metric | FP32 Keras Baseline | FP32 TFLite | Float16 TFLite | Full INT8 TFLite | INT8 Δ vs FP32 Keras |
|---|---|---|---|---|---|
| **Accuracy** | 0.936227 | 0.936227 | **0.936227** | **0.935694** | -0.000533 (-0.057%) |
| **Precision** | 0.974215 | 0.974215 | **0.974215** | **0.972388** | -0.001827 (-0.188%) |
| **Recall** | 0.950673 | 0.950673 | **0.950673** | **0.951918** | +0.001245 (+0.131%) |
| **Specificity** | 0.850259 | 0.850259 | **0.850259** | **0.839140** | -0.011119 (-1.308%) |
| **F1-Score** | 0.962300 | 0.962300 | **0.962300** | **0.962044** | -0.000255 (-0.026%) |
| **ROC-AUC** | 0.969219 | 0.969219 | **0.969219** | **0.968339** | -0.000880 (-0.091%) |
| **PR-AUC** | 0.993701 | 0.993701 | **0.993702** | **0.993865** | +0.000164 (+0.016%) |
| **Brier Score** | 0.047995 | 0.047995 | **0.047976** | **0.048386** | +0.000391 (+0.815%) |

---

### 4.3 Measured Confusion Matrices (TEST Split, $\tau = 0.50$)

| Model Variant | True Positive (TP) | True Negative (TN) | False Positive (FP) | False Negative (FN) |
|---|---|---|---|---|
| **FP32 Keras Baseline** | 7,632 | 1,147 | 202 | 396 |
| **Float16 TFLite** | 7,632 | 1,147 | 202 | 396 |
| **Full INT8 TFLite** | 7,642 | 1,132 | 217 | 386 |

---

## 5. Functional & Integrity Verification

1. **Input/Output Specification Verification**:
   - Both Float16 and INT8 TFLite models were verified to accept input tensor shape `[1, 24, 24, 1]` with `dtype=float32` in range $[0, 1]$.
   - Both models produce output tensor shape `[1, 1]` with `dtype=float32` representing the sigmoid probability $P(\text{sleepy})$.
2. **Repository Research Integrity Invariants (I1–I6)**:
   - `python3 evaluation/verify_integrity.py` passed with exit code 0 (`ALL INVARIANTS HOLD`).
   - No duplicate `.tflite` model files detected (I4).
3. **Unit & Smoke Tests**:
   - `python3 -m unittest tests.test_suite`: 17/17 OK.
   - `python3 tests/smoke_test.py`: 3/3 OK.

---

## 6. Target Deliverables Summary

- `models/eye_state_model_fp16.tflite` (44,500 bytes) — Float16 quantized TFLite model.
- `models/eye_state_model_int8.tflite` (26,160 bytes) — Full INT8 quantized TFLite model.
- `experiments/EXP-003_quantization/` (contains `metrics.json`, `quantization_report.json`, `fp32_vs_fp16_vs_int8.csv`, `conversion.log`, `verification_report.json`).
- `reports/EXP-003_REPORT.md` (this report).

---

## 7. Next Steps (EXP-004)

With EXP-003 successfully closed, the repository possesses certified Float16 and INT8 TFLite model weights. The system is ready to proceed to **EXP-004: Benchmark Evaluation on NTHU-DDD** (Full Hybrid System vs Baseline Heuristic).
