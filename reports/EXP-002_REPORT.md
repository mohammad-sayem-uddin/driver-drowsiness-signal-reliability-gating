# EXP-002 — MicroEyeNet Baseline Training Report

**Experiment ID:** EXP-002
**Title:** MicroEyeNet eye-state classifier — baseline supervised training
**Date:** 2026-07-28
**Status:** COMPLETED (training run finished, exit code 0)
**Author:** Research engineering (reproducible experiment)

> All numbers in this report are transcribed verbatim from the measured artifacts
> produced by the completed training run
> (`experiments/EXP-002_microeyenet_baseline/exp002_metrics.json`,
> `logs/EXP-002_run_stdout.log`, `logs/EXP-002_training_log.csv`,
> `experiments/EXP-002_microeyenet_baseline/model_summary.txt`).
> No value is projected, estimated, or fabricated. Where a quantity was not
> measured in this experiment, it is explicitly marked **NOT MEASURED**.

---

## 1. Objective

Train the frozen MicroEyeNet CNN to classify a single cropped eye image as
**open (label 0, "awake")** or **closed (label 1, "sleepy")**, on the
subject-disjoint MRL Eye split, under the frozen training specification, and
record the measured performance for the experiment registry.

This experiment covers **training and evaluation only**. TFLite export and
on-device latency measurement are out of scope (EXP-003) and are reported here
as **NOT MEASURED**.

---

## 2. Model Architecture (frozen — unchanged)

Input: 24×24×1 grayscale. Output: sigmoid = P(eye closed).

| Layer            | Output shape       | Params  |
|------------------|--------------------|---------|
| conv2d (Conv2D 8, 3×3, ReLU, same)   | (None, 24, 24, 8)  | 80      |
| max_pooling2d (2×2)                  | (None, 12, 12, 8)  | 0       |
| conv2d_1 (Conv2D 16, 3×3, ReLU, same)| (None, 12, 12, 16) | 1,168   |
| max_pooling2d_1 (2×2)                | (None, 6, 6, 16)   | 0       |
| flatten                              | (None, 576)        | 0       |
| dense (Dense 32, ReLU)               | (None, 32)         | 18,464  |
| dropout (0.3)                        | (None, 32)         | 0       |
| dense_1 (Dense 1, Sigmoid)           | (None, 1)          | 33      |

**Total measured parameters: 19,745** (Trainable: 19,745; Non-trainable: 0).

Source: `model_summary.txt` and the runtime log line
`[EXP-002] MicroEyeNet built. Params = 19745`.

---

## 3. Dataset & Split (frozen — subject-disjoint)

Split files: `Data/mrl_eye/splits_subject_disjoint/{train,val,test}.csv`
(seed 42, TRAIN_FRAC 0.70 / VAL_FRAC 0.15). Class map: `{"awake": 0, "sleepy": 1}`.
Subject disjointness and file existence were verified in Phase 3
(`reports/verification/EXP-002_DATASET_VERIFICATION.md`).

| Split | Images | Class 0 (open) | Class 1 (closed) |
|-------|--------|----------------|------------------|
| Train | 70,551 | 38,658         | 31,893           |
| Val   | 4,970  | 2,945          | 2,025            |
| Test  | 9,377  | 1,349          | 8,028            |

Tensor shapes loaded at runtime (from stdout):
`train=(70551, 24, 24, 1) val=(4970, 24, 24, 1) test=(9377, 24, 24, 1)`,
loaded in 26.2 s.

> Note: the test split is class-imbalanced (~86% closed). This is a property of
> the subject-disjoint split and is reported as measured; no rebalancing or
> class weighting was applied (frozen spec: `class_weights = null`).

---

## 4. Training Configuration (frozen — from `training_config.json`)

| Hyperparameter        | Value                                              |
|-----------------------|----------------------------------------------------|
| Optimizer             | Adam                                               |
| Learning rate (init)  | 1e-3                                               |
| Gradient clipnorm     | 1.0                                                |
| Batch size            | 64                                                 |
| Max epochs            | 30                                                 |
| Loss                  | Binary cross-entropy                               |
| Dropout               | 0.3                                                |
| LR scheduler          | ReduceLROnPlateau (factor 0.5, patience 3, val_loss)|
| Early stopping        | val_loss, patience 5, restore_best_weights = true  |
| Mixed precision       | OFF                                                |
| Weight decay          | null                                               |
| Class weights         | null                                               |
| Seed                  | 42                                                 |
| TensorFlow version    | 2.17.1                                             |

---

## 5. Training Outcome (measured)

- **Epochs run:** 13 (of max 30)
- **Best epoch:** 8 (minimum val_loss = 0.17927)
- **Early stopped:** true (val_loss patience 5 triggered; best weights restored)
- **Training time:** 102.777 s
- **Data load time:** 26.244 s

Learning-rate schedule as measured (ReduceLROnPlateau):
1e-3 for epochs 1–6, 5e-4 for epochs 7–11, 2.5e-4 for epochs 12–13.

Selected epoch trace (from `EXP-002_training_log.csv` / stdout):

| Epoch | acc    | loss   | val_acc | val_loss | lr      |
|-------|--------|--------|---------|----------|---------|
| 1     | 0.8816 | 0.2887 | 0.9093  | 0.2073   | 1e-3    |
| 3     | —      | —      | —       | 0.1799   | 1e-3    |
| 8*    | 0.9670 | 0.0915 | 0.9402  | 0.1793   | 5e-4    |
| 13    | 0.9726 | 0.0772 | 0.9425  | 0.1949   | 2.5e-4  |

\* Best epoch (restored weights). Val metrics below are reported from the
restored best-epoch model.

Artifacts confirming the run:
`[EXP-002] DONE. epochs_run=13 best_epoch=8 early_stopped=True train_time=102.8s`

---

## 6. Evaluation Results (measured)

### 6.1 Validation @ threshold 0.5

| Metric            | Value    |
|-------------------|----------|
| Accuracy          | 0.9402   |
| Balanced accuracy | 0.9372   |
| Precision         | 0.9316   |
| Recall            | 0.9210   |
| Specificity       | 0.9535   |
| F1                | 0.9262   |
| ROC-AUC           | 0.9824   |
| PR-AUC            | 0.9701   |
| Brier             | 0.0480   |

Confusion matrix (val): TP = 1,865, TN = 2,808, FP = 137, FN = 160.

### 6.2 Test @ threshold 0.5

| Metric            | Value    |
|-------------------|----------|
| Accuracy          | 0.9362   |
| Balanced accuracy | 0.9005   |
| Precision         | 0.9742   |
| Recall            | 0.9507   |
| Specificity       | 0.8503   |
| F1                | 0.9623   |
| ROC-AUC           | 0.9692   |
| PR-AUC            | 0.9937   |
| Brier             | 0.0480   |

Confusion matrix (test): TP = 7,632, TN = 1,147, FP = 202, FN = 396.

### 6.3 Test @ val-selected best-F1 threshold

Best-F1 threshold selected on **validation** = 0.3700 (val best-F1 = 0.9286),
then applied to the **test** set (no test-set tuning):

| Metric            | Value    |
|-------------------|----------|
| Accuracy          | 0.9433   |
| Balanced accuracy | 0.8892   |
| Precision         | 0.9685   |
| Recall            | 0.9651   |
| Specificity       | 0.8132   |
| F1                | 0.9668   |

Confusion matrix (test @ 0.37): TP = 7,748, TN = 1,097, FP = 252, FN = 280.

---

## 7. Artifacts Produced

Located in `experiments/EXP-002_microeyenet_baseline/`:

| File                          | Bytes  | Description                              |
|-------------------------------|--------|------------------------------------------|
| exp002_metrics.json           | 4,527  | Full measured metrics + config (authoritative) |
| training_config.json          | 650    | Frozen hyperparameters used              |
| model_summary.txt             | 3,205  | Keras layer/param summary (19,745)       |
| learning_curves.png           | 56,738 | Loss/accuracy vs epoch                    |
| roc_curve_test.png            | 26,294 | Test ROC curve                            |
| pr_curve_test.png             | 22,281 | Test precision-recall curve               |
| confusion_matrix_test.png     | 17,568 | Test confusion matrix @0.5                 |
| reliability_diagram_val.png   | 33,296 | Validation calibration reliability diagram |

Supporting logs: `logs/EXP-002_run_stdout.log`, `logs/EXP-002_training_log.csv`.
Checkpoints: `checkpoints/microeyenet_epoch{01,02,03,08}_valloss*.keras`
(best = epoch 08). TensorBoard: `tensorboard/EXP-002_microeyenet_baseline/`.

---

## 8. Deployment Metrics — NOT MEASURED

- **TFLite model size (EXP-002):** NOT MEASURED — TFLite export is EXP-003 scope.
  The existing `models/eye_state_model.tflite` (26,488 bytes) is from a prior
  experiment and was **not** produced or modified by EXP-002.
- **Inference latency:** NOT MEASURED for EXP-002. The only measured latency in
  this project is the EXP-001 full-pipeline latency of **3.205 ms on the
  Darwin-arm64 host** (not a Raspberry Pi 4). **No Raspberry Pi 4 latency has
  been measured, and none is reported.**

---

## 9. Environment

TensorFlow 2.17.1 (Python 3.11, `.venv`); numpy 1.26.4; OpenCV 4.11.0;
matplotlib 3.10.9. `scikit-learn` is not installed; ROC/PR/F1-threshold metrics
were computed with hand-rolled numpy trapezoidal routines in
`tools/train_exp002_microeyenet.py`.

---

## 10. Summary

MicroEyeNet (19,745 parameters) trained on the subject-disjoint MRL Eye split
under the frozen specification, early-stopping at epoch 13 with best weights from
epoch 8. Measured performance: **validation accuracy 0.9402 (F1 0.9262,
ROC-AUC 0.9824)** and **test accuracy 0.9362 (F1 0.9623, ROC-AUC 0.9692,
PR-AUC 0.9937)** at threshold 0.5. Deployment size and latency for this model
are **NOT MEASURED** and are deferred to EXP-003.
