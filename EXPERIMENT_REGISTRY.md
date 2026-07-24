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
| **EXP-000** | 2026-07-24 | Ingested Base | MicroEyeNet | 42 | N/A | N/A | N/A | N/A | Baseline | Baseline | 0 KB | <0.5 ms | Initial data foundation & validator initialized. |

---

## 3. Detailed Experiment Logs

### EXP-000: Data Infrastructure Verification & Baseline Setup
- **Date**: July 24, 2026
- **Status**: Completed (Phase 02 Data Pipeline)
- **Objective**: Establish pre-flight dataset validation and zero-leakage subject-independent partitioning.
- **Dataset**: Ingested open/closed eye patches (20 subjects, 200 crops, $24\times24$ grayscale).
- **Validator Output**: `tools/dataset_validator.py` verified 0 subject leakage across Train (14 subjs), Val (3 subjs), and Test (3 subjs).
- **Scientific Takeaway**: Research data foundation is certified and ready for Phase 3 CNN model training (`EXP-001`).

---

## 4. Planned Experiment Pipeline (Phase 3 & Phase 4)

- **EXP-001**: MicroEyeNet Baseline Training ($24\times24$ Grayscale, Adam LR=1e-3, 30 Epochs).
- **EXP-002**: MicroEyeNet Hyperparameter Search (LR Decay, Dropout 0.2 vs 0.4, Data Augmentation).
- **EXP-003**: MicroEyeNet Float16 Quantization & TFLite Export (`models/eye_state_model.tflite`).
- **EXP-004**: Benchmark Evaluation on NTHU-DDD (Full Hybrid System vs Baseline Heuristic).
- **EXP-005**: 4-Variant Ablation Study (Baseline vs. Speech Filter vs. RobustnessGuard vs. Proposed Hybrid).
