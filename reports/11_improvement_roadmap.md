# STAGE 10: SCIENTIFIC IMPROVEMENT ROADMAP

**Project**: Driver Drowsiness Detection System (v3.1)  
**Author**: Sayemuddin  
**Auditor**: Permanent AI Research Team (Research Project Management)  
**Date**: July 2026

---

## Ranked Action Plan for Paper Publication

Below is the prioritized, multi-phase execution roadmap required to elevate this repository from a **Research Prototype** to a **Peer-Reviewed Journal Publication (IEEE T-ITS)**.

---

### Task 1: Train MicroEyeNet & Populate `models/eye_state_model.tflite`
- **Priority**: 🔴 **CRITICAL (P0)**
- **Difficulty**: Low (2 / 10)
- **Estimated Effort**: 4 – 8 hours
- **Dependencies**: None
- **Actions Required**:
  1. Download public eye datasets (MRL Eye Dataset ~84k images or CEW ~4k images).
  2. Organize images into `data/eyes/open` and `data/eyes/closed`.
  3. Execute `python3 tools/train_eye_cnn.py` to train the ~9.5K parameter MicroEyeNet model.
  4. Verify export of `models/eye_state_model.tflite`.
- **Scientific Value**: 10/10 (Activates Tier 4 Selective CNN Validation).
- **Reviewer Impact**: Eliminates Reviewer B & C's primary technical rejection point.
- **Expected Outcome**: Working TFLite eye state classifier with $<0.5\text{ms}$ inference latency.

---

### Task 2: Build Automated Public Benchmark Evaluation Harness (NTHU-DDD & YawDD)
- **Priority**: 🔴 **CRITICAL (P0)**
- **Difficulty**: Medium (5 / 10)
- **Estimated Effort**: 2 – 3 days
- **Dependencies**: Task 1
- **Actions Required**:
  1. Download NTHU-DDD (National Tsing Hua University Driver Drowsiness Dataset) video sequences and ground-truth text labels.
  2. Write `tools/evaluate_benchmark.py` to process video clips frame-by-frame through `src/main.py` pipeline (in headless mode).
  3. Calculate frame-level Confusion Matrix, Accuracy, Precision, Recall, F1-Score, ROC-AUC, and episode-level False Alarm Rate.
- **Scientific Value**: 10/10 (Provides empirical ground truth required by IEEE journals).
- **Reviewer Impact**: Satisfies Reviewer C's mandatory empirical validation requirement.
- **Expected Outcome**: Quantitative performance tables for the manuscript.

---

### Task 3: Conduct Systematic Ablation Study
- **Priority**: 🟠 **HIGH (P1)**
- **Difficulty**: Medium (4 / 10)
- **Estimated Effort**: 1 – 2 days
- **Dependencies**: Task 2
- **Actions Required**:
  1. Run automated evaluation across 4 pipeline variants:
     - *Variant A*: Standard Heuristic EAR (Soukupová 2016).
     - *Variant B*: Heuristic + Wall-Clock + Speech Jitter Filter.
     - *Variant C*: Heuristic + RobustnessGuard Signal Quality Gating.
     - *Variant D*: Proposed Full System (Heuristic + RobustnessGuard + Selective MicroEyeNet).
  2. Plot ROC curves and compute metrics across all 4 variants to prove the individual contribution of each module.
- **Scientific Value**: 9/10 (Demonstrates rigorous experimental methodology).
- **Reviewer Impact**: High (Proves that each subsystem contributes to false-positive reduction).
- **Expected Outcome**: Ablation comparative table for Section V of the paper.

---

### Task 4: Edge Hardware Latency & Profiling Benchmark
- **Priority**: 🟡 **MEDIUM (P2)**
- **Difficulty**: Low (3 / 10)
- **Estimated Effort**: 1 day
- **Dependencies**: Task 1
- **Actions Required**:
  1. Deploy codebase to Raspberry Pi 4B (ARM Cortex-A72, 4GB RAM) running Raspberry Pi OS (64-bit).
  2. Run `main.py` with `enable_profiling=True` and record latency distributions (ms) for:
     - MediaPipe FaceMesh inference.
     - Geometric feature calculation (`detector.py` + `pose_estimator.py`).
     - Temporal analysis (`temporal_analyzer.py`).
     - Robustness guard (`robustness.py`).
     - Selective CNN inference (`cnn_validator.py`).
  3. Log CPU utilization %, temperature, and power draw.
- **Scientific Value**: 8/10 (Proves real-time edge feasibility claims).
- **Reviewer Impact**: High for IEEE IV / IEEE T-ITS ITS hardware reviewers.
- **Expected Outcome**: Latency breakdown table showing $<28\text{ms}$ total frame processing on Pi 4.

---

### Task 5: Draft IEEE Conference / Journal Manuscript
- **Priority**: 🟢 **FINAL PHASE (P3)**
- **Difficulty**: Medium (6 / 10)
- **Estimated Effort**: 1 week
- **Dependencies**: Tasks 1, 2, 3, 4
- **Actions Required**:
  1. Format paper using standard IEEE two-column LaTeX template (`IEEEtran.cls`).
  2. Draft sections: Abstract, Introduction, Related Work, System Architecture, Mathematical Formulations, Experimental Results & Ablation, Discussion, Conclusion.
  3. Insert architectural block diagrams, Mermaid charts, ROC curves, and benchmark tables.
- **Scientific Value**: 10/10 (Final publishable artifact).
- **Expected Outcome**: Submission-ready LaTeX manuscript for IEEE T-ITS / IEEE IV.
