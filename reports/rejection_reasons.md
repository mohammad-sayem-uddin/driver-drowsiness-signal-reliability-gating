# EXHAUSTIVE CATEGORIZED REJECTION REASONS

**Manuscript Title**: *A Lightweight Asymmetric Hybrid Architecture for Real-Time Driver Drowsiness Detection with Signal Reliability Gating*  
**Auditor**: Reviewer #2 (Red Team Review Panel)  
**Date**: July 2026

---

## 1. Category 1: Experimental & Benchmark Flaws (Highest Severity)

### R1.1: Complete Lack of Public Benchmark Dataset Evaluation
- **Severity**: 🔴 **CRITICAL / FATAL**
- **Likelihood Reviewers Notice It**: 100%
- **Description**: The repository contains zero evaluation runs on standard public driver drowsiness datasets (NTHU-DDD, YawDD, UTA-RLDD, DROZY). `data/eyes/` is empty and no evaluation scripts exist in `tools/`.
- **Impact**: Any accuracy or false-positive reduction claim is treated as unbacked paperware by IEEE reviewers.

### R1.2: Unbacked Accuracy and False Positive Reduction Figures
- **Severity**: 🔴 **CRITICAL / FATAL**
- **Likelihood Reviewers Notice It**: 100%
- **Description**: Stating ">95% accuracy" and "80% false-positive reduction" in the abstract/intro without presenting dataset confusion matrices, ROC curves, or precision-recall numbers violates IEEE scientific standards.

### R1.3: Absence of Subsystem Ablation Studies
- **Severity**: 🔴 **HIGH**
- **Likelihood Reviewers Notice It**: 95%
- **Description**: The paper claims that `RobustnessGuard` and speech MAR jitter filtering improve system robustness. Without a 4-variant ablation study, there is no empirical proof that these individual modules contribute to performance.

---

## 2. Category 2: Machine Learning & Codebase Flaws

### R2.1: Missing TFLite Model Weight Asset (`models/eye_state_model.tflite`)
- **Severity**: 🔴 **CRITICAL / FATAL**
- **Likelihood Reviewers Notice It**: 100%
- **Description**: The core machine learning validation layer (`src/cnn_validator.py`) depends on `models/eye_state_model.tflite`, which does not exist in the repository. When executed, the system degrades to pure heuristic mode.

### R2.2: Headless Mode CPU Overhead Bug in `src/main.py`
- **Severity**: 🟡 **MEDIUM**
- **Likelihood Reviewers Notice It**: 60%
- **Description**: In `src/main.py` lines 180–183, `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)` is executed unconditionally before checking `headless_mode=True`. This wastes CPU cycles during headless benchmark runs on embedded hardware.

---

## 3. Category 3: Novelty & Citation Flaws

### R3.1: Overstating Selective CNN Invocation Novelty
- **Severity**: 🔴 **HIGH**
- **Likelihood Reviewers Notice It**: 90%
- **Description**: Framing selective CNN invocation on EAR ambiguity as a primary novelty ignores Chen et al. (*Sensors*, Jan 2025), who previously published a dual-stage selective CNN pipeline.

### R3.2: Citation Metadata Errors and Misattributions
- **Severity**: 🟡 **MEDIUM**
- **Likelihood Reviewers Notice It**: 85%
- **Description**: Misattributing B. Reddy et al. as a 2021 3D-CNN paper (it is CVPRW 2017 model compression) and Horng et al. as a 2018 EAR paper (it is 2004 template matching) damages academic credibility.

---

## 4. Category 4: Physical & Mathematical Validation Flaws

### R4.1: Missing Empirical Proof for 2D vs. 3D Lip Depth Fix
- **Severity**: 🟡 **MEDIUM**
- **Likelihood Reviewers Notice It**: 75%
- **Description**: The manuscript claims 3D Euclidean metrics corrupt MAR due to MediaPipe $z$-depth divergence $>2.0$ during wide mouth opening. While theoretically correct, the paper lacks visual/quantitative plots comparing 2D MAR vs. 3D MAR across mouth gesture frames.
