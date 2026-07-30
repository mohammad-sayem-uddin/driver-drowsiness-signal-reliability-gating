# SIMULATED 5-REVIEWER PANEL COMMENTS

**Submission Target**: IEEE Transactions on Intelligent Transportation Systems / IEEE IV  
**Manuscript Title**: *A Lightweight Asymmetric Hybrid Architecture for Real-Time Driver Drowsiness Detection with Signal Reliability Gating*  
**Date**: July 2026

---

## Reviewer A — Embedded Systems Expert
**Recommendation**: Reject / Major Revision  
**Score**: 4 / 10

### Detailed Comments:
1. **Unverified Edge Performance**: The authors claim the pipeline processes video frames in $<28\text{ms}$ on a Raspberry Pi 4B. However, the repository contains no profiling logs from an actual ARM Cortex-A72 board running Linux. Desktop CPU latency cannot be linearly extrapolated to ARM edge processors.
2. **Headless Mode Inefficiency Bug**: In `src/main.py` (lines 180–183), the code performs color conversion (`cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)`) before checking if `headless_mode=True`. On constrained hardware like Raspberry Pi, performing unnecessary color space conversions on 720p frames severely degrades frame rate.
3. **Memory & Threading Concerns**: Loading MediaPipe Solutions allocates ~180MB–250MB RAM. The paper must include explicit CPU usage %, memory footprint, and thermal throttling stability benchmarks over continuous 1-hour driving simulations.

---

## Reviewer B — Computer Vision Expert
**Recommendation**: Weak Reject  
**Score**: 5 / 10

### Detailed Comments:
1. **Validation Needed for 2D vs. 3D Lip Depth Metric**: The authors claim that 3D Euclidean distances cause MAR values to inflate $>2.0$ due to uncalibrated $z$-depth divergence during yawning, and propose a 2D Euclidean metric fix in `src/detector.py`. While theoretically reasonable, the paper lacks visual or quantitative proof (e.g., plots comparing 2D MAR vs. 3D MAR across frames of open-mouth gestures).
2. **Landmark Jitter Under Adverse Lighting**: MediaPipe FaceMesh suffers severe landmark jitter under low light ($<30$ lux) or when drivers wear reflective glasses. While `RobustnessGuard` attempts to penalize jitter, there are no experiments evaluating landmark tracking breakdown under near-infrared (NIR) or nighttime driving conditions.

---

## Reviewer C — Machine Learning Expert
**Recommendation**: Reject  
**Score**: 3 / 10

### Detailed Comments:
1. **FATAL FLAW — Missing Trained Model Weights**: The core premise of the paper involves invoking a 9.5K-parameter MicroEyeNet CNN during EAR ambiguity. However, `models/eye_state_model.tflite` is absent from the repository. `CNNValidator` outputs a runtime warning and falls back to pure heuristics. Evaluating an unpopulated model renders the entire machine learning validation layer theoretical.
2. **Unbacked False Positive Reduction Claims**: The paper claims an 80% reduction in false positives via selective CNN invocation. Without training dataset split details, hyperparameter logs, or model evaluation metrics (ROC-AUC, Precision-Recall curves), this claim is completely unsupported.

---

## Reviewer D — Research Methodology Expert
**Recommendation**: Strong Reject  
**Score**: 2 / 10

### Detailed Comments:
1. **FATAL FLAW — Zero Benchmark Dataset Evaluation**: A research manuscript submitted to IEEE T-ITS or IEEE IV cannot be accepted without quantitative evaluation on public benchmark datasets (NTHU-DDD, YawDD, UTA-RLDD). The repository contains **ZERO evaluation scripts** and `data/eyes/` is completely empty.
2. **Unsupported >95% Accuracy Figure**: Stating ">95% accuracy" in the manuscript without providing confusion matrices, sensitivity, specificity, F1-scores, or statistical significance tests ($p$-values) is unacceptable scientific practice.
3. **Missing Baseline & Ablation Comparisons**: There are no comparative experiments evaluating the proposed system against existing baselines (Soukupová 2016, Reddy 2017, Hassan 2024).

---

## Reviewer E — IEEE Associate Editor
**Recommendation**: Desk Reject / Major Revision  
**Score**: 3 / 10

### Detailed Comments:
1. **Citation & Metadata Errors**: The manuscript contains significant citation inaccuracies (e.g., citing B. Reddy et al. as a 2021 3D-CNN paper instead of CVPRW 2017 model compression; citing Horng et al. as 2018 EAR instead of 2004 template matching).
2. **Novelty Reframing Required**: Selective CNN invocation is already anticipated in 2025 edge literature (Chen et al.). The paper must shift its primary claim to the `RobustnessGuard` multi-factor signal quality engine and physical metric separation.
3. **Editorial Summary**: In its current form, the paper is an unvalidated software prototype. It must undergo full benchmark evaluation before resubmission.
