# STAGE 8: PEER REVIEWER SIMULATION & COMMITTEE CONSENSUS

**Target Journal**: IEEE Transactions on Intelligent Transportation Systems (T-ITS)  
**Paper Title**: *A Lightweight Asymmetric Hybrid Architecture for Real-Time Driver Drowsiness Detection with Signal Reliability Gating*  
**Author**: Sayemuddin  
**Review Simulation Date**: July 2026

---

## 1. Reviewer A (Supportive / Applied ITS Specialist)

### Overall Recommendation: Weak Accept (Score: 7/10)

#### Strengths:
1. **Strong Practical Value for Embedded ADAS**: The paper addresses a major deployment hurdle in intelligent vehicles—running accurate drowsiness detection on low-cost CPUs without expensive GPUs.
2. **Elegant Architectural Hybridization**: Combining continuous wall-clock heuristics with selective CNN invocation is a smart, energy-efficient design.
3. **Signal Quality Monitor (`RobustnessGuard`)**: The multiplicative attenuation of fusion scores based on landmark jitter and brightness quality is a novel and intuitive approach to false-positive reduction.
4. **Clean Code Structure**: The repository architecture is remarkably modular, well-documented, and decoupled.

#### Weaknesses & Concerns:
1. **Lack of Public Dataset Benchmarking**: The submission claims high accuracy, but fails to present standardized confusion matrices, ROC curves, or F1-scores on public benchmarks like NTHU-DDD or YawDD.
2. **Missing Ablation Study for Weights**: The fusion weights ($w_{ear}=0.45, w_{pose}=0.30, w_{mar}=0.25$) appear ad-hoc. The author should present a sensitivity grid-search justifying these values.

---

## 2. Reviewer B (Neutral / Systems Engineering Specialist)

### Overall Recommendation: Borderline Reject (Score: 5/10)

#### Strengths:
1. **Solid Systems Engineering**: Replacing frame-count logic with wall-clock timing (`time.monotonic()`) solves the FPS-fluctuation problem on edge hardware.
2. **Thoughtful Mathematical Refinements**: Explaining and solving the 2D vs. 3D Euclidean metric difference for eye vs. mouth landmarks shows good analytical rigor.

#### Weaknesses & Concerns:
1. **Incremental CNN Selective Triggering**: The concept of selective CNN invocation for ambiguous states was already explored by Chen et al. (MDPI Sensors 2025). The author must highlight what distinguishes this work from Chen et al.
2. **Missing Trained Model File**: Upon inspecting the implementation artifact, the compiled TFLite model (`models/eye_state_model.tflite`) is absent, rendering the CNN validation experiments theoretical.
3. **Static Baseline Thresholds**: The EAR threshold ($0.21$) is static. How does the system handle diverse facial structures, eye shapes, or drivers wearing glasses?

---

## 3. Reviewer C (Highly Critical / Senior Computer Vision Reviewer)

### Overall Recommendation: Strong Reject (Score: 3/10)

#### Strengths:
1. Speech jitter filtering ($\sigma_{MAR}$) and pitch velocity gating are practical heuristics for noise suppression.

#### Weaknesses & Major Flaws:
1. **FATAL FLAW — Zero Empirical Benchmark Validation**: A manuscript submitted to IEEE T-ITS cannot be accepted without thorough empirical validation against established public benchmark datasets (NTHU-DDD, YawDD, UTA-RLDD, DROZY). Promising code architecture without statistical evaluation on public video datasets is insufficient for peer review.
2. **No Comparative Experimental Results**: The paper lacks direct baseline comparisons (e.g., Accuracy, Precision, Recall, FPS, Latency, CPU % Power) against recent SOTA methods (Reddy et al. 2021, Hassan et al. 2024, Chen et al. 2025).
3. **Unverified Hyperparameters**: Parameters such as hysteresis buffers ($0.03$), nod velocity thresholds ($-3^\circ/\text{s}$), and EMA alpha values ($0.3$) lack mathematical optimization proofs.

---

## 4. Meta-Reviewer Committee Consensus Decision

```
===================================================================================
                       IEEE T-ITS EDITORIAL DECISION
===================================================================================
DECISION: REJECT — REVISE AND RESUBMIT (Major Revision / Re-submission Required)

SUMMARY OF COMMITTEE CONSENSUS:
The Associate Editor and Review Panel acknowledge the strong engineering merit,
clean architecture, and genuine technical novelties of the proposed system (specifically
the RobustnessGuard signal quality engine, 2D depth divergence fix for MAR, and
speech/nod ambiguity filters).

However, in its CURRENT STATE, the project CANNOT BE PUBLISHED because it lacks 
empirical benchmark validation. Specifically:
  1. The CNN model file (models/eye_state_model.tflite) must be trained and included.
  2. The system MUST be evaluated on NTHU-DDD and YawDD datasets to generate 
     quantitative tables (Accuracy, Sensitivity, Specificity, F1-Score, FPS).
  3. An ablation study MUST prove the independent contributions of RobustnessGuard, 
     Selective CNN, and Speech Jitter Filtering.

IF THESE MANDATORY EXPERIMENTS ARE COMPLETED, THE MANUSCRIPT HAS HIGH POTENTIAL 
FOR ACCEPTANCE IN IEEE TRANSACTIONS ON INTELLIGENT TRANSPORTATION SYSTEMS OR IEEE IV.
===================================================================================
```
