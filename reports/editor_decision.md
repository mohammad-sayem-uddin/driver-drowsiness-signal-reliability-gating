# IEEE TRANSACTIONS ON INTELLIGENT TRANSPORTATION SYSTEMS

**EDITORIAL BOARD DECISION LETTER**

**Date**: July 24, 2026  
**Manuscript ID**: T-ITS-2026-07-0482  
**Manuscript Title**: *A Lightweight Asymmetric Hybrid Architecture for Real-Time Driver Drowsiness Detection with Signal Reliability Gating*  
**Author**: Sayemuddin  
**Handling Associate Editor**: Dr. Reviewer E (Senior IEEE Editor)

---

## 1. Official Editorial Decision

```
===================================================================================
                       IEEE T-ITS EDITORIAL BOARD DECISION
===================================================================================

DECISION: REJECT (Major Re-submission / Revision Required)

SUMMARY OF EVALUATION:
The Editorial Board and Review Panel have completed the peer review of your manuscript. 
While your submission presents an interesting systems engineering concept for edge-based 
driver fatigue monitoring (specifically the RobustnessGuard signal quality engine, 2D MAR 
metric fix, and wall-clock temporal analyzer), the manuscript in its CURRENT FORM 
CANNOT BE PUBLISHED.

Primary Grounds for Rejection:
  1. Complete absence of empirical evaluations on public benchmark datasets (NTHU-DDD / YawDD).
  2. Missing compiled TFLite model weights (models/eye_state_model.tflite).
  3. Unbacked claims regarding classification accuracy (>95%) and false positive reduction (80%).
  4. Citation metadata errors and overstating novelty regarding selective CNN triggering.

You are invited to perform the mandatory experimental remediation detailed below and 
resubmit the manuscript as a new submission.
===================================================================================
```

---

## 2. Reviewer Consensus Matrix

| Reviewer | Expertise Domain | Recommendation | Score (1-10) | Primary Concern |
|:---|:---|:---|:---|:---|
| **Reviewer A** | Embedded Systems | Reject / Major Revision | 4 / 10 | Unverified Pi 4 latency; headless mode CPU overhead bug. |
| **Reviewer B** | Computer Vision | Weak Reject | 5 / 10 | Lack of empirical proof for 2D vs 3D MAR depth divergence fix. |
| **Reviewer C** | Machine Learning | Reject | 3 / 10 | Missing `models/eye_state_model.tflite` model weights. |
| **Reviewer D** | Research Methodology | Strong Reject | 2 / 10 | **Zero public benchmark dataset runs** (NTHU-DDD / YawDD). |
| **Reviewer E** | Associate Editor | Desk Reject / Re-submission | 3 / 10 | Unbacked accuracy claims; citation metadata misattributions. |

---

## 3. Mandatory Requirements for Resubmission

To achieve a favorable decision upon resubmission, the authors MUST address the following requirements:

1. **Train and Include MicroEyeNet TFLite Weights**: Train the ~9.5K-parameter CNN model using `tools/train_eye_cnn.py` and populate `models/eye_state_model.tflite`.
2. **Execute Automated Benchmark Evaluations**: Evaluate the system frame-by-frame on **NTHU-DDD** and **YawDD** datasets, presenting full tables for Accuracy, Precision, Sensitivity (Recall), F1-Score, ROC-AUC, and FPR/hr.
3. **Conduct 4-Variant Ablation Study**: Provide empirical performance tables comparing Baseline EAR vs. Speech Jitter Filter vs. RobustnessGuard vs. Proposed Full Hybrid System.
4. **Fix Headless Mode Bug**: Move `cv2.cvtColor` inside the rendering condition in `src/main.py`.
5. **Correct Citation Metadata**: Fix citations for B. Reddy et al. (CVPRW 2017) and Horng et al. (ICNSC 2004), and cite missing SOTA papers (Zhang et al. T-ITS 2023, Happy & Routray T-IV 2022).
