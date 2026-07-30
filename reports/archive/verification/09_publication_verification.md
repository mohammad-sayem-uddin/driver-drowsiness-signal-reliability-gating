# STAGE 9: PUBLICATION CLAIM VERIFICATION & VENUE EVALUATION

**Auditing Body**: Scientific Verification Committee (IEEE Transactions Associate Editor & Committee Lead)  
**Scope**: Independent assessment of paper submission viability and venue readiness  
**Date**: July 2026

---

## 1. Official Peer Review Readiness Classification

```
===================================================================================
                   OFFICIAL VERIFICATION COMMITTEE CLASSIFICATION
===================================================================================

CURRENT STATUS: [ REJECT AS-IS (Major Revision Required Before Submission) ]

VERDICT BREAKDOWN:
  1. Submitted TODAY As-Is                --> [ INSTANT DESK REJECT / STRONG REJECT ]
     Reason: Missing TFLite model weights; 0 public benchmark dataset evaluation runs.

  2. Submitted AFTER 4-Step Remediation  --> [ CONFERENCE READY / JOURNAL CANDIDATE ]
     Target Venues: IEEE Intelligent Vehicles Symposium (IV 2027) / IEEE T-ITS.
===================================================================================
```

---

## 2. Definitive Answers to Publication Questions

### Question 1: Can this paper be submitted today?
**Answer**: **NO**. Submitting the repository in its current state will result in immediate rejection.

### Question 2: Would an Associate Editor desk-reject it today?
**Answer**: **YES**. An IEEE Transactions or IEEE Conference Associate Editor reviewing a manuscript without empirical accuracy, recall, and benchmark dataset comparison tables will desk-reject the submission within 48 hours for lacking experimental validation.

### Question 3: Would peer reviewers reject it?
**Answer**: **YES**. Reviewers will unanimously cite:
1. Missing trained model weights (`models/eye_state_model.tflite`).
2. Absence of statistical evaluations on standard datasets (NTHU-DDD, YawDD).
3. Lack of an empirical ablation study proving the value of `RobustnessGuard`.

### Question 4: Should the author target a Conference first or a Journal first?
**Answer**: **CONFERENCE FIRST (IEEE IV 2027 or IEEE ITSC 2027)**.
- A 6-page conference submission at IEEE IV / ITSC requires less benchmark depth than a 10-page IEEE Transactions journal paper.
- Achieving conference acceptance provides peer feedback, establishes priority of invention, and creates a clear path to expand the work into a full IEEE T-ITS journal manuscript later.

---

## 3. Mandatory Pre-Submission Action Checklist

| Phase | Milestone Task | Status | Required Output |
|:---|:---|:---|:---|
| **Phase 1** | Train MicroEyeNet | 🔴 Pending | Compiled `models/eye_state_model.tflite` (<9.5K params, <0.5ms latency). |
| **Phase 2** | NTHU-DDD Benchmark | 🔴 Pending | Script `tools/evaluate_benchmark.py` generating Accuracy, F1, FPR/hr metrics. |
| **Phase 3** | Ablation Study | 🔴 Pending | 4-Variant comparison table (Baseline vs. Speech Filter vs. Guard vs. Full Hybrid). |
| **Phase 4** | Pi 4 Latency Profiling | 🔴 Pending | Latency breakdown table on physical ARM Cortex-A72 hardware. |
| **Phase 5** | IEEE LaTeX Manuscript | 🔴 Pending | 6-Page double-column manuscript using `IEEEtran.cls`. |
