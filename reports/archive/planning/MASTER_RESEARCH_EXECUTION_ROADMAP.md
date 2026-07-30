# MASTER RESEARCH EXECUTION ROADMAP

Version: 1.0

Author: Sayemuddin

Project:
Real-Time Driver Drowsiness Detection with Signal Reliability Gating

Target Conference:
IEEE Intelligent Vehicles Symposium (IV)

Secondary Target:
IEEE Intelligent Transportation Systems Conference (ITSC)

Long-Term Target:
IEEE Transactions on Intelligent Transportation Systems (T-ITS)

---

# PROJECT MISSION

Build a scientifically rigorous, reproducible, lightweight driver drowsiness detection system suitable for publication in a top transportation or computer vision venue.

Scientific quality is more important than coding speed.

Evidence is more important than assumptions.

Every completed task should increase publication readiness.

---

# CURRENT STATUS

Repository

✅ Complete

Architecture

✅ Complete

Pipeline

✅ Complete

Documentation

✅ Complete

Literature Review

✅ Complete

Research Gap

✅ Complete

Novelty Analysis

✅ Complete

Verification Audit

✅ Complete

Red Team Review

✅ Complete

CNN Model

❌ Missing

Benchmark Evaluation

❌ Missing

Ablation Study

❌ Missing

Conference Paper

❌ Missing

---

# PHASE 1

Repository Stabilization

Objective

Freeze the architecture before experiments begin.

Tasks

- Fix all identified code bugs.
- Fix headless mode optimization.
- Remove dead code.
- Remove duplicated logic.
- Improve comments.
- Freeze repository version.
- Create release tag v4.0.

Deliverable

Stable research codebase.

Exit Criteria

No known implementation bugs.

---
Phase 1.5 – Research Asset Generation

This phase would focus on creating everything needed for the paper before running experiments:

Architecture diagrams (publication quality)
Pipeline diagrams
State machine diagrams
Dataset flowcharts
Benchmark evaluation scripts
Experiment automation scripts
Configuration snapshots
Logging infrastructure
Reproducibility checklist


# PHASE 2

Dataset Preparation

Objective

Prepare reproducible datasets.

Tasks

Download:

MRL Eye Dataset

CEW Dataset

NTHU-DDD

YawDD

(Optional)

UTA-RLDD

DROZY

Prepare folder structure.

Create dataset documentation.

Verify annotations.

Split:

Training

Validation

Testing

Document every split.

Deliverable

datasets/

annotations/

DATASET.md

Exit Criteria

Datasets reproducible.

---

# PHASE 3

CNN Training

Objective

Train MicroEyeNet.

Tasks

Train eye classifier.

Experiment:

Learning rate

Batch size

Image size

Augmentation

Export

TFLite

ONNX

PyTorch

Generate:

Training curves

Loss curves

Confusion matrix

ROC

Precision Recall

Deliverable

models/

eye_state_model.tflite

Exit Criteria

Model verified.

---

# PHASE 4

Benchmark Evaluation

Objective

Generate publishable experimental evidence.

Tasks

Run:

NTHU-DDD

YawDD

Measure:

Accuracy

Precision

Recall

Specificity

F1

ROC

AUC

Latency

CPU

RAM

FPS

False Positives per Hour

Deliverable

benchmark_results/

Exit Criteria

Complete benchmark tables.

---

# PHASE 5

Ablation Study

Objective

Prove every module contributes.

Variants

A

EAR only

B

EAR + MAR

C

EAR + MAR + Pose

D

+ RobustnessGuard

E

+ CNN

F

Complete pipeline

Compare

Accuracy

Latency

False positives

Recall

Deliverable

Ablation tables.

Exit Criteria

Every contribution experimentally justified.

---

# PHASE 6

Novelty Validation

Objective

Scientifically prove contributions.

Tasks

Validate:

RobustnessGuard

2D MAR

Speech Filter

Pitch Gate

Wall Clock Timing

CNN Validation

For every module answer:

Does it help?

By how much?

Is improvement statistically significant?

Deliverable

Contribution analysis.

Exit Criteria

Every contribution supported by experiments.

---

# PHASE 7

Paper Preparation

Write

Abstract

Introduction

Related Work

Methodology

Experiments

Discussion

Limitations

Future Work

Conclusion

Figures

Architecture

Pipeline

Flowcharts

ROC

Confusion Matrix

Ablation

Benchmark Tables

Deliverable

paper/

Exit Criteria

Conference-ready manuscript.

---

# PHASE 8

Internal Review

Run

Fact Checker

Red Team

Grammar Review

Citation Review

Reference Verification

IEEE Format Check

Reproducibility Audit

Fix every issue.

Deliverable

Submission Candidate.

Exit Criteria

No critical reviewer comments remain.

---

# PHASE 9

Conference Submission

Target

IEEE IV

Backup

IEEE ITSC

Prepare

IEEE PDF

Source Code

Supplementary

Video

GitHub Release

Submit.

---

# PHASE 10

Journal Extension

After conference acceptance

Improve

Dataset

Experiments

Cross-dataset evaluation

Statistical testing

Expand paper

Submit

IEEE T-ITS

or

Elsevier Pattern Recognition

---

# DAILY WORKFLOW

Every day

1.

Open this roadmap.

2.

Identify highest priority task.

3.

Complete only that task.

4.

Commit changes.

5.

Update documentation.

6.

Run tests.

7.

Mark task complete.

Never work outside this roadmap.

---

# SUCCESS CRITERIA

Conference Paper Accepted

↓

Journal Extension Completed

↓

Journal Accepted

↓

Open Source Release

↓

Research Impact

---

# GOLDEN RULES

Never fabricate experiments.

Never fabricate citations.

Never fabricate results.

Never overstate novelty.

Never skip benchmark evaluation.

Never skip ablation studies.

Never claim improvements without statistical evidence.

Scientific integrity is always more important than publication.