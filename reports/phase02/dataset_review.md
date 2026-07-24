# TASK 1 & 2: EXHAUSTIVE DATASET REVIEW & SELECTION ANALYSIS

**Auditing Body**: AI Research Engineer & Benchmark Infrastructure Architect  
**Scope**: Literature audit and selection analysis of 8 public driver drowsiness and eye patch datasets  
**Date**: July 2026

---

## 1. Candidate Dataset Evaluation Matrix

| Dataset Name | Domain / Modality | Scale / Subjects | Ground-Truth Annotations | Access & License | Citation | Strengths & Use Case | Key Weaknesses / Limitations |
|:---|:---|:---|:---|:---|:---|:---|:---|
| **MRL Eye Dataset** | Micro Eye Crop Patches | 84,898 images (37 subjects) | Binary (open/closed), glasses, reflection, lighting | CC BY 4.0 (Open Academic) | MRL (2018) | Ideal for MicroEyeNet CNN training ($24\times24$ inputs). Includes infrared & glasses. | Isolated eye crops; lacks temporal video sequences or head pose. |
| **Closed Eyes in The Wild (CEW)** | In-the-Wild Eye Crops | 4,846 images (2,423 subjects) | Binary (open/closed) | Academic Open Use | Song et al. (2014) | High diversity of facial expressions and unconstrained lighting. | Static images; resolution varies ($24\times24$ to $100\times100$). |
| **NTHU-DDD** | Infrared & RGB Driver Video | 36 subjects (5 scenarios: night, glasses, bare face, yawning, talking) | Frame-level drowsiness, blinking, yawning | Restricted Academic Agreement | Chen et al. (CVPRW 2016) | **Gold Standard Automotive Benchmark**. Infrared night vision and real driver actions. | Restricted access (requires EULA license); large download size (~15GB). |
| **YawDD** | In-Car Video Sequences | 30+ subjects (Male/Female in real vehicle) | Frame-level yawning, talking, normal driving | Open Academic | Ablavatski et al. (2014) | Real driving conditions with natural sunlight, shadows, and conversation. | Camera mounted at varying angles (dash vs. mirror); 30 FPS AVI format. |
| **UTA-RLDD** | Real-Life Multi-Stage Drowsiness | 60 subjects (180 sequences, 30 hours) | Multi-stage alertness (alert, low, drowsy) | Open Academic (Request Access) | Ghoddoosian et al. (2019) | Multi-stage fatigue scale (PERCLOS validation). | Heavy video dataset; requires Google Drive authorization. |
| **DROZY** | NIR Video & EEG Signals | 14 subjects (KSS fatigue scale) | Frame-level + EEG/ECG ground truth | Academic Open Use | Massoz et al. (2016) | Multimodal validation pairing vision metrics with physiological EEG ground truth. | Small subject sample size (14 subjects). |
| **Driver Monitoring Dataset (DMD)** | Multi-Camera Interior Video | 37 drivers (Distraction, Drowsiness) | Pixel-level & bounding box | Commercial / Academic | Ortega et al. (2020) | Multi-view (infrared, depth, RGB). | Restricted commercial license. |
| **OpenEDS** | Eye Tracking Images | 12,641 high-res eye images | Semantic segmentation of iris/pupil | Meta Research Agreement | Garbin et al. (2019) | Precise pupil & iris boundaries for gaze tracking. | Head-mounted display domain (VR/AR), not automotive driver view. |

---

## 2. Evidence-Based Dataset Selection Rationale

Based on scientific requirements for lightweight edge detection, we assign candidate datasets to specific research roles:

1. **Training & Validation Set (`data/train`, `data/validation`)**:
   - **MRL Eye Dataset + CEW**: Selected as the primary training and validation corpus for training **MicroEyeNet** ($24\times24$ TFLite CNN). Provides 89,000+ balanced eye crops spanning night vision, glasses, reflections, and unconstrained lighting.
2. **Primary Benchmark Set (`benchmark/`)**:
   - **NTHU-DDD**: Selected as the primary gold-standard benchmark dataset for evaluating system accuracy, precision, recall, F1-score, and false-positive rates under night driving, glasses, and talking conditions.
3. **Secondary & Cross-Dataset Evaluation Set (`benchmark/yawdd`)**:
   - **YawDD**: Selected for evaluating 2D MAR depth metric fix and visual speech jitter filtering under unconstrained sunlight and jaw motion.
