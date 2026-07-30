# STAGE 10: MASTER FACT-CHECK & SCIENTIFIC VERIFICATION AUDIT

**Auditing Body**: Independent Scientific Verification Committee (Associate Editors & Research Integrity Officers)  
**Scope**: Master factual audit of all previous project reports  
**Date**: July 2026

---

## 1. Master Statement Fact-Check Classification Table

| Statement / Claim from Previous Reports | Fact-Check Verdict | Scientific Evidence & Justification |
|:---|:---|:---|
| *"MediaPipe FaceMesh tracks 468/478 3D landmarks in real time."* | ✅ **TRUE** | Verified in `src/main.py` and MediaPipe official API specs. |
| *"EAR uses 3D Euclidean distance for curvature compensation."* | ✅ **TRUE** | Verified in `src/detector.py` lines 56–61 (`calculate_distance` includes $z$-depth). |
| *"MAR uses 2D Euclidean distance to fix monocular z-depth divergence."* | ✅ **TRUE** | Verified in `src/detector.py` lines 102–124 (`_distance_2d` explicitly ignores $z$-depth). |
| *"Wall-clock timing uses time.monotonic() to achieve FPS independence."* | ✅ **TRUE** | Verified across `src/temporal_analyzer.py` (lines 18, 168, 274, 528). |
| *"RobustnessGuard computes system reliability via 4-subscore geometric mean."* | ✅ **TRUE** | Verified in `src/robustness.py` lines 207–213. |
| *"Speech MAR jitter filter penalizes confidence when σ_MAR > 0.05."* | ✅ **TRUE** | Verified in `src/temporal_analyzer.py` lines 291–294 & 339. |
| *"Pitch velocity nod gate requires v < -3°/s and a 3.0s cooldown."* | ✅ **TRUE** | Verified in `src/temporal_analyzer.py` lines 391 & 456. |
| *"Soukupová & Čech (2016) proposed the foundational EAR metric."* | ✅ **TRUE** | Verified exact paper match (CVWW 2016). |
| *"Selective CNN invocation is a novel concept unique to our system."* | ⚠️ **MISLEADING** | Selective CNN triggering was previously proposed by Chen et al. (MDPI Sensors 2025). |
| *"B. Reddy et al. published a heavy 3D ResNet-18 model in 2021."* | ⚠️ **MISLEADING** | B. Reddy et al. published deep model compression at IEEE CVPR Workshops in 2017 (14.9 FPS, 89.5% acc). |
| *"Horng et al. published PERCLOS and EAR in 2018."* | ❌ **FALSE** | Horng & Chen published template matching in 2004. EAR was introduced by Soukupová in 2016. |
| *"The system achieves >95% accuracy on driver drowsiness detection."* | ❌ **UNSUPPORTED** | Zero benchmark dataset evaluation scripts exist; `data/eyes/` is unpopulated. |
| *"MicroEyeNet CNN validation reduces false positives by 80%."* | ❌ **UNSUPPORTED** | `models/eye_state_model.tflite` is MISSING; CNN runs in fallback mode. |
| *"The system is ready for immediate conference submission."* | ❌ **FALSE** | Submitting today would result in instant desk rejection for zero benchmark numbers. |

---

## 2. Comprehensive Correction & Remediation Log

### 1. Citation Corrections
- **Fix Metadata for B. Reddy et al.**: Correct year to 2017 and venue to IEEE CVPR Workshops (*Real-Time Driver Drowsiness Detection for Embedded System Using Model Compression of Deep Neural Networks*, pp. 121–128).
- **Fix Metadata for Horng et al.**: Correct year to 2004 (*Driver Fatigue Detection Based on Eye Tracking and Dynamic Template Matching*, IEEE ICNSC).
- **Add Missing Papers**: Include Zhang et al. (IEEE T-ITS 2023), Happy & Routray (IEEE T-IV 2022), Khan et al. (Sensors 2025), and Patel & Sharma (IEEE ITSC 2024).

### 2. Novelty & Claim Reframing
- **Primary Novelty**: Position `RobustnessGuard` signal quality gating, 2D/3D Euclidean metric separation, and visual speech/nod ambiguity filters as the lead scientific contributions.
- **Secondary Feature**: Position selective CNN uncertainty resolution as an integrated subsystem rather than a standalone novel operator.

### 3. Codebase Bug Fix
- **Fix Headless Mode CPU Overhead**: In `src/main.py` lines 180–183, move `cv2.cvtColor` inside the condition checking whether frame rendering is required.

### 4. Mandatory Pre-Submission Roadmap
1. Train MicroEyeNet via `tools/train_eye_cnn.py` -> Export `models/eye_state_model.tflite`.
2. Build `tools/evaluate_benchmark.py` -> Run automated evaluation on NTHU-DDD & YawDD datasets.
3. Conduct 4-variant ablation study (Baseline vs. Speech Filter vs. RobustnessGuard vs. Full Hybrid).
4. Profile execution latency on Raspberry Pi 4 hardware.
5. Draft 6-page IEEE double-column manuscript (`IEEEtran.cls`) targeting **IEEE Intelligent Vehicles Symposium (IV 2027)** or **IEEE ITSC 2027**.
