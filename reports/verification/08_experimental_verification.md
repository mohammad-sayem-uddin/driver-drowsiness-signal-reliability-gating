# STAGE 8: EXPERIMENTAL CLAIM VERIFICATION AUDIT

**Auditing Body**: Scientific Verification Committee (Experimental Design Expert & Senior Reviewer)  
**Scope**: Verification of all experimental results, accuracy figures, latency metrics, and benchmark dataset runs  
**Date**: July 2026

---

## 1. Executive Experimental Findings

```
===================================================================================
                    EXPERIMENTAL CLAIM FACT-CHECK VERDICT
===================================================================================

1. Public Benchmark Dataset Runs (NTHU-DDD / YawDD) --> [ ZERO EXECUTED ]
   - No automated test scripts exist; data/eyes/ directory is empty.

2. Empirical Accuracy / F1-Score Figures             --> [ UNSUPPORTED PAPERWARE ]
   - Claimed accuracy figures (>95%) are theoretical estimates, not empirical runs.

3. System Latency & Processing FPS Claims           --> [ PARTIALLY VERIFIED ]
   - Desktop CPU runtime (<12ms) verified; Raspberry Pi 4 (<28ms) is theoretical.

4. MicroEyeNet Ablation Study                        --> [ MISSING ASSET ]
   - MicroEyeNet model file is missing; CNN ablation cannot be executed.

OVERALL EXPERIMENTAL RATING: ZERO EMPIRICAL BENCHMARK DATA.
No empirical claims can be defended under peer review until NTHU-DDD is evaluated.
===================================================================================
```

---

## 2. Detailed Audit of Experimental Statements

### Claim 1: "The system achieves >95% accuracy on driver drowsiness detection."
- **Fact-Check Status**: ❌ **UNSUPPORTED / PAPERWARE**.
- **Audit Findings**: The repository contains no dataset evaluation scripts, no confusion matrices, and no saved ground-truth test outputs. The claimed >95% accuracy figure is an unbacked theoretical assumption derived from literature estimates.

### Claim 2: "The system processes video at <12ms per frame on Desktop CPU and <28ms on Raspberry Pi 4."
- **Fact-Check Status**: 🟡 **PARTIALLY VERIFIED**.
- **Audit Findings**: Desktop CPU performance (<12ms for MediaPipe + pure math heuristics) was verified using `test_pipeline.py`. However, Raspberry Pi 4 performance (<28ms) has not been profiled on physical ARM hardware with active camera streaming.

### Claim 3: "Selective CNN invocation reduces false positive rates by 80% while saving 90% compute."
- **Fact-Check Status**: ❌ **UNSUPPORTED (MISSING MODEL ASSET)**.
- **Audit Findings**: Because `models/eye_state_model.tflite` is absent, `CNNValidator` operates in fallback mode. The 80% FP reduction claim is theoretical paperware until the model is trained and evaluated against NTHU-DDD clips.

---

## 3. Mandatory Experimental Remediation

Before any manuscript is submitted to IEEE Transactions on Intelligent Transportation Systems or IEEE IV:
1. **Train MicroEyeNet TFLite weights** (`tools/train_eye_cnn.py`).
2. **Execute automated benchmark evaluation** on **NTHU-DDD** (36 subjects, 5 scenarios: glasses, night, yawning, talking, bare face) and **YawDD**.
3. **Generate verified empirical tables**:
   - Accuracy %, Sensitivity (Recall), Specificity, Precision, F1-Score, ROC-AUC, and FPR/hr.
   - 4-Variant Ablation Table (Baseline vs. Speech Filter vs. RobustnessGuard vs. Full Hybrid).
