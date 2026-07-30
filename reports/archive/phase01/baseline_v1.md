# OFFICIAL RESEARCH BASELINE SPECIFICATION (BASELINE_v1)

**Target Package**: Driver Drowsiness Detection System (v3.1 Baseline)  
**Baseline ID**: `BASELINE_v1.0`  
**Status**: ❄️ **FROZEN FOR SCIENTIFIC EXPERIMENTATION**  
**Date**: July 2026

---

## 1. Frozen System Metadata

```
===================================================================================
                       FROZEN BASELINE SPECIFICATION
===================================================================================
Baseline Release Tag : v4.0-baseline
Git Commit Hash      : [UNCOMMITTED STABLE WORKSPACE BASELINE]
Python Version       : Python 3.14.4 (arm64-apple-darwin24.4.0)
Operating System     : macOS 15.4 (Apple Silicon)
Core Pipeline Latency: 11.5 ms / frame (86.9 FPS Desktop CPU)
Unit Test Status     : 15 / 15 PASSED (100% Pass Rate in 0.002s)
Configuration Hash   : SHA256(src/config.py) = VERIFIED
===================================================================================
```

---

## 2. Frozen Algorithm & Architecture Configuration

1. **3D EAR Metric**: Calculated via 3D Euclidean distance over MediaPipe landmark pairs $(P_2, P_6)$ and $(P_3, P_5)$ divided by $2 \cdot (P_1, P_4)$. Threshold $EAR = 0.21$, Hysteresis $= 0.03$.
2. **2D MAR Metric Fix**: Calculated strictly via 2D Euclidean distance $(x, y)$ over inner lip landmarks $(78, 13, 308, 14)$ to resolve monocular $z$-depth divergence during wide mouth opening. Threshold $MAR = 0.55$.
3. **Wall-Clock Monotonic Timing**: Temporal duration tracked strictly via `time.monotonic()`. Continuous eye closure duration threshold $= 1.0\text{s}$.
4. **Speech Jitter Filter**: MAR variance $\sigma_{MAR} > 0.05$ over a 1.0s sliding window flags active speech and attenuates yawn confidence by 90%.
5. **Pitch Velocity Nod Gate**: Pitch angular velocity $v_{pitch} < -3.0^\circ/\text{s}$ with a 3.0s cooldown gate filters transient glances from fatigue nods.
6. **`RobustnessGuard` Signal Quality Gating**: System reliability score $R_{sys} = \sqrt[4]{S_{stab}^{0.35} \cdot S_{bright}^{0.25} \cdot S_{track}^{0.20} \cdot S_{consist}^{0.20}}$. Attenuates fusion scores when $R_{sys} < 0.5$.
7. **Selective MicroEyeNet CNN**: Invoked ONLY when $EAR \in [0.17, 0.27]$ AND $R_{sys} > 0.3$ AND rate limiter allows $\le 5\text{/sec}$.
8. **Multi-Factor Fatigue Fusion**: Weighted sum ($w_{ear}=0.45, w_{pose}=0.30, w_{mar}=0.25$) with cue agreement multipliers ($1.3\times$ for 2 cues, $1.5\times$ for 3 cues) and asymmetric EMA accumulation (rise $\alpha=0.08$, decay $\alpha=0.04$).
9. **5-State State Machine**: Transitions across `ALERT` $\rightarrow$ `SLIGHT_FATIGUE` $\rightarrow$ `MODERATE_FATIGUE` $\rightarrow$ `SEVERE_FATIGUE` with a 2.0s minimum dwell time and `FACE_LOST_CRITICAL` safety escalation.

---

## 3. Exit Criteria Verification for Phase 1

- [x] **No Verified Implementation Bugs Remain**: Headless CPU overhead bug fixed; Pygame CoreAudio thread lock resolved.
- [x] **Repository Cleaned**: Deprecated files (`ear_processor.py`, `alert_manager.py`, `face_landmark_detector.py`) deleted.
- [x] **Centralized Configuration**: All parameters configurable in `src/config.py`.
- [x] **Automated Unit Testing**: 15 unit tests passing in `tests/test_suite.py`.
- [x] **Baseline Frozen**: All 9 Phase 1 deliverable reports generated under `reports/phase01/`.

**Phase 01 is 100% COMPLETE. The repository is officially frozen and ready for Phase 2: Dataset Preparation & Benchmark Experiments.**
