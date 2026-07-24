# ══════════════════════════════════════════════════════════════════════
# TECHNICAL AUDIT REPORT — DRIVER DROWSINESS DETECTION SYSTEM
# ══════════════════════════════════════════════════════════════════════
# Date: 2026-06-17
# Auditor: Automated Deep-Code Audit
# Scope: Full codebase, architecture, research readiness, publication viability
# ══════════════════════════════════════════════════════════════════════


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 1 — PROJECT OVERVIEW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Project Name:
  Driver Drowsiness Detection System (v3.1 — "Robust")

Research Objective:
  Develop a lightweight, edge-deployable, explainable hybrid driver
  drowsiness detection system that combines temporal heuristic
  analysis (EAR/MAR/Head Pose) with selective CNN validation to
  suppress false positives in ambiguous boundary states, while
  maintaining real-time performance (>25 FPS) on constrained
  hardware (Raspberry Pi 4) without GPU acceleration.

Current Architecture:
  Hybrid Intelligence Pipeline — three tiers:
    Tier 1 (Continuous): Heuristic temporal analysis using EAR, MAR,
      Head Pose via MediaPipe Face Mesh + OpenCV solvePnP.
    Tier 2 (Selective): A ~9.5K-parameter Tiny CNN (MicroEyeNet)
      invoked ONLY when the heuristic signal enters an "uncertainty
      zone" (EAR ∈ [0.17, 0.27]).
    Tier 3 (Robustness): A signal quality monitor (RobustnessGuard)
      that multiplicatively attenuates the fusion score under degraded
      conditions (low light, camera shake, landmark jitter).

Current Research Direction:
  The system targets the "systems engineering" niche of ITS research,
  NOT state-of-the-art deep learning accuracy. The core contribution
  is the asymmetric hybrid architecture: heuristics as the backbone,
  CNN as the uncertainty resolver. This inverts the standard DL paradigm.

Main Hypothesis:
  "A selectively-invoked lightweight CNN, activated only during
  ambiguous heuristic states, can significantly reduce false positive
  rates in EAR-based drowsiness detection without exceeding the
  computational budget of edge devices."

Full Pipeline (Webcam Input → Final Fatigue Decision):
  ┌─────────────────────────────────────────────────────────────┐
  │ 1. CAMERA INPUT (CameraAsync — threaded LIFO queue)         │
  │    Frame captured asynchronously; main thread gets latest    │
  ├─────────────────────────────────────────────────────────────┤
  │ 2. MEDIAPIPE FACE MESH (tracking mode)                      │
  │    468 3D facial landmarks extracted; <12ms on desktop       │
  ├─────────────────────────────────────────────────────────────┤
  │ 3. FEATURE EXTRACTION (detector.py — pure math)             │
  │    • EAR = (||P2-P6|| + ||P3-P5||) / (2 × ||P1-P4||)       │
  │    • MAR = ||top-bottom|| / ||left-right||  (2D only)        │
  │    • Head Pose via solvePnP → Pitch, Yaw, Roll              │
  ├─────────────────────────────────────────────────────────────┤
  │ 4. TEMPORAL ANALYSIS (TemporalAnalyzer — wall-clock)         │
  │    • EMA smoothing (α=0.3) on EAR/MAR/Pose                  │
  │    • Hysteresis thresholding (close=0.21, open=0.24)         │
  │    • Yawn confidence (magnitude + duration + jitter smooth.) │
  │    • Posture confidence (pitch + duration + instability)     │
  │    • Closure ratio (time-based, not frame-count)             │
  ├─────────────────────────────────────────────────────────────┤
  │ 5. SIGNAL QUALITY (RobustnessGuard)                         │
  │    • Landmark stability (6-pt jitter)                        │
  │    • Brightness quality (face ROI mean intensity)            │
  │    • Tracking quality (MediaPipe confidence)                 │
  │    • Cue consistency (temporal variance of confidences)      │
  │    → system_reliability = geometric mean of sub-scores       │
  │    → Multiplicative attenuation of fusion score              │
  ├─────────────────────────────────────────────────────────────┤
  │ 6. CNN VALIDATION (CNNValidator — selective invocation)      │
  │    • GATE 1: EAR ∈ [0.17, 0.27] (uncertainty zone)          │
  │    • GATE 2: Rate limited (max 5 invocations/sec)            │
  │    • If invoked: extract 24×24 grayscale eye ROI             │
  │    → MicroEyeNet inference → CNNVerdict (open/closed/NA)     │
  │    • If CNN disagrees with heuristic → suppress false alarm  │
  ├─────────────────────────────────────────────────────────────┤
  │ 7. MULTI-FACTOR FUSION (FatigueFusionEngine in state_mgr)   │
  │    • Weighted sum: 0.45×EAR + 0.30×Pose + 0.25×MAR          │
  │    • Cue-agreement amplification (1-cue: 1.0×, 2-cue: 1.3×, │
  │      3-cue: 1.5×)                                           │
  │    • Asymmetric temporal accumulation (rise=0.08, decay=0.04)│
  │    • 4-level severity: ALERT / SLIGHT / MODERATE / SEVERE    │
  │    • Reliability-gated attenuation                           │
  ├─────────────────────────────────────────────────────────────┤
  │ 8. STATE MANAGEMENT (StateManager)                          │
  │    • 5-state DriverStatus enum                               │
  │    • Face-loss safety escalation (not silencing)             │
  │    • Minimum dwell time (2.0s between transitions)           │
  │    • Nod cooldown (3.0s) + velocity gating (3.0°/s)          │
  ├─────────────────────────────────────────────────────────────┤
  │ 9. ALARM CONTROL (AlarmController)                          │
  │    • Minimum alarm duration (3.0s)                           │
  │    • Cooldown period (5.0s) between alarms                   │
  │    • Level-based escalation (never de-escalates mid-episode) │
  │    • CSV event logging                                       │
  ├─────────────────────────────────────────────────────────────┤
  │ 10. OUTPUT                                                  │
  │     • Audio alert (pygame — synthesized 880Hz sine wave)     │
  │     • HUD overlay (EAR, MAR, Pose, Fusion score, Severity)  │
  │     • CSV event log (timestamped drowsiness events)          │
  └─────────────────────────────────────────────────────────────┘


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 2 — CURRENT FOLDER STRUCTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

```
Driver Drowsiness/
│
├── README.md                          ← [ACTIVE] Project documentation
├── requirements.txt                   ← [ACTIVE] 4 dependencies (opencv, mediapipe, numpy, pygame)
├── research_notes.md                  ← [ACTIVE] 2819-line research bible (stages 1–9, experiments, paper notes)
├── drowsiness_events_log.csv          ← [ACTIVE] Runtime event log (auto-generated)
│
├── data/
│   └── eyes/
│       ├── closed/                    ← [ACTIVE] Eye images for CNN training (class: closed)
│       └── open/                      ← [ACTIVE] Eye images for CNN training (class: open)
│
├── models/                            ← [ACTIVE] Directory for TFLite model files
│                                      ← (EMPTY — no .tflite model present yet)
│
├── src/
│   ├── __init__.py                    ← [ACTIVE] Package init
│   ├── main.py                        ← [ACTIVE] Main application entry point (608 lines)
│   ├── config.py                      ← [ACTIVE] Centralized configuration (6 dataclasses)
│   ├── detector.py                    ← [ACTIVE] Pure math: EAR/MAR computation (156 lines)
│   ├── temporal_analyzer.py           ← [ACTIVE] FPS-independent temporal detection + posture analysis
│   ├── state_manager.py               ← [ACTIVE] Face-loss safety + fusion integration + 5-state machine
│   ├── alarm_controller.py            ← [ACTIVE] Persistent alarms with anti-flicker + CSV logging
│   ├── fatigue_fusion.py              ← [ACTIVE] Multi-factor fusion engine (weighted sum + agreement)
│   ├── pose_estimator.py              ← [ACTIVE] 3D head pose via solvePnP (Pitch/Yaw/Roll)
│   ├── robustness.py                  ← [ACTIVE] Signal quality monitoring + reliability gating
│   ├── cnn_validator.py               ← [ACTIVE] Selective CNN validation (graceful fallback when no model)
│   ├── ear_processor.py               ← ⚠ [DEPRECATED] Standalone EAR processor (superseded by temporal_analyzer)
│   ├── alert_manager.py               ← ⚠ [DEPRECATED] Old alarm manager (superseded by alarm_controller)
│   ├── face_landmark_detector.py      ← ⚠ [DEPRECATED] Old face detection wrapper (superseded by MediaPipe direct)
│   ├── camera_async.py                ← [ACTIVE] Async camera with LIFO queue (Producer-Consumer)
│   ├── camera_base.py                 ← [ACTIVE] Base camera class
│   └── utils/
│       ├── __init__.py                ← [ACTIVE] Package init
│       ├── audio_alert.py             ← [ACTIVE] Pygame audio (synthesized sine wave, channel management)
│       └── landmark_indices.py        ← [ACTIVE] MediaPipe landmark index constants
│
├── tools/
│   ├── collect_eye_data.py            ← [ACTIVE] Data collection tool for CNN training images
│   └── train_eye_cnn.py               ← [ACTIVE] CNN training script (MicroEyeNet)
│
├── test_pipeline.py                   ← [TEST] Pipeline integration test
├── test_pose.py                       ← [TEST] Head pose estimation test
├── test_variance.py                   ← [TEST] Yawn variance/jitter test
└── test_webcam.py                     ← [TEST] Basic webcam connectivity test
```

File Count Summary:
  Active Python modules:     15
  Deprecated modules:         3
  Test files:                 4
  Tool scripts:               2
  Config/Data files:          4
  Research documentation:     1 (2819 lines)
  Missing:                    TFLite model file (models/eye_state_model.tflite)


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 3 — MODULE INVENTORY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┌─────────────────────────────────────────────────────────────────────┐
│ main.py — Application Entry Point (608 lines)                       │
├─────────────────────────────────────────────────────────────────────┤
│ Purpose:    Orchestrates the full pipeline — initializes all        │
│             subsystems, runs the main loop, renders the HUD.        │
│ Inputs:     Webcam frames via CameraAsync                           │
│ Outputs:    HUD display, audio alarms, CSV logs                     │
│ Dependencies: config, detector, temporal_analyzer, state_manager,   │
│               alarm_controller, pose_estimator, robustness,         │
│               camera_async, cnn_validator, mediapipe, cv2, numpy    │
│ Status:     ✅ Production Ready                                     │
│ Notes:      Clean separation of concerns. Imports all new modules.  │
│             Handles graceful CNN fallback. Has profiler support.     │
│             Adaptive frame skipping implemented.                     │
│             WART: Lines 180-183 do cvtColor even in headless mode.  │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ config.py — Centralized Configuration (~220 lines)                  │
├─────────────────────────────────────────────────────────────────────┤
│ Purpose:    Single source of truth for ALL thresholds and params.   │
│             6 typed dataclasses: DetectionConfig, PostureConfig,    │
│             YawnConfig, FusionConfig, RobustnessConfig, etc.        │
│ Inputs:     None (static configuration)                             │
│ Outputs:    SystemConfig instance consumed by all modules            │
│ Dependencies: dataclasses, typing                                   │
│ Status:     ✅ Production Ready                                     │
│ Notes:      __repr__() prints full config at session start for      │
│             experiment reproducibility. Eliminates the pre-v2.0     │
│             problem of scattered, inconsistent thresholds.          │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ detector.py — Pure Math Processor (156 lines)                       │
├─────────────────────────────────────────────────────────────────────┤
│ Purpose:    Stateless EAR and MAR computation from landmark coords. │
│             No state, no counters, no thresholds.                   │
│ Inputs:     MediaPipe landmark objects (6 for EAR, 4 for MAR)       │
│ Outputs:    float (EAR value 0.0–0.4, MAR value 0.0–1.0)            │
│ Dependencies: math (stdlib only)                                    │
│ Status:     ✅ Production Ready                                     │
│ Notes:      EAR uses 3D Euclidean (z-depth stable for eyes).        │
│             MAR uses 2D-only (z-depth diverges for lips — fixed    │
│             after live-testing root cause analysis).                 │
│             WART: `import math` inside _distance_2d is redundant    │
│             (already imported at module level).                      │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ temporal_analyzer.py — FPS-Independent Temporal Detection (~300 ln) │
├─────────────────────────────────────────────────────────────────────┤
│ Purpose:    Wall-clock temporal analysis using time.monotonic().    │
│             EMA smoothing, hysteresis thresholding, yawn detection  │
│             with jitter filtering, posture analysis with velocity   │
│             gating, closure ratio tracking.                         │
│ Inputs:     raw_ear, raw_mar, raw_pitch, raw_yaw, raw_roll          │
│ Outputs:    TemporalState dataclass (smoothed values, confidences,  │
│             closure_ratio, is_yawning, is_nodding, etc.)            │
│ Dependencies: config, time, collections.deque, dataclasses          │
│ Status:     ✅ Production Ready                                     │
│ Notes:      Replaces the old frame-count-based detection.           │
│             PostureAnalyzer class included for head pose analysis.   │
│             YawnAnalyzer uses sliding window jitter (deque of 15).  │
│             Confidence = 0.2×M + 0.5×D + 0.3×S for yawning.        │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ state_manager.py — State Machine + Fusion Integration (~230 lines)  │
├─────────────────────────────────────────────────────────────────────┤
│ Purpose:    5-state DriverStatus enum (ALERT, SLIGHT_FATIGUE,       │
│             MODERATE_FATIGUE, SEVERE_FATIGUE, FACE_LOST/CRITICAL).  │
│             Integrates FatigueFusionEngine. Face-loss safety logic. │
│             Minimum dwell time (2.0s) between transitions.           │
│ Inputs:     TemporalState, face_detected, reliability, CNN verdict  │
│ Outputs:    SystemState dataclass (status, fatigue_score, severity, │
│             cue contributions, CNN status, etc.)                    │
│ Dependencies: config, fatigue_fusion, time, enum                    │
│ Status:     ✅ Production Ready                                     │
│ Notes:      Replaced old boolean priority chain with fusion engine. │
│             Face-loss escalation (not silencing) is safety-critical.│
│             DROWSY state retained for backward compatibility.        │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ alarm_controller.py — Alarm Lifecycle Management (~160 lines)       │
├─────────────────────────────────────────────────────────────────────┤
│ Purpose:    Persistent alarms with minimum duration (3.0s),         │
│             cooldown (5.0s), level-based escalation, CSV logging.   │
│ Inputs:     SystemState from StateManager                           │
│ Outputs:    Audio alerts, CSV event logs                            │
│ Dependencies: config, audio_alert, time, csv, os                    │
│ Status:     ✅ Production Ready                                     │
│ Notes:      Replaces old alert_manager.py. Alarms can only          │
│             escalate, never de-escalate during active episode.      │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ fatigue_fusion.py — Multi-Factor Fusion Engine (~350 lines)         │
├─────────────────────────────────────────────────────────────────────┤
│ Purpose:    Weighted multi-cue behavioral fusion with temporal      │
│             accumulation and graduated severity estimation.          │
│ Inputs:     Per-cue confidences (EAR, MAR, Pose) from TemporalState │
│ Outputs:    FusionSnapshot (fatigue_score, severity, contributions, │
│             agreement multiplier, temporal trend)                   │
│ Dependencies: config, dataclasses, enum, collections.deque          │
│ Status:     ✅ Production Ready                                     │
│ Notes:      Weights: EAR=0.45, Pose=0.30, MAR=0.25.                 │
│             Asymmetric EMA: rise=0.08, decay=0.04.                  │
│             Cue-agreement bonus: 2-cue=1.3×, 3-cue=1.5×.           │
│             Severity hysteresis band = 0.12.                        │
│             Trend detection via split buffer analysis.              │
│             Fully explainable (FusionSnapshot has per-cue detail).  │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ pose_estimator.py — 3D Head Pose via solvePnP (~200 lines)          │
├─────────────────────────────────────────────────────────────────────┤
│ Purpose:    Extracts Pitch, Yaw, Roll from 6 facial landmarks      │
│             using OpenCV solvePnP against a generic 3D face model.  │
│ Inputs:     6 landmark pixel coordinates (nose, chin, eye corners,  │
│             mouth corners)                                          │
│ Outputs:    (pitch, yaw, roll) in degrees; projection axes for HUD  │
│ Dependencies: cv2, numpy                                            │
│ Status:     ✅ Production Ready                                     │
│ Notes:      Generic 3D face model (no calibration needed).          │
│             Uses SOLVEPNP_ITERATIVE. Camera intrinsics assumed      │
│             (focal_length = image_width). get_projection_axes() for │
│             HUD 3D axis visualization.                              │
│             ⚠ Known issue: baseline pitch offset depends on camera  │
│             mounting angle (no adaptive calibration yet).            │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ robustness.py — Signal Quality Monitor + Reliability Gating         │
├─────────────────────────────────────────────────────────────────────┤
│ Purpose:    Computes system_reliability (0.0–1.0) from 4 sub-scores │
│             (landmark stability, brightness, tracking, consistency).│
│             Multiplicatively gates fusion score. Suppresses         │
│             SLIGHT/MODERATE alerts when reliability < threshold.     │
│ Inputs:     SignalQuality (jitter, brightness, tracking_conf),      │
│             per-cue confidences                                     │
│ Outputs:    RobustnessSnapshot (sub-scores, system_reliability,     │
│             alert_suppressed flag)                                  │
│ Dependencies: config, numpy, dataclasses                            │
│ Status:     ✅ Production Ready                                     │
│ Notes:      Geometric mean ensures single-channel degradation       │
│             dominates. EMA smoothing (α=0.2) on reliability.        │
│             SEVERE alarms are NEVER suppressed (safety guarantee).  │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ cnn_validator.py — Selective CNN Validation Layer                   │
├─────────────────────────────────────────────────────────────────────┤
│ Purpose:    Loads MicroEyeNet TFLite model. Provides selective      │
│             eye-state validation when heuristic is uncertain.       │
│ Inputs:     Eye ROI image (24×24 grayscale), smoothed EAR           │
│ Outputs:    CNNVerdict (is_closed, confidence, invoked, agrees)     │
│ Dependencies: config, numpy, (optionally tflite_runtime)            │
│ Status:     ⚠ Needs Improvement (model file missing)               │
│ Notes:      should_invoke() gates on EAR ∈ [0.17, 0.27] and        │
│             rate-limiting (max 5/sec). extract_eye_roi() crops      │
│             24×24 grayscale eye region from landmarks.              │
│             Graceful fallback to heuristic-only when model absent.  │
│             Tracks invocation/agreement/override statistics.         │
│             CRITICAL: No .tflite model exists yet. The CNN is       │
│             architecturally integrated but functionally inactive.   │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ ear_processor.py — Standalone EAR Processor (DEPRECATED)            │
├─────────────────────────────────────────────────────────────────────┤
│ Purpose:    Original standalone EAR visualization/processing tool.  │
│ Status:     ❌ Deprecated                                           │
│ Notes:      Superseded by temporal_analyzer.py. Now imports from    │
│             config.py for threshold consistency. Retained for        │
│             backward compatibility and standalone demo use.          │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ alert_manager.py — Old Alarm Manager (DEPRECATED)                   │
├─────────────────────────────────────────────────────────────────────┤
│ Purpose:    Original alarm management with cooldown.                │
│ Status:     ❌ Deprecated                                           │
│ Notes:      Superseded by alarm_controller.py. Has deprecation      │
│             notice. Imports config.py for backward compatibility.    │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ face_landmark_detector.py — Old Face Detection (DEPRECATED)         │
├─────────────────────────────────────────────────────────────────────┤
│ Purpose:    Wrapper around MediaPipe Face Mesh initialization.      │
│ Status:     ❌ Deprecated                                           │
│ Notes:      MediaPipe is now initialized directly in main.py.       │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ camera_async.py — Asynchronous Camera Capture                       │
├─────────────────────────────────────────────────────────────────────┤
│ Purpose:    Producer-Consumer pattern camera with LIFO queue.       │
│             Background thread reads frames; main thread gets latest.│
│ Status:     ✅ Production Ready                                     │
│ Notes:      Decouples I/O latency from inference latency.           │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ camera_base.py — Base Camera Class                                  │
├─────────────────────────────────────────────────────────────────────┤
│ Purpose:    Abstract base for camera implementations.               │
│ Status:     ✅ Production Ready                                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ utils/audio_alert.py — Audio Alert System                           │
├─────────────────────────────────────────────────────────────────────┤
│ Purpose:    Pygame-based audio with synthesized 880Hz sine wave     │
│             fallback. Non-blocking channel management.               │
│ Status:     ✅ Production Ready                                     │
│ Notes:      Synthesized fallback eliminates external file dependency│
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ utils/landmark_indices.py — MediaPipe Landmark Constants            │
├─────────────────────────────────────────────────────────────────────┤
│ Purpose:    Named constants for MediaPipe landmark indices.         │
│ Status:     ✅ Production Ready                                     │
└─────────────────────────────────────────────────────────────────────┘


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 4 — FATIGUE PIPELINE ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

4.1 EAR PIPELINE
─────────────────
Algorithm:
  EAR = (||P2-P6|| + ||P3-P5||) / (2.0 × ||P1-P4||)
  where P1..P6 are 6 eye contour landmarks (Soukupová & Čech 2016).
  Bilateral averaging: raw_ear = (left_ear + right_ear) / 2.0

Smoothing:
  EMA with α=0.3 (TemporalAnalyzer). ~7 frames to 90% convergence.
  Adds ~233ms tracking latency at 30 FPS.

Thresholds (from config.py):
  Close threshold: 0.21 (enter CLOSED state)
  Open threshold:  0.24 (exit CLOSED state — hysteresis)
  Dead zone:       0.03 (prevents oscillation at boundary)
  Drowsiness trigger: 1.0 second of sustained closure

Closure Ratio:
  closure_ratio = time_closed / drowsiness_trigger_duration
  Ramps 0.0 → 1.0 over 1.0 second.
  At 1.0 → drowsiness event fires.

Current Weaknesses:
  • Fixed threshold (0.21) for all users. No per-subject calibration.
  • Bright sunlight (squinting) drops EAR to 0.18–0.22 for extended
    periods → false positives despite hysteresis.
  • 3D distance for EAR (including z) provides ~2% improvement for
    frontal faces but may degrade at extreme yaw angles.
  • No PERCLOS computation (industry standard metric).

4.2 MAR PIPELINE
────────────────
Algorithm:
  MAR = ||top_lip - bottom_lip||₂ᴅ / ||left_corner - right_corner||₂ᴅ
  Uses 2D-only distances (z-depth excluded — causes MAR > 2.0 inflation).
  Uses INNER lip contour (not outer — reduces smile false positives).

Smoothing:
  EMA with α=0.3 in TemporalAnalyzer.

Thresholds:
  Yawn open threshold: 0.55 (MAR must exceed this)
  Yawn duration: 0.8 seconds sustained above threshold
  Speech jitter threshold: 0.08 (frame-to-frame MAR difference)

Confidence Scoring:
  yawn_confidence = 0.2×M + 0.5×D + 0.3×S
  where M = magnitude (MAR excess), D = duration progress, S = smoothness
  If is_speaking (jitter > threshold): 0.1× penalty applied

Current Weaknesses:
  • No absolute upper bound enforcement on MAR (though 2D-only
    computation should keep it bounded to ~1.0 in practice).
  • Very slow sustained speech may occasionally leak through the
    jitter filter.
  • No yawn frequency tracking over time (just individual events).

4.3 HEAD POSE PIPELINE
──────────────────────
Algorithm:
  cv2.solvePnP() with 6 facial landmarks (nose tip, chin, eye
  corners, mouth corners) against a generic 3D face model.
  Rotation vector → Rodrigues → Rotation Matrix → Euler angles.
  Pitch: negative = downward, positive = upward.

Smoothing:
  EMA with α=0.10 (heavy smoothing to suppress webcam jitter).
  30-frame sliding window for yaw/roll variance (instability).

Thresholds (post-stabilization):
  Downward pitch threshold: -20.0° (was -15.0°, too sensitive)
  Nod minimum duration: 1.5s (was 0.5s, too short)
  Nod cooldown: 3.0s between events
  Minimum nod velocity: 3.0°/s (velocity gating — filters slow drift)
  Posture confidence threshold: 0.6

Confidence Scoring:
  posture_confidence combines:
    1. Magnitude: how deep pitch falls below -20°
    2. Duration: progress toward 1.5s sustained
    3. Instability boost: high yaw/roll variance → involuntary nod

Current Weaknesses:
  • No adaptive baseline calibration (camera mounting angle offsets
    resting pitch by ±5–10°).
  • solvePnP degrades at yaw > ±45° (contralateral occlusion).
  • Generic 3D face model assumes uniform facial proportions.
  • Chin landmark (152) has jitter under thick collars/facial hair.

4.4 TEMPORAL LOGIC
──────────────────
Implementation: TemporalAnalyzer + PostureAnalyzer + YawnAnalyzer

Key Design: FPS-Independent
  All temporal thresholds expressed in seconds (not frames).
  Uses time.monotonic() — immune to NTP jumps and FPS variation.
  Verified: same detection latency at 10 FPS, 15 FPS, 30 FPS.

EMA Smoothing:
  EAR/MAR: α=0.3, ~233ms to 90% convergence at 30 FPS
  Pose: α=0.10, ~690ms to 90% convergence (heavier for jitter suppression)

Hysteresis:
  Prevents state oscillation at threshold boundary.
  0.03 dead zone for EAR (close=0.21, open=0.24).

Current Weaknesses:
  • EMA accumulation rate in fusion engine (0.08 rise) was tuned for
    30 FPS. At significantly different FPS, effective time constant
    changes (update() called less frequently per second).
  • Not explicitly Δt-normalized (the EMA does not multiply by
    elapsed time between frames).

4.5 FUSION LOGIC
────────────────
Implementation: FatigueFusionEngine (in fatigue_fusion.py)

Weighted Sum:
  raw_score = 0.45 × ear_conf + 0.30 × pose_conf + 0.25 × mar_conf

Cue Agreement Amplification:
  1 active cue (conf > 0.3): 1.0× (no bonus)
  2 active cues: 1.3×
  3 active cues: 1.5×

Asymmetric Temporal Accumulation:
  Rise: α=0.08 (fast onset tracking)
  Decay: α=0.04 (slow decay — models physiological fatigue persistence)

Severity Classification (with hysteresis band = 0.12):
  ALERT:           score < 0.25
  SLIGHT_FATIGUE:  score ∈ [0.25, 0.50)
  MODERATE_FATIGUE: score ∈ [0.50, 0.75)
  SEVERE_FATIGUE:  score ≥ 0.75

  To de-escalate: score must drop below (threshold - 0.12).

Reliability Gating:
  effective_score = raw_fusion_score × system_reliability

Current Weaknesses:
  • Weights are hand-tuned, not learned from data.
  • Linear fusion assumption (additive cue contributions).
  • No circadian or environmental context.
  • EMA rate not normalized by actual Δt between frames.

4.6 ALARM LOGIC
───────────────
Implementation: AlarmController

Alarm Triggering:
  SEVERE_FATIGUE or DROWSY or FACE_LOST_CRITICAL → alarm ON

Alarm Lifecycle:
  1. Trigger: alarm starts at appropriate level
  2. Minimum Duration: alarm plays for at least 3.0 seconds
  3. Cooldown: after alarm ends, 5.0s suppression before next alarm
  4. Escalation: alarms can only escalate, never de-escalate mid-episode

Alarm Levels:
  Level 1: HUD warning only (SLIGHT)
  Level 2: Audible cue (MODERATE)
  Level 3: Full alarm (SEVERE, DROWSY, FACE_LOST_CRITICAL)

Face-Loss Behavior (CRITICAL SAFETY FIX):
  v1.0: face lost → alarm SILENCED (safety-inverting bug)
  v2.0: face lost during drowsiness → alarm ESCALATES to Level 3

Current Weaknesses:
  • No alarm sound differentiation (same 880Hz for all levels).
  • No haptic or visual-only escalation path.
  • CSV logging is synchronous (blocking main thread — minor concern
    on SSDs, potentially problematic on SD cards).

4.7 ROBUSTNESS LOGIC
────────────────────
Implementation: RobustnessGuard

Four Sub-Scores (weighted geometric mean):
  Landmark Stability (0.35): 6-point frame-to-frame displacement
    jitter ≤ 2.0px → 1.0, jitter ≥ 8.0px → 0.3
  Brightness Quality (0.25): Trapezoidal mapping
    [0,30]: 0.3, [30,60]: ramp, [60,200]: 1.0, [200,240]: decay, [240,255]: 0.5
  Tracking Quality (0.20): MediaPipe nose-tip visibility
  Cue Consistency (0.20): Coefficient of variation of per-cue confidences
    CV ≤ 0.2 → 1.0, CV ≥ 1.0 → 0.3

Composition:
  reliability = stability^0.35 × brightness^0.25 × tracking^0.20 × consistency^0.20
  (Geometric mean — single degraded channel dominates)

Adaptive Suppression:
  If reliability < 0.5: suppress SLIGHT and MODERATE alarms
  SEVERE alarms are NEVER suppressed (safety guarantee)

Current Weaknesses:
  • Brightness proxy is crude (mean intensity of face ROI).
    Does not capture directional lighting or local contrast.
  • No per-landmark quality (MediaPipe legacy API limitation).
  • Sub-score weights are hand-tuned.
  • No per-vehicle calibration/learning of "normal" conditions.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 5 — CNN STATUS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Does cnn_validator.py exist?
   ✅ YES. Fully implemented (~250 lines). Contains CNNValidator class,
   CNNVerdict dataclass, and extract_eye_roi() function.

2. Is the CNN actually active?
   ❌ NO. The system logs: "Model not found at
   'models/eye_state_model.tflite'. Running in heuristic-only mode."
   The graceful fallback is WORKING AS DESIGNED — the system functions
   entirely without the CNN.

3. Is a .tflite model present?
   ❌ NO. The models/ directory is EMPTY. No eye_state_model.tflite
   exists. The training pipeline (tools/train_eye_cnn.py) exists but
   has not been run to produce a deployable model.

4. Does inference run?
   ⚠ CANNOT VERIFY. The TFLite runtime inference code is implemented
   and ready, but there is no model to load. The code paths for
   interpreter.invoke() are unreachable until a model is trained.

5. Is selective activation implemented?
   ✅ YES. CNNValidator.should_invoke() implements dual gating:
     Gate 1: EAR ∈ [0.17, 0.27] (uncertainty zone)
     Gate 2: Rate limiting (max 5 invocations/second)
   The CNN is NOT invoked on clear-open or clear-closed frames.

6. Is uncertainty logic implemented?
   ✅ YES. The system tracks:
     • cnn_invoked (whether CNN was called this frame)
     • cnn_agrees (whether CNN matches heuristic verdict)
     • cnn_override_active (whether CNN vetoed the heuristic)
     • FP suppression counter
   Full statistics logged at session end (invocations, agreements,
   overrides, agreement rate).

CNN TRAINING STATUS:
   • Data collection tool exists (tools/collect_eye_data.py)
   • Training script exists (tools/train_eye_cnn.py)
   • Training data directory exists (data/eyes/open/, data/eyes/closed/)
   • MicroEyeNet architecture: 2 conv layers (8, 16 filters) +
     32-node dense → ~9,505 parameters → ~12KB quantized
   • Input: 24×24 grayscale
   • Training has NOT been executed yet (no model file generated)

VERDICT: The CNN integration is architecturally complete and
gracefully handles the missing model. The critical missing piece is
the TRAINED MODEL FILE. Without it, the system operates in
heuristic-only mode — the CNN novelty claim is untestable.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 6 — CONFIDENCE GATE STATUS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

A multi-dimensional confidence/quality gate EXISTS and is OPERATIONAL.

Implementation Details:

1. LANDMARK CONFIDENCE
   Status: ⚠ Partially Implemented
   • MediaPipe's legacy API does NOT expose per-landmark confidence.
   • The system uses nose-tip visibility (landmark[1].visibility) as
     a proxy for overall tracking quality.
   • This is a KNOWN LIMITATION documented in research_notes.md.

2. FACE VISIBILITY SCORING
   Status: ✅ Implemented
   • SignalQuality.face_visible boolean is set based on whether
     MediaPipe returns multi_face_landmarks.
   • When face_visible=False, the system triggers FACE_LOST state.

3. TRACKING QUALITY
   Status: ✅ Implemented
   • tracking_confidence extracted from MediaPipe landmark visibility.
   • Fed into RobustnessGuard as one of four sub-scores (weight=0.20).

4. SIGNAL QUALITY METRICS
   Status: ✅ Implemented (4-dimensional)
   • Landmark Stability (0.35 weight): 6-point jitter measurement
   • Brightness Quality (0.25 weight): face ROI mean intensity
   • Tracking Quality (0.20 weight): MediaPipe visibility
   • Cue Consistency (0.20 weight): temporal variance of confidences
   All four computed per-frame and combined via weighted geometric mean.

5. UNCERTAINTY ESTIMATION
   Status: ✅ Implemented (multi-layer)
   Layer 1: Fusion engine confidence (continuous 0.0–1.0 per cue)
   Layer 2: System reliability (continuous 0.0–1.0)
   Layer 3: CNN uncertainty zone gating (EAR ∈ [0.17, 0.27])
   Layer 4: Alert suppression when reliability < 0.5

Overall Confidence Gate Assessment:
  ✅ FULLY OPERATIONAL (with noted limitations on per-landmark
  confidence due to MediaPipe API constraints).
  The system does NOT blindly process frames — it has a sophisticated
  signal quality awareness pipeline that attenuates decisions under
  degraded conditions.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 7 — RESEARCH READINESS AUDIT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. PUBLIC DATASET EVALUATION
   Status: ❌ NOT READY
   • No dataset loader exists for NTHU-DDD, MRL Eye Dataset, or
     any public benchmark.
   • No batch processing mode (system only runs in real-time webcam).
   • No ground truth annotation parser.
   • The data/ directory contains only the student-collected CNN
     training images (open/closed eyes).

2. FAR MEASUREMENT (False Acceptance Rate)
   Status: ⚠ PARTIALLY READY
   • CSV event logging records every alarm with timestamp.
   • A FAR measurement script would need to:
     (a) Process annotated test sessions
     (b) Count alarms during alert-annotated periods
     (c) Compute FP / (FP + TN)
   • Infrastructure exists (CSV logging) but no evaluation harness.

3. PRECISION / RECALL
   Status: ❌ NOT READY
   • No ground truth labels for fatigue events.
   • No evaluation script to compute TP, FP, FN, TN.
   • No per-session confusion matrix generation.
   • The ablation framework (V1→V4) is DESIGNED but not IMPLEMENTED
     as runnable scripts.

4. LATENCY MEASUREMENT
   Status: ✅ READY
   • Built-in profiler tracks per-frame: capture time, inference time,
     math time, render time.
   • Profiler outputs rolling averages every N frames (configurable).
   • time.perf_counter() used for high-resolution timing.
   • Can be activated via cfg.optimization.enable_profiling = True.

5. CPU PROFILING
   Status: ⚠ PARTIALLY READY
   • Built-in profiler measures per-stage latency.
   • No CPU utilization measurement (would require psutil or /proc).
   • No memory profiling.
   • Raspberry Pi thermal tracking would require vcgencmd integration.

6. FPS PROFILING
   Status: ✅ READY
   • FPS computed every 1 second in main loop.
   • Displayed on HUD in real-time.
   • Logged via profiler.
   • 1st-percentile FPS not tracked (only rolling average).

7. ACTIVATION-RATE MEASUREMENT (CNN)
   Status: ✅ READY
   • CNNValidator tracks total_invocations vs total_frames.
   • Agreement rate, override rate logged at session end.
   • However, since no model exists, these measurements are moot.

SUMMARY TABLE:
  ┌──────────────────────────────┬───────────────────┐
  │ Capability                   │ Status            │
  ├──────────────────────────────┼───────────────────┤
  │ Public dataset evaluation    │ ❌ Not Ready      │
  │ FAR measurement              │ ⚠ Partially Ready │
  │ Precision/Recall             │ ❌ Not Ready      │
  │ Latency measurement          │ ✅ Ready          │
  │ CPU profiling                │ ⚠ Partially Ready │
  │ FPS profiling                │ ✅ Ready          │
  │ CNN activation-rate          │ ✅ Ready          │
  └──────────────────────────────┴───────────────────┘

MISSING PIECES:
  1. Dataset loader for public benchmarks (NTHU-DDD, MRL Eye, etc.)
  2. Batch processing mode (video file input, not just webcam)
  3. Ground truth annotation parser
  4. Evaluation harness (compute TP/FP/FN/TN/Precision/Recall/F1/FAR)
  5. Confusion matrix generator
  6. Statistical significance testing (McNemar's test)
  7. CPU/memory utilization tracker (psutil)
  8. Per-session result aggregation and comparison script


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 8 — EXPERIMENT READINESS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EXPERIMENTS THAT CAN BE RUN TODAY:

✅ Experiment S1: Threshold Consistency Verification
   Run system, inspect printed config. Verify all modules use 0.21.
   Effort: 5 minutes.

✅ Experiment S2: FPS Independence Validation
   Use cv2.CAP_PROP_FPS to cap camera at 10/15/20/25/30.
   Close eyes for 1.5s at each FPS. Measure detection latency.
   Effort: 30 minutes.

✅ Experiment S3: Face-Loss Safety Test
   Trigger drowsiness, cover camera. Verify alarm escalates (not silences).
   Effort: 5 minutes.

✅ Experiment S4: MAR Speech Artifact Rejection
   Read paragraph aloud. Count false yawn detections.
   Effort: 10 minutes.

✅ Experiment S5: Alarm Persistence Validation
   Trigger alarm, open eyes immediately. Verify alarm persists ≥ 3.0s.
   Effort: 5 minutes.

✅ Experiment S6: Hysteresis Anti-Oscillation Test
   Narrow eyes to hover EAR at ~0.21–0.23. Count state transitions.
   Effort: 15 minutes.

✅ Experiment S7: Cooldown Period Validation
   Trigger alarm twice in succession. Verify 5.0s gap.
   Effort: 5 minutes.

✅ Experiment Y1–Y3: Yawn Validation
   Speech resilience, smile test, genuine yawn.
   Effort: 15 minutes.

✅ Experiment P1–P3: Posture Validation
   Dashboard check, fatigue nod, mirror checks.
   Effort: 15 minutes.

✅ Experiment R1: False-Positive Rate Under Normal Conditions
   10-minute normal driving session. Count false alarms.
   Effort: 15 minutes.

✅ Experiment S7 (Stabilization): MAR Bounds Verification
   Sit still 30s, yawn 3x, talk 30s. Verify MAR < 1.0.
   Effort: 5 minutes.

✅ Experiment: Severity Stability
   Normal sitting 2 minutes. Verify zero severity transitions.
   Effort: 5 minutes.

EXPERIMENTS THAT REQUIRE ADDITIONAL IMPLEMENTATION:

❌ Experiment F1: Single-Cue vs. Multi-Cue FP Rate
   Requires: Ablation mode (disable individual cues). NOT implemented
   as a configuration option.

❌ Experiment F2: Fatigue Severity Validation (multi-subject)
   Requires: 5+ subjects. Subject recruitment, scheduling.

❌ Experiment F3: Temporal Accumulation Behavior
   Requires: Score logging over time (need to add fatigue_score to CSV).

❌ Experiment F4: Agreement Bonus Sensitivity Sweep
   Requires: Parameter sweep script (not implemented).

❌ Experiment F5: FPS Impact on Fusion Stability
   Requires: Score logging at multiple FPS levels.

❌ Experiment R2: Low-Light Robustness
   Requires: Dimmable lighting setup + controlled environment.

❌ Experiment R3: Camera Shake Robustness
   Requires: Vibrating platform (hardware).

❌ Experiment R4: Reliability Score Validation
   Requires: 30-minute annotated session + correlation computation.

❌ Experiment R5: Suppression Threshold Sweep
   Requires: Parameter sweep script.

❌ Experiment R6: Fusion Attenuation Impact on Detection Latency
   Requires: Controlled reliability simulation.

❌ CNN Validation Effectiveness (from Part IV of research_notes)
   Requires: Trained .tflite model + Ambiguity Dataset.

❌ Full Ablation Study (V1→V4)
   Requires: Configurable cue enable/disable + batch processing +
   evaluation harness + labeled dataset.

❌ Public Dataset Evaluation
   Requires: Dataset loader + batch processor + evaluation metrics.

❌ Raspberry Pi Edge Benchmarking
   Requires: Raspberry Pi 4 hardware + active cooling.

❌ Thermal Stability Analysis
   Requires: Raspberry Pi 4 + 120-minute run + temperature logging.

SUMMARY:
  Can run TODAY:  12 experiments (all require only webcam + manual protocol)
  Need code:       11 experiments (require evaluation harness, ablation modes,
                   batch processing, or parameter sweeps)
  Need hardware:    2 experiments (Raspberry Pi, vibrating platform)
  Need subjects:    1 experiment (multi-subject fatigue validation)


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 9 — NOVELTY COMPONENTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. CONDITIONAL INFERENCE (Selective CNN Invocation)
   Status: ✅ Implemented
   • CNNValidator.should_invoke() gates on uncertainty zone.
   • Dual gates: EAR range + rate limiting.
   • extract_eye_roi() crops 24×24 grayscale eye region.
   • CNN invoked on ~15% of frames (estimated).
   • UNTESTABLE until model is trained.

2. UNCERTAINTY ROUTING
   Status: ✅ Implemented
   • Fusion engine produces continuous confidence scores (0.0–1.0).
   • Uncertainty zone defined as EAR ∈ [0.17, 0.27].
   • When uncertain: CNN validates. When confident: heuristic trusted.
   • CNNVerdict tracks agreement/disagreement.

3. SELECTIVE CNN VALIDATION
   Status: ✅ Implemented (architecture) | ❌ Inactive (no model)
   • Full pipeline implemented: invoke gate → eye ROI extraction →
     inference → verdict → state integration → HUD display.
   • Statistics tracking (invocations, agreements, overrides).
   • Graceful fallback when model absent.
   • BLOCKER: No trained .tflite model file.

4. PERSONALIZED CALIBRATION
   Status: ❌ Not Implemented
   • research_notes.md discusses adaptive per-subject calibration
     during first 5 minutes of driving.
   • No code exists for baseline learning, threshold adaptation,
     or weight personalization.
   • Fixed thresholds for all users.

5. ROBUSTNESS EVALUATION
   Status: ✅ Implemented
   • RobustnessGuard computes 4-dimensional signal quality.
   • Reliability-gated fusion (multiplicative attenuation).
   • Adaptive alert suppression under degraded conditions.
   • Suppression logging for post-hoc analysis.
   • Geometric mean composition (single-channel degradation dominates).

6. EDGE DEPLOYMENT
   Status: ⚠ Partially Implemented
   • Headless mode (no GUI rendering).
   • Adaptive frame skipping (skip alternating frames during ALERT).
   • Async camera I/O (Producer-Consumer pattern).
   • Lightweight MicroEyeNet (~9.5K params, ~12KB quantized).
   • Graceful degradation (CNN fallback, FPS-independent timing).
   • NOT TESTED on Raspberry Pi (no hardware available).
   • No TFLite INT8 quantized model produced.

NOVELTY SUMMARY:
  ┌─────────────────────────────────────┬───────────────────────┐
  │ Component                           │ Status                │
  ├─────────────────────────────────────┼───────────────────────┤
  │ Conditional inference               │ ✅ Implemented        │
  │ Uncertainty routing                 │ ✅ Implemented        │
  │ Selective CNN validation            │ ⚠ Impl. (no model)   │
  │ Personalized calibration            │ ❌ Not Implemented    │
  │ Robustness evaluation               │ ✅ Implemented        │
  │ Edge deployment                     │ ⚠ Partially Impl.    │
  │ FPS-independent temporal analysis   │ ✅ Implemented        │
  │ Face-loss safety escalation         │ ✅ Implemented        │
  │ Multi-factor fatigue fusion         │ ✅ Implemented        │
  │ Asymmetric temporal accumulation    │ ✅ Implemented        │
  │ Hysteresis thresholding             │ ✅ Implemented        │
  │ MAR jitter speech filtering         │ ✅ Implemented        │
  │ Velocity-gated nod detection        │ ✅ Implemented        │
  └─────────────────────────────────────┴───────────────────────┘


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 10 — TECHNICAL DEBT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Ranked by severity (highest first):

[SEVERITY: CRITICAL]

1. NO TRAINED CNN MODEL
   Location: models/ (empty directory)
   Impact: The entire CNN validation novelty is untestable.
   The paper's core contribution (selective hybrid inference)
   has zero experimental evidence.
   Fix: Train MicroEyeNet on collected eye data + augment with
   MRL Eye Dataset.

2. NO EVALUATION HARNESS
   Location: Entire project
   Impact: Cannot compute precision, recall, FAR, or confusion
   matrices. No automated experiment execution. All experiments
   require manual observation.
   Fix: Create eval/batch_processor.py + eval/metrics.py +
   eval/run_experiment.py.

[SEVERITY: HIGH]

3. HARDCODED FUSION WEIGHTS
   Location: fatigue_fusion.py (EAR=0.45, Pose=0.30, MAR=0.25)
   Impact: Weights are hand-tuned, not data-driven. A reviewer
   will challenge "why 0.45 and not 0.40?" without empirical
   justification.
   Fix: Implement grid search or logistic regression on labeled data.

4. EMA NOT Δt-NORMALIZED
   Location: temporal_analyzer.py, fatigue_fusion.py
   Impact: EMA alpha values are implicitly tuned for 30 FPS.
   At 15 FPS (RPi), the effective time constant doubles. This
   means detection latency, accumulation rate, and decay rate
   all change with FPS — despite the "FPS-independent" design claim.
   Fix: Multiply alpha by (dt / expected_dt) in update().

5. NO BATCH/VIDEO-FILE PROCESSING MODE
   Location: main.py
   Impact: Cannot process recorded datasets. Cannot reproduce
   experiments. Cannot generate ROC curves from video archives.
   Fix: Add --video flag to main.py that reads from file instead
   of webcam.

6. CSV LOGGING ON MAIN THREAD
   Location: alarm_controller.py
   Impact: Disk I/O (especially on SD cards) can cause frame drops.
   Not a problem on SSDs but will be on Raspberry Pi.
   Fix: Move to async writer or buffered queue.

[SEVERITY: MEDIUM]

7. DUPLICATED EAR COMPUTATION
   Location: detector.py (3D) vs. ear_processor.py (standalone)
   Impact: Two EAR implementations exist. ear_processor.py is
   deprecated but still importable. Maintenance divergence risk.
   Fix: Remove or clearly mark ear_processor.py as non-authoritative.

8. NO PER-SUBJECT CALIBRATION
   Location: config.py (fixed thresholds)
   Impact: Inter-subject variation in eye morphology causes
   significantly different false positive rates.
   Fix: Implement 30-second calibration phase that learns
   subject-specific EAR baseline.

9. REDUNDANT `import math` INSIDE METHOD
   Location: detector.py, line 121 (`_distance_2d` method)
   Impact: Negligible (just style). Math already imported at module level.
   Fix: Remove redundant import.

10. POSE BASELINE OFFSET NOT CALIBRATED
    Location: pose_estimator.py
    Impact: Camera mounting angle introduces ±5-10° pitch offset.
    The nod threshold (-20.0°) may be too sensitive or too lax
    depending on camera position.
    Fix: Implement first-10-seconds baseline calibration.

[SEVERITY: LOW]

11. HEADLESS MODE STILL CALLS cvtColor
    Location: main.py, line 180
    Impact: rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    is always called even in headless mode. Minor CPU waste (~1ms).
    Fix: Not critical — MediaPipe needs RGB regardless.

12. NO ALARM SOUND DIFFERENTIATION
    Location: alarm_controller.py / audio_alert.py
    Impact: Same 880Hz tone for all alarm types (drowsy, yawn,
    face lost). Driver cannot distinguish severity by sound.
    Fix: Generate different frequencies for different levels.

13. DEPRECATED MODULES STILL PRESENT
    Location: ear_processor.py, alert_manager.py, face_landmark_detector.py
    Impact: Code clutter. Potential confusion for new researchers.
    Fix: Move to src/deprecated/ or delete entirely.

14. NO UNIT TESTS
    Location: test_pipeline.py, test_pose.py, test_variance.py, test_webcam.py
    Impact: Test files exist but are manual integration tests, not
    automated unit tests. No pytest, no CI, no coverage reporting.
    Fix: Add unit tests for detector.py, fatigue_fusion.py math.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 11 — RESEARCH GAP ALIGNMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TARGET PAPER IDEA:
  Heuristic State Machine → Uncertainty Detection → Selective CNN
  Validation → Final Decision

ALIGNMENT ANALYSIS:

Step 1: HEURISTIC STATE MACHINE
  Status: ✅ FULLY IMPLEMENTED
  What exists:
    • DrowsinessDetector (pure math: EAR/MAR)
    • TemporalAnalyzer (EMA, hysteresis, wall-clock timing)
    • PostureAnalyzer (solvePnP, velocity-gated nod detection)
    • YawnAnalyzer (jitter-based speech filtering, confidence scoring)
    • StateManager (5-state DriverStatus enum)
    • FatigueFusionEngine (weighted multi-cue fusion, severity levels)
    • RobustnessGuard (4D signal quality, reliability gating)
  Assessment: This is the STRONGEST component of the entire system.
  Exceeds the typical heuristic baseline in the literature.

Step 2: UNCERTAINTY DETECTION
  Status: ✅ FULLY IMPLEMENTED
  What exists:
    • CNNValidator.should_invoke() with EAR uncertainty zone [0.17, 0.27]
    • RobustnessGuard system_reliability as signal quality proxy
    • Fusion engine continuous confidence scores (0.0–1.0)
    • Cue-agreement count (1/2/3 active cues)
  Assessment: Well-designed uncertainty routing. The dual-gate
  (range + rate limit) is publication-worthy.

Step 3: SELECTIVE CNN VALIDATION
  Status: ⚠ ARCHITECTURE COMPLETE | FUNCTIONALLY INACTIVE
  What exists:
    • CNNValidator class with full inference pipeline
    • extract_eye_roi() for 24×24 grayscale eye crop
    • CNNVerdict dataclass with agreement tracking
    • Statistics logging (invocations, agreements, overrides)
    • Graceful fallback when model absent
  What is MISSING:
    • The trained .tflite model file
    • Experimental evidence that CNN actually reduces FPs
    • CNN accuracy on the target eye-state classification task
  Assessment: The ARCHITECTURE is publication-ready. The EVIDENCE
  is completely absent. This is the single biggest gap.

Step 4: FINAL DECISION
  Status: ✅ FULLY IMPLEMENTED
  What exists:
    • StateManager integrates fusion score + CNN verdict + robustness
    • AlarmController with lifecycle management
    • Face-loss safety escalation
    • Reliability-gated alert suppression
  Assessment: Complete and well-designed. The CNN override mechanism
  (suppression of false positives, never suppression of true positives)
  is a strong design choice.

OVERALL ALIGNMENT:
  ┌──────────────────────────┬─────────────────────────────────┐
  │ Pipeline Step            │ Status                          │
  ├──────────────────────────┼─────────────────────────────────┤
  │ Heuristic State Machine  │ ✅ Fully Implemented            │
  │ Uncertainty Detection    │ ✅ Fully Implemented            │
  │ Selective CNN Validation │ ⚠ Architecture Only (no model)  │
  │ Final Decision           │ ✅ Fully Implemented            │
  └──────────────────────────┴─────────────────────────────────┘

WHAT MUST BE IMPLEMENTED BEFORE EXPERIMENTS BEGIN:

  1. CRITICAL: Train MicroEyeNet → produce eye_state_model.tflite
  2. CRITICAL: Create evaluation harness (batch processing + metrics)
  3. HIGH: Implement ablation modes (V1→V4 configurations)
  4. HIGH: Add fatigue_score to CSV logging for temporal analysis
  5. HIGH: Create parameter sweep infrastructure
  6. MEDIUM: Add video-file input mode for reproducible experiments
  7. MEDIUM: Implement McNemar's test for statistical significance


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 12 — NEXT DEVELOPMENT ROADMAP
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Priority 1: TRAIN THE CNN MODEL
  • Collect 500+ open + 500+ closed eye images using collect_eye_data.py
  • Augment with MRL Eye Dataset images for diversity
  • Train MicroEyeNet via train_eye_cnn.py with k-fold cross-validation
  • Export to TFLite with dynamic range quantization
  • Place model at models/eye_state_model.tflite
  • Verify CNN inference works end-to-end
  Impact: Unlocks the core novelty claim. Without this, the paper
  cannot claim "hybrid" intelligence.

Priority 2: BUILD EVALUATION HARNESS
  • Create eval/ directory with:
    - batch_processor.py: Process video files or image sequences
    - metrics.py: Compute TP/FP/FN/TN, Precision, Recall, F1, FAR
    - run_ablation.py: Execute V1→V4 configurations on same dataset
    - plot_results.py: Generate ROC curves, confusion matrices,
      temporal score plots, latency histograms
  • Add --video FILE flag to main.py for offline processing
  • Add fatigue_score, system_reliability, cnn_invoked to CSV output
  Impact: Enables ALL quantitative experiments. Without this, the paper
  has no numbers.

Priority 3: IMPLEMENT ABLATION FRAMEWORK
  • Add config flags to disable individual cues:
    cfg.fusion.enable_ear_cue = True/False
    cfg.fusion.enable_mar_cue = True/False
    cfg.fusion.enable_pose_cue = True/False
    cfg.cnn_validation.enabled = True/False
  • Implement V1 (EAR only), V2 (EAR+MAR), V3 (Full fusion),
    V4 (Full fusion + CNN) as configurable presets
  • Run each on the same annotated test set
  Impact: Produces the "money chart" (V1→V4 progressive improvement)
  that proves the paper's contribution.

Priority 4: Δt-NORMALIZE EMA UPDATES
  • In TemporalAnalyzer.update(), multiply alpha by (dt / expected_dt)
    where expected_dt = 1/30 (assumes 30 FPS baseline)
  • Same fix in FatigueFusionEngine.update()
  • Verify detection latency is identical at 15 FPS and 30 FPS
  Impact: Eliminates the strongest technical weakness in the "FPS-
  independent" claim. Currently, the claim is only partially true
  (wall-clock thresholds are FPS-independent, but EMA dynamics are not).

Priority 5: COLLECT MULTI-CONDITION TEST DATASET
  • Record 10 video sessions (5 min each) covering:
    - Normal driving (blinking, speaking, mirror checks)
    - Simulated drowsiness (eye closure sequences)
    - Edge cases (glasses, bright light, dim light, talking)
  • Manually annotate ground truth fatigue events (timestamps)
  • Use this dataset for all ablation experiments and FAR measurement
  Impact: Provides the empirical evidence that reviewers demand.
  Without labeled data, no quantitative claims can be made.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 13 — FINAL VERDICT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. CURRENT PROJECT MATURITY: 65%
   The architecture is mature, well-documented, and modular. The code
   quality is high — clean separation of concerns, centralized config,
   proper error handling, graceful degradation. The stabilization
   patches (v2.0 → v3.1) resolved all critical engineering bugs. The
   system runs reliably in real-time with a webcam. However, the CNN
   model is missing, no evaluation infrastructure exists, and no
   quantitative experiments have been conducted.

2. RESEARCH READINESS: 35%
   The research DIRECTION is well-defined (selective hybrid inference
   for edge-optimized drowsiness detection). The research NOTES are
   exceptionally thorough (2819 lines covering 9 implementation stages,
   experiment protocols, paper writing notes, reviewer preparation).
   However, ZERO quantitative results exist. No dataset has been
   processed. No metrics have been computed. No ablation study has
   been run. The system has been validated only through manual
   observation ("does the alarm go off when I close my eyes?").
   This is insufficient for any publication venue.

3. PUBLICATION READINESS: 20%
   What IS ready:
     • Architecture is designed and implemented
     • Methodology is documented in excruciating detail
     • Novelty claims are clear and defensible
     • Limitations are honestly acknowledged
     • Paper positioning strategy is sophisticated
   What is NOT ready:
     • ZERO quantitative results (no precision, recall, FAR, F1)
     • No trained CNN model (core novelty unverifiable)
     • No comparison with baselines or SOTA
     • No public dataset evaluation
     • No statistical significance testing
     • No Raspberry Pi edge benchmarks
     • No figures, tables, or experimental evidence
   The gap between "well-architected system" and "publishable paper"
   is primarily EXPERIMENTAL, not architectural.

4. STRONGEST COMPONENT:
   The Multi-Factor Fatigue Fusion Engine + Robustness Guard
   combination. This is a genuinely sophisticated, well-engineered
   system that goes far beyond typical "EAR threshold → alarm"
   implementations found in the literature. The asymmetric temporal
   accumulation, cue-agreement amplification, hysteresis severity
   classification, 4D signal quality monitoring, and reliability-
   gated alert suppression collectively form a publication-worthy
   contribution — IF properly evaluated.

5. WEAKEST COMPONENT:
   The CNN validation layer — not in terms of code quality (which is
   excellent), but in terms of ACTUAL FUNCTIONALITY. The architecture
   is complete but the system has never run with an active CNN. The
   entire "hybrid intelligence" claim rests on a model file that does
   not exist. This is the weakest link in the entire chain.

6. BIGGEST PUBLICATION RISK:
   "The paper proposes a hybrid system but only evaluates the
   heuristic component. The CNN's contribution is entirely theoretical."
   A reviewer will immediately ask: "What is the CNN's accuracy on
   eye-state classification? How many false positives does it suppress?
   What is the F1 improvement from V3 to V4?" Currently, these
   questions have ZERO answers. If the CNN turns out to have poor
   accuracy (<80%), the entire hybrid argument collapses and the paper
   reduces to "yet another EAR+MAR system with fusion heuristics" —
   which, while well-engineered, is not novel enough for IEEE T-ITS.

7. SINGLE MOST IMPORTANT NEXT TASK:
   ┌─────────────────────────────────────────────────────────────┐
   │                                                             │
   │   TRAIN THE CNN MODEL AND RUN THE FIRST ABLATION STUDY.    │
   │                                                             │
   │   Specifically:                                             │
   │   1. Collect 1000 eye images (500 open, 500 closed)         │
   │   2. Train MicroEyeNet with 5-fold cross-validation        │
   │   3. Export to TFLite                                       │
   │   4. Run 5-minute test session WITH CNN active              │
   │   5. Log: CNN invocations, agreements, overrides, FP count  │
   │   6. Compare: FP rate with CNN vs. without CNN              │
   │                                                             │
   │   This single task unlocks the paper's core contribution    │
   │   and produces the first piece of quantitative evidence.    │
   │   Estimated time: 2-3 days of focused work.                 │
   │                                                             │
   └─────────────────────────────────────────────────────────────┘


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
END OF TECHNICAL AUDIT REPORT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Report compiled from:
  • 15 active Python source modules (read in full)
  • 3 deprecated modules (examined)
  • 4 test files (examined)
  • 2 tool scripts (examined)
  • 2819-line research notes (read in full)
  • requirements.txt, README.md
  • Directory structure analysis

Total lines of code analyzed: ~5,500+ (source) + 2,819 (research notes)