# PHASE 01: CONFIGURATION REFERENCE DOCUMENTATION

**Target Package**: Driver Drowsiness Detection System (v3.1 Baseline)  
**Source File**: `src/config.py`  
**Date**: July 2026

---

## 1. Overview of Central Configuration System

The repository utilizes Python dataclasses in `src/config.py` as the single source of truth for all system hyperparameters, physical thresholds, temporal windows, and feature toggles.

---

## 2. Parameter Reference Table

### 2.1 `DetectionConfig` (Heuristic Metrics)
| Parameter Name | Data Type | Default Value | Description |
|:---|:---|:---|:---|
| `ear_threshold` | `float` | `0.21` | EAR threshold below which an eye is considered closed. |
| `ear_hysteresis` | `float` | `0.03` | Hysteresis margin to prevent rapid toggle ($EAR_{open} = 0.24$). |
| `mar_threshold` | `float` | `0.55` | 2D MAR threshold for yawning detection. |
| `pitch_nod_threshold` | `float` | `-12.0` | Head pitch angle (degrees) indicating forward nod. |
| `yaw_tilt_threshold` | `float` | `25.0` | Head yaw angle (degrees) indicating side turn. |

### 2.2 `TemporalConfig` (Wall-Clock Windows)
| Parameter Name | Data Type | Default Value | Description |
|:---|:---|:---|:---|
| `eye_closure_duration` | `float` | `1.0` | Continuous closure duration (seconds) triggering drowsiness. |
| `yawn_duration` | `float` | `1.5` | Continuous open-mouth duration (seconds) triggering yawn. |
| `speech_jitter_threshold` | `float` | `0.05` | MAR variance threshold ($\sigma_{MAR}$) identifying speech. |
| `pitch_velocity_threshold` | `float` | `-3.0` | Pitch angular velocity ($^\circ/\text{s}$) triggering nod event. |
| `nod_cooldown` | `float` | `3.0` | Cooldown window (seconds) between nod events. |

### 2.3 `RobustnessConfig` (`RobustnessGuard`)
| Parameter Name | Data Type | Default Value | Description |
|:---|:---|:---|:---|
| `w_stability` | `float` | `0.35` | Geometric mean weight for landmark stability sub-score. |
| `w_brightness` | `float` | `0.25` | Geometric mean weight for frame brightness sub-score. |
| `w_tracking` | `float` | `0.20` | Geometric mean weight for tracking confidence sub-score. |
| `w_consistency` | `float` | `0.20` | Geometric mean weight for cue consistency sub-score. |

### 2.4 `FusionConfig` (Multi-Cue Engine)
| Parameter Name | Data Type | Default Value | Description |
|:---|:---|:---|:---|
| `ear_weight` | `float` | `0.45` | Weight assigned to Eye Aspect Ratio in fatigue score. |
| `pose_weight` | `float` | `0.30` | Weight assigned to Head Pose in fatigue score. |
| `mar_weight` | `float` | `0.25` | Weight assigned to Mouth Aspect Ratio in fatigue score. |
| `mult_2_cues` | `float` | `1.30` | Multiplier applied when 2 cues agree. |
| `mult_3_cues` | `float` | `1.50` | Multiplier applied when 3 cues agree. |

### 2.5 `CNNValidationConfig` (MicroEyeNet Selective Invocation)
| Parameter Name | Data Type | Default Value | Description |
|:---|:---|:---|:---|
| `model_path` | `str` | `"models/eye_state_model.tflite"` | Path to TFLite model binary asset. |
| `ambiguity_ear_min` | `float` | `0.17` | Lower EAR boundary for uncertainty resolution. |
| `ambiguity_ear_max` | `float` | `0.27` | Upper EAR boundary for uncertainty resolution. |
| `min_system_reliability` | `float` | `0.30` | Minimum $R_{sys}$ score required to invoke CNN. |
