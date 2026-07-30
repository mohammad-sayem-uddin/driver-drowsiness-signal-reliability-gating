# PHASE 01: DEPENDENCY AUDIT & COMPATIBILITY REVIEW

**Target Package**: Driver Drowsiness Detection System (v3.1 Baseline)  
**Manifest File**: `requirements.txt`  
**Date**: July 2026

---

## 1. Pinned Dependencies & Purpose

```
===================================================================================
                       PINNED DEPENDENCY MANIFEST
===================================================================================
opencv-python>=4.8.0.76     # Core image I/O, color conversion, solvePnP Head Pose
mediapipe>=0.10.7           # 3D FaceMesh 468-landmark tracking engine
numpy>=1.24.3               # Vectorized Euclidean math & EMA array operations
pygame>=2.5.2               # Real-time audio alert playback & tone generation
tensorflow>=2.14.0          # MicroEyeNet TFLite model execution engine
scipy>=1.11.3               # Signal processing & stats functions
pytest>=7.4.3               # Automated unit test framework
===================================================================================
```

---

## 2. Environment Compatibility Verification

- **Python Version**: Python 3.10+ compatible (tested on Python 3.14 on macOS Apple Silicon).
- **ARM Edge Compatibility**: All listed dependencies have native pre-compiled wheel binaries available for ARM64 Linux (Raspberry Pi OS 64-bit).
- **Virtual Environment**: Isolated environment managed via `.venv`.
