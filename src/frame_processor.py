"""
Headless Frame Processor (shared per-frame pipeline core)
=========================================================
Single source of truth for running ONE frame through the full detection
pipeline WITHOUT any camera, window, or rendering:

    frame(BGR) + timestamp
        -> MediaPipe FaceMesh
        -> geometry (EAR 3D, MAR 2D, head pose)
        -> SignalQuality (3 components) -> RobustnessGuard (reliability)
        -> TemporalAnalyzer (video-clock timing; speech-jitter filter)
        -> selective CNN validation (optional ablation arm)
        -> StateManager (reliability-gated fusion, face-loss escalation)

Both the live app and the offline benchmark run the SAME geometry/gating/
temporal logic through this class, so evaluation cannot silently diverge from
deployment. Timestamps are injected explicitly: the benchmark passes the video
clock (frame_index / fps) rather than wall-clock, per the frozen protocol.

This module measures nothing and fabricates nothing; it only computes the
per-frame FrameResult. Metric aggregation and latency measurement live in the
evaluation harness.
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Optional, List

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import SystemConfig
from src.detector import DrowsinessDetector
from src.temporal_analyzer import TemporalAnalyzer
from src.robustness import RobustnessGuard, SignalQuality
from src.state_manager import StateManager, DriverStatus
from src.pose_estimator import HeadPoseEstimator
from src.cnn_validator import CNNValidator, CNNVerdict, extract_eye_roi
import src.utils.landmark_indices as landmarks

# Mouth corners/lips used for the 2D MAR (matches src/main.py exactly).
_MOUTH_MAR_IDX = [78, 13, 308, 14]


@dataclass
class FrameResult:
    """Per-frame pipeline output (no timing; the harness times externally)."""
    face_detected: bool = False
    raw_ear: float = 0.0
    raw_mar: float = 0.0
    smoothed_ear: float = 0.0
    smoothed_mar: float = 0.0
    reliability: float = 0.0
    alert_suppressed: bool = False
    is_drowsy: bool = False
    is_speaking: bool = False
    yawn_confidence: float = 0.0
    closure_ratio: float = 0.0
    fatigue_score: float = 0.0   # accumulated 0–1 score; ROC threshold variable
    status: DriverStatus = DriverStatus.ALERT
    cnn_invoked: bool = False

    # --- Additive alarm-decision exposure (EXP-005 event-level evaluation) ---
    # Purely additive; populated verbatim from the StateManager SystemState
    # returned by ``state_mgr.update``. No pipeline logic changes; ``fatigue_score``
    # (the EXP-004 swept variable) is untouched. Defaults keep the dataclass
    # backward-compatible for existing readers (loso_harness reads only
    # ``.fatigue_score``). See Appendix A5 of the frozen spec.
    should_alarm: bool = False        # StateManager alarm boolean (event source)
    alarm_level: int = 0              # 0 silent / 1 beep / 2 alarm / 3 escalated
    cnn_override_active: bool = False # CNN suppressed a SLIGHT/MODERATE alarm
    alarm_suppressed_actual: bool = False  # reliability-gate suppression that
                                           # ACTUALLY flipped should_alarm (state
                                           # .alert_suppressed), distinct from the
                                           # guard's recommendation above
    face_visible: bool = True
    seconds_since_face_lost: float = 0.0


class FrameProcessor:
    """
    Stateful headless pipeline. Construct once per subject/clip (so temporal
    and state history reset between subjects — required for LOSO), then call
    ``process(frame_bgr, timestamp)`` per frame in order.
    """

    def __init__(self, cfg: Optional[SystemConfig] = None,
                 enable_cnn: bool = False):
        self.cfg = cfg or SystemConfig()
        self.detector = DrowsinessDetector()
        self.analyzer = TemporalAnalyzer(self.cfg)
        self.state_mgr = StateManager(self.cfg)
        self.robustness = RobustnessGuard(self.cfg)
        # CNN arm is OFF by default; it is a prior-art ablation, not the
        # default path (frozen spec §2). Only construct it when requested.
        self.enable_cnn = enable_cnn
        self.cnn = CNNValidator(self.cfg) if enable_cnn else None

        # Lazily built once the first frame reveals the resolution.
        self._pose: Optional[HeadPoseEstimator] = None
        self._prev_key_landmarks: Optional[np.ndarray] = None

        # MediaPipe FaceMesh — one instance per processor.
        import mediapipe as mp
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=self.cfg.face_mesh.max_num_faces,
            refine_landmarks=self.cfg.face_mesh.refine_landmarks,
            min_detection_confidence=self.cfg.face_mesh.min_detection_confidence,
            min_tracking_confidence=self.cfg.face_mesh.min_tracking_confidence,
        )

    def close(self):
        if self._face_mesh is not None:
            self._face_mesh.close()
            self._face_mesh = None

    def process(self, frame_bgr, timestamp: float) -> FrameResult:
        """Run one frame. ``timestamp`` is the video-clock time in seconds."""
        import cv2
        h, w = frame_bgr.shape[:2]
        if self._pose is None:
            self._pose = HeadPoseEstimator(w, h)

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._face_mesh.process(rgb)

        raw_ear = raw_mar = 0.0
        raw_pitch = raw_yaw = raw_roll = 0.0
        face_detected = False

        if results.multi_face_landmarks:
            face_detected = True
            face_lm = results.multi_face_landmarks[0]

            left_eye = [face_lm.landmark[i] for i in landmarks.LEFT_EYE_CONTOUR]
            right_eye = [face_lm.landmark[i] for i in landmarks.RIGHT_EYE_CONTOUR]
            mouth = [face_lm.landmark[i] for i in _MOUTH_MAR_IDX]

            raw_ear = (self.detector.calculate_ear(left_eye)
                       + self.detector.calculate_ear(right_eye)) / 2.0
            raw_mar = self.detector.calculate_mar(mouth)

            pose_coords = [
                (int(face_lm.landmark[i].x * w), int(face_lm.landmark[i].y * h))
                for i in landmarks.POSE_LANDMARKS
            ]
            raw_pitch, raw_yaw, raw_roll = self._pose.estimate_pose(pose_coords)

            # Landmark jitter: mean displacement of key points vs previous frame.
            cur = np.array(pose_coords, dtype=np.float32)
            jitter = 0.0
            if (self._prev_key_landmarks is not None
                    and len(cur) == len(self._prev_key_landmarks)):
                jitter = float(np.mean(
                    np.linalg.norm(cur - self._prev_key_landmarks, axis=1)))
            self._prev_key_landmarks = cur

            # Brightness: mean intensity of the face bounding box.
            xs = [p[0] for p in pose_coords]
            ys = [p[1] for p in pose_coords]
            x0, x1 = max(0, min(xs) - 20), min(w, max(xs) + 20)
            y0, y1 = max(0, min(ys) - 20), min(h, max(ys) + 20)
            if x1 > x0 and y1 > y0:
                gray = cv2.cvtColor(frame_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
                brightness = float(np.mean(gray))
            else:
                brightness = 128.0

            sig = SignalQuality(landmark_jitter=jitter,
                                frame_brightness=brightness,
                                face_visible=True)
        else:
            self._prev_key_landmarks = None
            sig = SignalQuality(face_visible=False)

        # Temporal analysis with the VIDEO clock (not wall-clock).
        ts = self.analyzer.update(raw_ear, raw_mar, raw_pitch, raw_yaw,
                                  raw_roll, timestamp=timestamp)

        snap = self.robustness.update(
            sig,
            ear_conf=ts.closure_ratio,
            mar_conf=ts.yawn_confidence,
            pose_conf=ts.posture_confidence,
        )

        # Ablation V0/V1: bypass the reliability gate (force reliability=1.0).
        # SEVERE exemption is unaffected; this only removes the multiplicative
        # attenuation of the fused score (frozen protocol §3).
        if self.cfg.ablation.reliability_gate_enabled:
            gate_reliability = snap.system_reliability
        else:
            gate_reliability = 1.0

        cnn_verdict = CNNVerdict()
        if (self.enable_cnn and face_detected and self.cnn is not None
                and self.cnn.should_invoke(ts.smoothed_ear,
                                           snap.system_reliability)):
            roi = extract_eye_roi(frame_bgr, results.multi_face_landmarks[0],
                                  w, h,
                                  target_size=self.cfg.cnn_validation.input_size)
            if roi is not None:
                cnn_verdict = self.cnn.validate_eye_state(
                    roi, ts.smoothed_ear,
                    ear_threshold=self.cfg.detection.ear_threshold)

        state = self.state_mgr.update(
            ts, face_detected,
            reliability=gate_reliability,
            alert_suppressed=snap.alert_suppressed,
            landmark_jitter=sig.landmark_jitter,
            frame_brightness=sig.frame_brightness,
            cnn_verdict=cnn_verdict,
            timestamp=timestamp,
        )

        return FrameResult(
            face_detected=face_detected,
            raw_ear=raw_ear,
            raw_mar=raw_mar,
            smoothed_ear=ts.smoothed_ear,
            smoothed_mar=ts.smoothed_mar,
            reliability=snap.system_reliability,
            alert_suppressed=snap.alert_suppressed,
            is_drowsy=ts.is_drowsy,
            is_speaking=ts.is_speaking,
            yawn_confidence=ts.yawn_confidence,
            closure_ratio=ts.closure_ratio,
            fatigue_score=state.fatigue_score,
            status=state.status,
            cnn_invoked=cnn_verdict.invoked,
            # --- Additive alarm-decision exposure (EXP-005), verbatim from state ---
            should_alarm=state.should_alarm,
            alarm_level=state.alarm_level,
            cnn_override_active=state.cnn_override_active,
            alarm_suppressed_actual=state.alert_suppressed,
            face_visible=state.face_visible,
            seconds_since_face_lost=state.seconds_since_face_lost,
        )
