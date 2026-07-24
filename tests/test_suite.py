"""
Comprehensive Unit Test Suite for Driver Drowsiness Detection System (v3.1 Baseline)
=====================================================================================
Verifies mathematical correctness, temporal determinism, signal quality calculations,
and state machine transitions across all core package modules.

Execution:
    python3 -m unittest tests/test_suite.py
"""

import unittest
import time
import math
from unittest.mock import MagicMock

from src.config import SystemConfig, DetectionConfig, FusionConfig, RobustnessConfig
from src.detector import DrowsinessDetector
from src.temporal_analyzer import TemporalAnalyzer, EMASmoother, TemporalState
from src.robustness import RobustnessGuard, SignalQuality, RobustnessSnapshot
from src.fatigue_fusion import FatigueFusionEngine, FatigueSeverity, FusionSnapshot
from src.state_manager import StateManager, DriverStatus
from src.cnn_validator import CNNValidator, CNNVerdict


class DummyLandmark:
    """Mock MediaPipe landmark object with x, y, z attributes."""
    def __init__(self, x: float, y: float, z: float = 0.0):
        self.x = x
        self.y = y
        self.z = z


class TestConfiguration(unittest.TestCase):
    """Test central configuration defaults and dataclass consistency."""

    def test_default_config_instantiation(self):
        cfg = SystemConfig()
        self.assertEqual(cfg.detection.ear_threshold, 0.21)
        self.assertEqual(cfg.detection.ear_hysteresis, 0.03)
        self.assertEqual(cfg.detection.mar_threshold, 0.55)
        self.assertEqual(cfg.temporal.eye_closure_duration, 1.0)
        self.assertEqual(cfg.fusion.ear_weight, 0.45)
        self.assertEqual(cfg.fusion.pose_weight, 0.30)
        self.assertEqual(cfg.fusion.mar_weight, 0.25)

    def test_open_threshold_calculation(self):
        cfg = SystemConfig()
        open_thresh = cfg.detection.ear_threshold + cfg.detection.ear_hysteresis
        self.assertAlmostEqual(open_thresh, 0.24, places=4)


class TestDrowsinessDetector(unittest.TestCase):
    """Test pure math functions in detector.py (EAR & MAR calculations)."""

    def setUp(self):
        self.detector = DrowsinessDetector()

    def test_euclidean_distance_2d(self):
        p1 = (0.0, 0.0)
        p2 = (3.0, 4.0)
        dist = self.detector.calculate_distance(p1, p2)
        self.assertAlmostEqual(dist, 5.0, places=4)

    def test_euclidean_distance_3d_landmark(self):
        p1 = DummyLandmark(0.0, 0.0, 0.0)
        p2 = DummyLandmark(1.0, 2.0, 2.0)
        dist = self.detector.calculate_distance(p1, p2)
        self.assertAlmostEqual(dist, 3.0, places=4)

    def test_ear_open_eye(self):
        # 6 landmark layout: [outer, upper_left, upper_right, inner, lower_right, lower_left]
        eye_open = [
            DummyLandmark(0.0, 0.5, 0.0),   # P1
            DummyLandmark(0.2, 0.7, 0.0),   # P2
            DummyLandmark(0.4, 0.7, 0.0),   # P3
            DummyLandmark(0.6, 0.5, 0.0),   # P4
            DummyLandmark(0.4, 0.3, 0.0),   # P5
            DummyLandmark(0.2, 0.3, 0.0),   # P6
        ]
        ear = self.detector.calculate_ear(eye_open)
        self.assertGreater(ear, 0.25)

    def test_mar_2d_depth_divergence_mitigation(self):
        # Verify 2D MAR distance ignores z-depth distortion
        mouth_landmarks = [
            DummyLandmark(0.2, 0.5, 0.0),   # Left corner
            DummyLandmark(0.5, 0.7, 5.0),   # Top lip (extreme z depth)
            DummyLandmark(0.8, 0.5, 0.0),   # Right corner
            DummyLandmark(0.5, 0.3, -5.0),  # Bottom lip (extreme z depth)
        ]
        mar = self.detector.calculate_mar(mouth_landmarks)
        # Vertical 2D dist = 0.4, Horizontal 2D dist = 0.6 => MAR = 0.4 / 0.6 = 0.666...
        self.assertAlmostEqual(mar, 0.6666667, places=4)
        self.assertLess(mar, 1.0)


class TestTemporalAnalyzer(unittest.TestCase):
    """Test wall-clock duration tracking and speech/nod filtering."""

    def setUp(self):
        self.cfg = SystemConfig()
        self.analyzer = TemporalAnalyzer(self.cfg)

    def test_ema_smoother(self):
        smoother = EMASmoother(alpha=0.5)
        v1 = smoother.update(10.0)
        self.assertEqual(v1, 10.0)
        v2 = smoother.update(20.0)
        self.assertEqual(v2, 15.0)

    def test_eye_closure_wall_clock_drowsiness(self):
        # Test EyeClosureAnalyzer directly with deterministic wall-clock timestamps
        analyzer = self.analyzer.eye_analyzer
        t0 = time.monotonic()

        # Eye initially open
        smoothed, closed, dur, drowsy, blinks, ratio, hist = analyzer.update(raw_ear=0.30, timestamp=t0)
        self.assertFalse(closed)
        self.assertFalse(drowsy)

        # Eye closes at t0 + 0.1s (takes 2-3 frames for EMA smoother α=0.3 to drop below 0.21)
        t1 = t0 + 0.1
        for _ in range(5):
            smoothed, closed, dur, drowsy, blinks, ratio, hist = analyzer.update(raw_ear=0.10, timestamp=t1)
        self.assertTrue(closed)
        self.assertFalse(drowsy)

        # Eye stays closed past closure threshold (t0 + 1.2s > 1.0s eye_closure_duration)
        t2 = t0 + 1.2
        smoothed, closed, dur, drowsy, blinks, ratio, hist = analyzer.update(raw_ear=0.10, timestamp=t2)
        self.assertTrue(closed)
        self.assertTrue(drowsy)
        self.assertGreaterEqual(dur, 1.0)

    def test_speech_jitter_filtering(self):
        # High frequency MAR fluctuation (speech artifact)
        mars = [0.2, 0.6, 0.2, 0.7, 0.1, 0.6, 0.2, 0.6]
        for mar in mars:
            state = self.analyzer.update(raw_ear=0.30, raw_mar=mar)

        self.assertTrue(state.is_speaking)
        self.assertLess(state.yawn_confidence, 0.3)


class TestRobustnessGuard(unittest.TestCase):
    """Test RobustnessGuard signal quality and geometric mean reliability score."""

    def setUp(self):
        self.cfg = SystemConfig()
        self.guard = RobustnessGuard(self.cfg)

    def test_perfect_signal_reliability(self):
        sq = SignalQuality(
            landmark_jitter=1.0,      # Perfect (<2.0px)
            frame_brightness=120.0,   # Ideal (60–200)
            tracking_confidence=1.0,  # Max confidence
            face_visible=True
        )
        snap = self.guard.update(sq, ear_conf=0.0, mar_conf=0.0, pose_conf=0.0)
        self.assertGreaterEqual(snap.system_reliability, 0.8)
        self.assertFalse(snap.alert_suppressed)

    def test_degraded_signal_attenuation(self):
        sq = SignalQuality(
            landmark_jitter=10.0,     # High jitter
            frame_brightness=20.0,    # Very dark
            tracking_confidence=0.3,
            face_visible=True
        )
        # Allow EMA (alpha=0.2) to converge over 15 frames
        for _ in range(15):
            snap = self.guard.update(sq, ear_conf=0.5, mar_conf=0.5, pose_conf=0.5)
        self.assertLess(snap.system_reliability, 0.5)

    def test_learned_reliability_equivalence(self):
        from src.robustness import LearnedReliabilityEstimator
        estimator = LearnedReliabilityEstimator(weights=(0.35, 0.25, 0.20, 0.20), bias=0.0, temperature=1.0)
        score = estimator.estimate(stability=0.8, brightness=0.9, tracking=1.0, consistency=0.7)
        expected_geom = (0.8 ** 0.35) * (0.9 ** 0.25) * (1.0 ** 0.20) * (0.7 ** 0.20)
        self.assertAlmostEqual(score, expected_geom, places=4)


class TestFatigueFusionEngine(unittest.TestCase):
    """Test multi-cue weighted fusion score and agreement multipliers."""

    def setUp(self):
        self.cfg = SystemConfig()
        self.engine = FatigueFusionEngine(self.cfg)

    def test_weighted_sum_and_agreement_bonus(self):
        ts = TemporalState(
            closure_ratio=0.8,
            is_drowsy=True,
            yawn_confidence=0.8,
            posture_confidence=0.8
        )
        # 3 active cues -> agreement bonus 1.5x
        snap = self.engine.update(ts, reliability=1.0)
        self.assertEqual(snap.active_cue_count, 3)
        self.assertEqual(snap.agreement_multiplier, 1.5)
        self.assertGreater(snap.fatigue_score, 0.0)


class TestStateManager(unittest.TestCase):
    """Test 5-state machine transitions and face loss safety escalation."""

    def setUp(self):
        self.cfg = SystemConfig()
        self.mgr = StateManager(self.cfg)

    def test_alert_state_default(self):
        ts = TemporalState(is_drowsy=False)
        state = self.mgr.update(ts, face_detected=True, reliability=1.0)
        self.assertEqual(state.status, DriverStatus.ALERT)

    def test_face_loss_critical_escalation(self):
        # Trigger severe fatigue first
        ts = TemporalState(is_drowsy=True, closure_ratio=1.0)
        for _ in range(20):
            state = self.mgr.update(ts, face_detected=True, reliability=1.0)

        # Suddenly lose face during active drowsiness episode
        state_lost = self.mgr.update(ts, face_detected=False, reliability=1.0)
        self.assertEqual(state_lost.status, DriverStatus.FACE_LOST_CRITICAL)


class TestCNNValidatorFallback(unittest.TestCase):
    """Test graceful fallback behavior when TFLite model is missing."""

    def setUp(self):
        self.cfg = SystemConfig()
        self.validator = CNNValidator(self.cfg)

    def test_missing_model_fallback(self):
        # models/eye_state_model.tflite is missing
        self.assertFalse(self.validator.is_available)
        should_invoke = self.validator.should_invoke(smoothed_ear=0.20, system_reliability=1.0)
        self.assertFalse(should_invoke)
        verdict = self.validator.validate_eye_state(eye_roi=None, smoothed_ear=0.20)
        self.assertFalse(verdict.invoked)


if __name__ == "__main__":
    unittest.main()
