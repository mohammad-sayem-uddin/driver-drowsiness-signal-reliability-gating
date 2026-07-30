"""
Drowsiness Detector — Pure Math Processor
==========================================
Stateless mathematical utility for computing Eye Aspect Ratio (EAR)
and Mouth Aspect Ratio (MAR) from facial landmark coordinates.

This module contains NO state machine logic, NO counters, and NO
thresholds.  All temporal analysis is handled by TemporalAnalyzer.
All threshold values are defined in config.py.

The EAR formula follows Soukupová & Čech (2016):
    EAR = (||P2 - P6|| + ||P3 - P5||) / (2.0 × ||P1 - P4||)

The MAR formula:
    MAR = ||top_lip - bottom_lip|| / ||left_corner - right_corner||

Usage:
    from src.detector import DrowsinessDetector
    detector = DrowsinessDetector()
    ear = detector.calculate_ear(eye_landmarks)
    mar = detector.calculate_mar(mouth_landmarks)
"""

import math


class DrowsinessDetector:
    """
    Stateless math processor for EAR and MAR computation.

    Performs pure 2D geometric calculations on facial landmark
    coordinates. Standardizing to 2D Planar Euclidean geometry across both EAR
    and MAR eliminates metric divergence caused by MediaPipe's uncalibrated z-depth.
    """

    @staticmethod
    def calculate_distance(p1, p2):
        """
        Computes 2D Planar Euclidean distance between two points.

        Using 2D planar coordinates for both EAR and MAR provides higher
        stability and eliminates pitch/yaw dependent ratio distortion caused by
        MediaPipe's uncalibrated relative z-depth estimate.

        Args:
            p1, p2: Points as MediaPipe landmarks or (x, y) tuples.

        Returns:
            float: 2D Euclidean distance.
        """
        if hasattr(p1, 'x'):
            return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def calculate_ear(self, eye_landmarks):
        """
        Calculates the Eye Aspect Ratio (EAR) from 6 landmark points (2D Planar).

        Point layout:
                  P2          P3
                   \\          /
              P1 --==========-- P4
                   /          \\
                  P6          P5

        Args:
            eye_landmarks (list): 6 points ordered as:
                [P1=outer_corner, P2=upper_left, P3=upper_right,
                 P4=inner_corner, P5=lower_right, P6=lower_left]

        Returns:
            float: EAR value (typically 0.0–0.4).
                   Returns 0.0 if input is invalid.
        """
        if len(eye_landmarks) < 6:
            return 0.0

        # Vertical distances (upper lid ↔ lower lid) using 2D geometry
        v1 = self.calculate_distance(eye_landmarks[1], eye_landmarks[5])
        v2 = self.calculate_distance(eye_landmarks[2], eye_landmarks[4])

        # Horizontal distance (corner ↔ corner)
        h = self.calculate_distance(eye_landmarks[0], eye_landmarks[3])

        if h == 0.0:
            return 0.0

        return (v1 + v2) / (2.0 * h)

    @staticmethod
    def _distance_2d(p1, p2):
        """
        Computes 2D Euclidean distance. Standardized alias for calculate_distance.
        """
        if hasattr(p1, 'x'):
            return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def calculate_mar(self, mouth_landmarks):
        """
        Calculates Mouth Aspect Ratio (MAR) from 4 landmark points (2D Planar).

        Layout: [left_corner, top_inner_lip, right_corner, bottom_inner_lip]
        Formula: MAR = ||top - bottom||₂ᴅ / ||left - right||₂ᴅ

        Args:
            mouth_landmarks (list): 4 points ordered as:
                [left_corner, top_inner_lip, right_corner, bottom_inner_lip]

        Returns:
            float: MAR value (typically 0.0–1.0).
                   Returns 0.0 if input is invalid.
        """
        if len(mouth_landmarks) < 4:
            return 0.0

        v = self._distance_2d(mouth_landmarks[1], mouth_landmarks[3])
        h = self._distance_2d(mouth_landmarks[0], mouth_landmarks[2])

        if h == 0.0:
            return 0.0

        return v / h


class CalibrationManager:
    """
    Dynamic Subject Baseline Normalization Engine.

    Collects initial neutral frame metrics to establish subject-specific
    baselines (EAR_base, MAR_base), eliminating subject-dependent geometric bias.
    """

    def __init__(self, target_samples: int = 90):
        self.target_samples = target_samples
        self.ear_samples = []
        self.mar_samples = []
        self.is_calibrated = False
        self.ear_baseline = 0.30
        self.mar_baseline = 0.15

    def add_sample(self, ear: float, mar: float) -> bool:
        """
        Adds a neutral frame sample to calibration buffer.

        Returns True when calibration becomes complete.
        """
        if self.is_calibrated:
            return True

        if ear > 0.0:
            self.ear_samples.append(ear)
        if mar > 0.0:
            self.mar_samples.append(mar)

        if len(self.ear_samples) >= self.target_samples:
            self.ear_baseline = max(0.15, float(sum(self.ear_samples) / len(self.ear_samples)))
            self.mar_baseline = max(0.05, float(sum(self.mar_samples) / len(self.mar_samples)))
            self.is_calibrated = True
            return True

        return False

    def normalize_ear(self, ear: float) -> float:
        """Normalizes EAR relative to calibrated subject baseline."""
        if not self.is_calibrated or self.ear_baseline == 0:
            return ear
        return ear / self.ear_baseline

    def normalize_mar(self, mar: float) -> float:
        """Normalizes MAR relative to calibrated subject baseline."""
        if not self.is_calibrated or self.mar_baseline == 0:
            return mar
        return mar / self.mar_baseline

