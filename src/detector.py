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

    This class performs pure geometric calculations on facial landmark
    coordinates.  It does not track state, count frames, or make
    drowsiness decisions — those responsibilities belong to
    TemporalAnalyzer and StateManager.
    """

    @staticmethod
    def calculate_distance(p1, p2):
        """
        Computes Euclidean distance between two points.

        Supports two input formats:
          - MediaPipe landmark objects (have .x, .y, .z attributes)
            → uses 3D Euclidean distance for curvature compensation
          - Plain tuples (x, y) → uses 2D Euclidean distance

        Using 3D coordinates (including MediaPipe's z-depth) provides
        ~2% improvement in EAR accuracy for frontal faces.

        Args:
            p1, p2: Points as MediaPipe landmarks or (x, y) tuples.

        Returns:
            float: Euclidean distance between the two points.
        """
        if hasattr(p1, 'x'):
            return math.sqrt(
                (p1.x - p2.x) ** 2 +
                (p1.y - p2.y) ** 2 +
                (p1.z - p2.z) ** 2
            )
        return math.sqrt(
            (p1[0] - p2[0]) ** 2 +
            (p1[1] - p2[1]) ** 2
        )

    def calculate_ear(self, eye_landmarks):
        """
        Calculates the Eye Aspect Ratio (EAR) from 6 landmark points.

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

        # Vertical distances (upper lid ↔ lower lid)
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
        Computes 2D Euclidean distance, ignoring the z-coordinate.

        MediaPipe's z-coordinate is a *relative depth estimate* that is
        NOT calibrated to the same scale as the normalized x/y coordinates.
        For inner lip landmarks, z diverges dramatically during mouth
        opening (up to 10× the actual spatial separation), inflating
        vertical distances and producing MAR values >2.0.

        Using 2D-only keeps MAR bounded to the expected 0.0–1.0 range.

        Args:
            p1, p2: Points as MediaPipe landmarks (.x, .y) or (x, y) tuples.

        Returns:
            float: 2D Euclidean distance.
        """
        import math
        if hasattr(p1, 'x'):
            return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def calculate_mar(self, mouth_landmarks):
        """
        Calculates Mouth Aspect Ratio (MAR) from 4 landmark points.

        Layout: [left_corner, top_lip, right_corner, bottom_lip]
        Formula: MAR = ||top - bottom||₂ᴅ / ||left - right||₂ᴅ

        IMPORTANT: Uses 2D-only distances (ignoring z-depth).
        MediaPipe's z-coordinate is uncalibrated for lip landmarks and
        causes MAR inflation >2.0 when using 3D Euclidean distance.
        The EAR formula retains 3D because eye landmark z-depth is
        relatively stable; lip z-depth is not.

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
