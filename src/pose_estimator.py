"""
Head Pose Estimator — 3D Posture Extraction
============================================
Lightweight extraction of Pitch, Yaw, and Roll Euler angles using
a standard 3D generic facial model and cv2.solvePnP.

This module is stateless and only performs matrix transformations.
Temporal analysis (nodding detection) is handled by TemporalAnalyzer.

Usage:
    from src.pose_estimator import HeadPoseEstimator
    pose_estimator = HeadPoseEstimator(img_w, img_h)
    pitch, yaw, roll = pose_estimator.estimate_pose(landmarks)
"""

import cv2
import numpy as np

class HeadPoseEstimator:
    """
    Extracts Euler angles from 2D facial landmarks by registering them
    against a static 3D generic facial model.
    """
    def __init__(self, img_w: int, img_h: int):
        self.img_w = img_w
        self.img_h = img_h

        # Generic 3D face model points (in arbitrary world units)
        # Coordinates: X (right), Y (down), Z (forward)
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, 330.0, -65.0),         # Chin
            (-225.0, -170.0, -135.0),    # Left eye left corner
            (225.0, -170.0, -135.0),     # Right eye right corner
            (-150.0, 150.0, -125.0),     # Left Mouth corner
            (150.0, 150.0, -125.0)       # Right mouth corner
        ], dtype=np.float64)

        # Approximate camera matrix based on frame dimensions
        focal_length = self.img_w
        center = (self.img_w / 2, self.img_h / 2)
        self.camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)

        # Assuming no significant lens distortion for webcam
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        # State cache for visualization
        self.last_rvec = None
        self.last_tvec = None

    def estimate_pose(self, image_points_2d):
        """
        Calculates Pitch, Yaw, and Roll.

        Args:
            image_points_2d (list of tuples): [(x,y), ...] for the 6 POSE_LANDMARKS.

        Returns:
            tuple: (pitch, yaw, roll) in degrees.
                   Negative Pitch = Downward tilt (nodding).
                   Positive Pitch = Upward tilt.
        """
        if len(image_points_2d) != 6:
            return 0.0, 0.0, 0.0

        image_points = np.array(image_points_2d, dtype=np.float64)

        # Solve Perspective-n-Point
        success, rvec, tvec = cv2.solvePnP(
            self.model_points,
            image_points,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return 0.0, 0.0, 0.0

        self.last_rvec = rvec
        self.last_tvec = tvec

        # Get rotation matrix
        rmat, _ = cv2.Rodrigues(rvec)

        # Extract Euler angles
        pose_mat = cv2.hconcat((rmat, tvec))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)

        pitch = float(euler_angles[0])
        yaw = float(euler_angles[1])
        roll = float(euler_angles[2])

        # Adjust pitch coordinate system so Negative = Downward
        # Depending on decomposeProjectionMatrix implementation, we might need to negate.
        # Standard decomposeProjectionMatrix: positive pitch is usually tilting down,
        # but let's invert it if needed to match our config preference.
        # Wait, if model Y is up (-330), looking down means nose moves down (pos Y in image).
        # So we explicitly invert pitch so chin down = negative.
        pitch = -pitch

        return pitch, yaw, roll

    def get_projection_axes(self, line_length=300.0):
        """
        Projects 3D axes (X, Y, Z) onto the 2D image for visualization.

        Returns:
            tuple: (nose_end_point2D, x_axis2D, y_axis2D, z_axis2D)
        """
        if self.last_rvec is None or self.last_tvec is None:
            return None

        # 3D points of the axes (origin + X, Y, Z directions)
        axes_3d = np.float32([
            [0, 0, 0],                 # Nose tip (origin)
            [line_length, 0, 0],       # X-axis (Right)
            [0, -line_length, 0],      # Y-axis (Up/Down)
            [0, 0, line_length]        # Z-axis (Forward)
        ])

        # Project 3D points to 2D image plane
        points_2d, _ = cv2.projectPoints(
            axes_3d,
            self.last_rvec,
            self.last_tvec,
            self.camera_matrix,
            self.dist_coeffs
        )

        p2d = points_2d.reshape(-1, 2).astype(int)
        
        # Origin, X (red), Y (green), Z (blue)
        return tuple(p2d[0]), tuple(p2d[1]), tuple(p2d[2]), tuple(p2d[3])
