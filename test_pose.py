from src.pose_estimator import HeadPoseEstimator
import math

est = HeadPoseEstimator(1280, 720)

# Simulate face looking straight
straight_pts = [
    (640, 360), # Nose
    (640, 460), # Chin
    (540, 260), # L Eye
    (740, 260), # R Eye
    (560, 400), # L Mouth
    (720, 400)  # R Mouth
]
p, y, r = est.estimate_pose(straight_pts)
print(f"Straight: Pitch={p:.1f}, Yaw={y:.1f}, Roll={r:.1f}")

# Simulate chin dropping (Nose stays, Chin moves up slightly relative to eyes, or eyes move up)
down_pts = [
    (640, 360), # Nose
    (640, 420), # Chin (moved up relative to center, nose dropped)
    (540, 220), # L Eye (moved up)
    (740, 220), # R Eye (moved up)
    (560, 380), # L Mouth
    (720, 380)  # R Mouth
]
p2, y2, r2 = est.estimate_pose(down_pts)
print(f"Look Down: Pitch={p2:.1f}, Yaw={y2:.1f}, Roll={r2:.1f}")

