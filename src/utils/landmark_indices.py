"""
MediaPipe Face Mesh Landmark Indices mapping.
This module provides standard landmark groups for eyes, lips, face oval, and irises
which are critical for calculating EAR (Eye Aspect Ratio) and MAR (Mouth Aspect Ratio).
"""

# LEFT EYE Landmarks (MediaPipe indices)
# Contour: [33, 160, 158, 133, 153, 144]
LEFT_EYE_CONTOUR = [33, 160, 158, 133, 153, 144]
LEFT_EYE_TOP_BOTTOM = [159, 145] # Vertical pair for rapid EAR calculation
LEFT_EYE_LEFT_RIGHT = [33, 133]  # Horizontal pair

# RIGHT EYE Landmarks
# Contour: [362, 385, 387, 263, 373, 380]
RIGHT_EYE_CONTOUR = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_TOP_BOTTOM = [386, 374] # Vertical pair
RIGHT_EYE_LEFT_RIGHT = [263, 362]  # Horizontal pair

# MOUTH/LIPS Landmarks
# Inner Lip Contour for Mouth Aspect Ratio (MAR)
LIPS_INNER_CONTOUR = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13]
LIPS_VERTICAL = [13, 14]       # Center top and bottom inner lips
LIPS_HORIZONTAL = [78, 308]    # Corner-to-corner inner lips

# LEFT IRIS Landmarks (MediaPipe v2 refined landmarks)
# Center: 468, Surround: 469, 470, 471, 472
LEFT_IRIS = [468, 469, 470, 471, 472]

# RIGHT IRIS Landmarks
# Center: 473, Surround: 474, 475, 476, 477
RIGHT_IRIS = [473, 474, 475, 476, 477]

# HEAD POSE Landmarks
# Ordered specifically for cv2.solvePnP alignment with a generic 3D face model:
# [Nose Tip, Chin, Left Eye Corner, Right Eye Corner, Left Mouth Corner, Right Mouth Corner]
POSE_LANDMARKS = [1, 152, 33, 263, 61, 291]
