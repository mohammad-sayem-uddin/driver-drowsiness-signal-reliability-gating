#!/usr/bin/env python3
"""
Eye Data Collection Tool
=========================
Webcam-based utility for collecting labeled eye-crop images to train
the MicroEyeNet Tiny CNN.

Uses the existing MediaPipe Face Mesh pipeline to detect the eye region,
crops and preprocesses it, and saves labeled images on keypress.

Controls:
    'o' → Save current eye crop as OPEN  (data/eyes/open/)
    'c' → Save current eye crop as CLOSED (data/eyes/closed/)
    'q' / ESC → Quit

Workflow:
    1. Run: python3 tools/collect_eye_data.py
    2. Keep eyes OPEN, press 'o' repeatedly (~500 times).
    3. Close eyes / simulate drowsiness, press 'c' repeatedly.
    4. Target: ~500 images per class (feasible in ~30 minutes).

The saved images are 24×24 grayscale PNGs — ready for training.
"""

import os
import sys
import time
import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import mediapipe as mp
except ImportError:
    print("[Error] mediapipe not installed. pip install mediapipe")
    sys.exit(1)

# Eye landmark indices for bounding box
LEFT_EYE_IDX = [33, 133, 160, 159, 158, 144, 145, 153]
RIGHT_EYE_IDX = [362, 263, 387, 386, 385, 373, 374, 380]

# Output directories
OPEN_DIR = os.path.join("data", "eyes", "open")
CLOSED_DIR = os.path.join("data", "eyes", "closed")
TARGET_SIZE = 24
MARGIN = 8


def extract_eye_crop(frame, face_lm, img_w, img_h):
    """Extract a merged bilateral eye ROI as a grayscale crop."""
    all_indices = LEFT_EYE_IDX + RIGHT_EYE_IDX
    xs = [int(face_lm.landmark[i].x * img_w) for i in all_indices]
    ys = [int(face_lm.landmark[i].y * img_h) for i in all_indices]

    x_min = max(0, min(xs) - MARGIN)
    x_max = min(img_w, max(xs) + MARGIN)
    y_min = max(0, min(ys) - MARGIN)
    y_max = min(img_h, max(ys) + MARGIN)

    if x_max <= x_min or y_max <= y_min:
        return None

    crop = frame[y_min:y_max, x_min:x_max]
    if crop.size == 0:
        return None

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)
    return resized


def count_images(directory):
    """Count PNG images in a directory."""
    if not os.path.isdir(directory):
        return 0
    return len([f for f in os.listdir(directory) if f.endswith(".png")])


def main():
    print("=" * 60)
    print("  EYE DATA COLLECTION TOOL — MicroEyeNet Training Data")
    print("=" * 60)
    print(f"  OPEN  dir: {OPEN_DIR}")
    print(f"  CLOSED dir: {CLOSED_DIR}")
    print(f"  Target size: {TARGET_SIZE}×{TARGET_SIZE} grayscale")
    print()
    print("  Controls:")
    print("    'o' → Save as OPEN eye")
    print("    'c' → Save as CLOSED eye")
    print("    'q' / ESC → Quit")
    print("=" * 60)

    # Ensure output directories exist
    os.makedirs(OPEN_DIR, exist_ok=True)
    os.makedirs(CLOSED_DIR, exist_ok=True)

    # Initialize MediaPipe
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Error] Cannot access webcam.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    open_count = count_images(OPEN_DIR)
    closed_count = count_images(CLOSED_DIR)
    status_msg = ""
    status_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Process face mesh
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        eye_crop = None
        if results.multi_face_landmarks:
            face_lm = results.multi_face_landmarks[0]
            eye_crop = extract_eye_crop(frame, face_lm, w, h)

            # Draw eye landmarks for visual feedback
            for idx in LEFT_EYE_IDX + RIGHT_EYE_IDX:
                lm = face_lm.landmark[idx]
                px = int(lm.x * w)
                py = int(lm.y * h)
                cv2.circle(frame, (px, py), 2, (0, 255, 0), -1)

        # Display preview of the crop
        if eye_crop is not None:
            # Scale up for visibility
            preview = cv2.resize(eye_crop, (120, 120), interpolation=cv2.INTER_NEAREST)
            frame[10:130, w - 130:w - 10] = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(frame, (w - 130, 10), (w - 10, 130), (0, 255, 0), 2)
            cv2.putText(frame, "Eye ROI", (w - 125, 145),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # HUD
        cv2.rectangle(frame, (0, 0), (w, 40), (25, 25, 25), -1)
        cv2.putText(frame, f"OPEN: {open_count} | CLOSED: {closed_count}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if status_msg and (time.time() - status_time) < 1.5:
            cv2.putText(frame, status_msg, (10, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(frame, "'o'=Open  'c'=Closed  'q'=Quit", (10, h - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

        cv2.imshow("Eye Data Collection", frame)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), 27):
            break

        if eye_crop is not None:
            if key == ord('o'):
                filename = os.path.join(OPEN_DIR, f"open_{open_count:05d}.png")
                cv2.imwrite(filename, eye_crop)
                open_count += 1
                status_msg = f"Saved OPEN #{open_count}"
                status_time = time.time()

            elif key == ord('c'):
                filename = os.path.join(CLOSED_DIR, f"closed_{closed_count:05d}.png")
                cv2.imwrite(filename, eye_crop)
                closed_count += 1
                status_msg = f"Saved CLOSED #{closed_count}"
                status_time = time.time()

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()

    print(f"\n{'=' * 60}")
    print(f"  Collection complete!")
    print(f"  OPEN images:   {open_count}")
    print(f"  CLOSED images: {closed_count}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
