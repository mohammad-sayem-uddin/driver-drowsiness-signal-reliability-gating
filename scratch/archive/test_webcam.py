#!/usr/bin/env python3
"""
Driver Drowsiness Detection System - Environment Verification Script
This script verifies that your webcam, OpenCV, MediaPipe, NumPy, and Pygame
are all correctly installed and functioning in your development environment.
"""

import sys
import time
import math

print("=" * 60)
print("  DRIVER DROWSINESS DETECTION SYSTEM - DIAGNOSTIC UTILITY  ")
print("=" * 60)

# 1. Verify Python & Core Libraries
print("[1/5] Checking Core Dependencies...")

libs_status = {
    "Python": sys.version.split()[0],
    "NumPy": "Not Installed",
    "OpenCV (cv2)": "Not Installed",
    "MediaPipe": "Not Installed",
    "Pygame": "Not Installed"
}

# Check NumPy
try:
    import numpy as np
    libs_status["NumPy"] = np.__version__
    print("  ✓ NumPy is available (v{})".format(np.__version__))
except ImportError:
    print("  ✗ NumPy is NOT installed.")

# Check OpenCV
try:
    import cv2
    libs_status["OpenCV (cv2)"] = cv2.__version__
    print("  ✓ OpenCV is available (v{})".format(cv2.__version__))
except ImportError:
    print("  ✗ OpenCV is NOT installed.")

# Check MediaPipe
try:
    import mediapipe as mp
    libs_status["MediaPipe"] = mp.__version__
    print("  ✓ MediaPipe is available (v{})".format(mp.__version__))
except ImportError:
    print("  ✗ MediaPipe is NOT installed.")

# Check Pygame
try:
    import pygame
    libs_status["Pygame"] = pygame.__version__
    print("  ✓ Pygame is available (v{})".format(pygame.__version__))
except ImportError:
    print("  ✗ Pygame is NOT installed.")

print("-" * 60)

# Check if OpenCV is available before continuing to camera test
if "Not Installed" in libs_status["OpenCV (cv2)"]:
    print("CRITICAL: OpenCV is not installed. Cannot run camera test.")
    print("Please install requirements first using: pip install -r requirements.txt")
    sys.exit(1)

# Initialize Pygame Mixer if available (for alert sound test)
pygame_audio_works = False
if "Not Installed" not in libs_status["Pygame"]:
    try:
        pygame.mixer.init()
        pygame_audio_works = True
        print("  ✓ Pygame audio mixer initialized successfully.")
    except Exception as e:
        print("  ✗ Pygame audio mixer initialization failed: {}".format(e))

# Setup MediaPipe Face Mesh if available
mp_face_mesh = None
face_mesh = None
if "Not Installed" not in libs_status["MediaPipe"]:
    try:
        mp_face_mesh = mp.solutions.face_mesh
        # Use single-face tracking for lightweight verification
        face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("  ✓ MediaPipe Face Mesh initialized successfully.")
    except Exception as e:
        print("  ✗ MediaPipe Face Mesh initialization failed: {}".format(e))

print("-" * 60)
print("[2/5] Initializing Webcam Test...")
print("Attempting to open default camera (ID 0)...")
print("NOTE: On macOS, your terminal or VS Code must have Camera permissions allowed.")
print("      A system popup should prompt you for permission.")

# Try to capture from the default camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("\n[!] ERROR: Could not open camera (ID 0).")
    print("Common Mac issues:")
    print("  1. System Preferences -> Privacy & Security -> Camera -> Enable VS Code / Terminal.")
    print("  2. If another application (Zoom, FaceTime) is using the webcam, close it.")
    print("  3. Try changing the camera index in this script (e.g., cv2.VideoCapture(1)).")
    sys.exit(1)

# Get camera properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
print("  ✓ Camera opened successfully!")
print("  ✓ Resolution: {} x {}".format(width, height))
print("  ✓ Reported FPS: {}".format(fps))
print("\nPress 'q' or 'ESC' in the video window to EXIT the test.")
print("Press 's' to play a sample diagnostic sound (tests pygame).")
print("=" * 60)

# Create a window
window_name = "Driver Drowsiness System - Environment Verification"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 800, 600)

frame_count = 0
start_time = time.time()
display_fps = 0.0

# Generate a simple beep sound sample in-memory if pygame works
sound_channel = None
sound_object = None
if pygame_audio_works:
    try:
        # Create a simple 440Hz sine wave beep in pygame
        import array
        sample_rate = 44100
        duration = 0.3
        frequency = 880.0 # 880 Hz beep
        num_samples = int(sample_rate * duration)
        
        # Formulate a sine wave sample array
        buf = array.array('h', [0] * num_samples)
        for i in range(num_samples):
            t = i / sample_rate
            val = int(32767 * math.sin(2 * math.pi * frequency * t))
            buf[i] = val
            
        sound_buffer = bytes(buf)
        sound_object = pygame.mixer.Sound(buffer=sound_buffer)
        print("  ✓ Generated diagnostic beep sound in memory.")
    except Exception as e:
        print("  ✗ Could not generate diagnostic sound: {}".format(e))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("[!] Failed to grab frame from camera.")
        break
        
    frame_count += 1
    # Flip the frame horizontally for a more natural mirror view
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    # Calculate operational FPS
    elapsed = time.time() - start_time
    if elapsed >= 1.0:
        display_fps = frame_count / elapsed
        frame_count = 0
        start_time = time.time()
        
    # Process with MediaPipe Face Mesh if available
    face_detected = False
    iris_landmarks_detected = False
    
    if face_mesh is not None:
        # MediaPipe requires RGB format, OpenCV uses BGR
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            face_detected = True
            for face_landmarks in results.multi_face_landmarks:
                # Draw standard facial landmarks (just a subset for clean diagnostic view)
                # Let's draw dots on eyes and iris landmarks to demonstrate precision
                # Left eye, Right eye, and Iris index ranges in MediaPipe Face Mesh:
                # Left Iris: 468, 469, 470, 471, 472
                # Right Iris: 473, 474, 475, 476, 477
                
                # Draw facial landmark points
                for idx, lm in enumerate(face_landmarks.landmark):
                    # Only draw a subset of points (e.g., eye contours & lips) to keep it clean
                    # Left Eye: 33, 133, 159, 145, etc. Right Eye: 362, 263, 386, 374, etc.
                    # We will draw points that have index divisble by 10 to show structure
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    
                    if idx in [468, 473]: # Iris center
                        cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1) # Red for iris center
                        iris_landmarks_detected = True
                    elif idx % 8 == 0: # Sparse mesh
                        cv2.circle(frame, (cx, cy), 1, (0, 255, 0), -1) # Green for mesh
                        
    # --- RENDER MODERN DASHBOARD OVERLAY ---
    # Top overlay bar for environment status
    cv2.rectangle(frame, (0, 0), (w, 105), (30, 30, 30), -1)
    cv2.line(frame, (0, 105), (w, 105), (0, 255, 0), 2)
    
    # Title
    cv2.putText(frame, "DRIVER DROWSINESS DETECTION - DIAGNOSTIC UTILITY", (15, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
    # Dependency Checkboxes
    def draw_status_dot(img, text, status_ok, pos):
        color = (0, 255, 0) if status_ok else (0, 0, 255)
        cv2.circle(img, (pos[0], pos[1] - 5), 6, color, -1)
        cv2.putText(img, text, (pos[0] + 15, pos[1]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
                    
    draw_status_dot(frame, "OpenCV (v{})".format(libs_status["OpenCV (cv2)"]), True, (15, 55))
    draw_status_dot(frame, "NumPy (v{})".format(libs_status["NumPy"]), "Not" not in libs_status["NumPy"], (160, 55))
    draw_status_dot(frame, "MediaPipe (v{})".format(libs_status["MediaPipe"]), "Not" not in libs_status["MediaPipe"], (320, 55))
    draw_status_dot(frame, "Pygame (v{})".format(libs_status["Pygame"]), "Not" not in libs_status["Pygame"], (500, 55))

    # Real-time Stats
    draw_status_dot(frame, "Camera Stream ({}x{})".format(w, h), True, (15, 85))
    draw_status_dot(frame, "MediaPipe Face Tracking", face_detected, (220, 85))
    draw_status_dot(frame, "Iris Mesh (High Precision)", iris_landmarks_detected, (470, 85))
    
    # Frame Rate
    cv2.rectangle(frame, (w - 110, 15), (w - 15, 45), (10, 10, 10), -1)
    cv2.putText(frame, "FPS: {:.1f}".format(display_fps), (w - 100, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                
    # Quick Instructions on Screen Footer
    cv2.rectangle(frame, (0, h - 35), (w, h), (20, 20, 20), -1)
    cv2.putText(frame, "Press 's' to trigger Alarm Sound Test  |  Press 'q' or 'ESC' to Exit Dashboard", 
                (15, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                
    # Show Visual Alarm Overlay if sound is testing
    if sound_channel and sound_channel.get_busy():
        # Flash a red warning border
        cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 10)
        cv2.putText(frame, "!!! ALARM TESTING !!!", (w // 2 - 140, h // 2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

    # Render frame
    cv2.imshow(window_name, frame)
    
    # Process key inputs
    key = cv2.waitKey(1) & 0xFF
    if key in [ord('q'), 27]: # 'q' or ESC
        break
    elif key == ord('s'): # Sound trigger
        if pygame_audio_works and sound_object:
            print("[Diagnostic] Playing alert sound...")
            sound_channel = sound_object.play()
        else:
            print("[Diagnostic] Sound test skipped (pygame audio failed to initialize).")

# Clean up
print("\n[5/5] Releasing resources...")
cap.release()
cv2.destroyAllWindows()
if face_mesh is not None:
    face_mesh.close()
print("Verification complete. Thank you!")
print("=" * 60)
