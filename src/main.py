#!/usr/bin/env python3
"""
Driver Drowsiness Detection System — Main Application Entry Point
===================================================================
Real-time driver drowsiness and yawn monitoring using MediaPipe Face Mesh,
EAR/MAR analysis, and time-based temporal detection.

Stabilized Architecture (v2.0):
    config.py            → Centralized thresholds (single source of truth)
    temporal_analyzer.py → FPS-independent wall-clock detection
    state_manager.py     → Face-loss safety + unified state machine
    alarm_controller.py  → Persistent alarms with anti-flicker logic
    detector.py          → Pure math (EAR/MAR computation)

Run:
    python3 -m src.main
"""

import sys
import time
import cv2

# ─── Local package imports ────────────────────────────────────────────
try:
    from src.config import SystemConfig
    from src.detector import DrowsinessDetector
    from src.temporal_analyzer import TemporalAnalyzer
    from src.state_manager import StateManager, DriverStatus
    from src.alarm_controller import AlarmController
    from src.pose_estimator import HeadPoseEstimator
    from src.robustness import RobustnessGuard, SignalQuality
    from src.camera_async import CameraAsync
    from src.cnn_validator import CNNValidator, CNNVerdict, extract_eye_roi
    import src.utils.landmark_indices as landmarks
except ImportError as e:
    print(f"[Error] Failed to import internal modules: {e}")
    print("        Ensure you run from the project root: python3 -m src.main")
    sys.exit(1)

# ─── External dependencies ───────────────────────────────────────────
try:
    import mediapipe as mp
    import numpy as np
except ImportError as e:
    print(f"[Error] Missing external dependencies: {e}")
    print("        pip install -r requirements.txt")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════
# Utility
# ═══════════════════════════════════════════════════════════════════════

def get_eye_coords(landmarks_list, indices, img_w, img_h):
    """Extracts landmark indices as pixel-coordinate tuples."""
    return [
        (int(landmarks_list[idx].x * img_w), int(landmarks_list[idx].y * img_h))
        for idx in indices
    ]


# ═══════════════════════════════════════════════════════════════════════
# HUD Colors (BGR)
# ═══════════════════════════════════════════════════════════════════════

_GREEN = (0, 255, 0)
_RED = (0, 0, 255)
_ORANGE = (0, 165, 255)
_YELLOW = (0, 255, 255)
_WHITE = (255, 255, 255)
_GRAY = (200, 200, 200)
_LIGHT_GRAY = (180, 180, 180)
_DARK = (25, 25, 25)
_DARKER = (20, 20, 20)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  DRIVER DROWSINESS DETECTION SYSTEM — v3.1 (Robust)")
    print("=" * 60)

    # ─── 1. Load centralized configuration ────────────────────────────
    cfg = SystemConfig()
    print(cfg)  # Log all parameters for experiment reproducibility

    # ─── 2. Initialize subsystems ─────────────────────────────────────
    detector = DrowsinessDetector()              # Pure math (EAR/MAR)
    analyzer = TemporalAnalyzer(cfg)             # Time-based detection
    state_mgr = StateManager(cfg)                # Face-loss safety
    alarm_ctrl = AlarmController(cfg)            # Persistent alarms
    robustness_guard = RobustnessGuard(cfg)      # Signal quality monitor
    cnn_validator = CNNValidator(cfg)            # Hybrid CNN validation

    # ─── 3. Initialize MediaPipe Face Mesh ────────────────────────────
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=cfg.face_mesh.max_num_faces,
        refine_landmarks=cfg.face_mesh.refine_landmarks,
        min_detection_confidence=cfg.face_mesh.min_detection_confidence,
        min_tracking_confidence=cfg.face_mesh.min_tracking_confidence,
    )

    # ─── 4. Initialize camera ────────────────────────────────────────
    print("[System] Accessing webcam stream...")
    try:
        cap = CameraAsync(
            camera_id=cfg.camera.camera_id,
            width=cfg.camera.capture_width,
            height=cfg.camera.capture_height
        )
    except RuntimeError as e:
        print(f"[CRITICAL] {e}")
        sys.exit(1)

    actual_w = cap.width
    actual_h = cap.height
    print(f"[System] Stream initialized: {actual_w}x{actual_h}")
    
    # ─── 4.5 Initialize Head Pose Estimator ──────────────────────────
    pose_estimator = HeadPoseEstimator(actual_w, actual_h)

    print("[System] Press 'q' or ESC to exit.")
    print("=" * 60)

    window_name = "Drowsiness Detection System v3.1"
    if not cfg.optimization.headless_mode:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, cfg.camera.display_width, cfg.camera.display_height)
    else:
        print("[System] Running in HEADLESS MODE (No GUI). Check logs for events.")

    # ─── 5. FPS tracking ─────────────────────────────────────────────
    frame_count = 0
    fps_start = time.time()
    fps = 0.0

    # ─── 6. Landmark jitter tracking ─────────────────────────────────
    prev_key_landmarks = None

    # ─── 7. Profiling trackers ───────────────────────────────────────
    t_cap_sum = t_inf_sum = t_math_sum = t_rndr_sum = 0.0
    profile_frames = 0
    skip_next_frame = False

    # ─────────────────────────────────────────────────────────────────
    # MAIN LOOP
    # ─────────────────────────────────────────────────────────────────
    while cap.isOpened():
        t0 = time.perf_counter()
        ret, frame = cap.read()
        t_cap = time.perf_counter() - t0
        
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        frame_count += 1
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # FPS calculation (1-second window)
        elapsed = time.time() - fps_start
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            fps_start = time.time()

        # ─── Adaptive Frame Skipping ───────────────────────────────────
        # Skip processing if we are in an alert state, to save CPU.
        if cfg.optimization.adaptive_frame_skipping and skip_next_frame:
            skip_next_frame = False
            continue

        t1 = time.perf_counter()
        # ─── MediaPipe inference ──────────────────────────────────────
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = face_mesh.process(rgb_frame)
        rgb_frame.flags.writeable = True
        t_inf = time.perf_counter() - t1

        t2 = time.perf_counter()
        # ─── Metric computation ───────────────────────────────────────
        raw_ear = 0.0
        raw_mar = 0.0
        face_detected = False

        if results.multi_face_landmarks:
            face_detected = True
            face_lm = results.multi_face_landmarks[0]

            # Extract landmarks
            left_eye_pts = [face_lm.landmark[idx] for idx in landmarks.LEFT_EYE_CONTOUR]
            right_eye_pts = [face_lm.landmark[idx] for idx in landmarks.RIGHT_EYE_CONTOUR]
            mouth_pts = [face_lm.landmark[idx] for idx in [78, 13, 308, 14]]

            # Compute raw metrics (pure math — no state)
            left_ear = detector.calculate_ear(left_eye_pts)
            right_ear = detector.calculate_ear(right_eye_pts)
            raw_ear = (left_ear + right_ear) / 2.0
            raw_mar = detector.calculate_mar(mouth_pts)

            # Compute Head Pose (Pitch, Yaw, Roll)
            pose_pts = [face_lm.landmark[idx] for idx in landmarks.POSE_LANDMARKS]
            pose_coords = get_eye_coords(face_lm.landmark, landmarks.POSE_LANDMARKS, w, h)
            raw_pitch, raw_yaw, raw_roll = pose_estimator.estimate_pose(pose_coords)

            # ─── Signal quality extraction ───────────────────────────────
            # 1. Landmark jitter: mean displacement from previous frame
            cur_key_landmarks = np.array(pose_coords, dtype=np.float32)
            jitter = 0.0
            if prev_key_landmarks is not None and len(cur_key_landmarks) == len(prev_key_landmarks):
                displacements = np.linalg.norm(cur_key_landmarks - prev_key_landmarks, axis=1)
                jitter = float(np.mean(displacements))
            prev_key_landmarks = cur_key_landmarks

            # 2. Frame brightness: mean intensity of face bounding box
            xs = [int(face_lm.landmark[idx].x * w) for idx in landmarks.POSE_LANDMARKS]
            ys = [int(face_lm.landmark[idx].y * h) for idx in landmarks.POSE_LANDMARKS]
            x_min, x_max = max(0, min(xs) - 20), min(w, max(xs) + 20)
            y_min, y_max = max(0, min(ys) - 20), min(h, max(ys) + 20)
            if x_max > x_min and y_max > y_min:
                gray_roi = cv2.cvtColor(frame[y_min:y_max, x_min:x_max], cv2.COLOR_BGR2GRAY)
                face_brightness = float(np.mean(gray_roi))
            else:
                face_brightness = 128.0

            # NOTE: A per-frame tracking-confidence signal was removed
            # from the reliability gate. MediaPipe FaceMesh does not
            # populate a real per-landmark visibility/confidence value,
            # so it would only ever be a constant (freeze-report
            # precondition 4). Reliability now uses 3 real components:
            # landmark stability, brightness, and cue consistency.
            sig_quality = SignalQuality(
                landmark_jitter=jitter,
                frame_brightness=face_brightness,
                face_visible=True,
            )
        else:
            raw_pitch = raw_yaw = raw_roll = 0.0
            prev_key_landmarks = None
            sig_quality = SignalQuality(face_visible=False)

        # ─── Temporal analysis (time-based, FPS-independent) ────────────
        temporal_state = analyzer.update(raw_ear, raw_mar, raw_pitch, raw_yaw, raw_roll)

        # ─── Robustness assessment ─────────────────────────────────────
        robustness_snap = robustness_guard.update(
            sig_quality,
            ear_conf=temporal_state.closure_ratio,
            mar_conf=temporal_state.yawn_confidence,
            pose_conf=temporal_state.posture_confidence,
        )

        # ─── CNN Validation (selective — uncertainty resolver) ───────────
        cnn_verdict = CNNVerdict()  # Default: not invoked
        if (face_detected
                and cnn_validator.should_invoke(
                    temporal_state.smoothed_ear,
                    robustness_snap.system_reliability)):
            eye_roi = extract_eye_roi(
                frame, results.multi_face_landmarks[0], w, h,
                target_size=cfg.cnn_validation.input_size
            )
            if eye_roi is not None:
                cnn_verdict = cnn_validator.validate_eye_state(
                    eye_roi, temporal_state.smoothed_ear,
                    ear_threshold=cfg.detection.ear_threshold
                )

        # ─── State management (face-loss safety + reliability-gated fusion)
        system_state = state_mgr.update(
            temporal_state, face_detected,
            reliability=robustness_snap.system_reliability,
            alert_suppressed=robustness_snap.alert_suppressed,
            landmark_jitter=sig_quality.landmark_jitter,
            frame_brightness=sig_quality.frame_brightness,
            cnn_verdict=cnn_verdict,
        )

        # ─── Alarm control (persistent, anti-flicker) ────────────────
        alarm_ctrl.update(system_state)
        t_math = time.perf_counter() - t2

        # ─── Determine next frame skip ───────────────────────────────
        if cfg.optimization.adaptive_frame_skipping:
            # Only skip if fully alert and no face loss
            if system_state.status == DriverStatus.ALERT and face_detected:
                skip_next_frame = True

        # ─── Profiler Logging ─────────────────────────────────────────
        if cfg.optimization.enable_profiling:
            t_cap_sum += t_cap
            t_inf_sum += t_inf
            t_math_sum += t_math
            profile_frames += 1

        # ─── Render annotations ──────────────────────────────────────
        if cfg.optimization.headless_mode:
            # End of loop for headless, check for exit if cv2 was somehow running
            key = cv2.waitKey(1) & 0xFF if not cfg.optimization.headless_mode else -1
            if key in (ord('q'), 27):
                break
            
            # Print profiler anyway
            if cfg.optimization.enable_profiling and profile_frames >= cfg.optimization.profiling_interval:
                print(f"[Profiler] FPS: {fps:.1f} | Cap: {(t_cap_sum/profile_frames)*1000:.1f}ms | "
                      f"Inf: {(t_inf_sum/profile_frames)*1000:.1f}ms | Math: {(t_math_sum/profile_frames)*1000:.1f}ms")
                t_cap_sum = t_inf_sum = t_math_sum = t_rndr_sum = 0.0
                profile_frames = 0
            continue
            
        t3 = time.perf_counter()
        
        if face_detected:
            face_lm = results.multi_face_landmarks[0]

            # Eye contour color: green=open, red=closed
            eye_color = _RED if system_state.is_eye_closed else _GREEN
            left_poly = np.array(get_eye_coords(face_lm.landmark, landmarks.LEFT_EYE_CONTOUR, w, h), np.int32)
            right_poly = np.array(get_eye_coords(face_lm.landmark, landmarks.RIGHT_EYE_CONTOUR, w, h), np.int32)
            cv2.polylines(frame, [left_poly], True, eye_color, 1)
            cv2.polylines(frame, [right_poly], True, eye_color, 1)

            # Mouth contour (Inner Lip)
            if system_state.is_yawning:
                mouth_color = _RED
            elif system_state.is_speaking:
                mouth_color = _ORANGE  # Speech detected
            elif system_state.yawn_confidence > 0.3:
                mouth_color = _YELLOW
            else:
                mouth_color = _GREEN

            mouth_poly = np.array(get_eye_coords(face_lm.landmark, landmarks.LIPS_INNER_CONTOUR, w, h), np.int32)
            cv2.polylines(frame, [mouth_poly], True, mouth_color, 1)

            # Sparse face mesh
            for idx in range(0, 468, 15):
                lm = face_lm.landmark[idx]
                cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 1, _GRAY, -1)

            # Draw 3D Head Pose Axes
            axes = pose_estimator.get_projection_axes()
            if axes is not None:
                origin, p_x, p_y, p_z = axes
                cv2.line(frame, origin, p_x, (0, 0, 255), 2)  # X: Red
                cv2.line(frame, origin, p_y, (0, 255, 0), 2)  # Y: Green
                cv2.line(frame, origin, p_z, (255, 0, 0), 2)  # Z: Blue

        # ═════════════════════════════════════════════════════════════
        # HUD OVERLAY
        # ═════════════════════════════════════════════════════════════

        # --- Top panel ---
        cv2.rectangle(frame, (0, 0), (w, 90), _DARK, -1)
        _sev_colors = {
            DriverStatus.ALERT: _GREEN,
            DriverStatus.SLIGHT_FATIGUE: _YELLOW,
            DriverStatus.MODERATE_FATIGUE: _ORANGE,
            DriverStatus.SEVERE_FATIGUE: _RED,
            DriverStatus.FACE_LOST_CRITICAL: _RED,
        }
        accent = _sev_colors.get(system_state.status, _GREEN)
        cv2.line(frame, (0, 90), (w, 90), accent, 2)

        # Title
        cv2.putText(frame, "DRIVER ATTENTION MONITORING v3.0 — Fusion", (20, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, _WHITE, 2)

        # EAR readout (smoothed)
        ear_color = _GREEN if system_state.smoothed_ear >= cfg.detection.ear_threshold else _RED
        cv2.putText(frame, f"EAR: {system_state.smoothed_ear:.3f}", (20, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, ear_color, 2)

        # MAR readout (smoothed)
        mar_color = _RED if system_state.is_yawning else (_ORANGE if system_state.is_speaking else _GREEN)
        cv2.putText(frame, f"MAR: {system_state.smoothed_mar:.3f}", (150, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, mar_color, 2)

        # Yawn Confidence
        conf_color = _GREEN if system_state.yawn_confidence < 0.3 else (_YELLOW if system_state.yawn_confidence < cfg.yawn.confidence_threshold else _RED)
        cv2.putText(frame, f"YwnConf: {system_state.yawn_confidence:.2f}", (420, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, conf_color, 1)

        # Posture metrics
        pitch_color = _RED if system_state.smoothed_pitch < cfg.posture.downward_pitch_threshold else _GREEN
        cv2.putText(frame, f"P: {system_state.smoothed_pitch:5.1f}", (550, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, pitch_color, 1)
        cv2.putText(frame, f"Y: {system_state.smoothed_yaw:5.1f}", (550, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, _LIGHT_GRAY, 1)
        cv2.putText(frame, f"R: {system_state.smoothed_roll:5.1f}", (550, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.45, _LIGHT_GRAY, 1)

        # Posture Confidence
        p_conf_color = _GREEN if system_state.posture_confidence < 0.3 else (_YELLOW if system_state.posture_confidence < cfg.posture.confidence_threshold else _RED)
        cv2.putText(frame, f"PstConf: {system_state.posture_confidence:.2f}", (630, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, p_conf_color, 1)

        # Speech Indicator
        if system_state.is_speaking:
            cv2.putText(frame, "SPEAKING", (420, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, _ORANGE, 2)

        # Closure duration (seconds, not frames)
        duration_text = f"Closure: {system_state.eye_closure_duration:.1f}s"
        cv2.putText(frame, duration_text, (280, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, _LIGHT_GRAY, 1)

        # Closure progress bar
        bar_x, bar_y, bar_w, bar_h = 280, 65, 100, 12
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
        fill_w = int(bar_w * system_state.closure_ratio)
        if fill_w > 0:
            bar_color = _GREEN if system_state.closure_ratio < 0.5 else (_YELLOW if system_state.closure_ratio < 0.8 else _RED)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bar_color, -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), _LIGHT_GRAY, 1)

        # Blink + Yawn counters
        cv2.putText(frame, f"Blinks: {system_state.total_blinks}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        cv2.putText(frame, f"Yawns: {system_state.total_yawns}", (130, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        # FPS + Reliability
        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 110, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, _LIGHT_GRAY, 1)
        rel_val = system_state.system_reliability
        rel_color = _GREEN if rel_val > 0.7 else (_YELLOW if rel_val > 0.5 else _RED)
        cv2.putText(frame, f"Rel: {rel_val:.2f}", (w - 110, 43),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, rel_color, 1)
        if system_state.alert_suppressed:
            cv2.putText(frame, "SUPPRESSED", (w - 110, 56),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, _ORANGE, 1)

        # CNN validation indicator
        if system_state.cnn_invoked:
            cnn_color = _GREEN if system_state.cnn_agrees else _YELLOW
            cnn_label = "CNN:AGR" if system_state.cnn_agrees else "CNN:OVR"
            cv2.putText(frame, cnn_label, (w - 110, 68),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, cnn_color, 1)
        if system_state.cnn_override_active:
            cv2.putText(frame, "FP SUPPRESSED", (w - 150, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, _YELLOW, 1)

        # Alarm indicator
        if alarm_ctrl.is_active:
            cv2.putText(frame, f"ALARM [{alarm_ctrl.alarm_duration:.1f}s]", (w - 180, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, _RED, 1)

        # ═════════════════════════════════════════════════════════════
        # FUSION HUD (second panel below top bar)
        # ═════════════════════════════════════════════════════════════
        fus_y = 95  # Fusion panel starts right below the accent line
        cv2.rectangle(frame, (0, fus_y), (w, fus_y + 40), _DARKER, -1)

        # Driver status badge (top-right). `status` is reused by the
        # full-screen alert block below, so define it here.
        status = system_state.status
        if status in (DriverStatus.SEVERE_FATIGUE, DriverStatus.DROWSY,
                      DriverStatus.FACE_LOST_CRITICAL):
            badge_color, badge_text = _RED, "DRIVER STATUS: DANGER"
        elif status in (DriverStatus.MODERATE_FATIGUE, DriverStatus.SLIGHT_FATIGUE,
                        DriverStatus.FACE_LOST):
            badge_color, badge_text = _ORANGE, "DRIVER STATUS: WARNING"
        else:
            badge_color, badge_text = (0, 150, 0), "DRIVER STATUS: OK"

        badge_w = 230
        cv2.rectangle(frame, (w - badge_w - 10, 62), (w - 10, 88), badge_color, -1)
        cv2.putText(frame, badge_text, (w - badge_w - 2, 82),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, _WHITE, 2)

        # --- Full-screen alert for SEVERE_FATIGUE / FACE_LOST_CRITICAL ---
        if status in (DriverStatus.SEVERE_FATIGUE, DriverStatus.DROWSY, DriverStatus.FACE_LOST_CRITICAL):
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), _RED, 4)
            banner_y = h // 2 - 25
            cv2.rectangle(frame, (w // 2 - 220, banner_y), (w // 2 + 220, banner_y + 50), _RED, -1)
            if status == DriverStatus.SEVERE_FATIGUE:
                alert_text = f"SEVERE FATIGUE [{system_state.fatigue_score:.2f}]"
            elif status == DriverStatus.DROWSY:
                alert_text = "DROWSINESS DETECTED!"
            else:
                alert_text = "FACE LOST — ESCALATION!"
            cv2.putText(frame, alert_text, (w // 2 - 200, banner_y + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, _WHITE, 2)
            cv2.putText(frame, "!!! WAKE UP !!!", (w // 2 - 110, banner_y + 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, _WHITE, 2)

        # --- Bottom bar ---
        cv2.rectangle(frame, (0, h - 30), (w, h), _DARKER, -1)
        if not face_detected:
            lost_text = "CRITICAL: NO FACE DETECTED! Position camera in front of face."
            if system_state.seconds_since_face_lost > 0:
                lost_text += f" ({system_state.seconds_since_face_lost:.1f}s)"
            cv2.putText(frame, lost_text, (15, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, _RED, 2)
        else:
            cv2.putText(frame, "Press 'q' or ESC to exit  |  Fusion Architecture v3.0",
                        (15, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, _LIGHT_GRAY, 1)

        # ─── Display ─────────────────────────────────────────────────
        cv2.imshow(window_name, frame)
        
        if cfg.optimization.enable_profiling:
            t_rndr = time.perf_counter() - t3
            t_rndr_sum += t_rndr
            if profile_frames >= cfg.optimization.profiling_interval:
                print(f"[Profiler] FPS: {fps:.1f} | Cap: {(t_cap_sum/profile_frames)*1000:.1f}ms | "
                      f"Inf: {(t_inf_sum/profile_frames)*1000:.1f}ms | Math: {(t_math_sum/profile_frames)*1000:.1f}ms | "
                      f"Rndr: {(t_rndr_sum/profile_frames)*1000:.1f}ms")
                t_cap_sum = t_inf_sum = t_math_sum = t_rndr_sum = 0.0
                profile_frames = 0

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break

    # ─── Cleanup ──────────────────────────────────────────────────────
    print("[System] Terminating drowsiness monitor. Cleaning up...")

    # Print CNN validation statistics for research logging
    if cnn_validator.is_available:
        stats = cnn_validator.get_stats()
        print(f"[CNN Stats] Invocations: {stats['total_invocations']}")
        print(f"[CNN Stats] Agreements:  {stats['total_agreements']}")
        print(f"[CNN Stats] Overrides:   {stats['total_overrides']}")
        print(f"[CNN Stats] Agreement Rate: {stats['agreement_rate']:.2%}")

    cap.release()
    if not cfg.optimization.headless_mode:
        cv2.destroyAllWindows()
    face_mesh.close()
    alarm_ctrl.shutdown()
    print("[System] Shutdown complete. Stay safe!")
    print("=" * 60)


if __name__ == "__main__":
    main()
