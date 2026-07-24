"""
Alarm Controller — Persistent, Anti-Flicker Alarm Management
==============================================================
Wraps AudioAlertSystem with lifecycle management that prevents the
alarm instability problems identified in the system review:

    1. ALARM FLICKERING:  Brief eye opening silenced the alarm instantly.
       Fix: Minimum alarm duration (3.0s) — once triggered, the alarm
       plays for at least 3 seconds regardless of EAR recovery.

    2. RAPID RE-TRIGGERING:  EAR oscillating near threshold caused
       start/stop/start/stop alarm cycling.
       Fix: Cooldown period (5.0s) — after an alarm ends, no new alarm
       can start for 5 seconds.

    3. FACE-LOSS SILENCING:  Face disappearance stopped the alarm.
       Fix: Alarm level 3 (escalation) overrides normal stop logic.

    4. NO ALARM LOGGING:  Events were not consistently recorded.
       Fix: All alarm lifecycle events logged to CSV with timestamps.

Architecture:
    AlarmController wraps AudioAlertSystem and is the ONLY module that
    directly calls play_alert() / stop_alert().  All other modules
    communicate alarm intent through SystemState.should_alarm.

Usage:
    from src.alarm_controller import AlarmController
    controller = AlarmController(cfg)
    controller.update(system_state)     # call per frame
    controller.shutdown()               # cleanup
"""

import time
import os
from datetime import datetime

from src.config import SystemConfig
from src.state_manager import SystemState, DriverStatus
from src.utils.audio_alert import AudioAlertSystem


class AlarmController:
    """
    Manages the alarm lifecycle with persistence, cooldown, and logging.

    State diagram:
        IDLE ──(should_alarm=True, cooldown expired)──> ACTIVE
          │                                                │
          └──(should_alarm but cooldown active)──> SUPPRESSED  │
                                                           │
        ACTIVE ──(alarm_duration >= min_duration            │
                  AND should_alarm=False)──> COOLDOWN ─────>│
                                                           │
        ACTIVE ──(alarm_duration < min_duration)──> stays ACTIVE
                                                           │
        COOLDOWN ──(cooldown_elapsed >= cooldown_period)──> IDLE
    """

    def __init__(self, cfg: SystemConfig):
        self.cfg = cfg
        self.audio = AudioAlertSystem()

        # --- Alarm lifecycle state ---
        self._is_active = False
        self._alarm_start_time = 0.0
        self._alarm_stop_time = 0.0
        self._current_level = 0

        # --- Yawn tracking for logging ---
        self._previous_total_yawns = 0
        self._current_yawn_max_mar = 0.0
        self._current_yawn_max_conf = 0.0
        self._current_yawn_duration = 0.0

        # --- Nod tracking for logging ---
        self._previous_is_nodding = False
        self._current_nod_min_pitch = 0.0
        self._current_nod_max_conf = 0.0
        self._current_nod_duration = 0.0
        self._nod_start_time_local = 0.0

        # --- Fusion severity tracking for logging ---
        self._previous_severity = None

        # --- Suppression tracking ---
        self._previous_suppressed = False

        # --- Event logging ---
        self._log_path = cfg.alarm.log_file_path
        self._initialize_log()

        print(f"[AlarmController] Initialized. Min duration: {cfg.alarm.min_alarm_duration}s, "
              f"Cooldown: {cfg.alarm.cooldown_period}s")

    # ═══════════════════════════════════════════════════════════════════
    # Public API
    # ═══════════════════════════════════════════════════════════════════

    def update(self, state: SystemState):
        """
        Per-frame alarm update.  This is the ONLY method that should be
        called from the main loop.

        Args:
            state: SystemState from StateManager.update().
        """
        now = time.monotonic()

        if state.should_alarm:
            self._handle_alarm_request(state, now)
        else:
            self._handle_alarm_release(now)

        self._handle_yawn_logging(state)
        self._handle_nod_logging(state)
        self._handle_fusion_logging(state)
        self._handle_suppression_logging(state)

    def shutdown(self):
        """Release audio resources."""
        if self._is_active:
            self.audio.stop_alert()
            self._is_active = False
        self.audio.shutdown()
        print("[AlarmController] Shutdown complete.")

    @property
    def is_active(self) -> bool:
        """Whether the alarm is currently sounding."""
        return self._is_active

    @property
    def current_level(self) -> int:
        """Current alarm severity level (0–3)."""
        return self._current_level

    @property
    def alarm_duration(self) -> float:
        """How long the current alarm has been active (seconds)."""
        if not self._is_active or self._alarm_start_time == 0.0:
            return 0.0
        return time.monotonic() - self._alarm_start_time

    # ═══════════════════════════════════════════════════════════════════
    # Internal Logic
    # ═══════════════════════════════════════════════════════════════════

    def _handle_alarm_request(self, state: SystemState, now: float):
        """Handle a frame where should_alarm is True."""
        if self._is_active:
            # Already alarming — check for escalation
            if state.alarm_level > self._current_level:
                self._current_level = state.alarm_level
                self._log_event("ALARM_ESCALATED", f"Level {self._current_level} ({state.status.value})")
            return

        # Not currently active — check cooldown
        if self._alarm_stop_time > 0:
            time_since_stop = now - self._alarm_stop_time
            if time_since_stop < self.cfg.alarm.cooldown_period:
                # Still in cooldown — suppress
                return

        # Start the alarm
        self._is_active = True
        self._alarm_start_time = now
        self._current_level = state.alarm_level
        self.audio.play_alert(loop=True)
        self._log_event("ALARM_STARTED", f"Level {state.alarm_level} ({state.status.value})")

    def _handle_alarm_release(self, now: float):
        """Handle a frame where should_alarm is False."""
        if not self._is_active:
            return

        # Check minimum duration — don't stop early
        elapsed = now - self._alarm_start_time
        if elapsed < self.cfg.alarm.min_alarm_duration:
            # Still within minimum duration — keep alarming
            return

        # Minimum duration met and condition cleared — stop alarm
        self._is_active = False
        self._alarm_stop_time = now
        self._current_level = 0
        self.audio.stop_alert()
        self._log_event("ALARM_STOPPED", f"Duration: {elapsed:.2f}s")

    def _handle_yawn_logging(self, state: SystemState):
        """Track and log yawn metrics (confidence, duration, max MAR)."""
        if state.is_yawning:
            # Track peaks during the active yawn
            if state.raw_mar > self._current_yawn_max_mar:
                self._current_yawn_max_mar = state.raw_mar
            if state.yawn_confidence > self._current_yawn_max_conf:
                self._current_yawn_max_conf = state.yawn_confidence
            self._current_yawn_duration = state.yawn_duration

        # When total_yawns increments, the yawn just finished.
        if state.total_yawns > self._previous_total_yawns:
            details = (
                f"Conf: {self._current_yawn_max_conf:.2f}, "
                f"Dur: {self._current_yawn_duration:.2f}s, "
                f"MaxMAR: {self._current_yawn_max_mar:.2f}"
            )
            self._log_event("YAWN_DETECTED", details)
            
            # Reset trackers for next yawn
            self._previous_total_yawns = state.total_yawns
            self._current_yawn_max_mar = 0.0
            self._current_yawn_max_conf = 0.0
            self._current_yawn_duration = 0.0

    def _handle_nod_logging(self, state: SystemState):
        """Track and log nodding metrics (pitch, confidence, duration)."""
        if state.is_nodding and not self._previous_is_nodding:
            # Nod just started
            self._nod_start_time_local = time.monotonic()
            self._current_nod_min_pitch = state.raw_pitch
            self._current_nod_max_conf = state.posture_confidence
            
        elif state.is_nodding and self._previous_is_nodding:
            # Nod continuing — track peaks
            self._current_nod_duration = time.monotonic() - self._nod_start_time_local
            if state.raw_pitch < self._current_nod_min_pitch:
                self._current_nod_min_pitch = state.raw_pitch
            if state.posture_confidence > self._current_nod_max_conf:
                self._current_nod_max_conf = state.posture_confidence
            
        elif not state.is_nodding and self._previous_is_nodding:
            # Nod finished
            details = (
                f"Conf: {self._current_nod_max_conf:.2f}, "
                f"Dur: {self._current_nod_duration:.2f}s, "
                f"MinPitch: {self._current_nod_min_pitch:.1f}°"
            )
            self._log_event("NOD_DETECTED", details)
            
            self._current_nod_min_pitch = 0.0
            self._current_nod_max_conf = 0.0
            self._current_nod_duration = 0.0
            
        self._previous_is_nodding = state.is_nodding

    def _handle_fusion_logging(self, state: SystemState):
        """Log severity transitions with fusion details."""
        current = state.fatigue_severity
        if self._previous_severity is not None and current != self._previous_severity:
            direction = "ESCALATED" if current.value > self._previous_severity.value else "DEESCALATED"
            details = (
                f"{self._previous_severity.name} -> {current.name}, "
                f"Score: {state.fatigue_score:.3f}, "
                f"EAR: {state.ear_contribution:.2f}, "
                f"MAR: {state.mar_contribution:.2f}, "
                f"Pose: {state.pose_contribution:.2f}, "
                f"Cues: {state.active_cue_count}, "
                f"Trend: {state.temporal_trend}"
            )
            self._log_event(f"SEVERITY_{direction}", details)
        self._previous_severity = current

    def _handle_suppression_logging(self, state: SystemState):
        """Log when alerts are suppressed due to low reliability."""
        if state.alert_suppressed and not self._previous_suppressed:
            details = (
                f"Reliability: {state.system_reliability:.3f}, "
                f"Score: {state.fatigue_score:.3f}, "
                f"Status: {state.status.value}, "
                f"Jitter: {state.landmark_jitter:.1f}px, "
                f"Brightness: {state.frame_brightness:.0f}"
            )
            self._log_event("ALERT_SUPPRESSED", details)
        elif not state.alert_suppressed and self._previous_suppressed:
            self._log_event("ALERT_UNSUPPRESSED",
                            f"Reliability recovered: {state.system_reliability:.3f}")
        self._previous_suppressed = state.alert_suppressed

    # ═══════════════════════════════════════════════════════════════════
    # Event Logging
    # ═══════════════════════════════════════════════════════════════════

    def _initialize_log(self):
        """Create CSV log file with header if it doesn't exist."""
        file_exists = (
            os.path.isfile(self._log_path)
            and os.path.getsize(self._log_path) > 0
        )
        if not file_exists:
            try:
                with open(self._log_path, 'w') as f:
                    f.write("Timestamp,Event_Type,Details\n")
            except Exception as e:
                print(f"[AlarmController] Could not create log file: {e}")

    def _log_event(self, event_type: str, details: str = ""):
        """Append a timestamped event to the research log."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        try:
            with open(self._log_path, 'a') as f:
                f.write(f"{timestamp},{event_type},{details}\n")
            print(f"[LOG] {timestamp} | {event_type} | {details}")
        except Exception as e:
            print(f"[AlarmController] Log write error: {e}")
