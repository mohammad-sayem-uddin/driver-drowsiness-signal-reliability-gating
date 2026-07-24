"""
State Manager — Unified State Machine with Face-Loss Safety
=============================================================
Consumes TemporalState from the TemporalAnalyzer and manages the
system-level driver status, including face-presence safety logic.

CRITICAL SAFETY FIX:
    The original system silenced the alarm when the face disappeared.
    This is *safety-inverting*: a drowsy driver slumping forward (face
    exits camera FOV) would cause the alarm to stop precisely when it
    is most needed.

    This module implements the opposite behavior:
      - Face loss during drowsiness → ESCALATE alarm (FACE_LOST_CRITICAL)
      - Face loss without prior drowsiness → warning only (FACE_LOST)
      - Face present → normal state machine (ALERT / DROWSY / YAWNING)

Architecture:
    - DriverStatus enum : Possible system states
    - FacePresenceTracker : Tracks face visibility with timestamps
    - StateManager : Top-level state machine producing SystemState
    - SystemState dataclass : Output consumed by AlarmController + HUD

Usage:
    from src.state_manager import StateManager
    manager = StateManager(cfg)
    system_state = manager.update(temporal_state, face_visible=True)
"""

import time
from dataclasses import dataclass
from enum import Enum

from src.config import SystemConfig
from src.temporal_analyzer import TemporalState
from src.fatigue_fusion import FatigueFusionEngine, FusionSnapshot, FatigueSeverity
from src.cnn_validator import CNNVerdict


# ═══════════════════════════════════════════════════════════════════════
# Driver Status Enum
# ═══════════════════════════════════════════════════════════════════════

class DriverStatus(Enum):
    """
    System-level driver status.  Ordered by severity.

    State transition diagram:

        ALERT ──(drowsiness detected)──> DROWSY
          │                                 │
          │                                 ├──(face lost)──> FACE_LOST_CRITICAL
          │                                 │
          │                                 └──(eyes open for min_alarm_duration)──> ALERT
          │
          ├──(yawn detected)──> YAWNING ──(yawn ends)──> ALERT
          │
          ├──(nod detected)──> NODDING ──(nod ends)──> ALERT
          │
          ├──(fusion: slight)──> SLIGHT_FATIGUE
          ├──(fusion: moderate)──> MODERATE_FATIGUE
          ├──(fusion: severe)──> SEVERE_FATIGUE
          │
          └──(face lost, no drowsiness)──> FACE_LOST ──(face returns)──> ALERT
                                              │
                                              └──(timeout exceeded)──> stays FACE_LOST
    """
    ALERT = "ALERT"                       # Normal — driver is attentive
    SLIGHT_FATIGUE = "SLIGHT_FATIGUE"     # Early warning (fusion score 0.25–0.50)
    MODERATE_FATIGUE = "MODERATE_FATIGUE" # Sustained multi-cue (fusion 0.50–0.75)
    SEVERE_FATIGUE = "SEVERE_FATIGUE"     # Strong agreement (fusion > 0.75)
    YAWNING = "YAWNING"                   # Yawn detected (informational)
    NODDING = "NODDING"                   # Head dropping / micro-sleep nod
    DROWSY = "DROWSY"                     # Sustained eye closure (legacy / direct)
    FACE_LOST = "FACE_LOST"               # Face not visible (no prior drowsiness)
    FACE_LOST_CRITICAL = "FACE_LOST_CRITICAL"  # Face lost during drowsiness sequence


# ═══════════════════════════════════════════════════════════════════════
# System State — Output snapshot
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class SystemState:
    """
    Complete system state snapshot for one frame.
    Consumed by AlarmController (for alarm decisions) and the HUD renderer.
    """
    # --- Driver classification ---
    status: DriverStatus = DriverStatus.ALERT
    warning_text: str = ""

    # --- Alarm control signals ---
    should_alarm: bool = False
    alarm_level: int = 0       # 0=silent, 1=beep(yawn), 2=alarm(drowsy), 3=escalated(face-lost-critical)

    # --- Face presence ---
    face_visible: bool = True
    seconds_since_face_lost: float = 0.0

    # --- Temporal metrics (forwarded for HUD) ---
    smoothed_ear: float = 0.0
    smoothed_mar: float = 0.0
    raw_ear: float = 0.0
    raw_mar: float = 0.0
    eye_closure_duration: float = 0.0
    closure_ratio: float = 0.0
    yawn_duration: float = 0.0
    total_blinks: int = 0
    total_yawns: int = 0
    is_eye_closed: bool = False
    is_yawning: bool = False
    yawn_confidence: float = 0.0
    is_speaking: bool = False
    ear_history: list = None
    
    # --- Posture metrics ---
    raw_pitch: float = 0.0
    raw_yaw: float = 0.0
    raw_roll: float = 0.0
    smoothed_pitch: float = 0.0
    smoothed_yaw: float = 0.0
    smoothed_roll: float = 0.0
    is_nodding: bool = False
    posture_confidence: float = 0.0
    posture_instability: float = 0.0

    # --- Fusion metrics ---
    fatigue_score: float = 0.0
    fatigue_severity: FatigueSeverity = FatigueSeverity.ALERT
    ear_contribution: float = 0.0
    mar_contribution: float = 0.0
    pose_contribution: float = 0.0
    active_cue_count: int = 0
    temporal_trend: str = "stable"
    agreement_multiplier: float = 1.0

    # --- Robustness metrics ---
    system_reliability: float = 1.0
    landmark_jitter: float = 0.0
    frame_brightness: float = 128.0
    alert_suppressed: bool = False

    # --- CNN Validation Layer ---
    cnn_invoked: bool = False
    cnn_probability_closed: float = -1.0
    cnn_agrees: bool = True
    cnn_override_active: bool = False    # True if CNN overrode heuristic


# ═══════════════════════════════════════════════════════════════════════
# Face Presence Tracker
# ═══════════════════════════════════════════════════════════════════════

class FacePresenceTracker:
    """
    Tracks face visibility state with timestamps.

    Provides:
      - Whether the face is currently visible
      - How long the face has been missing (seconds)
      - Whether the face was lost during an active drowsiness sequence
    """

    def __init__(self):
        self._face_visible = False
        self._last_seen_time = 0.0
        self._lost_time = 0.0
        self._was_drowsy_at_loss = False

    def update(self, face_detected: bool, is_drowsy: bool, timestamp: float):
        """
        Update face presence state.

        Args:
            face_detected: Whether MediaPipe detected a face this frame.
            is_drowsy: Whether the temporal analyzer detected drowsiness
                       *before* the face was lost.
            timestamp: time.monotonic() for this frame.
        """
        if face_detected:
            self._face_visible = True
            self._last_seen_time = timestamp
            self._lost_time = 0.0
            self._was_drowsy_at_loss = False
        else:
            if self._face_visible:
                # Face just disappeared this frame
                self._lost_time = timestamp
                self._was_drowsy_at_loss = is_drowsy
            self._face_visible = False

    @property
    def is_visible(self) -> bool:
        return self._face_visible

    @property
    def was_drowsy_at_loss(self) -> bool:
        return self._was_drowsy_at_loss

    def seconds_since_lost(self, current_time: float) -> float:
        """Returns how long the face has been missing."""
        if self._face_visible or self._lost_time == 0.0:
            return 0.0
        return current_time - self._lost_time

    def reset(self):
        self._face_visible = False
        self._last_seen_time = 0.0
        self._lost_time = 0.0
        self._was_drowsy_at_loss = False


# ═══════════════════════════════════════════════════════════════════════
# State Manager — Top-level state machine
# ═══════════════════════════════════════════════════════════════════════

class StateManager:
    """
    Central state machine that combines temporal analysis results with
    face-presence safety logic to produce a unified SystemState.

    Safety-critical behavior:
      - Face loss during drowsiness → FACE_LOST_CRITICAL → alarm escalates
      - Face loss without drowsiness → FACE_LOST → warning only
      - Face present → normal ALERT / DROWSY / YAWNING transitions
    """

    # Minimum time (seconds) that must elapse between severity transitions.
    # Prevents the MODERATE→SLIGHT→MODERATE oscillation observed in live
    # testing where the fusion score hovers near a threshold boundary.
    MIN_DWELL_TIME = 2.0

    def __init__(self, cfg: SystemConfig):
        self.cfg = cfg
        self.face_tracker = FacePresenceTracker()
        self.fusion_engine = FatigueFusionEngine(cfg)
        self._previous_status = DriverStatus.ALERT
        self._last_transition_time = 0.0  # Wall-clock of last severity change

    def update(self, temporal_state: TemporalState, face_detected: bool,
               reliability: float = 1.0, alert_suppressed: bool = False,
               landmark_jitter: float = 0.0, frame_brightness: float = 128.0,
               cnn_verdict: CNNVerdict = None) -> SystemState:
        """
        Compute system state for this frame.

        Args:
            temporal_state: Output from TemporalAnalyzer.update().
            face_detected: Whether a face was found in this frame.
            reliability: System reliability (0–1) from RobustnessGuard.
            alert_suppressed: Whether the RobustnessGuard recommends
                              suppressing non-severe alerts.
            landmark_jitter: Raw landmark jitter for HUD/logging.
            frame_brightness: Raw brightness for HUD/logging.

        Returns:
            SystemState snapshot.
        """
        now = time.monotonic()

        # --- Update face presence ---
        # Pass the drowsiness state *or* whether we were previously drowsy
        is_currently_drowsy = temporal_state.is_drowsy
        was_previously_drowsy = self._previous_status in (
            DriverStatus.DROWSY,
            DriverStatus.FACE_LOST_CRITICAL,
        )
        self.face_tracker.update(
            face_detected,
            is_currently_drowsy or was_previously_drowsy,
            now,
        )

        # --- Determine driver status ---
        status = DriverStatus.ALERT
        warning_text = ""
        should_alarm = False
        alarm_level = 0

        if not self.face_tracker.is_visible:
            # === FACE NOT VISIBLE ===
            seconds_lost = self.face_tracker.seconds_since_lost(now)

            if self.face_tracker.was_drowsy_at_loss and self.cfg.alarm.face_loss_escalation:
                # CRITICAL: Face lost during drowsiness → escalate
                status = DriverStatus.FACE_LOST_CRITICAL
                warning_text = "FACE LOST — DROWSINESS ESCALATION!"
                should_alarm = True
                alarm_level = 3  # Maximum severity
            elif seconds_lost > self.cfg.alarm.face_loss_timeout:
                # Face lost for extended period (no prior drowsiness)
                status = DriverStatus.FACE_LOST
                warning_text = "NO FACE DETECTED — Reposition camera"
                should_alarm = False
                alarm_level = 0
            else:
                # Grace period — face just disappeared, likely momentary
                status = self._previous_status if self._previous_status != DriverStatus.FACE_LOST else DriverStatus.ALERT
                warning_text = ""
                should_alarm = self._previous_status in (
                    DriverStatus.DROWSY, DriverStatus.SEVERE_FATIGUE
                )
                alarm_level = 2 if should_alarm else 0

            # Use empty fusion snapshot when face is lost
            fusion = FusionSnapshot()

        else:
            # === FACE VISIBLE — Run fusion engine ===
            fusion = self.fusion_engine.update(temporal_state, reliability=reliability)

            severity = fusion.severity

            # --- Minimum dwell time gate ---
            # Prevent severity changes faster than MIN_DWELL_TIME seconds.
            # Exception: ESCALATION to SEVERE is never delayed (safety).
            _STATUS_TO_SEVERITY = {
                DriverStatus.ALERT: FatigueSeverity.ALERT,
                DriverStatus.SLIGHT_FATIGUE: FatigueSeverity.SLIGHT_FATIGUE,
                DriverStatus.MODERATE_FATIGUE: FatigueSeverity.MODERATE_FATIGUE,
                DriverStatus.SEVERE_FATIGUE: FatigueSeverity.SEVERE_FATIGUE,
            }
            time_since_transition = now - self._last_transition_time
            prev_as_severity = _STATUS_TO_SEVERITY.get(self._previous_status)
            if prev_as_severity is not None and severity != prev_as_severity:
                if (time_since_transition < self.MIN_DWELL_TIME
                        and severity != FatigueSeverity.SEVERE_FATIGUE
                        and self._previous_status != DriverStatus.FACE_LOST_CRITICAL):
                    # Too soon — hold the previous severity
                    severity = prev_as_severity

            if severity == FatigueSeverity.SEVERE_FATIGUE:
                status = DriverStatus.SEVERE_FATIGUE
                warning_text = "SEVERE FATIGUE — WAKE UP!"
                should_alarm = True
                alarm_level = 2
            elif severity == FatigueSeverity.MODERATE_FATIGUE:
                status = DriverStatus.MODERATE_FATIGUE
                warning_text = "MODERATE FATIGUE — Take a break"
                should_alarm = True
                alarm_level = 1
            elif severity == FatigueSeverity.SLIGHT_FATIGUE:
                status = DriverStatus.SLIGHT_FATIGUE
                warning_text = "Slight fatigue detected"
                should_alarm = False
                alarm_level = 0
            else:
                status = DriverStatus.ALERT
                warning_text = ""
                should_alarm = False
                alarm_level = 0

        # --- Adaptive alert suppression ---
        # If RobustnessGuard recommends suppression (low reliability),
        # suppress non-SEVERE alarms.  SEVERE is NEVER suppressed.
        actual_suppressed = False
        if alert_suppressed and status != DriverStatus.SEVERE_FATIGUE:
            if should_alarm and alarm_level < 2:
                should_alarm = False
                actual_suppressed = True

        # --- CNN validation override ---
        # The CNN can suppress false positives in the SLIGHT/MODERATE zone.
        # It NEVER overrides SEVERE_FATIGUE or FACE_LOST_CRITICAL (safety).
        cnn_override_active = False
        if cnn_verdict is None:
            cnn_verdict = CNNVerdict()

        if (cnn_verdict.invoked
                and not cnn_verdict.cnn_agrees_with_heuristic
                and cnn_verdict.confidence > (self.cfg.cnn_validation.confidence_threshold - 0.5)
                and status not in (DriverStatus.SEVERE_FATIGUE,
                                   DriverStatus.FACE_LOST_CRITICAL,
                                   DriverStatus.FACE_LOST)):
            # CNN disagrees with heuristic in a non-critical state
            if not cnn_verdict.cnn_says_closed and status in (
                DriverStatus.SLIGHT_FATIGUE, DriverStatus.MODERATE_FATIGUE
            ):
                # CNN says eyes are OPEN but heuristic triggered fatigue
                # → Likely a false positive.  Suppress alarm.
                if should_alarm and alarm_level < 2:
                    should_alarm = False
                    cnn_override_active = True

        # Track severity transitions for dwell time gating
        if status != self._previous_status:
            self._last_transition_time = now
        self._previous_status = status

        return SystemState(
            status=status,
            warning_text=warning_text,
            should_alarm=should_alarm,
            alarm_level=alarm_level,
            face_visible=self.face_tracker.is_visible,
            seconds_since_face_lost=self.face_tracker.seconds_since_lost(now),
            smoothed_ear=temporal_state.smoothed_ear,
            smoothed_mar=temporal_state.smoothed_mar,
            raw_ear=temporal_state.raw_ear,
            raw_mar=temporal_state.raw_mar,
            eye_closure_duration=temporal_state.eye_closure_duration,
            closure_ratio=temporal_state.closure_ratio,
            yawn_duration=temporal_state.yawn_duration,
            total_blinks=temporal_state.total_blinks,
            total_yawns=temporal_state.total_yawns,
            is_eye_closed=temporal_state.is_eye_closed,
            is_yawning=temporal_state.is_yawning,
            yawn_confidence=temporal_state.yawn_confidence,
            is_speaking=temporal_state.is_speaking,
            ear_history=temporal_state.ear_history,
            raw_pitch=temporal_state.raw_pitch,
            raw_yaw=temporal_state.raw_yaw,
            raw_roll=temporal_state.raw_roll,
            smoothed_pitch=temporal_state.smoothed_pitch,
            smoothed_yaw=temporal_state.smoothed_yaw,
            smoothed_roll=temporal_state.smoothed_roll,
            is_nodding=temporal_state.is_nodding,
            posture_confidence=temporal_state.posture_confidence,
            posture_instability=temporal_state.posture_instability,
            fatigue_score=fusion.fatigue_score,
            fatigue_severity=fusion.severity,
            ear_contribution=fusion.ear_contribution,
            mar_contribution=fusion.mar_contribution,
            pose_contribution=fusion.pose_contribution,
            active_cue_count=fusion.active_cue_count,
            temporal_trend=fusion.temporal_trend,
            agreement_multiplier=fusion.agreement_multiplier,
            system_reliability=reliability,
            landmark_jitter=landmark_jitter,
            frame_brightness=frame_brightness,
            alert_suppressed=actual_suppressed,
            cnn_invoked=cnn_verdict.invoked,
            cnn_probability_closed=cnn_verdict.probability_closed,
            cnn_agrees=cnn_verdict.cnn_agrees_with_heuristic,
            cnn_override_active=cnn_override_active,
        )

    def reset(self):
        """Reset all state."""
        self.face_tracker.reset()
        self.fusion_engine.reset()
        self._previous_status = DriverStatus.ALERT
