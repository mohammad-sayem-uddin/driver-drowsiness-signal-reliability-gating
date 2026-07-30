"""
Temporal Analyzer — FPS-Independent Detection Engine
======================================================
Replaces all frame-count-based detection logic with wall-clock duration
tracking using ``time.monotonic()``.  This makes drowsiness and yawn
detection deterministic regardless of whether the system runs at 15 FPS
(Raspberry Pi) or 60 FPS (desktop GPU).

Architecture:
    - EyeClosureAnalyzer : Tracks eye closure duration; detects blinks
                           and drowsiness using hysteresis thresholding
                           and EMA-smoothed EAR.
    - YawnAnalyzer       : Tracks sustained mouth opening; filters
                           speech-induced MAR spikes via minimum duration.
    - TemporalAnalyzer   : Orchestrator that owns both sub-analyzers and
                           returns a unified TemporalState per frame.

All timing uses ``time.monotonic()`` which is immune to NTP clock
adjustments and guaranteed monotonically increasing.

Usage:
    from src.config import SystemConfig
    from src.temporal_analyzer import TemporalAnalyzer

    cfg = SystemConfig()
    analyzer = TemporalAnalyzer(cfg)

    # Per frame:
    state = analyzer.update(avg_ear, mar)
"""

import time
from dataclasses import dataclass, field
from collections import deque

from src.config import SystemConfig


# ═══════════════════════════════════════════════════════════════════════
# Output State — Immutable snapshot returned per frame
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class TemporalState:
    """
    Immutable snapshot of the temporal analysis for a single frame.
    Consumed by StateManager and the HUD renderer.
    """
    # --- Raw inputs (for HUD display) ---
    raw_ear: float = 0.0
    raw_mar: float = 0.0

    # --- Smoothed signals ---
    smoothed_ear: float = 0.0
    smoothed_mar: float = 0.0

    # --- Eye state ---
    is_eye_closed: bool = False
    eye_closure_duration: float = 0.0     # seconds of current closure
    closure_ratio: float = 0.0            # 0.0–1.0 progress toward drowsiness trigger
    is_drowsy: bool = False

    # --- Blink tracking ---
    is_blinking: bool = False             # True during a blink (closure < blink_max)
    total_blinks: int = 0

    # --- Yawn state ---
    is_mouth_open: bool = False
    yawn_duration: float = 0.0            # seconds of current mouth opening
    is_yawning: bool = False              # True when yawn_duration >= yawn_min_duration
    total_yawns: int = 0
    yawn_confidence: float = 0.0          # 0.0-1.0 confidence score based on temporal rules
    is_speaking: bool = False             # True if high MAR variance detected (speech artifact)
    mar_jitter: float = 0.0               # Sliding window frame-to-frame jitter of MAR

    # --- EAR history (for waveform visualization) ---
    ear_history: list = field(default_factory=list)

    # --- Posture & Head Pose ---
    raw_pitch: float = 0.0
    raw_yaw: float = 0.0
    raw_roll: float = 0.0
    smoothed_pitch: float = 0.0
    smoothed_yaw: float = 0.0
    smoothed_roll: float = 0.0
    posture_instability: float = 0.0      # Sliding window variance of yaw/roll
    is_nodding: bool = False              # True if chin dropped below threshold for duration
    nod_duration: float = 0.0
    posture_confidence: float = 0.0       # 0.0-1.0 confidence score for fatigue posture


# ═══════════════════════════════════════════════════════════════════════
# EMA Smoother — Reusable low-pass filter
# ═══════════════════════════════════════════════════════════════════════

class EMASmoother:
    """
    Exponential Moving Average filter.

    Formula:  smoothed = α × raw + (1 − α) × previous

    At α = 0.3:
      - Single-frame outlier is attenuated to 30% of its magnitude.
      - A sustained signal change converges within ~7 frames (~230ms at 30 FPS).
      - Perceptual delay is < 50ms — imperceptible to the driver.
    """

    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self._value = None

    def update(self, raw: float) -> float:
        if self._value is None:
            self._value = raw
        else:
            self._value = self.alpha * raw + (1.0 - self.alpha) * self._value
        return self._value

    @property
    def value(self) -> float:
        return self._value if self._value is not None else 0.0

    def reset(self):
        self._value = None


# ═══════════════════════════════════════════════════════════════════════
# Eye Closure Analyzer — Hysteresis + wall-clock timing
# ═══════════════════════════════════════════════════════════════════════

class EyeClosureAnalyzer:
    """
    Tracks eye closure state using hysteresis thresholding on the
    EMA-smoothed EAR signal, with wall-clock duration tracking.

    Hysteresis prevents rapid toggling when EAR hovers near the threshold:
      - CLOSE threshold : ear_threshold (e.g., 0.21)
      - OPEN threshold  : ear_threshold + ear_hysteresis (e.g., 0.24)
      - Transition OPEN→CLOSED requires EAR < 0.21
      - Transition CLOSED→OPEN requires EAR > 0.24

    Blink detection:
      - Closure lasting blink_min (0.05s) to blink_max (0.4s) = blink
      - Closure lasting > eye_closure_duration (1.0s) = drowsiness

    All durations are measured in seconds via time.monotonic().
    """

    def __init__(self, cfg: SystemConfig):
        self._close_threshold = cfg.detection.ear_threshold
        self._open_threshold = cfg.detection.ear_threshold + cfg.detection.ear_hysteresis
        self._drowsy_duration = cfg.temporal.eye_closure_duration
        self._blink_min = cfg.temporal.blink_min_duration
        self._blink_max = cfg.temporal.blink_max_duration

        # Smoother
        self._smoother = EMASmoother(cfg.smoothing.ear_ema_alpha)

        # State
        self._is_closed = False
        self._closure_start_time = 0.0
        self._is_drowsy = False
        self._total_blinks = 0

        # History buffer for visualization
        self._history = deque(maxlen=cfg.smoothing.ear_history_size)

    def update(self, raw_ear: float, timestamp: float):
        """
        Process one frame of EAR data.

        Args:
            raw_ear: Raw average EAR from the EAR calculator.
            timestamp: time.monotonic() value for this frame.

        Returns:
            Tuple of (smoothed_ear, is_closed, closure_duration, is_drowsy,
                       total_blinks, closure_ratio, history_list)
        """
        smoothed = self._smoother.update(raw_ear)
        self._history.append(smoothed)

        closure_duration = 0.0

        if not self._is_closed:
            # Currently OPEN — check if we should transition to CLOSED
            if smoothed < self._close_threshold:
                self._is_closed = True
                self._closure_start_time = timestamp
                self._is_drowsy = False
        else:
            # Currently CLOSED — check if we should transition to OPEN
            if smoothed > self._open_threshold:
                # Eye just reopened — classify the closure event
                closure_duration_final = timestamp - self._closure_start_time

                if self._blink_min <= closure_duration_final <= self._blink_max:
                    self._total_blinks += 1

                self._is_closed = False
                self._is_drowsy = False
                self._closure_start_time = 0.0
            else:
                # Still closed — update duration
                closure_duration = timestamp - self._closure_start_time

                if closure_duration >= self._drowsy_duration:
                    self._is_drowsy = True

        # Compute closure ratio (progress toward drowsiness trigger)
        if self._is_closed and self._closure_start_time > 0:
            closure_duration = timestamp - self._closure_start_time
        closure_ratio = min(closure_duration / self._drowsy_duration, 1.0) if self._drowsy_duration > 0 else 0.0

        return (
            smoothed,
            self._is_closed,
            closure_duration,
            self._is_drowsy,
            self._total_blinks,
            closure_ratio,
            list(self._history),
        )

    def reset(self):
        """Reset all state.  Used during re-calibration."""
        self._is_closed = False
        self._closure_start_time = 0.0
        self._is_drowsy = False
        self._total_blinks = 0
        self._smoother.reset()
        self._history.clear()


# ═══════════════════════════════════════════════════════════════════════
# Yawn Analyzer — Temporal filtering for MAR
# ═══════════════════════════════════════════════════════════════════════

class YawnAnalyzer:
    """
    Tracks yawn state with advanced temporal filtering to eliminate speech
    artifacts and compute confidence scores.

    Problem:  Speech produces MAR spikes lasting < 0.3 seconds with high jitter.
              Real yawns sustain MAR > threshold for 2–6 seconds with smooth curves.

    Solution: Computes a Sliding Window Jitter of MAR.
              Assigns a `yawn_confidence` score based on:
                1. MAR magnitude
                2. Sustained duration
                3. Curve smoothness (inverse of jitter)
              Severely penalizes confidence if high jitter (speech) is detected.
    """

    def __init__(self, cfg: SystemConfig):
        self._open_threshold = cfg.detection.mar_threshold
        self._close_threshold = cfg.detection.mar_threshold - cfg.detection.mar_hysteresis
        self._yawn_duration = cfg.temporal.yawn_min_duration
        self._conf_threshold = cfg.yawn.confidence_threshold
        self._speech_jitter_limit = cfg.yawn.speech_jitter_threshold
        # Ablation switch (frozen protocol §3). Default True == frozen behavior.
        # When False, the σ²(MAR) speech-jitter penalty is bypassed so the
        # V0/V2 baselines can be measured without the speech filter.
        self._speech_filter_enabled = cfg.ablation.speech_filter_enabled

        # Smoother & Jitter Buffer
        self._smoother = EMASmoother(cfg.smoothing.mar_ema_alpha)
        self._mar_buffer = deque(maxlen=cfg.yawn.sliding_window_size)

        # State
        self._is_open = False
        self._open_start_time = 0.0
        self._is_yawning = False
        self._total_yawns = 0
        self._yawn_confidence = 0.0
        self._is_speaking = False

    def update(self, raw_mar: float, timestamp: float):
        """
        Process one frame of MAR data with advanced temporal analysis.

        Args:
            raw_mar: Raw MAR value from the detector.
            timestamp: time.monotonic() value for this frame.

        Returns:
            Tuple of (smoothed_mar, is_mouth_open, yawn_duration,
                       is_yawning, total_yawns, yawn_confidence,
                       is_speaking, mar_jitter)
        """
        smoothed = self._smoother.update(raw_mar)
        self._mar_buffer.append(smoothed)

        # Calculate Sliding Window Jitter (Frame-to-Frame Absolute Difference)
        mar_jitter = 0.0
        if len(self._mar_buffer) > 1:
            deltas = [abs(self._mar_buffer[i] - self._mar_buffer[i-1]) for i in range(1, len(self._mar_buffer))]
            mar_jitter = sum(deltas) / len(deltas)

        # Detect speech (high jitter in the signal)
        self._is_speaking = mar_jitter > self._speech_jitter_limit
        if not self._speech_filter_enabled:
            # Ablation V0/V2: disable the speech-jitter contribution entirely.
            self._is_speaking = False

        yawn_duration = 0.0

        if not self._is_open:
            # Mouth currently closed — check if opening
            if smoothed > self._open_threshold:
                self._is_open = True
                self._open_start_time = timestamp
                self._is_yawning = False
                self._yawn_confidence = 0.0
        else:
            # Mouth currently open — check if closing
            if smoothed < self._close_threshold:
                # Mouth just closed — was it a genuine yawn?
                if self._is_yawning and self._yawn_confidence >= self._conf_threshold:
                    self._total_yawns += 1

                self._is_open = False
                self._is_yawning = False
                self._open_start_time = 0.0
                self._yawn_confidence = 0.0
            else:
                # Still open — update duration
                yawn_duration = timestamp - self._open_start_time

                # --- Confidence Scoring Logic ---
                # 1. Base magnitude (how far above threshold)
                mag_factor = min((smoothed - self._open_threshold) / 0.3, 1.0)
                if mag_factor < 0: mag_factor = 0.0

                # 2. Duration factor (builds up over time)
                dur_factor = min(yawn_duration / self._yawn_duration, 1.0)

                # 3. Smoothness factor (inverse of jitter)
                jitter_penalty = min(mar_jitter / (self._speech_jitter_limit * 1.5), 1.0)
                smoothness = 1.0 - jitter_penalty
                if not self._speech_filter_enabled:
                    # Ablation V0/V2: no variance-based penalty at all.
                    smoothness = 1.0

                # Combine: Need magnitude, duration, and smoothness.
                self._yawn_confidence = (mag_factor * 0.2 + dur_factor * 0.5 + smoothness * 0.3)

                # Heavy penalty if speech is actively detected
                if self._is_speaking:
                    self._yawn_confidence *= 0.1

                # Trigger boolean state if thresholds are met
                if yawn_duration >= self._yawn_duration and self._yawn_confidence >= self._conf_threshold and not self._is_yawning:
                    self._is_yawning = True

        # Compute current yawn duration for display
        if self._is_open and self._open_start_time > 0:
            yawn_duration = timestamp - self._open_start_time

        return (
            smoothed,
            self._is_open,
            yawn_duration,
            self._is_yawning,
            self._total_yawns,
            self._yawn_confidence,
            self._is_speaking,
            mar_jitter
        )

    def reset(self):
        """Reset all state."""
        self._is_open = False
        self._open_start_time = 0.0
        self._is_yawning = False
        self._total_yawns = 0
        self._yawn_confidence = 0.0
        self._is_speaking = False
        self._smoother.reset()
        self._mar_buffer.clear()


# ═══════════════════════════════════════════════════════════════════════
# Posture Analyzer — Nodding and Instability
# ═══════════════════════════════════════════════════════════════════════

class PostureAnalyzer:
    """
    Tracks head pitch, yaw, and roll over time to detect fatigue-induced
    posture behaviors such as downward nodding and general neck instability.

    Stabilization features (v3.2):
        - Heavier EMA smoothing (α=0.10) for pose signals to suppress
          MediaPipe landmark jitter that causes false nod events.
        - Nod cooldown timer (3.0s) prevents rapid re-triggering when
          the head oscillates near the pitch threshold.
        - Pitch velocity gating: requires ≥3°/s downward velocity to
          distinguish genuine fatigue head-drops from slow natural drift.
    """
    # Cooldown between consecutive nod events (seconds)
    NOD_COOLDOWN = 3.0

    # Minimum pitch velocity (degrees/second) to qualify as a fatigue nod.
    # Slow drift from -18° to -21° over 5 seconds is NOT a fatigue nod.
    # A genuine fatigue head-drop produces >5°/s velocity.
    MIN_NOD_VELOCITY = 3.0

    def __init__(self, cfg: SystemConfig):
        self.cfg = cfg
        self._down_threshold = cfg.posture.downward_pitch_threshold
        self._nod_duration = cfg.posture.nod_min_duration
        self._instability_limit = cfg.posture.posture_instability_limit

        # Heavier smoothing (0.10 vs. old 0.15) to suppress jitter
        self._smoother_pitch = EMASmoother(0.10)
        self._smoother_yaw = EMASmoother(0.10)
        self._smoother_roll = EMASmoother(0.10)

        self._yaw_buffer = deque(maxlen=30)
        self._roll_buffer = deque(maxlen=30)

        self._is_nodding_state = False
        self._nod_start_time = 0.0
        self._posture_confidence = 0.0

        # Cooldown state
        self._last_nod_end_time = 0.0

        # Velocity tracking
        self._prev_pitch = None
        self._prev_time = None

    def update(self, raw_pitch: float, raw_yaw: float, raw_roll: float, now: float):
        smoothed_pitch = self._smoother_pitch.update(raw_pitch)
        smoothed_yaw = self._smoother_yaw.update(raw_yaw)
        smoothed_roll = self._smoother_roll.update(raw_roll)

        self._yaw_buffer.append(smoothed_yaw)
        self._roll_buffer.append(smoothed_roll)

        # 1. Calculate Posture Instability (Sliding Window Variance of Yaw/Roll)
        instability = 0.0
        if len(self._yaw_buffer) > 1:
            mean_y = sum(self._yaw_buffer) / len(self._yaw_buffer)
            mean_r = sum(self._roll_buffer) / len(self._roll_buffer)
            var_y = sum((y - mean_y)**2 for y in self._yaw_buffer) / len(self._yaw_buffer)
            var_r = sum((r - mean_r)**2 for r in self._roll_buffer) / len(self._roll_buffer)
            instability = var_y + var_r

        # 2. Compute pitch velocity (degrees per second)
        pitch_velocity = 0.0
        if self._prev_pitch is not None and self._prev_time is not None:
            dt = now - self._prev_time
            if dt > 0:
                pitch_velocity = (smoothed_pitch - self._prev_pitch) / dt
        self._prev_pitch = smoothed_pitch
        self._prev_time = now

        # 3. Detect Downward Pitch (Nodding) with cooldown + velocity gate
        nod_duration = 0.0
        in_cooldown = (now - self._last_nod_end_time) < self.NOD_COOLDOWN

        if smoothed_pitch < self._down_threshold and not in_cooldown:
            if not self._is_nodding_state:
                # Only start a nod if pitch is actively dropping
                # (negative velocity = chin moving toward chest)
                if pitch_velocity < -self.MIN_NOD_VELOCITY:
                    self._is_nodding_state = True
                    self._nod_start_time = now
            if self._is_nodding_state:
                nod_duration = now - self._nod_start_time
        else:
            if self._is_nodding_state:
                # Nod just ended — start cooldown
                self._last_nod_end_time = now
            self._is_nodding_state = False
            self._nod_start_time = 0.0
            self._posture_confidence = 0.0

        # 4. Calculate Posture Fatigue Confidence
        if self._is_nodding_state:
            # Base magnitude: How far past the threshold?
            mag = (self._down_threshold - smoothed_pitch) / 10.0
            mag_factor = min(max(mag, 0.0), 1.0)

            # Duration factor
            dur_factor = min(nod_duration / self._nod_duration, 1.0)

            # Instability boost
            inst_boost = min(instability / self._instability_limit, 1.0) * 0.2

            self._posture_confidence = min((mag_factor * 0.4) + (dur_factor * 0.4) + inst_boost, 1.0)

            # Decay confidence if not fully meeting duration
            if nod_duration < self._nod_duration:
                self._posture_confidence *= (nod_duration / self._nod_duration)

        is_nodding_final = self._posture_confidence >= self.cfg.posture.confidence_threshold

        return (
            smoothed_pitch, smoothed_yaw, smoothed_roll,
            instability, is_nodding_final, nod_duration, self._posture_confidence
        )


# ═══════════════════════════════════════════════════════════════════════
# Temporal Analyzer — Orchestrator
# ═══════════════════════════════════════════════════════════════════════

class TemporalAnalyzer:
    """
    Top-level orchestrator that owns the EyeClosureAnalyzer and
    YawnAnalyzer.  Call ``update()`` once per frame with raw EAR and MAR
    values; it returns a TemporalState snapshot.

    Usage:
        analyzer = TemporalAnalyzer(cfg)
        state = analyzer.update(avg_ear=0.28, mar=0.15)
    """

    def __init__(self, cfg: SystemConfig):
        self.cfg = cfg
        self.eye_analyzer = EyeClosureAnalyzer(cfg)
        self.yawn_analyzer = YawnAnalyzer(cfg)
        self.posture_analyzer = PostureAnalyzer(cfg)

    def update(self, raw_ear: float, raw_mar: float, raw_pitch: float = 0.0, raw_yaw: float = 0.0, raw_roll: float = 0.0, timestamp: float = None) -> TemporalState:
        """
        Process one frame of EAR and MAR data.

        Args:
            raw_ear: Raw average EAR (bilateral mean).
            raw_mar: Raw MAR value.
            timestamp: Monotonic time for this frame. Defaults to
                time.monotonic() for the live camera path. Offline video
                evaluation MUST pass the video-clock time (frame_index / fps)
                so temporal integration reflects the recording, not the
                processing speed (frozen evaluation protocol).

        Returns:
            TemporalState snapshot for this frame.
        """
        now = time.monotonic() if timestamp is None else timestamp

        # --- Eye analysis ---
        (
            smoothed_ear,
            is_eye_closed,
            eye_closure_duration,
            is_drowsy,
            total_blinks,
            closure_ratio,
            ear_history,
        ) = self.eye_analyzer.update(raw_ear, now)

        # --- Yawn analysis ---
        (
            smoothed_mar,
            is_mouth_open,
            yawn_duration,
            is_yawning,
            total_yawns,
            yawn_confidence,
            is_speaking,
            mar_jitter,
        ) = self.yawn_analyzer.update(raw_mar, now)

        # --- Blink detection (closure in blink range) ---
        is_blinking = (
            is_eye_closed
            and eye_closure_duration <= self.cfg.temporal.blink_max_duration
        )

        # --- Posture analysis ---
        (
            smoothed_pitch,
            smoothed_yaw,
            smoothed_roll,
            posture_instability,
            is_nodding,
            nod_duration,
            posture_confidence
        ) = self.posture_analyzer.update(raw_pitch, raw_yaw, raw_roll, now)

        return TemporalState(
            raw_ear=raw_ear,
            raw_mar=raw_mar,
            raw_pitch=raw_pitch,
            raw_yaw=raw_yaw,
            raw_roll=raw_roll,
            smoothed_ear=smoothed_ear,
            smoothed_mar=smoothed_mar,
            smoothed_pitch=smoothed_pitch,
            smoothed_yaw=smoothed_yaw,
            smoothed_roll=smoothed_roll,
            is_eye_closed=is_eye_closed,
            eye_closure_duration=eye_closure_duration,
            closure_ratio=closure_ratio,
            is_drowsy=is_drowsy,
            is_blinking=is_blinking,
            total_blinks=total_blinks,
            is_mouth_open=is_mouth_open,
            yawn_duration=yawn_duration,
            is_yawning=is_yawning,
            total_yawns=total_yawns,
            yawn_confidence=yawn_confidence,
            is_speaking=is_speaking,
            mar_jitter=mar_jitter,
            ear_history=ear_history,
            posture_instability=posture_instability,
            is_nodding=is_nodding,
            nod_duration=nod_duration,
            posture_confidence=posture_confidence
        )

    def reset(self):
        """Reset both sub-analyzers."""
        self.eye_analyzer.reset()
        self.yawn_analyzer.reset()
