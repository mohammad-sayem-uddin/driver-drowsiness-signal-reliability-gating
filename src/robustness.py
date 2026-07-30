"""
Robustness Guard — Signal Quality Monitoring & FP Reduction
=============================================================
Computes a per-frame ``system_reliability`` score (0.0–1.0) that
quantifies how trustworthy the current sensor signal is.  The score
is consumed by the FatigueFusionEngine as a multiplicative gate on
the raw fatigue score, ensuring that degraded-signal conditions
(low light, camera shake, landmark hallucination, occlusion)
automatically suppress false alarms without reducing sensitivity
under good conditions.

Design principles:
    1. Graceful degradation — reliability is continuous, not binary.
       There is no hard cutoff where the system "gives up."
    2. Three independent sub-scores — landmark stability, frame
       brightness, and cue consistency — each capturing a different
       failure mode.  (A per-frame tracking-confidence channel was
       removed: MediaPipe FaceMesh does not populate a per-landmark
       ``visibility``/confidence value, so any such component would be
       a constant, not a real signal.  See CHANGELOG / freeze report
       precondition 4.)
    3. Multiplicative composition — the final reliability is the
       weighted product of sub-scores, so a single severely degraded
       channel can dominate.
    4. Temporal smoothing — the final reliability uses EMA to prevent
       single-frame drops from causing alarm flicker.
    5. Zero side effects — purely analytical, no alarms, no I/O.

Architecture:
    main.py extracts raw signal quality metrics (landmark positions,
    frame brightness, tracking confidence) each frame and passes them
    to RobustnessGuard.update().  The returned RobustnessSnapshot is
    forwarded to the fusion engine and state manager.

Usage:
    from src.robustness import RobustnessGuard, SignalQuality
    guard = RobustnessGuard(cfg)
    snapshot = guard.update(signal_quality)
"""

import math
from dataclasses import dataclass, field
from collections import deque
from typing import List, Tuple, Optional

from src.config import SystemConfig


# ═══════════════════════════════════════════════════════════════════════
# Signal Quality — Raw per-frame input
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class SignalQuality:
    """
    Raw signal quality metrics extracted in main.py each frame.

    These are the INPUTS to the RobustnessGuard.  main.py is
    responsible for computing them because it has direct access
    to the raw frame pixels and landmark coordinates.
    """
    # Mean frame-to-frame displacement of key landmarks (pixels).
    # Computed as: mean(||L_t - L_{t-1}||) over ~15 key points.
    # 0.0 if no previous frame or no face detected.
    landmark_jitter: float = 0.0

    # Mean pixel intensity (0–255) of the face bounding box region.
    # Computed on the grayscale face ROI.  0.0 if no face.
    frame_brightness: float = 128.0

    # Whether a face was detected this frame.
    face_visible: bool = True


# ═══════════════════════════════════════════════════════════════════════
# Robustness Snapshot — Per-frame output
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class RobustnessSnapshot:
    """
    Complete robustness assessment for one frame.  Every sub-score
    is exposed for HUD display, research logging, and ablation studies.
    """
    # --- Sub-scores (0–1 each, higher = better) ---
    landmark_stability: float = 1.0
    brightness_quality: float = 1.0
    cue_consistency: float = 1.0

    # --- Final reliability (0–1, higher = more trustworthy) ---
    system_reliability: float = 1.0
    heuristic_reliability: float = 1.0
    learned_reliability: float = 1.0
    estimator_mode: str = "geometric"

    # --- Raw inputs (forwarded for logging) ---
    landmark_jitter: float = 0.0
    frame_brightness: float = 128.0

    # --- Advisory flags ---
    alert_suppressed: bool = False


# ═══════════════════════════════════════════════════════════════════════
# Phase 2.5: Learned Reliability Estimator Framework
# ═══════════════════════════════════════════════════════════════════════

class LearnedReliabilityEstimator:
    """
    Parametric, Temperature-Calibrated Logistic Reliability Estimator.

    Formulates signal reliability as a calibrated probabilistic estimate:
        R_learned(S; w, b, T) = sigmoid((w^T * ln(S + epsilon) + b) / T)

    Theoretical Grounding:
    When w = [0.45, 0.30, 0.25], b = 0.0, and T = 1.0, this formulation
    analytically reduces to the weighted geometric mean:
        exp(0.45*ln S_stab + 0.30*ln S_bright + 0.25*ln S_consist)
      = S_stab^0.45 * S_bright^0.30 * S_consist^0.25
    """

    def __init__(
        self,
        weights: Tuple[float, float, float] = (0.45, 0.30, 0.25),
        bias: float = 0.0,
        temperature: float = 1.0,
        epsilon: float = 1e-6,
    ):
        self.weights = [float(w) for w in weights]
        self.bias = float(bias)
        self.temperature = max(1e-4, float(temperature))
        self.epsilon = float(epsilon)

    def estimate(
        self,
        stability: float,
        brightness: float,
        consistency: float,
    ) -> float:
        s_stab = max(self.epsilon, float(stability))
        s_bright = max(self.epsilon, float(brightness))
        s_consist = max(self.epsilon, float(consistency))

        # Log-space feature transformation
        log_stab = math.log(s_stab)
        log_bright = math.log(s_bright)
        log_consist = math.log(s_consist)

        logit = (
            self.weights[0] * log_stab
            + self.weights[1] * log_bright
            + self.weights[2] * log_consist
            + self.bias
        ) / self.temperature

        if self.bias == 0.0 and self.temperature == 1.0:
            # Analytical equivalence to geometric mean
            return max(0.0, min(1.0, math.exp(logit)))
        else:
            # Calibrated Logistic Sigmoid
            try:
                return float(1.0 / (1.0 + math.exp(-logit)))
            except OverflowError:
                return 0.0 if logit < 0 else 1.0


# ═══════════════════════════════════════════════════════════════════════
# Robustness Guard
# ═══════════════════════════════════════════════════════════════════════

class RobustnessGuard:
    """
    Signal quality monitor and reliability estimator.

    Per-frame pipeline:
        1. Score landmark stability (jitter → 0–1).
        2. Score frame brightness (too dark / too bright → penalty).
        3. Score cue consistency (variance of fusion confidences).
        4. Evaluate heuristic vs learned reliability.
        5. EMA-smooth the reliability to prevent flicker.
        6. Recommend alert suppression if reliability is very low.
    """

    def __init__(self, cfg: SystemConfig):
        rc = cfg.robustness
        self._cfg = cfg

        # Landmark stability parameters
        self._jitter_low = rc.jitter_low_threshold
        self._jitter_high = rc.jitter_high_threshold
        self._min_stability = rc.min_stability_score

        # Brightness parameters
        self._bright_low = rc.brightness_low
        self._bright_high = rc.brightness_high

        # Consistency tracking
        self._consistency_window = rc.consistency_window
        self._ear_conf_buffer = deque(maxlen=rc.consistency_window)
        self._mar_conf_buffer = deque(maxlen=rc.consistency_window)
        self._pose_conf_buffer = deque(maxlen=rc.consistency_window)

        # Reliability EMA
        self._ema_alpha = rc.reliability_ema_alpha
        self._smoothed_reliability = 1.0
        self._suppress_thresh = rc.alert_suppression_threshold

        # Frames-without-face counter (for consistency scoring)
        self._consecutive_no_face = 0

        # Phase 2.5: Learned Estimator Instance
        self.learned_estimator = LearnedReliabilityEstimator(
            weights=rc.learned_weights,
            bias=rc.learned_bias,
            temperature=rc.temperature,
        )
        self._estimator_mode = rc.reliability_estimator_mode

    # ───────────────────────────────────────────────────────────────────
    # Public API
    # ───────────────────────────────────────────────────────────────────

    def update(
        self,
        sq: SignalQuality,
        ear_conf: float = 0.0,
        mar_conf: float = 0.0,
        pose_conf: float = 0.0,
    ) -> RobustnessSnapshot:
        """
        Process one frame of signal quality data.

        Args:
            sq: SignalQuality metrics from main.py.
            ear_conf: EAR confidence from the fusion engine (for consistency).
            mar_conf: MAR confidence from the fusion engine.
            pose_conf: Pose confidence from the fusion engine.

        Returns:
            RobustnessSnapshot with all sub-scores and final reliability.
        """
        if not sq.face_visible:
            self._consecutive_no_face += 1
            # When face is lost, reliability degrades gradually
            face_penalty = max(0.3, 1.0 - (self._consecutive_no_face * 0.1))
            self._smoothed_reliability = (
                self._ema_alpha * face_penalty
                + (1.0 - self._ema_alpha) * self._smoothed_reliability
            )
            return RobustnessSnapshot(
                landmark_stability=0.0,
                brightness_quality=0.5,
                cue_consistency=0.5,
                system_reliability=self._smoothed_reliability,
                heuristic_reliability=self._smoothed_reliability,
                learned_reliability=self._smoothed_reliability,
                estimator_mode=self._estimator_mode,
                landmark_jitter=sq.landmark_jitter,
                frame_brightness=sq.frame_brightness,
                alert_suppressed=self._smoothed_reliability < self._suppress_thresh,
            )

        self._consecutive_no_face = 0

        # ── Step 1: Landmark stability ──────────────────────────────
        stability = self._score_landmark_stability(sq.landmark_jitter)

        # ── Step 2: Brightness quality ──────────────────────────────
        brightness = self._score_brightness(sq.frame_brightness)

        # ── Step 3: Cue consistency ─────────────────────────────────
        consistency = self._score_cue_consistency(ear_conf, mar_conf, pose_conf)

        # ── Step 4: Combine sub-scores ──────────────────────────────
        # Classical Heuristic Geometric Mean (3 real components,
        # weights renormalized to sum to 1.0 after dropping the phantom
        # tracking-confidence channel — see freeze-report precondition 4).
        heuristic_reliability = (
            (stability ** 0.45)
            * (brightness ** 0.30)
            * (consistency ** 0.25)
        )
        heuristic_reliability = max(0.0, min(1.0, heuristic_reliability))

        # Learned Estimator Score
        learned_reliability = self.learned_estimator.estimate(
            stability, brightness, consistency
        )

        # Select Mode
        if self._estimator_mode == "learned_logistic":
            target_reliability = learned_reliability
        elif self._estimator_mode == "ensemble":
            target_reliability = 0.5 * heuristic_reliability + 0.5 * learned_reliability
        else:
            target_reliability = heuristic_reliability

        # ── Step 5: EMA smooth ──────────────────────────────────────
        self._smoothed_reliability = (
            self._ema_alpha * target_reliability
            + (1.0 - self._ema_alpha) * self._smoothed_reliability
        )

        # ── Step 6: Alert suppression recommendation ────────────────
        suppressed = self._smoothed_reliability < self._suppress_thresh

        return RobustnessSnapshot(
            landmark_stability=stability,
            brightness_quality=brightness,
            cue_consistency=consistency,
            system_reliability=self._smoothed_reliability,
            heuristic_reliability=heuristic_reliability,
            learned_reliability=learned_reliability,
            estimator_mode=self._estimator_mode,
            landmark_jitter=sq.landmark_jitter,
            frame_brightness=sq.frame_brightness,
            alert_suppressed=suppressed,
        )

    def reset(self):
        """Reset all temporal state."""
        self._smoothed_reliability = 1.0
        self._consecutive_no_face = 0
        self._ear_conf_buffer.clear()
        self._mar_conf_buffer.clear()
        self._pose_conf_buffer.clear()

    # ───────────────────────────────────────────────────────────────────
    # Sub-score computations
    # ───────────────────────────────────────────────────────────────────

    def _score_landmark_stability(self, jitter: float) -> float:
        """
        Map landmark jitter (pixels) to a 0–1 stability score.

        Uses a linear ramp between jitter_low and jitter_high:
          - jitter <= low  → 1.0 (perfect)
          - jitter >= high → min_stability (floor)
          - between        → linear interpolation

        Rationale: Below 2px, jitter is within the noise floor of
        MediaPipe's landmark detector at 720p.  Above 8px, the
        landmarks are likely hallucinating or the camera is shaking
        severely.
        """
        if jitter <= self._jitter_low:
            return 1.0
        if jitter >= self._jitter_high:
            return self._min_stability

        # Linear interpolation
        t = (jitter - self._jitter_low) / (self._jitter_high - self._jitter_low)
        return 1.0 - t * (1.0 - self._min_stability)

    def _score_brightness(self, brightness: float) -> float:
        """
        Map face-ROI brightness to a 0–1 quality score.

        Uses a trapezoidal function:
          - [0, 30]      → 0.3 (very dark, near-total loss)
          - [30, low]     → linear ramp from 0.3 to 1.0
          - [low, high]   → 1.0 (ideal range)
          - [high, 240]   → linear decay from 1.0 to 0.5
          - [240, 255]    → 0.5 (overexposed but still some signal)

        Rationale: At very low brightness, webcam read noise dominates
        and landmark detection degrades.  At overexposure, facial
        features wash out, but landmarks are still partially usable
        because MediaPipe operates on the RGB image, not edges.
        """
        if brightness < 30:
            return 0.3
        if brightness < self._bright_low:
            t = (brightness - 30) / (self._bright_low - 30)
            return 0.3 + t * 0.7
        if brightness <= self._bright_high:
            return 1.0
        if brightness < 240:
            t = (brightness - self._bright_high) / (240 - self._bright_high)
            return 1.0 - t * 0.5
        return 0.5

    def _score_cue_consistency(
        self, ear_conf: float, mar_conf: float, pose_conf: float
    ) -> float:
        """
        Score the temporal consistency of per-cue confidences.

        High frame-to-frame variance in cue confidences indicates
        unreliable signals (e.g., flickering landmarks, intermittent
        occlusion).  Stable confidences — whether high or low —
        indicate a trustworthy signal.

        Uses the mean coefficient of variation (σ/μ) across cues
        over the consistency window.  Low CV = consistent = high score.
        """
        self._ear_conf_buffer.append(ear_conf)
        self._mar_conf_buffer.append(mar_conf)
        self._pose_conf_buffer.append(pose_conf)

        if len(self._ear_conf_buffer) < 5:
            # Not enough history — assume consistent
            return 1.0

        cvs = []
        for buf in [self._ear_conf_buffer, self._mar_conf_buffer, self._pose_conf_buffer]:
            vals = list(buf)
            mean_v = sum(vals) / len(vals)
            if mean_v < 0.01:
                # Near-zero mean — consistency is trivially high
                cvs.append(0.0)
                continue
            var_v = sum((v - mean_v) ** 2 for v in vals) / len(vals)
            std_v = math.sqrt(var_v)
            cv = std_v / mean_v  # Coefficient of variation
            cvs.append(cv)

        # Average CV across cues
        mean_cv = sum(cvs) / len(cvs)

        # Map CV to score: CV < 0.2 → 1.0, CV > 1.0 → 0.3
        if mean_cv <= 0.2:
            return 1.0
        if mean_cv >= 1.0:
            return 0.3
        t = (mean_cv - 0.2) / 0.8
        return 1.0 - t * 0.7
