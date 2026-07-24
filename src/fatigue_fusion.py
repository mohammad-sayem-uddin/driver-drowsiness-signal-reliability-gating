"""
Fatigue Fusion Engine — Multi-Factor Behavioral Fusion
========================================================
Combines per-cue confidence signals (EAR, MAR, Head Pose) into a
unified fatigue_score (0.0–1.0) and graduated FatigueSeverity level.

Design principles:
    1. Weighted fusion — cues contribute proportionally to their
       empirical reliability as fatigue predictors.
    2. Cue-agreement amplification — convergent evidence from
       independent channels boosts the fused score multiplicatively.
    3. Temporal accumulation with asymmetric rates — fatigue evidence
       builds faster than it decays, preventing transient spikes from
       triggering alerts while capturing sustained degradation.
    4. Hysteresis-based severity transitions — prevents oscillation at
       severity boundaries.
    5. Explainability — every FusionSnapshot contains per-cue
       contributions, active cue count, and temporal trend.

Architecture:
    This module is STATEFUL (maintains temporal accumulation buffer)
    but has NO side effects (no alarms, no logging, no I/O).
    StateManager consumes FusionSnapshot to determine DriverStatus.

Usage:
    from src.fatigue_fusion import FatigueFusionEngine
    engine = FatigueFusionEngine(cfg)
    snapshot = engine.update(temporal_state)
"""

import time
from dataclasses import dataclass
from enum import Enum, auto
from collections import deque

from src.config import SystemConfig
from src.temporal_analyzer import TemporalState


# ═══════════════════════════════════════════════════════════════════════
# Fatigue Severity Levels
# ═══════════════════════════════════════════════════════════════════════

class FatigueSeverity(Enum):
    """
    Graduated fatigue severity levels.

    Binary (drowsy/alert) systems suffer from two failure modes:
      1. Late detection — the threshold must be high enough to avoid
         false positives, which delays genuine fatigue alerts.
      2. Alert fatigue — a single alarm tone for all situations
         desensitizes the driver.

    Graduated severity addresses both:
      - SLIGHT triggers early HUD warnings before fatigue deepens.
      - MODERATE introduces audible cues without full alarm.
      - SEVERE fires the full alarm with maximum urgency.
    """
    ALERT = 0
    SLIGHT_FATIGUE = 1
    MODERATE_FATIGUE = 2
    SEVERE_FATIGUE = 3


# ═══════════════════════════════════════════════════════════════════════
# Fusion Snapshot — Per-frame output
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class FusionSnapshot:
    """
    Complete fusion state for one frame.  Consumed by StateManager
    and the HUD renderer.  Every field is designed for explainability —
    a researcher can inspect exactly how the final severity was reached.
    """
    # --- Unified fatigue score ---
    fatigue_score: float = 0.0          # Accumulated, 0.0–1.0
    raw_score: float = 0.0             # Instantaneous (before accumulation)
    severity: FatigueSeverity = FatigueSeverity.ALERT

    # --- Per-cue contributions (for explainability / HUD) ---
    ear_confidence: float = 0.0         # Normalized 0–1 per-cue confidence
    mar_confidence: float = 0.0
    pose_confidence: float = 0.0
    ear_contribution: float = 0.0       # Weighted: ear_confidence × ear_weight
    mar_contribution: float = 0.0
    pose_contribution: float = 0.0

    # --- Cue agreement ---
    active_cue_count: int = 0           # How many cues exceed cue_active_threshold
    agreement_multiplier: float = 1.0   # Applied bonus (1.0, 1.3, or 1.5)

    # --- Temporal trend ---
    temporal_trend: str = "stable"      # "rising", "falling", or "stable"

    # --- Reliability attenuation ---
    reliability_applied: float = 1.0    # The reliability factor that was applied


# ═══════════════════════════════════════════════════════════════════════
# Fatigue Fusion Engine
# ═══════════════════════════════════════════════════════════════════════

class FatigueFusionEngine:
    """
    Core intelligence system for multi-factor fatigue estimation.

    Per-frame pipeline:
        1. Extract per-cue confidence from TemporalState.
        2. Compute weighted sum.
        3. Apply cue-agreement bonus.
        4. Temporally accumulate (asymmetric EMA).
        5. Classify severity with hysteresis.
        6. Detect temporal trend.
    """

    def __init__(self, cfg: SystemConfig):
        self.cfg = cfg
        fc = cfg.fusion

        # Weights (normalized to sum to 1.0 for safety)
        total_w = fc.ear_weight + fc.mar_weight + fc.pose_weight
        self._w_ear = fc.ear_weight / total_w
        self._w_mar = fc.mar_weight / total_w
        self._w_pose = fc.pose_weight / total_w

        # Agreement
        self._bonus_2 = fc.agreement_bonus_2cue
        self._bonus_3 = fc.agreement_bonus_3cue
        self._cue_active_thresh = fc.cue_active_threshold

        # Temporal accumulation
        self._acc_rate = fc.temporal_accumulation_rate
        self._decay_rate = fc.temporal_decay_rate
        self._accumulated_score = 0.0

        # Severity thresholds
        self._thresh_slight = fc.slight_threshold
        self._thresh_moderate = fc.moderate_threshold
        self._thresh_severe = fc.severe_threshold
        self._hysteresis = fc.severity_hysteresis
        self._min_cue_agreement = fc.min_cue_agreement

        # Current severity (for hysteresis)
        self._current_severity = FatigueSeverity.ALERT

        # Trend detection buffer
        self._score_history = deque(maxlen=30)

    # ───────────────────────────────────────────────────────────────────
    # Public API
    # ───────────────────────────────────────────────────────────────────

    def update(self, ts: TemporalState, reliability: float = 1.0) -> FusionSnapshot:
        """
        Process one frame of temporal state into a fusion snapshot.

        Args:
            ts: TemporalState from TemporalAnalyzer.update().
            reliability: System reliability score (0–1) from RobustnessGuard.
                         Multiplicatively attenuates the raw score before
                         temporal accumulation.  1.0 = no attenuation.

        Returns:
            FusionSnapshot with fatigue score, severity, and per-cue details.
        """
        # ── Step 1: Extract per-cue confidence ──────────────────────
        ear_conf = self._extract_ear_confidence(ts)
        mar_conf = self._extract_mar_confidence(ts)
        pose_conf = self._extract_pose_confidence(ts)

        # ── Step 2: Weighted fusion ─────────────────────────────────
        ear_contrib = ear_conf * self._w_ear
        mar_contrib = mar_conf * self._w_mar
        pose_contrib = pose_conf * self._w_pose
        raw_score = ear_contrib + mar_contrib + pose_contrib

        # ── Step 3: Cue agreement bonus ─────────────────────────────
        active_count = sum(1 for c in [ear_conf, mar_conf, pose_conf]
                          if c > self._cue_active_thresh)

        if active_count >= 3:
            multiplier = self._bonus_3
        elif active_count >= 2:
            multiplier = self._bonus_2
        else:
            multiplier = 1.0

        raw_score = min(raw_score * multiplier, 1.0)

        # ── Step 3b: Reliability attenuation ─────────────────────────
        # Under degraded signal conditions (low light, jitter, etc.),
        # the RobustnessGuard produces reliability < 1.0, which dampens
        # the raw score.  This requires STRONGER fatigue evidence to
        # trigger alerts when the signal is unreliable.
        reliability = max(0.0, min(1.0, reliability))
        raw_score *= reliability

        # ── Step 4: Temporal accumulation (asymmetric EMA) ──────────
        if raw_score > self._accumulated_score:
            # Rising — accumulate at faster rate
            alpha = self._acc_rate
        else:
            # Falling — decay at slower rate
            alpha = self._decay_rate

        self._accumulated_score = (
            alpha * raw_score + (1.0 - alpha) * self._accumulated_score
        )
        # Clamp
        self._accumulated_score = max(0.0, min(1.0, self._accumulated_score))

        # ── Step 5: Severity classification with hysteresis ─────────
        severity = self._classify_severity(
            self._accumulated_score, active_count
        )
        self._current_severity = severity

        # ── Step 6: Trend detection ─────────────────────────────────
        self._score_history.append(self._accumulated_score)
        trend = self._detect_trend()

        return FusionSnapshot(
            fatigue_score=self._accumulated_score,
            raw_score=raw_score,
            severity=severity,
            ear_confidence=ear_conf,
            mar_confidence=mar_conf,
            pose_confidence=pose_conf,
            ear_contribution=ear_contrib,
            mar_contribution=mar_contrib,
            pose_contribution=pose_contrib,
            active_cue_count=active_count,
            agreement_multiplier=multiplier,
            temporal_trend=trend,
            reliability_applied=reliability,
        )

    def reset(self):
        """Reset all temporal state."""
        self._accumulated_score = 0.0
        self._current_severity = FatigueSeverity.ALERT
        self._score_history.clear()

    # ───────────────────────────────────────────────────────────────────
    # Per-cue confidence extraction
    # ───────────────────────────────────────────────────────────────────

    def _extract_ear_confidence(self, ts: TemporalState) -> float:
        """
        Convert EAR temporal state into a 0–1 confidence score.

        Uses closure_ratio (already 0–1, representing progress toward
        the drowsiness trigger duration) as the primary signal.  If the
        temporal analyzer has already flagged is_drowsy, we floor the
        confidence at 0.8 to ensure the fusion engine reflects this.
        """
        conf = ts.closure_ratio  # 0.0–1.0

        # If temporal analyzer already determined drowsiness, boost
        if ts.is_drowsy:
            conf = max(conf, 0.8)

        return min(conf, 1.0)

    def _extract_mar_confidence(self, ts: TemporalState) -> float:
        """
        Convert MAR temporal state into a 0–1 confidence score.

        Uses the yawn_confidence from the YawnAnalyzer (already 0–1,
        incorporating MAR magnitude, duration, and jitter filtering).

        CRITICAL: If the temporal analyzer detected speech (high MAR
        jitter), we suppress the MAR contribution entirely.  Speech
        is the #1 source of MAR false positives.
        """
        if ts.is_speaking:
            return 0.0

        return min(ts.yawn_confidence, 1.0)

    def _extract_pose_confidence(self, ts: TemporalState) -> float:
        """
        Convert head pose temporal state into a 0–1 confidence score.

        Uses posture_confidence from PostureAnalyzer (already 0–1,
        incorporating pitch magnitude, nod duration, and instability).
        """
        return min(ts.posture_confidence, 1.0)

    # ───────────────────────────────────────────────────────────────────
    # Severity classification
    # ───────────────────────────────────────────────────────────────────

    def _classify_severity(self, score: float, active_cues: int) -> FatigueSeverity:
        """
        Map accumulated score to severity level with hysteresis.

        Hysteresis prevents oscillation: to de-escalate from SEVERE to
        MODERATE, the score must drop below (severe_threshold - hysteresis).
        To escalate, the score only needs to exceed the threshold.

        Additionally, escalation beyond SLIGHT requires at least
        min_cue_agreement active cues.  This prevents a single noisy
        cue from triggering high-severity alarms.
        """
        current = self._current_severity
        h = self._hysteresis

        # --- Check for escalation (upward) ---
        if score >= self._thresh_severe and active_cues >= self._min_cue_agreement:
            return FatigueSeverity.SEVERE_FATIGUE

        if score >= self._thresh_moderate and active_cues >= self._min_cue_agreement:
            # Don't de-escalate from SEVERE unless below threshold - hysteresis
            if current == FatigueSeverity.SEVERE_FATIGUE:
                if score >= (self._thresh_severe - h):
                    return FatigueSeverity.SEVERE_FATIGUE
            return FatigueSeverity.MODERATE_FATIGUE

        if score >= self._thresh_slight:
            # Don't de-escalate from MODERATE unless below threshold - hysteresis
            if current == FatigueSeverity.MODERATE_FATIGUE:
                if score >= (self._thresh_moderate - h):
                    return FatigueSeverity.MODERATE_FATIGUE
            if current == FatigueSeverity.SEVERE_FATIGUE:
                if score >= (self._thresh_severe - h):
                    return FatigueSeverity.SEVERE_FATIGUE
                if score >= (self._thresh_moderate - h):
                    return FatigueSeverity.MODERATE_FATIGUE
            return FatigueSeverity.SLIGHT_FATIGUE

        # --- Below slight threshold ---
        # Apply hysteresis for de-escalation from SLIGHT
        if current == FatigueSeverity.SLIGHT_FATIGUE:
            if score >= (self._thresh_slight - h):
                return FatigueSeverity.SLIGHT_FATIGUE
        if current.value >= FatigueSeverity.MODERATE_FATIGUE.value:
            if score >= (self._thresh_slight - h):
                return FatigueSeverity.SLIGHT_FATIGUE

        return FatigueSeverity.ALERT

    # ───────────────────────────────────────────────────────────────────
    # Trend detection
    # ───────────────────────────────────────────────────────────────────

    def _detect_trend(self) -> str:
        """
        Classify the temporal trend of the accumulated score as
        rising, falling, or stable by comparing the recent mean
        to the older mean within the history buffer.
        """
        buf = self._score_history
        if len(buf) < 10:
            return "stable"

        mid = len(buf) // 2
        older = list(buf)[:mid]
        recent = list(buf)[mid:]

        mean_old = sum(older) / len(older)
        mean_new = sum(recent) / len(recent)
        delta = mean_new - mean_old

        if delta > 0.03:
            return "rising"
        elif delta < -0.03:
            return "falling"
        return "stable"
