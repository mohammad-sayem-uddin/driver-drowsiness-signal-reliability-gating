"""
Centralized Configuration — Single Source of Truth
====================================================
All tunable parameters for the Driver Drowsiness Detection System are
defined here.  Every other module imports from this file.  This guarantees:

    1. Threshold consistency across entry points (main.py, ear_processor.py).
    2. Reproducible experiments — print SystemConfig at session start.
    3. Clean ablation studies — change one value, re-run, compare.

Usage:
    from src.config import SystemConfig
    cfg = SystemConfig()          # all defaults
    cfg.detection.ear_threshold   # 0.21
"""

from dataclasses import dataclass, field
from typing import Tuple, List, Optional


# ═══════════════════════════════════════════════════════════════════════
# Detection Thresholds
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class DetectionConfig:
    """
    Metric thresholds for EAR and MAR classification.

    Attributes
    ----------
    ear_threshold : float
        EAR below this value → eye considered "closed".
        Literature range: 0.18–0.27 depending on eye morphology.
        Default 0.21 is a balanced baseline for MediaPipe Face Mesh
        landmarks (Soukupová & Čech, 2016).

    ear_hysteresis : float
        Added to ear_threshold to form the *upper* threshold for
        hysteresis.  The eye must exceed (ear_threshold + ear_hysteresis)
        to transition back to OPEN.  Prevents rapid toggling when EAR
        hovers near the threshold boundary.
        Effective open-threshold = 0.21 + 0.03 = 0.24.

    mar_threshold : float
        MAR above this value → mouth considered "open" (yawn candidate).
        Typical relaxed-mouth MAR: 0.15–0.30.
        Typical yawn MAR: 0.55–0.90.

    mar_hysteresis : float
        Subtracted from mar_threshold to form the *lower* threshold for
        hysteresis.  Mouth must close below (mar_threshold - mar_hysteresis)
        to reset yawn state.
    """
    ear_threshold: float = 0.21
    ear_hysteresis: float = 0.03
    mar_threshold: float = 0.55
    mar_hysteresis: float = 0.05


# ═══════════════════════════════════════════════════════════════════════
# Temporal Parameters (wall-clock, FPS-independent)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class TemporalConfig:
    """
    Time-based duration thresholds.  All values in **seconds**.
    These replace the old frame-count parameters (which were
    FPS-dependent and non-portable across devices).

    Attributes
    ----------
    eye_closure_duration : float
        Sustained closure time to trigger drowsiness alert.
        Literature range: 0.5–2.0 seconds.
        NHTSA/PERCLOS studies typically use 0.5–1.0 second.

    yawn_min_duration : float
        Minimum sustained mouth-open time to classify as yawn.
        Normal speech mouth openings: < 0.3 seconds.
        Real yawns: 2–6 seconds, but MAR exceeds threshold for ≥ 0.8s.
        Setting this to 0.8s eliminates virtually all speech artifacts.

    blink_min_duration : float
        Minimum closure to count as a blink (filters landmark jitter).
        A human blink lasts 100–400 ms.

    blink_max_duration : float
        Maximum closure to count as a blink (longer = drowsiness, not blink).
    """
    eye_closure_duration: float = 1.0
    yawn_min_duration: float = 0.8
    blink_min_duration: float = 0.05
    blink_max_duration: float = 0.4


# ═══════════════════════════════════════════════════════════════════════
# Advanced Yawning Analysis
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class YawnConfig:
    """
    Parameters for the advanced Temporal Yawning Analysis and Speech
    Differentiation module.

    Attributes
    ----------
    confidence_threshold : float
        The yawn_confidence score (0.0 to 1.0) required to trigger a
        formal yawn event. A higher score requires longer sustained MAR
        and lower variance (smoother mouth opening).
        Default: 0.6.

    speech_jitter_threshold : float
        The maximum allowed frame-to-frame jitter (Absolute Sum of Differences)
        of MAR. Speech creates high-frequency MAR spikes (high jitter > 0.08).
        Yawns create smooth, sustained curves (low jitter < 0.06).

    sliding_window_size : int
        The number of recent MAR samples used to calculate jitter.
        At 30 FPS, a window of 15 frames = 0.5 seconds of context.
    """
    confidence_threshold: float = 0.6
    speech_jitter_threshold: float = 0.05
    sliding_window_size: int = 15


# ═══════════════════════════════════════════════════════════════════════
# Posture & Head Pose Configuration
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class PostureConfig:
    """
    Parameters for the Head Pose Estimation and Fatigue Posture analysis.

    Attributes
    ----------
    downward_pitch_threshold : float
        The pitch angle (in degrees) below which the head is considered
        to be dropping/nodding. Negative pitch = chin towards chest.
        Typical resting pitch is 0 to -5. A severe nod is -15 to -25.
        Default: -15.0

    nod_min_duration : float
        Minimum sustained time (seconds) the head must be below the
        downward_pitch_threshold to trigger a fatigue nod event.
        Default: 0.5s (filters out rapid glances at the dashboard)

    posture_instability_limit : float
        The maximum allowed variance (jitter) in yaw/roll before the
        posture is considered "unstable" (bobbing). Extreme bobbing
        multiplies the fatigue confidence score.
        Default: 2.0 (degrees)

    confidence_threshold : float
        The posture_confidence score (0.0 to 1.0) required to trigger a
        formal NODDING state.
        Default: 0.6
    """
    downward_pitch_threshold: float = -20.0
    nod_min_duration: float = 1.5
    posture_instability_limit: float = 2.0
    confidence_threshold: float = 0.7


# ═══════════════════════════════════════════════════════════════════════
# Multi-Factor Fatigue Fusion
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class FusionConfig:
    """
    Parameters for the Multi-Factor Fatigue Fusion Engine.

    The fusion engine combines per-cue confidence scores (EAR, MAR, Head Pose)
    into a unified fatigue_score (0.0–1.0) using weighted summation, cue-agreement
    amplification, and temporal accumulation with asymmetric rise/decay rates.

    Attributes
    ----------
    ear_weight : float
        Weight assigned to the EAR (eye closure) confidence signal.
        Highest weight because sustained eye closure is the strongest
        single predictor of micro-sleep onset.

    mar_weight : float
        Weight assigned to the MAR (yawning) confidence signal.
        Lowest weight because yawning has the highest false-positive
        rate (speech artifacts) and is a weaker standalone fatigue indicator.

    pose_weight : float
        Weight assigned to the Head Pose (nodding/posture) confidence.
        Second highest because posture degradation frequently precedes
        eyelid closure during fatigue onset.

    agreement_bonus_2cue : float
        Multiplicative bonus applied when 2 cues are simultaneously active.
        Rewards convergent evidence from independent behavioral channels.

    agreement_bonus_3cue : float
        Multiplicative bonus when all 3 cues are active.

    temporal_accumulation_rate : float
        EMA-style rise rate. Controls how fast the accumulated score
        tracks upward toward the instantaneous raw score.
        Higher = faster response, lower = more inertia.

    temporal_decay_rate : float
        EMA-style decay rate when raw score drops below accumulated.
        Intentionally slower than accumulation_rate to create asymmetric
        behavior: fatigue evidence builds fast, clears slowly.

    slight_threshold : float
        Accumulated score above which the system enters SLIGHT_FATIGUE.

    moderate_threshold : float
        Accumulated score above which the system enters MODERATE_FATIGUE.

    severe_threshold : float
        Accumulated score above which the system enters SEVERE_FATIGUE.

    severity_hysteresis : float
        Required drop below the current severity threshold to de-escalate.
        Prevents oscillation at severity boundaries.

    min_cue_agreement : int
        Minimum number of active cues (confidence > 0.3) required to
        allow severity above SLIGHT_FATIGUE.  Set to 1 to allow
        single-cue escalation (e.g., prolonged eye closure alone), or 2
        to require multi-cue confirmation for higher severities.

    cue_active_threshold : float
        Per-cue confidence must exceed this value for the cue to count
        as "active" in the agreement calculation.
    """
    ear_weight: float = 0.45
    mar_weight: float = 0.25
    pose_weight: float = 0.30
    agreement_bonus_2cue: float = 1.3
    agreement_bonus_3cue: float = 1.5
    temporal_accumulation_rate: float = 0.08
    temporal_decay_rate: float = 0.04
    slight_threshold: float = 0.25
    moderate_threshold: float = 0.50
    severe_threshold: float = 0.75
    severity_hysteresis: float = 0.12
    min_cue_agreement: int = 1
    cue_active_threshold: float = 0.3


# ═══════════════════════════════════════════════════════════════════════
# Robustness & False-Positive Reduction
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class RobustnessConfig:
    """
    Parameters for the RobustnessGuard — signal quality monitoring and
    false-positive reduction through reliability-gated fusion.

    The RobustnessGuard computes a per-frame system_reliability (0–1)
    from three sub-scores: landmark stability, frame brightness, and
    cue consistency.  This reliability score multiplicatively
    attenuates the fusion engine's raw score, ensuring degraded-signal
    conditions (low light, camera shake, occlusion) automatically
    suppress false alarms without reducing sensitivity under good
    conditions.

    NOTE: A per-frame tracking-confidence sub-score was intentionally
    removed.  MediaPipe FaceMesh does not expose a real per-frame
    landmark ``visibility``/confidence value, so such a component would
    have been a constant (~0.9), inflating the reliability index with a
    phantom signal.  The three retained components are all genuinely
    measured each frame.

    Attributes
    ----------
    jitter_low_threshold : float
        Landmark jitter (mean px displacement) below this → perfect
        stability score (1.0).  Typical value: 2.0 px.

    jitter_high_threshold : float
        Jitter above this → minimum stability score.  Typical: 8.0 px.
        Values above 8px indicate severe camera shake or landmark
        hallucination.

    min_stability_score : float
        Floor for the landmark stability sub-score.  Even under extreme
        jitter, we don't drive reliability to zero — there is still
        some signal.

    brightness_low : int
        Face-ROI mean brightness below this → brightness penalty.
        Below ~60, most webcams produce significant read noise.

    brightness_high : int
        Face-ROI mean brightness above this → overexposure penalty.
        Above ~200, features wash out and landmarks lose precision.

    consistency_window : int
        Number of frames over which to track cue-confidence variance.

    reliability_ema_alpha : float
        EMA alpha for the final system_reliability score.  Lower values
        produce a more stable (slower-reacting) reliability metric.

    alert_suppression_threshold : float
        When system_reliability falls below this, non-severe alerts
        are suppressed.  SEVERE alerts are NEVER suppressed (safety).
    """
    jitter_low_threshold: float = 2.0
    jitter_high_threshold: float = 8.0
    min_stability_score: float = 0.3
    brightness_low: int = 60
    brightness_high: int = 200
    consistency_window: int = 15
    reliability_ema_alpha: float = 0.2
    alert_suppression_threshold: float = 0.5

    # --- Phase 2.5 Extensions: Learned Reliability Estimation Framework ---
    # Estimation mode: "geometric" (classical heuristic), "learned_logistic" (calibrated logistic), or "ensemble"
    # Weights correspond to (landmark_stability, brightness, cue_consistency)
    # and are renormalized to sum to 1.0 after dropping the phantom
    # tracking component (freeze-report precondition 4).
    reliability_estimator_mode: str = "geometric"
    learned_weights: Tuple[float, float, float] = (0.45, 0.30, 0.25)
    learned_bias: float = 0.0
    temperature: float = 1.0


# ═══════════════════════════════════════════════════════════════════════
# Alarm Behavior
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class AlarmConfig:
    """
    Alarm lifecycle parameters.

    Attributes
    ----------
    min_alarm_duration : float
        Once triggered, alarm plays for at least this many seconds even
        if the driver reopens their eyes.  Prevents alarm flickering and
        ensures the driver is fully alerted.

    cooldown_period : float
        After an alarm ends, no new alarm can trigger for this many
        seconds.  Prevents rapid re-triggering during EAR oscillation
        near the threshold boundary.

    face_loss_timeout : float
        If the face disappears for longer than this duration *without*
        a prior drowsiness sequence, the system enters FACE_LOST warning.
        If the face disappears *during* an active drowsiness sequence,
        the alarm immediately escalates (no timeout).

    face_loss_escalation : bool
        If True, losing the face during an active drowsiness sequence
        escalates the alarm rather than silencing it.  This handles the
        critical scenario where a drowsy driver slumps forward.

    log_file_path : str
        Path to the CSV event log for research data collection.
    """
    min_alarm_duration: float = 3.0
    cooldown_period: float = 5.0
    face_loss_timeout: float = 2.0
    face_loss_escalation: bool = True
    log_file_path: str = "drowsiness_events_log.csv"


# ═══════════════════════════════════════════════════════════════════════
# Signal Smoothing
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class SmoothingConfig:
    """
    Exponential Moving Average (EMA) parameters for signal filtering.

    EMA formula:  smoothed = α × raw + (1 − α) × previous_smoothed

    Attributes
    ----------
    ear_ema_alpha : float
        Smoothing factor for the EAR signal.  Lower α = heavier smoothing
        (more lag).  Higher α = less smoothing (more noise).
        0.3 provides a good balance: filters single-frame outliers while
        tracking genuine blinks with < 50ms perceptual delay.

    mar_ema_alpha : float
        Smoothing factor for the MAR signal.

    ear_history_size : int
        Number of recent EAR values stored for waveform visualization.
    """
    ear_ema_alpha: float = 0.3
    mar_ema_alpha: float = 0.3
    ear_history_size: int = 150


# ═══════════════════════════════════════════════════════════════════════
# Camera Configuration
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class CameraConfig:
    """
    Camera and display settings.

    Attributes
    ----------
    camera_id : int
        Device index for cv2.VideoCapture.  0 = default camera.

    capture_width : int
        Requested capture width (advisory; camera may use nearest supported).

    capture_height : int
        Requested capture height.

    display_width : int
        Window display width.

    display_height : int
        Window display height.
    """
    camera_id: int = 0
    capture_width: int = 1280
    capture_height: int = 720
    display_width: int = 960
    display_height: int = 720


# ═══════════════════════════════════════════════════════════════════════
# MediaPipe Face Mesh Configuration
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class FaceMeshConfig:
    """
    MediaPipe Face Mesh model parameters.

    Attributes
    ----------
    max_num_faces : int
        Maximum faces to track.  1 for driver-only monitoring.

    refine_landmarks : bool
        Enable iris landmarks (indices 468–477).

    min_detection_confidence : float
        BlazeFace detector threshold.

    min_tracking_confidence : float
        Frame-to-frame tracking threshold.
    """
    max_num_faces: int = 1
    refine_landmarks: bool = True
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


# ═══════════════════════════════════════════════════════════════════════
# CNN Validation Layer (Hybrid AI)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class CNNValidationConfig:
    """
    Parameters for the Tiny CNN Validation Layer.

    The CNN acts as a selective uncertainty resolver — it is invoked ONLY
    when the heuristic pipeline's EAR confidence falls into an ambiguous
    zone (near the hysteresis boundary).  It never replaces the temporal
    logic; it provides a learned second opinion for false-positive reduction.

    Attributes
    ----------
    enabled : bool
        Master toggle for the CNN validation layer.  When False, the
        system operates in pure heuristic mode (no CNN overhead).

    model_path : str
        Path to the TFLite model file.  If the file does not exist,
        the CNN layer degrades gracefully to heuristic-only mode.

    input_size : int
        Side length of the square grayscale input (input_size × input_size × 1).
        24×24 is sufficient for binary eye-state classification and keeps
        the parameter count under 10K.

    confidence_threshold : float
        CNN sigmoid output must exceed this threshold (or fall below
        1 - threshold) for the CNN verdict to be trusted.  Predictions
        between (1-threshold) and threshold are treated as "uncertain"
        and the heuristic verdict is preserved.

    uncertainty_zone_low : float
        Lower bound of the EAR uncertainty zone.  When smoothed_ear is
        between uncertainty_zone_low and uncertainty_zone_high, the CNN
        is eligible for invocation.

    uncertainty_zone_high : float
        Upper bound of the EAR uncertainty zone.

    max_invocations_per_second : int
        Rate-limiter to prevent CPU spikes from excessive CNN calls.
    """
    enabled: bool = True
    model_path: str = "models/eye_state_model.tflite"
    input_size: int = 24
    confidence_threshold: float = 0.75
    uncertainty_zone_low: float = 0.17
    uncertainty_zone_high: float = 0.27
    max_invocations_per_second: int = 5


# ═══════════════════════════════════════════════════════════════════════
# Optimization & Edge Deployment
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class OptimizationConfig:
    """
    Performance and deployment optimizations for edge hardware (e.g., Raspberry Pi).

    Attributes
    ----------
    headless_mode : bool
        If True, disables all cv2 rendering and GUI windows. Drastically
        reduces CPU overhead for production deployment.
        
    enable_profiling : bool
        If True, tracks pipeline latency and prints performance metrics
        to the terminal periodically.
        
    adaptive_frame_skipping : bool
        If True, intelligently skips MediaPipe processing on alternating
        frames when the driver is fully alert, saving CPU cycles.
        
    profiling_interval : int
        How often (in frames) to print the profiling summary.
    """
    headless_mode: bool = False
    enable_profiling: bool = True
    adaptive_frame_skipping: bool = True
    profiling_interval: int = 150


@dataclass
class AblationConfig:
    """
    Ablation switches for the frozen experimental protocol (§3, variants V0–V4).

    BOTH default to True, i.e. the full proposed system exactly as frozen — so
    the live app and every non-ablation run are UNCHANGED. The LOSO/ablation
    harness flips these to isolate each contribution:

        V0 baseline          : both False  (weighted fusion only)
        V1 + speech filter    : speech_filter_enabled=True,  gate off
        V2 + reliability gate : gate on, speech off
        V3 full               : both True   (== default frozen behavior)

    These flags NEVER alter the research design; they only enable/disable an
    already-frozen component so its individual effect can be measured. Turning
    a component OFF must never make the system less safe than the baseline —
    the SEVERE-never-suppressed invariant holds regardless.
    """
    # When False, the σ²(MAR) speech-jitter penalty in the yawn detector is
    # bypassed (talking-induced MAR spikes are no longer down-weighted).
    speech_filter_enabled: bool = True
    # When False, the multiplicative reliability gate is bypassed (reliability
    # is forced to 1.0 before fusion; SEVERE exemption is unaffected).
    reliability_gate_enabled: bool = True


@dataclass
class DatasetPathsConfig:
    """
    Centralized Dataset Path Configuration.
    Single source of truth for all dataset locations under Data/.
    """
    base_dir: str = "Data"
    mrl_eye_path: str = "Data/mrl_eye"
    drowsiness_detection_path: str = "Data/drowsiness_detection"
    nthu_ddd_path: str = "Data/nthu_ddd"
    yawdd_path: str = "Data/yawdd"


@dataclass
class SystemConfig:
    """
    Aggregated system configuration.  Instantiate once at startup and
    pass to all subsystems.  Print at session start for reproducibility.

    Example:
        cfg = SystemConfig()
        print(cfg)  # logs all parameter values for experiment records
    """
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    yawn: YawnConfig = field(default_factory=YawnConfig)
    posture: PostureConfig = field(default_factory=PostureConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    robustness: RobustnessConfig = field(default_factory=RobustnessConfig)
    alarm: AlarmConfig = field(default_factory=AlarmConfig)
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    face_mesh: FaceMeshConfig = field(default_factory=FaceMeshConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    cnn_validation: CNNValidationConfig = field(default_factory=CNNValidationConfig)
    dataset_paths: DatasetPathsConfig = field(default_factory=DatasetPathsConfig)

    ablation: AblationConfig = field(default_factory=AblationConfig)

    def __repr__(self):
        lines = [
            "=" * 60,
            "  SYSTEM CONFIGURATION (Experiment Record)",
            "=" * 60,
            "",
            "--- Detection Thresholds ---",
            f"  EAR Threshold          : {self.detection.ear_threshold}",
            f"  EAR Hysteresis         : {self.detection.ear_hysteresis}",
            f"  EAR Open-Threshold     : {self.detection.ear_threshold + self.detection.ear_hysteresis}",
            f"  MAR Threshold          : {self.detection.mar_threshold}",
            f"  MAR Hysteresis         : {self.detection.mar_hysteresis}",
            "",
            "--- Temporal Parameters (seconds) ---",
            f"  Eye Closure Duration   : {self.temporal.eye_closure_duration}s",
            f"  Yawn Min Duration      : {self.temporal.yawn_min_duration}s",
            f"  Blink Min Duration     : {self.temporal.blink_min_duration}s",
            f"  Blink Max Duration     : {self.temporal.blink_max_duration}s",
            "",
            "--- Advanced Yawning Analysis ---",
            f"  Yawn Confidence Thresh : {self.yawn.confidence_threshold}",
            f"  Speech Jitter Limit    : {self.yawn.speech_jitter_threshold}",
            f"  Sliding Window Size    : {self.yawn.sliding_window_size} frames",
            "",
            "--- Head Pose & Posture ---",
            f"  Downward Pitch Thresh  : {self.posture.downward_pitch_threshold}°",
            f"  Nod Min Duration       : {self.posture.nod_min_duration}s",
            f"  Posture Instability Lim: {self.posture.posture_instability_limit}°",
            f"  Posture Conf Thresh    : {self.posture.confidence_threshold}",
            "",
            "--- Multi-Factor Fusion ---",
            f"  EAR Weight             : {self.fusion.ear_weight}",
            f"  MAR Weight             : {self.fusion.mar_weight}",
            f"  Pose Weight            : {self.fusion.pose_weight}",
            f"  Agreement Bonus (2-cue): {self.fusion.agreement_bonus_2cue}x",
            f"  Agreement Bonus (3-cue): {self.fusion.agreement_bonus_3cue}x",
            f"  Accumulation Rate      : {self.fusion.temporal_accumulation_rate}",
            f"  Decay Rate             : {self.fusion.temporal_decay_rate}",
            f"  Slight Threshold       : {self.fusion.slight_threshold}",
            f"  Moderate Threshold     : {self.fusion.moderate_threshold}",
            f"  Severe Threshold       : {self.fusion.severe_threshold}",
            f"  Severity Hysteresis    : {self.fusion.severity_hysteresis}",
            f"  Min Cue Agreement      : {self.fusion.min_cue_agreement}",
            "",
            "--- Robustness & FP Reduction ---",
            f"  Jitter Low Thresh      : {self.robustness.jitter_low_threshold} px",
            f"  Jitter High Thresh     : {self.robustness.jitter_high_threshold} px",
            f"  Min Stability Score    : {self.robustness.min_stability_score}",
            f"  Brightness Low         : {self.robustness.brightness_low}",
            f"  Brightness High        : {self.robustness.brightness_high}",
            f"  Consistency Window     : {self.robustness.consistency_window} frames",
            f"  Reliability EMA Alpha  : {self.robustness.reliability_ema_alpha}",
            f"  Alert Suppress Thresh  : {self.robustness.alert_suppression_threshold}",
            "",
            "--- Alarm Behavior ---",
            f"  Min Alarm Duration     : {self.alarm.min_alarm_duration}s",
            f"  Cooldown Period        : {self.alarm.cooldown_period}s",
            f"  Face-Loss Timeout      : {self.alarm.face_loss_timeout}s",
            f"  Face-Loss Escalation   : {self.alarm.face_loss_escalation}",
            f"  Log File               : {self.alarm.log_file_path}",
            "",
            "--- Smoothing ---",
            f"  EAR EMA Alpha          : {self.smoothing.ear_ema_alpha}",
            f"  MAR EMA Alpha          : {self.smoothing.mar_ema_alpha}",
            f"  EAR History Buffer     : {self.smoothing.ear_history_size}",
            "",
            "--- Camera ---",
            f"  Camera ID              : {self.camera.camera_id}",
            f"  Capture Resolution     : {self.camera.capture_width}x{self.camera.capture_height}",
            f"  Display Resolution     : {self.camera.display_width}x{self.camera.display_height}",
            "",
            "--- MediaPipe Face Mesh ---",
            f"  Max Faces              : {self.face_mesh.max_num_faces}",
            f"  Refine Landmarks       : {self.face_mesh.refine_landmarks}",
            f"  Detection Confidence   : {self.face_mesh.min_detection_confidence}",
            f"  Tracking Confidence    : {self.face_mesh.min_tracking_confidence}",
            "",
            "--- Edge Optimization ---",
            f"  Headless Mode          : {self.optimization.headless_mode}",
            f"  Enable Profiling       : {self.optimization.enable_profiling}",
            f"  Adaptive Skipping      : {self.optimization.adaptive_frame_skipping}",
            "",
            "--- CNN Validation Layer ---",
            f"  CNN Enabled            : {self.cnn_validation.enabled}",
            f"  Model Path             : {self.cnn_validation.model_path}",
            f"  Input Size             : {self.cnn_validation.input_size}x{self.cnn_validation.input_size}",
            f"  CNN Confidence Thresh  : {self.cnn_validation.confidence_threshold}",
            f"  Uncertainty Zone       : [{self.cnn_validation.uncertainty_zone_low}, {self.cnn_validation.uncertainty_zone_high}]",
            f"  Max CNN/sec            : {self.cnn_validation.max_invocations_per_second}",
            "=" * 60,
        ]
        return "\n".join(lines)
