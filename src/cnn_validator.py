"""
CNN Validator — Tiny CNN Eye-State Validation Layer
=====================================================
A lightweight learned validation module that acts as an uncertainty
resolver for the heuristic EAR-based eye closure detection.

Architecture Role:
    This module does NOT replace the temporal EAR/MAR/Pose pipeline.
    It is invoked SELECTIVELY — only when the heuristic pipeline's
    EAR confidence falls into an ambiguous zone (near the hysteresis
    boundary where false positives concentrate).

    The CNN provides a binary "open/closed" verdict on a cropped
    grayscale eye ROI (24×24).  This verdict is fused with the
    heuristic's EAR classification to reduce false positives:

        Heuristic says CLOSED + CNN says OPEN (high conf) → suppress FP
        Heuristic says OPEN   + CNN says CLOSED (high conf) → boost score
        CNN uncertain → defer to heuristic (default behavior)

CNN Architecture (MicroEyeNet):
    Input:  24×24×1 grayscale
    Conv2D(8, 3×3, ReLU) → MaxPool(2×2)
    Conv2D(16, 3×3, ReLU) → MaxPool(2×2)
    Flatten → Dense(32, ReLU) → Dropout(0.3) → Dense(1, Sigmoid)
    Total: ~9.5K parameters
    Inference: <0.5ms on Raspberry Pi 4 (ARM Cortex-A72)

Safety Guarantees:
    - CNN NEVER overrides SEVERE_FATIGUE or FACE_LOST_CRITICAL.
    - CNN operates only in the ALERT ↔ SLIGHT_FATIGUE boundary.
    - If the TFLite model file is missing, the system degrades
      gracefully to pure heuristic mode (no crash).

Usage:
    from src.cnn_validator import CNNValidator
    validator = CNNValidator(cfg)
    if validator.should_invoke(smoothed_ear, reliability):
        verdict = validator.validate_eye_state(eye_roi_bgr, smoothed_ear)
"""

import os
import time
import logging
from dataclasses import dataclass

import cv2
import numpy as np

from src.config import SystemConfig

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# CNN Verdict — Output dataclass
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class CNNVerdict:
    """
    Result of a CNN validation query.

    Attributes
    ----------
    invoked : bool
        Whether the CNN was actually called this frame.  False if the
        EAR was outside the uncertainty zone or the rate-limiter blocked.

    probability_closed : float
        Raw sigmoid output: 0.0 = high confidence OPEN, 1.0 = high
        confidence CLOSED.  -1.0 if not invoked.

    cnn_says_closed : bool
        Binary CNN verdict after thresholding.

    cnn_agrees_with_heuristic : bool
        Whether the CNN verdict matches the EAR-based classification.
        True if both agree OR if the CNN was not invoked (no conflict).

    confidence : float
        Distance from 0.5 — how certain the CNN is.  Range [0.0, 0.5].
        Higher = more confident.  0.0 = maximally uncertain.
    """
    invoked: bool = False
    probability_closed: float = -1.0
    cnn_says_closed: bool = False
    cnn_agrees_with_heuristic: bool = True
    confidence: float = 0.0


# ═══════════════════════════════════════════════════════════════════════
# Eye ROI Extraction
# ═══════════════════════════════════════════════════════════════════════

# MediaPipe Face Mesh eye landmark indices for bounding box extraction.
# Using the outer contour landmarks for a tight crop.
_LEFT_EYE_BBOX_IDX = [33, 133, 160, 159, 158, 144, 145, 153]
_RIGHT_EYE_BBOX_IDX = [362, 263, 387, 386, 385, 373, 374, 380]


def extract_eye_roi(frame, face_landmarks, img_w, img_h, target_size=24, margin=5):
    """
    Extract a grayscale, square-cropped eye ROI from the frame.

    Uses the bilateral eye landmarks to compute a merged bounding box,
    then crops, pads to square, converts to grayscale, and resizes.

    Args:
        frame: BGR frame from the camera.
        face_landmarks: MediaPipe face landmarks object.
        img_w, img_h: Frame dimensions.
        target_size: Output resolution (target_size × target_size).
        margin: Pixel margin around the eye bounding box.

    Returns:
        numpy array of shape (target_size, target_size, 1), float32,
        normalized to [0, 1].  Returns None if extraction fails.
    """
    try:
        # Collect all eye landmark pixel coordinates
        all_eye_indices = _LEFT_EYE_BBOX_IDX + _RIGHT_EYE_BBOX_IDX
        xs = []
        ys = []
        for idx in all_eye_indices:
            lm = face_landmarks.landmark[idx]
            xs.append(int(lm.x * img_w))
            ys.append(int(lm.y * img_h))

        # Compute tight bounding box with margin
        x_min = max(0, min(xs) - margin)
        x_max = min(img_w, max(xs) + margin)
        y_min = max(0, min(ys) - margin)
        y_max = min(img_h, max(ys) + margin)

        if x_max <= x_min or y_max <= y_min:
            return None

        # Crop the eye region
        eye_crop = frame[y_min:y_max, x_min:x_max]

        if eye_crop.size == 0:
            return None

        # Convert to grayscale
        gray = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2GRAY)

        # Resize to target
        resized = cv2.resize(gray, (target_size, target_size),
                             interpolation=cv2.INTER_AREA)

        # Normalize to [0, 1] and add channel dimension
        normalized = resized.astype(np.float32) / 255.0
        return normalized.reshape(target_size, target_size, 1)

    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════
# CNN Validator
# ═══════════════════════════════════════════════════════════════════════

class CNNValidator:
    """
    Selective CNN-based eye-state validator.

    Manages TFLite model loading, rate-limiting, uncertainty zone
    checking, and inference execution.  Designed for graceful
    degradation: if the model file is missing, all calls to
    should_invoke() return False.

    Per-frame pipeline (when invoked):
        1. Check if EAR is in uncertainty zone.
        2. Check rate-limiter.
        3. Extract eye ROI from frame.
        4. Run TFLite inference (~0.5ms).
        5. Return CNNVerdict with agreement analysis.
    """

    def __init__(self, cfg: SystemConfig):
        self.cfg = cfg
        cc = cfg.cnn_validation

        self._enabled = cc.enabled
        self._input_size = cc.input_size
        self._conf_threshold = cc.confidence_threshold
        self._zone_low = cc.uncertainty_zone_low
        self._zone_high = cc.uncertainty_zone_high
        self._max_per_sec = cc.max_invocations_per_second

        # Rate-limiter state
        self._last_invoke_time = 0.0
        self._min_interval = 1.0 / max(self._max_per_sec, 1)

        # Statistics for profiling / research logging
        self.total_invocations = 0
        self.total_agreements = 0
        self.total_overrides = 0

        # TFLite interpreter
        self._interpreter = None
        self._input_details = None
        self._output_details = None
        self._model_loaded = False

        if self._enabled:
            self._load_model(cc.model_path)

    def _load_model(self, model_path: str):
        """
        Attempt to load the TFLite model.  Degrades gracefully if
        the file is missing or TFLite runtime is unavailable.
        """
        if not os.path.isfile(model_path):
            print(f"[CNN Validator] Model not found at '{model_path}'. "
                  f"Running in heuristic-only mode.")
            self._enabled = False
            return

        try:
            # Try tflite_runtime first (lightweight, preferred on Pi)
            try:
                import tflite_runtime.interpreter as tflite
                self._interpreter = tflite.Interpreter(model_path=model_path)
            except ImportError:
                # Fall back to full TensorFlow
                try:
                    import tensorflow as tf
                    self._interpreter = tf.lite.Interpreter(model_path=model_path)
                except ImportError:
                    print("[CNN Validator] Neither tflite_runtime nor tensorflow "
                          "installed. Running in heuristic-only mode.")
                    self._enabled = False
                    return

            self._interpreter.allocate_tensors()
            self._input_details = self._interpreter.get_input_details()
            self._output_details = self._interpreter.get_output_details()
            self._model_loaded = True
            print(f"[CNN Validator] Model loaded: {model_path}")
            print(f"[CNN Validator] Input shape: {self._input_details[0]['shape']}")

        except Exception as e:
            print(f"[CNN Validator] Failed to load model: {e}. "
                  f"Running in heuristic-only mode.")
            self._enabled = False

    # ───────────────────────────────────────────────────────────────────
    # Public API
    # ───────────────────────────────────────────────────────────────────

    @property
    def is_available(self) -> bool:
        """Whether the CNN is loaded and ready for inference."""
        return self._enabled and self._model_loaded

    def should_invoke(self, smoothed_ear: float, system_reliability: float = 1.0) -> bool:
        """
        Determine if the CNN should be invoked this frame.

        The CNN is only invoked when:
          1. The model is loaded and enabled.
          2. The smoothed EAR is within the uncertainty zone.
          3. The rate-limiter allows another invocation.
          4. System reliability is above a minimum (avoid wasting CPU
             when signal quality is already poor).

        Args:
            smoothed_ear: EMA-smoothed EAR value from the temporal analyzer.
            system_reliability: Reliability score from RobustnessGuard.

        Returns:
            True if CNN should be invoked this frame.
        """
        if not self.is_available:
            return False

        # Check uncertainty zone
        if not (self._zone_low <= smoothed_ear <= self._zone_high):
            return False

        # Don't waste CNN calls when signal is very unreliable
        if system_reliability < 0.3:
            return False

        # Rate-limiter
        now = time.monotonic()
        if (now - self._last_invoke_time) < self._min_interval:
            return False

        return True

    def validate_eye_state(self, eye_roi: np.ndarray, smoothed_ear: float,
                           ear_threshold: float = 0.21) -> CNNVerdict:
        """
        Run CNN inference on a preprocessed eye ROI.

        Args:
            eye_roi: Preprocessed eye ROI from extract_eye_roi().
                     Shape: (input_size, input_size, 1), float32, [0,1].
            smoothed_ear: Current smoothed EAR for agreement analysis.
            ear_threshold: EAR threshold for heuristic classification.

        Returns:
            CNNVerdict with inference results and agreement analysis.
        """
        if eye_roi is None or not self.is_available:
            return CNNVerdict()

        try:
            # Prepare input tensor (add batch dimension)
            input_data = np.expand_dims(eye_roi, axis=0).astype(np.float32)

            # Run inference
            self._interpreter.set_tensor(
                self._input_details[0]['index'], input_data
            )
            self._interpreter.invoke()

            # Get output
            output = self._interpreter.get_tensor(
                self._output_details[0]['index']
            )
            prob_closed = float(output[0][0])

            # Update rate-limiter
            self._last_invoke_time = time.monotonic()
            self.total_invocations += 1

            # Determine CNN verdict
            cnn_says_closed = prob_closed >= 0.5
            confidence = abs(prob_closed - 0.5)

            # Determine heuristic verdict
            heuristic_says_closed = smoothed_ear < ear_threshold

            # Check agreement
            agrees = (cnn_says_closed == heuristic_says_closed)
            if agrees:
                self.total_agreements += 1
            else:
                self.total_overrides += 1

            return CNNVerdict(
                invoked=True,
                probability_closed=prob_closed,
                cnn_says_closed=cnn_says_closed,
                cnn_agrees_with_heuristic=agrees,
                confidence=confidence,
            )

        except Exception as e:
            logger.warning(f"CNN inference failed: {e}")
            return CNNVerdict()

    def get_stats(self) -> dict:
        """Return invocation statistics for profiling / research logging."""
        agreement_rate = (
            self.total_agreements / max(self.total_invocations, 1)
        )
        return {
            "total_invocations": self.total_invocations,
            "total_agreements": self.total_agreements,
            "total_overrides": self.total_overrides,
            "agreement_rate": agreement_rate,
        }
"""
This module uses the following utility function to compute the CNN verdict:

Usage:
    from src.cnn_validator import CNNValidator, extract_eye_roi
    validator = CNNValidator(cfg)
    if validator.should_invoke(smoothed_ear, reliability):
        eye_roi = extract_eye_roi(frame, face_lm, w, h, cfg.cnn_validation.input_size)
        verdict = validator.validate_eye_state(eye_roi, smoothed_ear)
"""
