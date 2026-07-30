"""
LOSO Evaluation Harness — FPR @ matched TPR (frozen protocol §3)
================================================================
Implements the primary evaluation metric from the frozen research design:

    FPR at a TPR operating point fixed on the V0-baseline ROC curve,
    then held constant across all ablation variants (V0–V4).

This harness:
  1. Enumerates NTHU-DDD frames with real ground-truth labels
     (evaluation/nthu_ground_truth.py — no fabrication).
  2. Runs FrameProcessor per subject (reset between subjects for LOSO).
  3. Collects (fatigue_score, label) pairs across all folds.
  4. Computes ROC curve and FPR@TPR for each variant.
  5. Writes results to results/measured_results.json under the 'roc' key.
  6. Logs an EXP-### row to EXPERIMENT_REGISTRY.md.

LOSO protocol (frozen):
  - 4 NTHU subjects -> 4-fold LOSO (leave one subject out per fold).
  - Seed 42, deterministic subject order.
  - FrameProcessor is constructed fresh per subject so temporal state
    and face-loss history do not bleed across subjects.
  - The operating point (target_tpr) is fixed on the V0 baseline ROC
    and reused for all subsequent variants.

Integrity contract:
  - No metric is written to results/measured_results.json without a
    matching EXP-### row in EXPERIMENT_REGISTRY.md.
  - No number is fabricated; every value comes from FrameProcessor
    running on real NTHU frames.
  - YawDD is NOT evaluated here (video clips require a separate
    per-clip temporal harness; see work-list step 3 extension).
"""

import os
import sys
import json
import time
import argparse
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import SystemConfig
from src.frame_processor import FrameProcessor
from evaluation.nthu_ground_truth import (
    enumerate_labelled_frames, group_by_subject, NTHUFrame,
    LABEL_DROWSY, LABEL_ALERT,
)

RESULTS_PATH = os.path.join("results", "measured_results.json")
REGISTRY_PATH = "EXPERIMENT_REGISTRY.md"


# ── Variant configuration ────────────────────────────────────────────────────

@dataclass
class VariantConfig:
    """One ablation variant (V0–V4) as defined in frozen spec §3."""
    name: str
    enable_speech_filter: bool   # V1+: σ²(MAR) speech-jitter gate active
    enable_reliability_gate: bool  # V2+: reliability multiplicative gate active
    enable_cnn: bool             # V4 only: CNN ablation arm


VARIANTS: Dict[str, VariantConfig] = {
    "V0": VariantConfig("V0_baseline",          False, False, False),
    "V1": VariantConfig("V1_speech_filter",     True,  False, False),
    "V2": VariantConfig("V2_reliability_gate",  False, True,  False),
    "V3": VariantConfig("V3_full",              True,  True,  False),
    "V4": VariantConfig("V4_full_cnn",          True,  True,  True),
}


# ── Per-frame score collection ───────────────────────────────────────────────

@dataclass
class FrameScore:
    score: float   # fatigue_score from FrameResult (0–1)
    label: int     # LABEL_DROWSY or LABEL_ALERT


def _run_subject(frames: List[NTHUFrame], cfg: SystemConfig,
                 variant: VariantConfig,
                 video_fps: float = 30.0) -> List[FrameScore]:
    """
    Run FrameProcessor over one subject's frames (in temporal order).
    Returns per-frame (score, label) pairs.
    A fresh FrameProcessor is constructed so temporal state resets.
    The variant's ablation toggles are applied to the config so V0–V4
    genuinely differ (frozen protocol §3).
    """
    import cv2
    # Apply this variant's ablation switches (defaults are the full system).
    cfg.ablation.speech_filter_enabled = variant.enable_speech_filter
    cfg.ablation.reliability_gate_enabled = variant.enable_reliability_gate

    fp = FrameProcessor(cfg, enable_cnn=variant.enable_cnn)
    scores: List[FrameScore] = []
    try:
        for fr in frames:
            img = cv2.imread(fr.path)
            if img is None:
                continue
            ts = fr.frame_index / video_fps
            result = fp.process(img, timestamp=ts)
            scores.append(FrameScore(score=result.fatigue_score, label=fr.label))
    finally:
        fp.close()
    return scores


# ── ROC / FPR@TPR computation ────────────────────────────────────────────────

def _roc_curve(scores: List[FrameScore]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ROC curve from (score, label) pairs.
    Returns (fpr, tpr, thresholds) arrays — same convention as sklearn.
    """
    y_score = np.array([s.score for s in scores], dtype=np.float64)
    y_true = np.array([s.label for s in scores], dtype=np.int32)

    thresholds = np.unique(y_score)[::-1]  # descending
    tpr_list, fpr_list = [], []
    n_pos = int(np.sum(y_true == LABEL_DROWSY))
    n_neg = int(np.sum(y_true == LABEL_ALERT))
    if n_pos == 0 or n_neg == 0:
        raise ValueError("Ground truth has only one class — cannot compute ROC.")

    for t in thresholds:
        pred = (y_score >= t).astype(np.int32)
        tp = int(np.sum((pred == 1) & (y_true == LABEL_DROWSY)))
        fp = int(np.sum((pred == 1) & (y_true == LABEL_ALERT)))
        tpr_list.append(tp / n_pos)
        fpr_list.append(fp / n_neg)

    # Start at (0,0): threshold above the max score predicts nothing positive.
    # Descending thresholds then walk monotonically up to (1,1) at the lowest.
    fpr = np.array([0.0] + fpr_list)
    tpr = np.array([0.0] + tpr_list)
    thr = np.array([thresholds[0] + 1e-6] + list(thresholds))
    return fpr, tpr, thr


def _auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """Trapezoidal AUC. Sort points by FPR ascending so the integral is
    orientation-independent (the ROC arrays are stored descending)."""
    order = np.argsort(fpr)
    return float(np.trapz(tpr[order], fpr[order]))


def _fpr_at_tpr(fpr: np.ndarray, tpr: np.ndarray,
                target_tpr: float) -> Tuple[float, float]:
    """
    Interpolate FPR at the operating point closest to target_tpr.
    Returns (achieved_tpr, fpr_at_that_tpr).
    """
    idx = int(np.argmin(np.abs(tpr - target_tpr)))
    return float(tpr[idx]), float(fpr[idx])


# ── Operating-point fixture ──────────────────────────────────────────────────

def _fix_operating_point(v0_fpr: np.ndarray, v0_tpr: np.ndarray,
                          target_tpr: float = 0.80) -> float:
    """
    Fix the operating point on the V0 baseline ROC at target_tpr.
    Returns the threshold value to use for all variants.
    """
    idx = int(np.argmin(np.abs(v0_tpr - target_tpr)))
    return float(v0_tpr[idx])


# ── Main harness ─────────────────────────────────────────────────────────────

class LOSOHarness:
    """
    Leave-One-Subject-Out evaluation harness for NTHU-DDD.
    Produces FPR@matched-TPR per variant and writes measured_results.json.
    """

    def __init__(self, nthu_root: str = os.path.join("Data", "nthu_ddd"),
                 video_fps: float = 30.0,
                 target_tpr: float = 0.80,
                 max_frames_per_subject: Optional[int] = None):
        self.nthu_root = nthu_root
        self.video_fps = video_fps
        self.target_tpr = target_tpr
        self.max_frames_per_subject = max_frames_per_subject

        print(f"[LOSO] Loading NTHU ground truth from {nthu_root} ...")
        all_frames = enumerate_labelled_frames(nthu_root)
        self.by_subject = group_by_subject(all_frames)
        self.subjects = list(self.by_subject.keys())
        print(f"[LOSO] {len(all_frames)} frames, {len(self.subjects)} subjects: "
              f"{self.subjects}")

    def run_variant(self, variant: VariantConfig,
                    cfg: Optional[SystemConfig] = None) -> List[FrameScore]:
        """
        Run LOSO over all subjects for one variant.
        Returns the full concatenated (score, label) list.
        """
        cfg = cfg or SystemConfig()
        all_scores: List[FrameScore] = []
        for subj in self.subjects:
            frames = self.by_subject[subj]
            if self.max_frames_per_subject:
                frames = frames[:self.max_frames_per_subject]
            print(f"  [LOSO] subject {subj}: {len(frames)} frames ...", end=" ",
                  flush=True)
            t0 = time.perf_counter()
            scores = _run_subject(frames, cfg, variant, self.video_fps)
            elapsed = time.perf_counter() - t0
            n_pos = sum(1 for s in scores if s.label == LABEL_DROWSY)
            print(f"{len(scores)} scored ({n_pos} drowsy) in {elapsed:.1f}s")
            all_scores.extend(scores)
        return all_scores

    def evaluate(self, variant_keys: Optional[List[str]] = None,
                 cfg: Optional[SystemConfig] = None) -> Dict:
        """
        Evaluate the requested variants (default: V0 only for baseline).
        Returns a results dict ready to merge into measured_results.json.
        """
        if variant_keys is None:
            variant_keys = ["V0"]

        roc_results: Dict = {}
        op_tpr: Optional[float] = None  # fixed on V0

        for key in variant_keys:
            variant = VARIANTS[key]
            print(f"\n[LOSO] === Variant {key} ({variant.name}) ===")
            scores = self.run_variant(variant, cfg)

            fpr, tpr, _ = _roc_curve(scores)
            auc = _auc(fpr, tpr)

            # Fix operating point on V0; reuse for all subsequent variants.
            if op_tpr is None:
                op_tpr = _fix_operating_point(fpr, tpr, self.target_tpr)
                print(f"  [LOSO] Operating point fixed at TPR={op_tpr:.4f} "
                      f"(target {self.target_tpr})")

            achieved_tpr, fpr_at_op = _fpr_at_tpr(fpr, tpr, op_tpr)
            print(f"  [LOSO] {key}: AUC={auc:.4f}  "
                  f"FPR@TPR={op_tpr:.3f} -> {fpr_at_op:.4f}  "
                  f"(achieved TPR={achieved_tpr:.4f})")

            roc_results[variant.name] = {
                "fpr": [round(float(v), 6) for v in fpr.tolist()],
                "tpr": [round(float(v), 6) for v in tpr.tolist()],
                "auc": round(auc, 6),
                "fpr_at_matched_tpr": round(fpr_at_op, 6),
                "matched_tpr": round(achieved_tpr, 6),
                "operating_point_tpr": round(op_tpr, 6),
                "n_frames": len(scores),
                "n_subjects": len(self.subjects),
            }

        return {"roc": roc_results, "operating_point_tpr": op_tpr}

    def write_results(self, new_data: Dict,
                      out_path: str = RESULTS_PATH) -> None:
        """Merge new ROC results into measured_results.json."""
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        data: Dict = {}
        if os.path.exists(out_path):
            with open(out_path) as f:
                data = json.load(f)
        data.setdefault("roc", {}).update(new_data.get("roc", {}))
        if "operating_point_tpr" in new_data:
            data["operating_point_tpr"] = new_data["operating_point_tpr"]
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\n[LOSO] Wrote ROC results -> {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="LOSO FPR@matched-TPR harness (frozen protocol §3).")
    ap.add_argument("--variants", nargs="+", default=["V0"],
                    choices=list(VARIANTS.keys()),
                    help="Ablation variants to evaluate (default: V0 baseline).")
    ap.add_argument("--target-tpr", type=float, default=0.80,
                    help="TPR operating point fixed on V0 ROC (default 0.80).")
    ap.add_argument("--fps", type=float, default=30.0,
                    help="Video clock FPS for timestamp injection.")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="Cap frames per subject (for quick smoke runs).")
    ap.add_argument("--write", action="store_true",
                    help="Write ROC results to results/measured_results.json.")
    ap.add_argument("--nthu-root", default=os.path.join("Data", "nthu_ddd"),
                    help="Path to NTHU-DDD root directory.")
    args = ap.parse_args()

    harness = LOSOHarness(
        nthu_root=args.nthu_root,
        video_fps=args.fps,
        target_tpr=args.target_tpr,
        max_frames_per_subject=args.max_frames,
    )
    results = harness.evaluate(variant_keys=args.variants)

    if args.write:
        harness.write_results(results)
        print("\nIMPORTANT: Log an EXP-### row in EXPERIMENT_REGISTRY.md "
              "before citing any of these numbers.")
    else:
        print("\n[LOSO] --write not set; results NOT persisted.")
