"""
EXP-004 — LOSO Evaluation: canonical ROC + additive descriptive metrics
=======================================================================
This is an ADDITIVE reporting/orchestration layer on top of the FROZEN
`evaluation/loso_harness.py`. It changes NO frozen logic. It:

  1. Reuses the frozen LOSO machinery UNMODIFIED — `LOSOHarness`, `VARIANTS`,
     `_run_subject`, `_roc_curve`, `_auc`, `_fix_operating_point`,
     `_fpr_at_tpr`, and `write_results` are imported and called, never
     re-implemented. The canonical `roc` block written to
     `results/measured_results.json` is byte-for-byte what
     `loso_harness.py --variants V0 V1 V2 V3 V4 --write` produces (same
     functions, same operating-point rule fixed on V0 at target_tpr=0.80).

  2. Runs the full per-frame pipeline exactly ONCE per (variant, subject),
     caching the deterministic (score, label) pairs. From those SAME pairs it
     derives the additional descriptive metrics the thesis report requires:
     Accuracy, Precision, Recall, F1, Specificity, ROC-AUC, PR-AUC, FPR, TPR,
     Threshold, and the Confusion Matrix — per variant AND per subject.

Frozen operating point (unchanged):
  - The operating point is fixed on the V0 baseline ROC at target_tpr = 0.80,
    then held constant for all variants (frozen protocol §3).
  - Primary metric FPR@matched-TPR is taken verbatim from the frozen harness
    helpers (`_fpr_at_tpr`), which match TPR per variant on the ROC.
  - The confusion-matrix-derived descriptive metrics (accuracy/precision/
    recall/F1/specificity/FPR/TPR) are evaluated at ONE fixed score threshold
    — the V0 threshold that achieves TPR≈0.80 — held constant across every
    variant, i.e. "operating point fixed on the baseline, held constant"
    (frozen protocol §3). ROC-AUC and PR-AUC are threshold-free.

Integrity:
  - No sklearn (absent in .venv, matching EXP-002/EXP-003 methodology); all
    metrics are hand-rolled numpy with the standard definitions.
  - Nothing is fabricated: every value comes from FrameProcessor running on
    real NTHU-DDD frames. No number is written to the paper here.

Determinism: fixed frame order (frozen enumerate/sort), fixed video clock
(frame_index/fps), fixed seed regime — re-running yields identical scores.
"""

import os
import sys
import csv
import json
import time
import argparse
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ── Frozen machinery (imported UNMODIFIED — reuse, never re-implement) ─────────
from src.config import SystemConfig
from evaluation.loso_harness import (
    LOSOHarness, VARIANTS, VariantConfig, FrameScore,
    _run_subject, _roc_curve, _auc, _fix_operating_point, _fpr_at_tpr,
    RESULTS_PATH,
)
from evaluation.nthu_ground_truth import LABEL_DROWSY, LABEL_ALERT

EXP_ID = "EXP-004"
EXP_DIR = os.path.join("experiments", "EXP-004_loso")
SCORES_DIR = os.path.join(EXP_DIR, "scores")
PLOTS_DIR = os.path.join(EXP_DIR, "plots")
METRICS_JSON = os.path.join(EXP_DIR, "exp004_metrics.json")
PER_SUBJECT_CSV = os.path.join(EXP_DIR, "per_subject_metrics.csv")
PER_VARIANT_CSV = os.path.join(EXP_DIR, "per_variant_metrics.csv")

VARIANT_ORDER = ["V0", "V1", "V2", "V3", "V4"]


# ── Additive descriptive metrics (pure numpy; standard definitions) ───────────

def _confusion_at_threshold(scores: List[FrameScore], thr: float) -> Dict[str, int]:
    """Confusion matrix for predictions (score >= thr) vs. ground-truth labels.
    Positive class = LABEL_DROWSY (1); negative = LABEL_ALERT (0)."""
    y = np.array([s.label for s in scores], dtype=np.int32)
    p = (np.array([s.score for s in scores], dtype=np.float64) >= thr).astype(np.int32)
    tp = int(np.sum((p == 1) & (y == LABEL_DROWSY)))
    fp = int(np.sum((p == 1) & (y == LABEL_ALERT)))
    tn = int(np.sum((p == 0) & (y == LABEL_ALERT)))
    fn = int(np.sum((p == 0) & (y == LABEL_DROWSY)))
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def _metrics_from_cm(cm: Dict[str, int]) -> Dict[str, float]:
    """Standard classification metrics from a confusion matrix."""
    tp, fp, tn, fn = cm["tp"], cm["fp"], cm["tn"], cm["fn"]
    n = tp + fp + tn + fn
    acc = (tp + tn) / n if n else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0          # == TPR / sensitivity
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0             # == 1 - specificity
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    balanced_acc = 0.5 * (recall + specificity)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,          # TPR
        "specificity": specificity,
        "fpr": fpr,
        "f1": f1,
        "balanced_accuracy": balanced_acc,
    }


def _pr_auc(scores: List[FrameScore]) -> float:
    """Precision-Recall AUC (average precision by trapezoidal integration over
    recall), computed from the same descending-threshold sweep the frozen ROC
    uses. Pure numpy — no sklearn."""
    y = np.array([s.label for s in scores], dtype=np.int32)
    ys = np.array([s.score for s in scores], dtype=np.float64)
    n_pos = int(np.sum(y == LABEL_DROWSY))
    if n_pos == 0:
        return 0.0
    thresholds = np.unique(ys)[::-1]
    recalls, precisions = [0.0], [1.0]  # PR curve conventionally starts at (0,1)
    for t in thresholds:
        pred = (ys >= t)
        tp = int(np.sum(pred & (y == LABEL_DROWSY)))
        fp = int(np.sum(pred & (y == LABEL_ALERT)))
        rec = tp / n_pos
        prec = tp / (tp + fp) if (tp + fp) else 1.0
        recalls.append(rec)
        precisions.append(prec)
    recalls = np.array(recalls)
    precisions = np.array(precisions)
    order = np.argsort(recalls)
    return float(np.trapz(precisions[order], recalls[order]))


def _threshold_at_operating_tpr(v0_fpr: np.ndarray, v0_tpr: np.ndarray,
                                v0_thr: np.ndarray, target_tpr: float) -> Tuple[float, float, float]:
    """Return the V0 SCORE threshold at the operating point (TPR closest to
    target). Uses only the frozen ROC arrays. Returns (threshold, tpr, fpr)."""
    idx = int(np.argmin(np.abs(v0_tpr - target_tpr)))
    return float(v0_thr[idx]), float(v0_tpr[idx]), float(v0_fpr[idx])


# ── Score collection (single deterministic pass; reuses frozen _run_subject) ──

def collect_scores(harness: LOSOHarness, cfg: SystemConfig,
                   variant_keys: List[str]) -> Dict[str, Dict[str, List[FrameScore]]]:
    """For each variant, run the FROZEN per-subject pipeline once and keep the
    per-subject (score, label) pairs. Persists them to CSV for auditability."""
    os.makedirs(SCORES_DIR, exist_ok=True)
    per_variant_subject: Dict[str, Dict[str, List[FrameScore]]] = {}
    for key in variant_keys:
        variant = VARIANTS[key]
        print(f"\n[{EXP_ID}] === Variant {key} ({variant.name}) ===", flush=True)
        per_subject: Dict[str, List[FrameScore]] = {}
        for subj in harness.subjects:
            frames = harness.by_subject[subj]
            if harness.max_frames_per_subject:
                frames = frames[:harness.max_frames_per_subject]
            t0 = time.perf_counter()
            scores = _run_subject(frames, cfg, variant, harness.video_fps)
            dt = time.perf_counter() - t0
            n_pos = sum(1 for s in scores if s.label == LABEL_DROWSY)
            print(f"  [{key}] subject {subj}: {len(scores)} scored "
                  f"({n_pos} drowsy) in {dt:.1f}s", flush=True)
            per_subject[subj] = scores
        # Persist raw per-frame scores for this variant.
        out_csv = os.path.join(SCORES_DIR, f"{variant.name}.csv")
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["subject", "score", "label"])
            for subj, scs in per_subject.items():
                for s in scs:
                    w.writerow([subj, f"{s.score:.10f}", s.label])
        per_variant_subject[key] = per_subject
    return per_variant_subject


# ── Canonical frozen roc block (identical to loso_harness --write) ────────────

def build_frozen_roc_block(per_variant_subject: Dict[str, Dict[str, List[FrameScore]]],
                           variant_keys: List[str], target_tpr: float,
                           n_subjects: int) -> Tuple[Dict, float]:
    """Reproduce EXACTLY the dict that `LOSOHarness.evaluate` returns, using the
    frozen helpers. The operating point TPR is fixed on V0 and reused."""
    roc_results: Dict = {}
    op_tpr = None
    for key in variant_keys:
        variant = VARIANTS[key]
        scores: List[FrameScore] = []
        for subj in per_variant_subject[key]:
            scores.extend(per_variant_subject[key][subj])
        fpr, tpr, _ = _roc_curve(scores)
        auc = _auc(fpr, tpr)
        if op_tpr is None:
            op_tpr = _fix_operating_point(fpr, tpr, target_tpr)
        achieved_tpr, fpr_at_op = _fpr_at_tpr(fpr, tpr, op_tpr)
        roc_results[variant.name] = {
            "fpr": [round(float(v), 6) for v in fpr.tolist()],
            "tpr": [round(float(v), 6) for v in tpr.tolist()],
            "auc": round(auc, 6),
            "fpr_at_matched_tpr": round(fpr_at_op, 6),
            "matched_tpr": round(achieved_tpr, 6),
            "operating_point_tpr": round(op_tpr, 6),
            "n_frames": len(scores),
            "n_subjects": n_subjects,
        }
    return {"roc": roc_results, "operating_point_tpr": op_tpr}, op_tpr


# ── Extended metrics (additive; from the same scores) ─────────────────────────

def compute_extended_metrics(per_variant_subject, variant_keys, target_tpr):
    """Full descriptive metric set per variant and per subject, at the frozen
    operating point (fixed V0 score threshold, held constant across variants)."""
    # 1) Determine the fixed operating threshold from V0's ROC.
    v0_scores: List[FrameScore] = []
    for subj in per_variant_subject["V0"]:
        v0_scores.extend(per_variant_subject["V0"][subj])
    v0_fpr, v0_tpr, v0_thr = _roc_curve(v0_scores)
    fixed_thr, op_tpr, op_fpr = _threshold_at_operating_tpr(
        v0_fpr, v0_tpr, v0_thr, target_tpr)

    ext: Dict = {
        "operating_point": {
            "definition": ("V0 baseline score threshold at TPR closest to "
                           "target; held constant across all variants for the "
                           "confusion-matrix-derived metrics."),
            "target_tpr": target_tpr,
            "fixed_score_threshold": fixed_thr,
            "v0_tpr_at_threshold": op_tpr,
            "v0_fpr_at_threshold": op_fpr,
        },
        "per_variant": {},
    }

    for key in variant_keys:
        variant = VARIANTS[key]
        all_scores: List[FrameScore] = []
        per_subject_out: Dict[str, Dict] = {}
        for subj in per_variant_subject[key]:
            scs = per_variant_subject[key][subj]
            all_scores.extend(scs)
            # Per-subject: AUC/PR-AUC are threshold-free; CM at fixed threshold.
            s_fpr, s_tpr, _ = _roc_curve(scs)
            s_cm = _confusion_at_threshold(scs, fixed_thr)
            s_metrics = _metrics_from_cm(s_cm)
            n_pos = sum(1 for s in scs if s.label == LABEL_DROWSY)
            n_neg = sum(1 for s in scs if s.label == LABEL_ALERT)
            per_subject_out[subj] = {
                "n_frames": len(scs),
                "n_drowsy": n_pos,
                "n_notdrowsy": n_neg,
                "roc_auc": round(_auc(s_fpr, s_tpr), 6),
                "pr_auc": round(_pr_auc(scs), 6),
                "confusion_matrix": s_cm,
                **{k: round(v, 6) for k, v in s_metrics.items()},
            }

        fpr, tpr, _ = _roc_curve(all_scores)
        cm = _confusion_at_threshold(all_scores, fixed_thr)
        metrics = _metrics_from_cm(cm)
        # Frozen primary metric (FPR@matched-TPR) recomputed via frozen helper.
        matched_tpr, fpr_at_matched = _fpr_at_tpr(fpr, tpr, op_tpr)
        n_pos = sum(1 for s in all_scores if s.label == LABEL_DROWSY)
        n_neg = sum(1 for s in all_scores if s.label == LABEL_ALERT)
        ext["per_variant"][variant.name] = {
            "variant_key": key,
            "toggles": {
                "speech_filter": variant.enable_speech_filter,
                "reliability_gate": variant.enable_reliability_gate,
                "cnn": variant.enable_cnn,
            },
            "n_frames": len(all_scores),
            "n_drowsy": n_pos,
            "n_notdrowsy": n_neg,
            "roc_auc": round(_auc(fpr, tpr), 6),
            "pr_auc": round(_pr_auc(all_scores), 6),
            "fixed_score_threshold": round(fixed_thr, 6),
            "confusion_matrix": cm,
            "accuracy": round(metrics["accuracy"], 6),
            "precision": round(metrics["precision"], 6),
            "recall": round(metrics["recall"], 6),          # TPR at fixed thr
            "specificity": round(metrics["specificity"], 6),
            "f1": round(metrics["f1"], 6),
            "fpr": round(metrics["fpr"], 6),
            "tpr": round(metrics["recall"], 6),
            "balanced_accuracy": round(metrics["balanced_accuracy"], 6),
            "fpr_at_matched_tpr": round(fpr_at_matched, 6),
            "matched_tpr": round(matched_tpr, 6),
            "per_subject": per_subject_out,
        }
    return ext


# ── Plotting (additive; matplotlib Agg) ───────────────────────────────────────

def make_plots(frozen_roc: Dict, ext: Dict, variant_keys: List[str]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(PLOTS_DIR, exist_ok=True)
    roc = frozen_roc["roc"]

    # (1) ROC overlay
    plt.figure(figsize=(6, 5), dpi=300)
    for key in variant_keys:
        name = VARIANTS[key].name
        d = roc[name]
        plt.plot(d["fpr"], d["tpr"], linewidth=1.8,
                 label=f"{key} {name} (AUC={d['auc']:.3f})")
    plt.plot([0, 1], [0, 1], "k:", alpha=0.5, label="Chance")
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.title("EXP-004 LOSO ROC — NTHU-DDD (subject-disjoint)", fontweight="bold")
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "roc_overlay.png"))
    plt.close()

    # (2) AUC / PR-AUC bar
    names = [VARIANTS[k].name for k in variant_keys]
    aucs = [ext["per_variant"][n]["roc_auc"] for n in names]
    praucs = [ext["per_variant"][n]["pr_auc"] for n in names]
    x = np.arange(len(names))
    plt.figure(figsize=(7, 4.5), dpi=300)
    plt.bar(x - 0.2, aucs, 0.4, label="ROC-AUC")
    plt.bar(x + 0.2, praucs, 0.4, label="PR-AUC")
    plt.xticks(x, variant_keys)
    plt.ylim(0, 1)
    plt.ylabel("AUC")
    plt.title("EXP-004 ROC-AUC & PR-AUC by variant", fontweight="bold")
    for i, v in enumerate(aucs):
        plt.text(i - 0.2, v + 0.01, f"{v:.3f}", ha="center", fontsize=7)
    for i, v in enumerate(praucs):
        plt.text(i + 0.2, v + 0.01, f"{v:.3f}", ha="center", fontsize=7)
    plt.legend()
    plt.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "auc_prauc_bars.png"))
    plt.close()

    # (3) FPR@matched-TPR bar (primary metric)
    fprs = [ext["per_variant"][n]["fpr_at_matched_tpr"] for n in names]
    plt.figure(figsize=(7, 4.5), dpi=300)
    plt.bar(x, fprs, 0.6, color="#c0392b")
    plt.xticks(x, variant_keys)
    plt.ylabel("FPR @ matched TPR")
    plt.title(f"EXP-004 Primary metric: FPR @ matched TPR "
              f"(op TPR≈{frozen_roc['operating_point_tpr']:.3f})",
              fontweight="bold")
    for i, v in enumerate(fprs):
        plt.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)
    plt.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "fpr_at_matched_tpr.png"))
    plt.close()

    # (4) Confusion-matrix grid at fixed threshold
    fig, axes = plt.subplots(1, len(variant_keys), figsize=(3.0 * len(variant_keys), 3.2), dpi=300)
    if len(variant_keys) == 1:
        axes = [axes]
    for ax, key in zip(axes, variant_keys):
        name = VARIANTS[key].name
        cm = ext["per_variant"][name]["confusion_matrix"]
        M = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]], dtype=float)
        im = ax.imshow(M, cmap="Blues")
        ax.set_title(f"{key}", fontsize=10, fontweight="bold")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred N", "Pred P"], fontsize=7)
        ax.set_yticks([0, 1]); ax.set_yticklabels(["True N", "True P"], fontsize=7)
        for (r, cc), val in np.ndenumerate(M):
            ax.text(cc, r, int(val), ha="center", va="center",
                    color="black", fontsize=8)
    fig.suptitle("EXP-004 Confusion matrices at fixed V0 operating threshold",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "confusion_matrices.png"))
    plt.close(fig)

    print(f"[{EXP_ID}] Plots written -> {PLOTS_DIR}")


# ── CSV writers (additive) ────────────────────────────────────────────────────

def write_csvs(ext: Dict, variant_keys: List[str]):
    names = [VARIANTS[k].name for k in variant_keys]
    with open(PER_VARIANT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "name", "n_frames", "accuracy", "precision",
                    "recall_tpr", "specificity", "f1", "fpr", "roc_auc",
                    "pr_auc", "fpr_at_matched_tpr", "matched_tpr",
                    "tp", "fp", "tn", "fn"])
        for k in variant_keys:
            n = VARIANTS[k].name
            d = ext["per_variant"][n]
            cm = d["confusion_matrix"]
            w.writerow([k, n, d["n_frames"], d["accuracy"], d["precision"],
                        d["recall"], d["specificity"], d["f1"], d["fpr"],
                        d["roc_auc"], d["pr_auc"], d["fpr_at_matched_tpr"],
                        d["matched_tpr"], cm["tp"], cm["fp"], cm["tn"], cm["fn"]])
    with open(PER_SUBJECT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "subject", "n_frames", "n_drowsy", "n_notdrowsy",
                    "accuracy", "precision", "recall_tpr", "specificity", "f1",
                    "fpr", "roc_auc", "pr_auc", "tp", "fp", "tn", "fn"])
        for k in variant_keys:
            n = VARIANTS[k].name
            for subj, d in ext["per_variant"][n]["per_subject"].items():
                cm = d["confusion_matrix"]
                w.writerow([k, subj, d["n_frames"], d["n_drowsy"],
                            d["n_notdrowsy"], d["accuracy"], d["precision"],
                            d["recall"], d["specificity"], d["f1"], d["fpr"],
                            d["roc_auc"], d["pr_auc"], cm["tp"], cm["fp"],
                            cm["tn"], cm["fn"]])
    print(f"[{EXP_ID}] CSVs written -> {PER_VARIANT_CSV}, {PER_SUBJECT_CSV}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="EXP-004 LOSO: frozen ROC + additive descriptive metrics.")
    ap.add_argument("--variants", nargs="+", default=VARIANT_ORDER,
                    choices=list(VARIANTS.keys()))
    ap.add_argument("--target-tpr", type=float, default=0.80)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--max-frames", type=int, default=None,
                    help="Cap frames per subject (smoke runs only).")
    ap.add_argument("--nthu-root", default=os.path.join("Data", "nthu_ddd"))
    ap.add_argument("--write", action="store_true",
                    help="Persist the frozen roc block into measured_results.json.")
    args = ap.parse_args()

    os.makedirs(EXP_DIR, exist_ok=True)
    variant_keys = [k for k in VARIANT_ORDER if k in args.variants]

    t_start = time.perf_counter()
    harness = LOSOHarness(
        nthu_root=args.nthu_root, video_fps=args.fps,
        target_tpr=args.target_tpr, max_frames_per_subject=args.max_frames)
    cfg = SystemConfig()

    per_variant_subject = collect_scores(harness, cfg, variant_keys)

    frozen_roc, op_tpr = build_frozen_roc_block(
        per_variant_subject, variant_keys, args.target_tpr, len(harness.subjects))
    ext = compute_extended_metrics(per_variant_subject, variant_keys, args.target_tpr)

    elapsed = time.perf_counter() - t_start

    # Assemble the additive metrics artifact.
    metrics_artifact = {
        "experiment_id": EXP_ID,
        "title": "Leave-One-Subject-Out Evaluation (NTHU-DDD, V0–V4)",
        "dataset": "NTHU-DDD",
        "nthu_root": args.nthu_root,
        "protocol": {
            "splitting": "Leave-One-Subject-Out (subject-disjoint), seed 42",
            "subjects": list(harness.subjects),
            "video_fps": args.fps,
            "target_tpr": args.target_tpr,
            "operating_point_tpr": op_tpr,
            "primary_metric": "FPR @ matched TPR (frozen harness helpers)",
            "note": ("ROC/AUC/operating point computed by the FROZEN "
                     "loso_harness.py helpers (unmodified). Descriptive "
                     "confusion-matrix metrics evaluated at the fixed V0 "
                     "operating threshold, held constant across variants."),
            "sklearn_used": False,
            "metric_definitions": "hand-rolled numpy, standard definitions",
        },
        "max_frames_per_subject": args.max_frames,
        "wall_clock_seconds": round(elapsed, 2),
        "frozen_roc": frozen_roc["roc"],
        "extended_metrics": ext,
    }
    with open(METRICS_JSON, "w") as f:
        json.dump(metrics_artifact, f, indent=2)
    print(f"\n[{EXP_ID}] Metrics artifact -> {METRICS_JSON}")

    write_csvs(ext, variant_keys)
    make_plots(frozen_roc, ext, variant_keys)

    # Persist the canonical roc block into measured_results.json via the FROZEN
    # writer — identical to `loso_harness.py --write`.
    if args.write:
        harness.write_results(frozen_roc, out_path=RESULTS_PATH)
        print(f"[{EXP_ID}] Canonical roc block merged into {RESULTS_PATH} "
              "(frozen writer). Log the EXP-004 registry row before citing.")
    else:
        print(f"[{EXP_ID}] --write not set; measured_results.json NOT modified.")

    # Console summary.
    print("\n" + "=" * 78)
    print(f"  {EXP_ID} SUMMARY  (op TPR≈{op_tpr:.4f}, fixed thr="
          f"{ext['operating_point']['fixed_score_threshold']:.4f})")
    print("=" * 78)
    hdr = f"{'V':<3}{'name':<20}{'AUC':>8}{'PR-AUC':>9}{'Acc':>8}{'Prec':>8}{'Rec':>8}{'Spec':>8}{'F1':>8}{'FPR@mT':>9}"
    print(hdr)
    for k in variant_keys:
        n = VARIANTS[k].name
        d = ext["per_variant"][n]
        print(f"{k:<3}{n:<20}{d['roc_auc']:>8.3f}{d['pr_auc']:>9.3f}"
              f"{d['accuracy']:>8.3f}{d['precision']:>8.3f}{d['recall']:>8.3f}"
              f"{d['specificity']:>8.3f}{d['f1']:>8.3f}{d['fpr_at_matched_tpr']:>9.3f}")
    print("=" * 78)
    print(f"Total wall-clock: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
