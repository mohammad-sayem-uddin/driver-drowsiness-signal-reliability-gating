"""
Publication-Quality Figure Generator (measured-data only)
=========================================================
Generates paper-ready figures STRICTLY from committed, measured result
artifacts. This module intentionally contains NO synthetic, analytic, or
hardcoded performance numbers.

Contract
--------
- All figures are rendered from a results JSON produced by the real
  evaluation harness (default: ``results/measured_results.json``).
- If that file does not exist, the generators REFUSE to produce a figure
  and exit non-zero. This prevents fabricated figures from ever being
  regenerated (freeze-report precondition 1).

Expected results JSON schema (populated by the evaluation harness only):
    {
      "roc": {
        "<variant_name>": {"fpr": [...], "tpr": [...], "auc": <float>},
        ...
      },
      "latency_ms": {
        "stages": ["Frame Grab", "MediaPipe Mesh", ...],
        "<device_name>": [<float>, ...],
        ...
      }
    }
"""

import os
import sys
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

DEFAULT_RESULTS = os.path.join("results", "measured_results.json")


def _load_results(results_path):
    if not os.path.exists(results_path):
        raise FileNotFoundError(
            f"No measured results at '{results_path}'. Figures are generated "
            "ONLY from committed measured experiments. Run the evaluation "
            "harness first and record results per EXPERIMENT_REGISTRY.md. "
            "Synthetic/placeholder figures are not permitted."
        )
    with open(results_path, "r") as f:
        return json.load(f)


def generate_roc_curve_fig(results_path=DEFAULT_RESULTS,
                           output_path="results/fig_roc_curves.png"):
    """Render ROC curves from measured per-variant (fpr, tpr, auc) arrays."""
    results = _load_results(results_path)
    roc = results.get("roc")
    if not roc:
        raise KeyError("results JSON has no 'roc' section; nothing to plot.")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(6, 5), dpi=300)

    for variant, data in roc.items():
        fpr, tpr = data["fpr"], data["tpr"]
        auc = data.get("auc")
        label = f"{variant} (AUC = {auc:.3f})" if auc is not None else variant
        plt.plot(fpr, tpr, linewidth=1.8, label=label)

    plt.plot([0, 1], [0, 1], 'k:', alpha=0.5, label='Random Chance')
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=11)
    plt.title('ROC Curves Across Detection Architecture Variants',
              fontsize=12, fontweight='bold')
    plt.legend(loc='lower right', fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[Plot Figures] Saved ROC Curve figure to: {output_path}")


def generate_latency_breakdown_fig(results_path=DEFAULT_RESULTS,
                                   output_path="results/fig_latency_breakdown.png"):
    """Render per-stage latency bars from measured device timings."""
    results = _load_results(results_path)
    lat = results.get("latency_ms")
    if not lat or "stages" not in lat:
        raise KeyError("results JSON has no 'latency_ms.stages'; nothing to plot.")

    import numpy as np
    stages = lat["stages"]
    devices = [k for k in lat.keys() if k != "stages"]
    if not devices:
        raise KeyError("results JSON 'latency_ms' has no device measurements.")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(6, 4.5), dpi=300)

    x = np.arange(len(stages))
    width = 0.8 / max(1, len(devices))
    for i, dev in enumerate(devices):
        offset = (i - (len(devices) - 1) / 2) * width
        plt.bar(x + offset, lat[dev], width, label=dev)

    plt.ylabel('Execution Latency (ms)', fontsize=11)
    plt.title('Per-Stage Execution Latency Breakdown',
              fontsize=12, fontweight='bold')
    plt.xticks(x, stages, rotation=25, ha='right', fontsize=9)
    plt.legend(loc='upper right', fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.4, axis='y')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[Plot Figures] Saved Latency Breakdown figure to: {output_path}")


if __name__ == "__main__":
    try:
        generate_roc_curve_fig()
        generate_latency_breakdown_fig()
    except (FileNotFoundError, KeyError) as e:
        print(f"[Plot Figures] REFUSING to generate figures: {e}")
        sys.exit(1)
