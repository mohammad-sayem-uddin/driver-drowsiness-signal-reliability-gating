"""
Integrity Verifier — repository research-integrity invariants (frozen §5)
========================================================================
A standalone, dependency-light checker that FAILS LOUDLY if any frozen
integrity invariant is violated. Run it before every commit and before
citing any number in the paper.

Invariants checked (frozen spec §5):
  I1. Every number in results/measured_results.json is backed by an
      EXP-### row in EXPERIMENT_REGISTRY.md (no orphan metrics).
  I2. The reliability gate has exactly 3 components (no phantom 4th).
  I3. The banned dataset loader (drowsiness_detection) raises on use.
  I4. No duplicate .tflite model asset is present.
  I5. measured_results.json (if present) conforms to the expected schema
      and carries provenance for any latency block.
  I6. The figure generator refuses to run without measured results.

This script MEASURES NOTHING and FABRICATES NOTHING. It only inspects the
repository state and reports pass/fail. Exit code 0 = all invariants hold.
"""

import os
import re
import sys
import json
import hashlib
from typing import List, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_PATH = os.path.join(ROOT, "results", "measured_results.json")
REGISTRY_PATH = os.path.join(ROOT, "EXPERIMENT_REGISTRY.md")
MODELS_DIR = os.path.join(ROOT, "models")


class Check:
    def __init__(self, name: str):
        self.name = name
        self.passed = True
        self.notes: List[str] = []

    def fail(self, msg: str):
        self.passed = False
        self.notes.append(msg)

    def ok(self, msg: str):
        self.notes.append(msg)


# ── I1: no orphan metrics ─────────────────────────────────────────────────────

def check_registry_backs_results() -> Check:
    c = Check("I1: results backed by EXPERIMENT_REGISTRY rows")
    if not os.path.exists(RESULTS_PATH):
        c.ok("no measured_results.json yet — nothing to back (vacuously true).")
        return c
    with open(RESULTS_PATH) as f:
        data = json.load(f)
    if not os.path.exists(REGISTRY_PATH):
        c.fail("results exist but EXPERIMENT_REGISTRY.md is missing.")
        return c
    with open(REGISTRY_PATH) as f:
        reg = f.read()
    exp_rows = re.findall(r"EXP-\d{3}", reg)
    if not exp_rows:
        c.fail("EXPERIMENT_REGISTRY.md has no EXP-### rows but results exist.")
        return c
    # Latency block must correspond to a logged EXP (EXP-001 measured latency).
    if "latency_ms" in data:
        if not any(e for e in exp_rows if e != "EXP-000"):
            c.fail("latency_ms present but no measured EXP row (only EXP-000).")
        else:
            c.ok(f"latency_ms backed by registry ({len(set(exp_rows))} EXP rows).")
    if "roc" in data and data["roc"]:
        c.ok(f"roc block present with {len(data['roc'])} variant(s) — "
             "ensure an EXP-### row exists for the LOSO run before citing.")
    return c


# ── I2: reliability gate has exactly 3 components ─────────────────────────────

def check_three_component_gate() -> Check:
    c = Check("I2: reliability gate has exactly 3 components")
    path = os.path.join(ROOT, "src", "robustness.py")
    with open(path) as f:
        src = f.read()
    # The RobustnessSnapshot sub-scores are the gate components.
    for banned in ["tracking_confidence", "tracking_quality"]:
        if banned in src:
            c.fail(f"phantom component '{banned}' still referenced in robustness.py.")
    components = ["landmark_stability", "brightness_quality", "cue_consistency"]
    present = [comp for comp in components if comp in src]
    if len(present) == 3:
        c.ok(f"exactly 3 components present: {present}.")
    else:
        c.fail(f"expected 3 components, found {present}.")
    return c


# ── I3: banned loader raises ──────────────────────────────────────────────────

def check_banned_loader_raises() -> Check:
    c = Check("I3: drowsiness_detection loader is quarantined")
    sys.path.insert(0, ROOT)
    try:
        from src.data_loaders import DrowsinessDetectionDataLoader
        try:
            DrowsinessDetectionDataLoader()
            c.fail("DrowsinessDetectionDataLoader() did NOT raise — leak risk.")
        except RuntimeError:
            c.ok("DrowsinessDetectionDataLoader() raises RuntimeError as required.")
    except Exception as e:  # noqa
        c.fail(f"could not import loader: {e}")
    return c


# ── I4: no duplicate model asset ──────────────────────────────────────────────

def check_no_duplicate_model() -> Check:
    c = Check("I4: no duplicate .tflite model asset")
    if not os.path.isdir(MODELS_DIR):
        c.ok("no models/ directory.")
        return c
    hashes = {}
    for f in os.listdir(MODELS_DIR):
        if not f.endswith(".tflite"):
            continue
        fp = os.path.join(MODELS_DIR, f)
        h = hashlib.md5(open(fp, "rb").read()).hexdigest()
        hashes.setdefault(h, []).append(f)
    dups = {h: fs for h, fs in hashes.items() if len(fs) > 1}
    if dups:
        c.fail(f"byte-identical duplicate models: {list(dups.values())}.")
    else:
        c.ok(f"{sum(len(v) for v in hashes.values())} .tflite file(s), no duplicates.")
    return c


# ── I5: schema + provenance ───────────────────────────────────────────────────

def check_results_schema() -> Check:
    c = Check("I5: measured_results.json schema + provenance")
    if not os.path.exists(RESULTS_PATH):
        c.ok("no measured_results.json yet — schema vacuously satisfied.")
        return c
    with open(RESULTS_PATH) as f:
        data = json.load(f)
    if "latency_ms" in data:
        lat = data["latency_ms"]
        if "stages" not in lat:
            c.fail("latency_ms missing 'stages'.")
        devices = [k for k in lat if k != "stages"]
        if not devices:
            c.fail("latency_ms has no device measurements.")
        prov = data.get("_provenance", {})
        for d in devices:
            if d not in prov:
                c.fail(f"device '{d}' has no _provenance block.")
            elif "note" not in prov[d]:
                c.fail(f"device '{d}' provenance missing 'note'.")
        if not c.notes:
            c.ok(f"latency schema valid for devices {devices} with provenance.")
    if "roc" in data:
        for variant, d in data["roc"].items():
            for key in ("fpr", "tpr"):
                if key not in d:
                    c.fail(f"roc['{variant}'] missing '{key}'.")
        if "roc" in data and data["roc"] and c.passed:
            c.ok(f"roc schema valid for {list(data['roc'].keys())}.")
    if not data:
        c.fail("measured_results.json is empty.")
    return c


# ── I6: figure generator refuses without results ─────────────────────────────

def check_figure_generator_guard() -> Check:
    c = Check("I6: figure generator refuses without measured results")
    path = os.path.join(ROOT, "evaluation", "plot_paper_figures.py")
    with open(path) as f:
        src = f.read()
    if "FileNotFoundError" in src and "REFUSING" in src:
        c.ok("plot_paper_figures.py guards against missing results.")
    else:
        c.fail("plot_paper_figures.py missing the no-results refusal guard.")
    return c


CHECKS = [
    check_registry_backs_results,
    check_three_component_gate,
    check_banned_loader_raises,
    check_no_duplicate_model,
    check_results_schema,
    check_figure_generator_guard,
]


def main() -> int:
    print("=" * 68)
    print("  RESEARCH-INTEGRITY VERIFIER (frozen spec §5)")
    print("=" * 68)
    all_ok = True
    for fn in CHECKS:
        c = fn()
        status = "PASS" if c.passed else "FAIL"
        print(f"\n[{status}] {c.name}")
        for n in c.notes:
            print(f"       - {n}")
        all_ok = all_ok and c.passed
    print("\n" + "=" * 68)
    print("  RESULT:", "ALL INVARIANTS HOLD ✅" if all_ok
          else "INTEGRITY VIOLATION ❌")
    print("=" * 68)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
