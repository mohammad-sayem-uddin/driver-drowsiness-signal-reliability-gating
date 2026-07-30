# PROJECT CONTEXT — The Single Source of Truth

> **Read this first.** Assume you are an AI agent or researcher who has been
> handed *only this file*, six months after the last commit, with no
> conversation history and no memory of prior work. This document is written so
> that you can understand the entire project — its research, its architecture,
> its history, and how to safely continue it — **without reading the source
> code first**. Read the source to *act*, not to *understand*.
>
> **Truth policy:** every performance number in this file traces to a logged
> experiment or a committed artifact. Nothing is fabricated, projected, or
> estimated. Unmeasured claims are marked **NOT MEASURED**.

---

## 0. Executive Summary (read this even if you read nothing else)

- **What it is:** a real-time, webcam-based **driver drowsiness detector**
  (Python, MediaPipe FaceMesh + classical geometry + temporal fusion).
- **The research contribution is NOT a classifier.** It is a **signal-reliability
  gate**: on every frame, the system estimates how *trustworthy* the fatigue
  evidence is (given lighting, landmark stability, cue agreement) and
  **attenuates that evidence multiplicatively before accumulating it over
  time**. Claim: this reduces false-positive rate (FPR) at a matched
  true-positive rate (TPR) vs. plain weighted fusion.
- **Second novelty:** a **variance-based speech-jitter MAR filter** that
  suppresses talking-induced yawn false positives via σ²(MAR) gating.
- **Current state (2026-07-29): the first full experiment cycle is complete
  (EXP-000 … EXP-004), plus an independent scientific audit of EXP-004.** The
  pipeline, LOSO harness, CNN training, quantization, and the full V0–V4
  ablation have all been run, measured, and logged. Measured results now on
  record (each with an `EXP-###` row + committed artifact):
  - **EXP-001** — host latency **3.205 ms/frame** (Darwin-arm64 host, **NOT** a Pi 4).
  - **EXP-002** — MicroEyeNet trained on subject-disjoint MRL: TEST F1 **0.9623**,
    measured **19,745 params**.
  - **EXP-003** — quantization: INT8 **25.55 KB** (3.18× smaller, −0.026% F1).
  - **EXP-004** — LOSO ablation V0–V4 on NTHU-DDD: **honest negative result.**
    The reliability gate does **not** reduce FPR@matched-TPR=0.80 (V2 0.6244 vs
    V0 0.6241, flat), the speech filter *raises* FPR (V1/V3/V4 ≈0.669), and
    ROC-AUC never beats baseline (V0 0.629 highest). V4≡V3 byte-identical.
- **What's next (EXP-005):** an **event-level alarm evaluation** (FPR/hour +
  episode latency over V0–V4), where the CNN arm and the gate's spurious-alarm
  protection can actually be observed; then a **gate redesign evaluation
  (EXP-006)**, **Raspberry Pi deployment evaluation (EXP-007)**, and an optional
  **second-dataset validation (EXP-008)**. See the official roadmap in
  `EXPERIMENT_REGISTRY.md §4`. Each run MUST be logged as an `EXP-###` row before
  it is cited.
- **What is frozen:** the research design and the implementation spec
  (`reports/IMPLEMENTATION_SPECIFICATION_FROZEN.md`). Do **not** redesign.

---

## 1. Research History & Major Decisions (with rejected alternatives)

The project went through a hard integrity correction before being frozen.
Understanding *why* each decision was made prevents you from re-introducing a
retracted mistake.

| Decision | What was chosen | Rejected alternative | Why |
|---|---|---|---|
| **Core contribution** | Signal-reliability gating (evidence trust) | "Yet another CNN drowsiness classifier" | The literature is saturated with classifiers; the gap is *robustness under bad measurement conditions*, which is a gating problem, not a modelling one. |
| **Reliability composition** | Weighted **geometric mean** of 3 sub-scores | Arithmetic mean / learned MLP | Geometric mean makes any single collapsing signal (brightness→0) pull reliability→0 ("one bad signal poisons trust"). An MLP would need labels and defeat interpretability. |
| **Gate components** | Exactly **3**: landmark-stability, brightness-quality, cue-consistency | A 4th "tracking_confidence/tracking_quality" component | The 4th was a **phantom** — it was never actually computed. It was removed; weights renormalized to (0.45, 0.30, 0.25). A test + the integrity verifier now *forbid* a 4th component. |
| **Safety behaviour** | SEVERE fatigue states are **never** suppressed by the gate | Uniformly gate all states | A reliability gate that could silence a genuine SEVERE alert is unsafe. The asymmetry is a safety argument. |
| **Yawn robustness** | Variance-based σ²(MAR) speech filter | Fixed MAR threshold only | Talking produces high-variance mouth motion without sustained opening; variance separates speech from yawns better than a static threshold. |
| **MAR geometry** | 2D image-plane MAR; EAR uses 3D | Inflating MAR with the z-axis | Documented as a deliberate 2D/3D metric separation (supporting point, not a claimed novelty). Do not "fix" MAR to 3D. |
| **Timing** | Injectable clock: wall-clock live, **video clock (`frame_index/fps`) offline** | Frame-count or processing-wall-clock offline | Offline temporal integration must reflect the *recording's* time, not the machine's processing speed, or LOSO temporal metrics are meaningless. |
| **Splitting** | Subject-disjoint LOSO / GroupKFold, seed 42 | Random frame split | Random splits leak subject identity → inflated accuracy. This was an actual bug (see §7). |
| **CNN role** | Optional, ablation-only, **OFF by default**, selective eye validator | CNN on the default detection path | The thesis claim is about the gate, not the CNN. The CNN is prior-art baseline for the V4 ablation only. |
| **`drowsiness_detection` dataset** | Quarantined; loader raises `RuntimeError` | Use it as an extra dataset | It is a 100% byte-duplicate of MRL Eye; using it would be silent double-counting. |
| **Results discipline** | Nothing citable without an `EXP-###` row + committed artifact | "Fill in the numbers, measure later" | The project previously shipped fabricated "0 leakage / <0.5 ms" claims (EXP-000). These were retracted; the whole integrity regime exists to prevent recurrence. |

---

## 2. Per-Module Architecture

Pipeline order is fixed (do not reorder):

```
frame → FaceMesh → geometry (EAR, MAR 2D, head pose solvePnP)
      → SignalQuality (3 sub-scores)
      → RobustnessGuard → reliability r ∈ [0,1]
      → TemporalAnalyzer (speech-jitter filter; injectable clock)
      → FatigueFusionEngine (weighted sum × agreement × r; SEVERE-exempt)
      → StateManager (5-state hysteresis machine)
      → alert / UI
```

### `src/detector.py` — geometry
- **Purpose:** compute EAR, MAR (2D image-plane), run calibration.
- **Inputs:** FaceMesh landmarks. **Outputs:** EAR, MAR scalars.
- **Rationale/constraint:** MAR must stay 2D (mouth idx `[78,13,308,14]`); do
  not inflate with z. EAR is planar/3D as implemented.

### `src/pose_estimator.py` — head pose
- **Purpose:** head pose (pitch/yaw/roll) via `cv2.solvePnP`.
- **Outputs:** Euler angles; feeds the "downward pitch" posture cue.

### `src/robustness.py` — **the core novelty**
- **Purpose:** compute the 3 component sub-scores and the reliability index.
- **Inputs:** landmark stability (jitter), frame brightness, cue consistency.
- **Outputs:** `RobustnessSnapshot(landmark_stability, brightness_quality,
  cue_consistency, system_reliability, estimator_mode="geometric",
  alert_suppressed)`.
- **Math:** `r = stability^0.45 * brightness^0.30 * consistency^0.25`
  (weighted geometric mean). On face loss it emits
  `landmark_stability=0.0, brightness_quality=0.5, cue_consistency=0.5` and a
  `face_penalty = max(0.3, 1.0 - consecutive_no_face*0.1)`. `alert_suppressed`
  fires when smoothed reliability `< alert_suppression_threshold (0.5)`.
- **Do NOT:** add a 4th component; use an arithmetic mean; suppress SEVERE.

### `src/temporal_analyzer.py` — temporal cues
- **Purpose:** monotonic/injected-clock durations, blink/yawn timing, PERCLOS,
  and the **σ²(MAR) speech-jitter gate**.
- **Key param:** `speech_jitter_threshold = 0.05`.
- **Enabling change:** `update()` accepts an optional `timestamp`; defaults to
  `time.monotonic()` (live), receives `frame_index/fps` offline.

### `src/fatigue_fusion.py` — fusion
- **Purpose:** weighted evidence sum × cue-agreement × reliability `r`.
- **Weights:** `ear_weight=0.45, pose_weight=0.30, mar_weight=0.25`.
- **Constraint:** SEVERE states bypass the reliability attenuation.

### `src/state_manager.py` — decision
- **Purpose:** 5-state hysteresis machine (ALERT … SEVERE) + face-loss
  escalation. Prevents state chatter.

### `src/cnn_validator.py` — optional CNN (ablation only)
- **Purpose:** selective MicroEyeNet eye-state validation (24×24×1 grayscale
  TFLite, **19,745 params measured** — see `reports/EXP-002_PARAMETER_AUDIT.md`;
  INT8 export is 25.55 KB per EXP-003) invoked only as an uncertainty resolver.
- **Default:** OFF. Built only when `enable_cnn=True`. Falls back to
  heuristic-only if the model file is absent.

### `src/frame_processor.py` — headless core (critical)
- **Purpose:** the exact per-frame flow, camera-free, shared by the live app
  and the benchmark so they cannot diverge.
- **Outputs:** `FrameResult` dataclass, including `fatigue_score: float`
  (the continuous ROC threshold variable).
- **Reliability bypass logic:**
  ```python
  if self.cfg.ablation.reliability_gate_enabled:
      gate_reliability = snap.system_reliability
  else:
      gate_reliability = 1.0
  ```
- Constructed **fresh once per subject/clip** in the harness.

### `src/config.py` — centralized configuration
- All thresholds/weights live here (no magic numbers in logic modules). Key
  blocks: `DetectionConfig`, `TemporalConfig`, `YawnConfig`, `PostureConfig`,
  `FusionConfig`, `RobustnessConfig` (`learned_weights=(0.45,0.30,0.25)`,
  `alert_suppression_threshold=0.5`), `SmoothingConfig`, `CNNValidationConfig`,
  `AblationConfig` (`speech_filter_enabled=True`, `reliability_gate_enabled=True`
  as *app* defaults — the harness overrides these per variant), `SystemConfig`.

### `src/main.py` — live application
- Camera → `FrameProcessor` → UI/alarm. Entry point for the interactive demo.

### Support: `src/data_loaders.py`, `src/dataset_manager.py`,
`src/camera_*.py`, `src/alarm_controller.py` — I/O, dataset access (incl. the
quarantined `DrowsinessDetectionDataLoader`), camera abstraction, and the
audible alarm.

---

## 3. Research Design (frozen)

- **Research question:** does a decomposed signal-reliability gate + a
  speech-jitter MAR filter reduce FPR at a matched TPR vs. a weighted-fusion
  baseline, while staying real-time on a Raspberry Pi 4 CPU?
- **Primary metric:** FPR @ matched TPR. The operating point is fixed on the
  **V0 baseline** ROC at `target_tpr = 0.80`, then held constant for all
  variants (so variants are compared at the *same* sensitivity).
- **Secondary:** ROC-AUC (trapezoidal), FPR/hour.
- **Feasibility:** Pi 4 per-stage latency, FPS, memory — **NOT MEASURED** yet
  (only host latency exists).
- **Sweep variable:** `FrameResult.fatigue_score ∈ [0,1]`.
- **Ablation V0–V4** with toggles `(enable_speech_filter,
  enable_reliability_gate, enable_cnn)`:
  V0 `(F,F,F)` → V1 `(T,F,F)` → V2 `(F,T,F)` → V3 `(T,T,F)` → V4 `(T,T,T)`.

---

## 4. Dataset Strategy

| Dataset | Path | Role | Facts (on-disk) |
|---|---|---|---|
| NTHU-DDD | `Data/nthu_ddd/` | Primary temporal eval | 66,521 JPG, 4 subjects (001/002/005/006), 36,030 drowsy / 30,491 notdrowsy |
| MRL Eye | `Data/mrl_eye/` | CNN-ablation training only | 84,898 PNG, 37 subjects; subject-disjoint split under `splits_subject_disjoint/` |
| YawDD | `Data/yawdd/` | Video/yawn eval | 348 AVI |
| drowsiness_detection | `Data/drowsiness_detection/` | **BANNED** | 100% MRL duplicate; loader raises `RuntimeError` |

- **NTHU labels** are parsed from filenames by `evaluation/nthu_ground_truth.py`
  (`<subject>_<glasses>_<condition>_<frameindex>_<label>.jpg`; `drowsy→1`,
  `notdrowsy→0`). No pixel is read and nothing is fabricated by that module.
- **Known label limitation:** clip-condition-derived frame labels make
  frame-level FPR conservative. State this in the paper.
- **Leakage fix:** MRL subject-disjoint split regenerated by
  `tools/build_subject_disjoint_splits.py` (seed 42, asserted 0 overlap).

---

## 5. Engineering Architecture (evaluation & integrity)

| File | Role |
|---|---|
| `evaluation/nthu_ground_truth.py` | Enumerate + label NTHU frames (metadata only). |
| `evaluation/loso_harness.py` | LOSO/GroupKFold, V0–V4 variants, ROC/AUC, fixed operating point. `--write` to persist. |
| `evaluation/benchmark_nthan_yawdd.py` | Latency benchmark (`measure_nthu_latency`, `write_latency_artifact`). Source of EXP-001. |
| `evaluation/plot_paper_figures.py` | Regenerates figures ONLY from `results/measured_results.json`; refuses (raises) if missing. |
| `evaluation/verify_integrity.py` | 6 invariants I1–I6 (see §8). Exit 0 iff all hold. |
| `tools/train_cnn.py` | Canonical MicroEyeNet trainer (reads `Data/mrl_eye/`). |
| `tools/build_subject_disjoint_splits.py` | Leak-free MRL split generator. |
| `EXPERIMENT_REGISTRY.md` | The experiment ledger. Nothing citable without a row here. |
| `results/measured_results.json` | Committed results artifact (latency + ROC). Per-experiment artifacts also live under `experiments/EXP-00X_*/`. |
| `tests/test_suite.py`, `tests/smoke_test.py` | 17 unit + 3 smoke tests. |

**LOSO harness internals worth knowing:** `VARIANTS` dict V0–V4;
`FrameScore(score,label)`; `_run_subject()` applies toggles to `cfg.ablation`,
builds a fresh `FrameProcessor` per subject, uses `ts = fr.frame_index /
video_fps`; `_roc_curve()` walks (0,0)→(1,1) and raises `ValueError` if only one
class present; `_auc()` = trapezoid; `_fix_operating_point(target_tpr=0.80)` on
V0; `write_results()` merges a `roc` block into `measured_results.json`.

---

## 6. Chronological Implementation History (summary — full detail in `IMPLEMENTATION_LOG.md`)

1. Initial system built (detector, fusion, state machine, live app).
2. Research framing settled on reliability gating as the novelty.
3. **Integrity crisis discovered:** fabricated "0 leakage / <0.5 ms" claims
   (EXP-000), 100% MRL subject leakage, a phantom 4th gate component, a
   byte-duplicate `.tflite`, a broken figure/benchmark path.
4. **Freeze preconditions cleared** (see frozen spec §0): purge fabricated
   results; restore executability; remove leakage; honest 3-component gate;
   remove duplicate model asset.
5. **Spec frozen** 2026-07-28.
6. **Pre-CNN engineering built:** NTHU ground truth, LOSO harness with V0–V4,
   latency benchmark, results schema, figure guard, integrity verifier.
7. **EXP-001** measured (first honest latency, 3.205 ms host).
8. **Cleanup:** removed 3 dead/broken files
   (`evaluation/ablation_runner.py`, `evaluation/latency_memory_profiler.py`,
   `tools/train_eye_cnn.py`); fixed a stale path in `tools/verify_integrity.py`.
9. **EXP-002** — MicroEyeNet trained on subject-disjoint MRL (TEST F1 0.9623,
   19,745 params measured; parameter count audited in
   `reports/EXP-002_PARAMETER_AUDIT.md`).
10. **EXP-003** — INT8 quantization (25.55 KB, −0.026% F1).
11. **EXP-004** — full V0–V4 LOSO ablation on NTHU-DDD. **Negative result:** the
    reliability gate does not reduce frame-level FPR; the speech filter raises
    it; AUC never beats baseline. V4≡V3 byte-identical (CNN only flips the
    `should_alarm` boolean, not the swept `fatigue_score`).
12. **Independent scientific audit of EXP-004**
    (`reports/EXP-004_AUDIT/`) — confirms the negative result, diagnoses why
    (frame-level metric can't see episode-level behaviour; subject 006
    discriminates below chance), and recommends an event-level re-evaluation.
13. **Handover documentation authored + kept current** (this file + HANDOVER /
    LOG / MEMORY / registry).

---

## 7. Design Principles

**NEVER change (frozen):**
- The reliability gate = exactly 3 components, weighted geometric mean, weights
  (0.45, 0.30, 0.25).
- SEVERE states are never suppressed.
- Subject-disjoint LOSO everywhere; seed 42.
- `drowsiness_detection` stays quarantined.
- No performance number without an `EXP-###` row + committed artifact.
- CNN is ablation-only and OFF by default.
- MAR stays 2D.

**MAY extend (with care, and without violating the above):**
- Add new `EXP-###` rows and results artifacts (this is the whole point of the
  next phase).
- Add on-device profiling numbers once measured.
- Add datasets *if* subject-disjoint and logged.

**Frozen documents:**
- `reports/IMPLEMENTATION_SPECIFICATION_FROZEN.md` (the engineering contract).
- The research design described therein.

---

## 8. Current Status & Integrity Invariants

**Status (verified 2026-07-29):** 17/17 unit tests OK, 3/3 smoke OK, 6/6
integrity invariants hold. The first full experiment cycle (EXP-000 … EXP-004)
is complete and logged; measured results = EXP-001 latency, EXP-002 training,
EXP-003 quantization, EXP-004 LOSO ablation (negative). No Pi 4 or event-level
number exists yet.

The integrity verifier (`evaluation/verify_integrity.py`) enforces:
- **I1** — every result number is backed by an `EXP-###` row.
- **I2** — reliability gate has exactly 3 components (bans
  `tracking_confidence`/`tracking_quality`).
- **I3** — `DrowsinessDetectionDataLoader` raises `RuntimeError`.
- **I4** — no byte-identical duplicate `.tflite`.
- **I5** — `measured_results.json` schema + latency provenance present.
- **I6** — `plot_paper_figures.py` refuses to run without measured results.

---

## 9. Precise Continuation Instructions

EXP-001…004 are done (see §6). The remaining work follows the **official
roadmap** in `EXPERIMENT_REGISTRY.md §4`:

1. **EXP-005 — Event-Level Alarm Evaluation.** The EXP-004 audit showed the
   frame-level FPR@TPR metric cannot see the gate's or the CNN's intended
   behaviour (V4≡V3 at frame level because the CNN only flips `should_alarm`).
   Re-evaluate V0–V4 at the *episode* level: FPR/hour and per-episode alarm
   latency. Log as `EXP-005`.
2. **EXP-006 — Gate Redesign Evaluation.** Re-architect the gate as an additive
   decision-layer term (the follow-up the EXP-004 audit recommends; the audit
   calls it "EXP-005", but the official ID is **EXP-006**). Log as `EXP-006`.
3. **EXP-007 — Raspberry Pi Deployment Evaluation.** Run the latency benchmark
   on real Pi 4 hardware; record per-stage latency/memory/thermal. Log as
   `EXP-007`.
4. **EXP-008 — Second Dataset Validation (optional).** Reproduce the primary
   evaluation on another subject-disjoint dataset. Log as `EXP-008`.
5. **Populate the paper:** fill `paper/main.tex` results ONLY from committed
   artifacts (`results/measured_results.json` + `experiments/EXP-00X_*/`);
   regenerate figures with `plot_paper_figures.py`. The paper must report the
   EXP-004 negative result honestly. (Writing task — no `EXP-###` of its own.)
6. **After every step:** run `python3 evaluation/verify_integrity.py`,
   `python3 -m unittest tests.test_suite`, and `python3 tests/smoke_test.py` —
   all must stay green.

**Rule of thumb:** land code → run → log `EXP-###` → commit artifact. Never
batch numbers ahead of measurement.

---

## 10. Common Mistakes (do not repeat these)

- ❌ Adding a 4th reliability component "to be more thorough." (Fails CI; the
  3-component gate is the frozen novelty.)
- ❌ Writing a Raspberry Pi 4 latency number. **None has been measured.** The
  3.205 ms figure is a Darwin-arm64 host.
- ❌ Using an arithmetic mean for reliability. It must be the weighted geometric
  mean.
- ❌ Suppressing SEVERE states with the gate.
- ❌ Loading `drowsiness_detection` (it hard-raises by design).
- ❌ Putting a number in the paper/report before it has an `EXP-###` row and a
  committed artifact.
- ❌ Using a random (subject-mixing) split.
- ❌ Reviving `evaluation/ablation_runner.py`, `latency_memory_profiler.py`, or
  `tools/train_eye_cnn.py` — they were deleted as dead/broken/contradictory.
  Use `loso_harness.py`, `benchmark_nthan_yawdd.py`, and `tools/train_cnn.py`.
- ❌ Making MAR 3D.

---

## 11. FAQ

**Q: Is there a Raspberry Pi 4 result?** No. Only host latency (EXP-001). Pi 4
profiling is a pending task.

**Q: Which file is the "real" trainer?** EXP-002 was run with
`tools/train_exp002_microeyenet.py` (subject-disjoint MRL). `tools/train_cnn.py`
is the general trainer. The old `tools/train_eye_cnn.py` was deleted.

**Q: Why does `loso_harness.py` not save results by default?** To prevent
accidental un-logged results. Pass `--write` *and* log an `EXP-###` row.

**Q: Is the experiment numbering settled?** Yes. §2/§3 of the registry are
authoritative for completed work: EXP-000 … EXP-004 are done and measured. An
earlier stale block that listed *planned* "EXP-001 MicroEyeNet training … EXP-005
ablation" (a numbering that collided with the real measured rows) has been
removed. Future experiments now follow the **official roadmap** in
`EXPERIMENT_REGISTRY.md §4`.

**Q: What are EXP-005 – EXP-008?** The official roadmap:
- **EXP-005** — Event-Level Alarm Evaluation (next).
- **EXP-006** — Gate Redesign Evaluation (the audit's gate-redesign follow-up).
- **EXP-007** — Raspberry Pi Deployment Evaluation.
- **EXP-008** — Second Dataset Validation (optional).

**Q: The EXP-004 audit and the CNN spec use different EXP-005/006/007 — which
wins?** The official roadmap in `EXPERIMENT_REGISTRY.md §4` wins; a reconciliation
table there maps every provisional ID onto the official one. Notably, the audit's
"EXP-005" (gate redesign) is officially **EXP-006**, and the CNN spec's "EXP-005"
(V4 ablation) was already completed inside EXP-004. Historical documents keep
their original text; use the registry table when reading them.

**Q: Did the reliability-gate hypothesis hold?** Not at frame level. EXP-004 is
an honest negative result: the gate does not reduce FPR@matched-TPR and the
speech filter raises it. Treat this as the starting point for EXP-005, not
something to retune until the number improves.

**Q: Can I start the CNN now?** It is already trained (EXP-002) and quantized
(EXP-003). It remains ablation-only and OFF by default, and every run must be
logged.

**Q: What must I never touch?** `reports/IMPLEMENTATION_SPECIFICATION_FROZEN.md`
(except its Appendix A change-log), the frozen research design, and anything
enforced by the integrity verifier.

---

## 12. Files to Open First / Never Modify

**Open first (to understand):** this file → `HANDOVER.md` →
`reports/IMPLEMENTATION_SPECIFICATION_FROZEN.md` → `EXPERIMENT_REGISTRY.md` →
`src/frame_processor.py` → `src/robustness.py`.

**Never modify (without design sign-off):** the frozen spec's design sections,
`src/robustness.py`'s 3-component/geometric-mean contract, the LOSO seed, the
`drowsiness_detection` quarantine, and the integrity invariants.
