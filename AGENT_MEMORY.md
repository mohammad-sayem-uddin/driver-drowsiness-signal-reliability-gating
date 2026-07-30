# AGENT_MEMORY.md — Fast Start for an AI Agent

> **You have no conversation history.** This file exists to make you productive
> in one read. It is deliberately terse and imperative. For the *full* picture,
> read `PROJECT_CONTEXT.md`. For *why* things happened, read
> `IMPLEMENTATION_LOG.md`. For the *contract*, read
> `reports/IMPLEMENTATION_SPECIFICATION_FROZEN.md`.
>
> **One rule dominates all others: never write a performance number that is not
> backed by an `EXP-###` row in `EXPERIMENT_REGISTRY.md` and a committed
> artifact in `results/`.** The project was almost sunk by fabricated numbers;
> the whole integrity regime exists to stop that.

---

## Project identity

- **Name:** Real-Time Driver Drowsiness Detection via Signal Reliability Gating.
- **Type:** Master's / IEEE research project (Python, MediaPipe FaceMesh +
  classical geometry + temporal fusion).
- **The contribution is a *gate*, not a *classifier*.** It scores how
  *trustworthy* the fatigue evidence is each frame (lighting, landmark
  stability, cue agreement) and multiplicatively attenuates that evidence before
  accumulating it. Claim: lower FPR at matched TPR vs. plain weighted fusion.
- **Second novelty:** variance-based σ²(MAR) speech-jitter filter (suppresses
  talking-as-yawn false positives).

## Current milestone (as of 2026-07-29)

- **IMPLEMENTATION + FIRST EXPERIMENT CYCLE COMPLETE (EXP-000 … EXP-004).**
  Pipeline, harness, CNN training, quantization, and the LOSO ablation have all
  been run, measured, and logged. An independent scientific re-audit of EXP-004
  is also complete (`reports/EXP-004_AUDIT/`).
- **Measured results that now exist** (all with `EXP-###` rows + artifacts):
  - **EXP-001** latency = **3.205 ms/frame, Darwin-arm64 host — NOT a Pi 4.**
  - **EXP-002** MicroEyeNet trained on subject-disjoint MRL: VAL acc 0.9402 /
    F1 0.9262; TEST acc 0.9362 / F1 0.9623. Measured **19,745 params** (not the
    older "~9.5K" prose — see `reports/EXP-002_PARAMETER_AUDIT.md`).
  - **EXP-003** quantization: INT8 25.55 KB (3.18× smaller, −0.026% F1); FP16
    43.46 KB (0.0% F1 loss).
  - **EXP-004** LOSO ablation V0–V4 on NTHU-DDD: **NEGATIVE result** — the
    reliability gate does *not* reduce FPR@matched-TPR=0.80 (V2 0.6244 vs V0
    0.6241, flat), the speech filter *raises* FPR (V1/V3/V4 ≈0.669), and AUC
    never beats baseline (V0 0.629 highest). V4≡V3 byte-identical (CNN routes
    only to the alarm boolean, not `fatigue_score`).
- **Still NOT measured:** any Raspberry Pi 4 number; event-/episode-level FPR.
- Green: 6/6 integrity invariants, 17/17 unit tests, 3/3 smoke tests.

## Frozen decisions (do NOT change)

1. Reliability gate = **exactly 3** components (landmark_stability,
   brightness_quality, cue_consistency), **weighted geometric mean**, weights
   **(0.45, 0.30, 0.25)**. No 4th component (a phantom `tracking_confidence`/
   `tracking_quality` was removed and is now banned by CI).
2. **SEVERE** fatigue states are **never** suppressed by the gate (safety).
3. **Subject-disjoint LOSO** everywhere; **seed 42**.
4. Offline timing uses the **video clock** (`frame_index/fps`), not wall-clock.
5. **MAR stays 2D** (mouth idx `[78,13,308,14]`); EAR as implemented.
6. **CNN is ablation-only, OFF by default.**
7. `drowsiness_detection` dataset is **quarantined** (loader raises).
8. Ablation variants **V0–V4 are frozen** (toggles: speech_filter,
   reliability_gate, cnn): V0 (F,F,F) → V1 (T,F,F) → V2 (F,T,F) → V3 (T,T,F) →
   V4 (T,T,T).
9. Primary metric = **FPR @ matched TPR**, operating point fixed on **V0** at
   `target_tpr=0.80`; secondary = ROC-AUC (trapezoid), FPR/hour.

## Conventions

- All thresholds/weights live in `src/config.py` — no magic numbers in logic.
- `src/frame_processor.py` is the single per-frame path shared by app + harness.
  It is built **fresh once per subject** in the LOSO harness.
- ROC sweep variable = `FrameResult.fatigue_score ∈ [0,1]`.
- Every experiment: **land code → run → log `EXP-###` → commit artifact.** Never
  batch numbers ahead of measurement.

## Known findings / limitations (current)

- **The primary hypothesis did not hold at frame level (EXP-004).** The gate
  does not reduce frame-level FPR@matched-TPR; the speech filter worsens it.
  Treat this as the honest starting point for EXP-005, not something to "fix" by
  retuning until the number improves.
- **No Raspberry Pi 4 numbers exist.** Do not invent them. The 3.205 ms is host.
- **V4 ≡ V3 (byte-identical) is structural, not a bug.** The CNN verdict feeds
  only `StateManager`'s `should_alarm` boolean, never the swept `fatigue_score`,
  so it cannot move a frame-level ROC. Any CNN effect needs an event-level metric.
- **Subject 006 is a below-chance outlier** (per-subject AUC ≈0.30–0.37; inverted
  label balance) and drags down the aggregate. See the EXP-004 audit.
- **NTHU labels are clip-condition derived** → frame-level FPR is *conservative*
  (an open-eyed frame in a "yawning" clip is still labelled drowsy). Document,
  don't "fix."
- **Future experiment IDs are now official and fixed** (see
  `EXPERIMENT_REGISTRY.md §4`): **EXP-005** Event-Level Alarm Evaluation,
  **EXP-006** Gate Redesign Evaluation, **EXP-007** Raspberry Pi Deployment
  Evaluation, **EXP-008** Second Dataset Validation (optional). This supersedes
  the earlier ID clash — the EXP-004 audit's "EXP-005" (gate redesign) is
  officially **EXP-006**, and the CNN spec's "EXP-005" (V4 ablation) was already
  done inside EXP-004. A reconciliation table in the registry maps the old IDs.

## Pending tasks (the next phase, official roadmap — `EXPERIMENT_REGISTRY.md §4`)

1. **EXP-005 — Event-Level Alarm Evaluation.** Move from frame-level
   `fatigue_score` ROC to an alarm-event metric (FPR/hour + episode latency)
   over V0–V4, where the CNN arm and the gate's spurious-alarm protection can
   actually be observed. Log as `EXP-005`.
2. **EXP-006 — Gate Redesign Evaluation.** Re-architect the gate as an additive
   decision-layer term (the EXP-004 audit's recommended follow-up; the audit
   calls it "EXP-005", official ID is `EXP-006`). Log as `EXP-006`.
3. **EXP-007 — Raspberry Pi Deployment Evaluation.** Profile on a real Raspberry
   Pi 4 (per-stage latency/memory/thermal); log as `EXP-007`.
4. **EXP-008 — Second Dataset Validation (optional).** Reproduce on another
   subject-disjoint dataset; log as `EXP-008`.
5. Populate `paper/main.tex` results ONLY from committed artifacts
   (`results/measured_results.json` + `experiments/EXP-00X_*/`); regenerate
   figures with `evaluation/plot_paper_figures.py`. State EXP-004 honestly.
   (Writing task — no `EXP-###` of its own.)

## Expected outputs

- Results land in `results/measured_results.json` (latency block already there;
  a `roc` block appears after a `--write` LOSO run).
- Figures come only from that JSON (`plot_paper_figures.py` refuses otherwise).

## Reading priority

1. This file (`AGENT_MEMORY.md`) — orientation.
2. `PROJECT_CONTEXT.md` — the full single source of truth.
3. `reports/IMPLEMENTATION_SPECIFICATION_FROZEN.md` — the frozen contract.
4. `EXPERIMENT_REGISTRY.md` — what has actually been measured.
5. `IMPLEMENTATION_LOG.md` — why the repo looks like it does.

## Files to open first (to act)

- `src/frame_processor.py` — the shared per-frame pipeline.
- `src/robustness.py` — the core novelty (3-component geometric-mean gate).
- `evaluation/loso_harness.py` — the V0–V4 evaluation harness.
- `evaluation/nthu_ground_truth.py` — NTHU label mapping (metadata only).
- `src/config.py` — every threshold/weight.

## Files / things to NEVER modify (without design sign-off)

- The design sections of `reports/IMPLEMENTATION_SPECIFICATION_FROZEN.md`
  (only its Appendix A change-log may be appended to).
- The 3-component / geometric-mean / weights (0.45,0.30,0.25) contract in
  `src/robustness.py`.
- The LOSO seed (42), the V0–V4 definitions, the FPR@TPR operating-point rule.
- The `drowsiness_detection` quarantine (`src/data_loaders.py`).
- Anything enforced by `evaluation/verify_integrity.py` (invariants I1–I6).
- Do NOT revive the deleted files `evaluation/ablation_runner.py`,
  `evaluation/latency_memory_profiler.py`, `tools/train_eye_cnn.py`.

## How to continue safely (checklist)

1. Make the smallest change that advances one pending task.
2. Run all three gates and keep them green:
   - `python3 evaluation/verify_integrity.py` (→ 6/6)
   - `python3 -m unittest tests.test_suite` (→ 17/17)
   - `python3 tests/smoke_test.py` (→ 3/3)
3. If you produced a number: add an `EXP-###` row, commit the artifact, THEN
   cite it. Never the other way round.
4. Never fabricate, project, or estimate a performance number.
5. Do not implement/train the CNN unless the task explicitly asks — and even
   then it stays ablation-only and OFF by default.

## Common mistakes (do not repeat)

- ❌ Adding a 4th reliability component. ❌ Arithmetic mean. ❌ Suppressing SEVERE.
- ❌ Writing a Pi 4 latency number. ❌ Random (subject-mixing) split.
- ❌ Loading `drowsiness_detection`. ❌ Making MAR 3D.
- ❌ Any paper/report number without an `EXP-###` row + committed artifact.
