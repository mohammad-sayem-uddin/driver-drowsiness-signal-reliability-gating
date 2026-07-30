# CONTRADICTION REPORT

**Project:** Real-Time Driver Drowsiness Detection via Signal-Reliability Gating
**Date:** 2026-07-30
**Purpose:** For every topic where two or more files disagree, name the **one
authoritative file** and the **obsolete** one(s), and give the resolution the
next researcher should apply. This is a *record-consistency* audit only.

> **Scope guardrails (held throughout):** This report does **not** modify any
> scientific conclusion, redesign any algorithm, or propose new experiments. The
> negative EXP-004 result and the EXP-005 findings stand exactly as measured.
> Where a *document* describes the *code* incorrectly, the resolution is always
> "correct the prose to match the committed code/artifact," never "change the
> code to match the prose." Each item states the authoritative source and the
> evidence.

---

## 1. Contradiction severity legend

| Sev | Meaning |
|---|---|
| **BLOCKER** | Misleads a reader about project *status* or *what a number means*; must fix before handover |
| **MAJOR** | A claim in a governing doc contradicts the committed code/artifact |
| **MINOR** | Localized inconsistency (a stray number, a stale count) |

---

## 2. Master contradiction table

| # | Topic | Authoritative | Obsolete / wrong | Sev |
|---|---|---|---|---|
| K1 | EXP-005 status | EXP-005 reports + audit (done, ACCEPT) | README, HANDOVER, REGISTRY, AGENT_MEMORY, IMPL_LOG, paper (all say "planned") | BLOCKER |
| K2 | EXP-005 root cause | `EXP-005_REPORT.md` (66,521 frames, recall 0.122) | `EXP005_ROOT_CAUSE_ANALYSIS.md` (10,800 frames, recall 0.0) | BLOCKER |
| C1 | SEVERE "never suppressed" | `src/fatigue_fusion.py:196-197` (multiply is unconditional) | README/HANDOVER "SEVERE-exempt at score level" | MAJOR |
| C2 | Speech filter math | `src/temporal_analyzer.py:294-298` (mean\|ΔMAR\|) | "variance-based σ²(MAR)" in README/HANDOVER/paper | MAJOR |
| C3 | EAR dimensionality | `src/detector.py:55-123` (2D) | "3D EAR" in `paper/main.tex:39,46` | MAJOR |
| C4 | YawDD usage | EXP registry (never run) | `paper/main.tex:22` lists YawDD as evaluated | MAJOR |
| C5 | Matched operating point | `loso_harness.py` (per-variant nearest TPR) | "held constant across variants" in README/HANDOVER | MAJOR |
| C9 | Subject-006 AUC | `EXP-004_REPORT.md` (0.372733) | `EXP-004_AUDIT` (0.304675) | MINOR |
| T1 | Test count | `tests/` (17 + 3 + ~65 event) | "17 unit + 3 smoke" everywhere | MINOR |

---

## 3. BLOCKER contradictions (fix before handover)

### K1 — EXP-005 is done, but every navigation doc says "planned"

**Evidence.** EXP-005 is fully executed: `reports/EXP-005_REPORT.md`,
`reports/EXP-005_AUDIT.md` (verdict ACCEPT), `experiments/EXP-005_*` (44 MB:
episodes, event streams, metrics JSON, run log), the `evaluation/event_metrics.py`
+ `exp005_event_report.py` code, and ~65 tests in `tests/test_event_metrics.py`.
Yet:
- `README.md:20-22` — "The next step (EXP-005) is an event-level alarm evaluation."
- `HANDOVER.md` §5 (lines 175-185) — EXP-005 under "Not yet done."
- `EXPERIMENT_REGISTRY.md` — EXP-005 as "Planned (next)."
- `AGENT_MEMORY.md`, `IMPLEMENTATION_LOG.md`, `paper/main.tex:116,135-137` — all future-tense.

**Authoritative:** the EXP-005 report + audit + committed artifacts (a run that
happened cannot be un-happened by prose).

**Resolution (documentation only):** flip EXP-005 from "planned" to "complete
(audited ACCEPT)" in all six navigation docs; renumber the "next" roadmap to
begin at EXP-006. No scientific claim changes — EXP-005's own findings
(recall 0.122, FA/hr 6.5-9.7, all three observability gates FAIL) are reported
as-measured. Detailed edits in `DOCUMENTATION_CLEANUP.md`.

### K2 — Two different EXP-005 "root causes" on the record

**Evidence.** `reports/EXP005_ROOT_CAUSE_ANALYSIS.md` narrates a run of **10,800
frames, recall 0.0, 16/16 ground-truth episodes missed**. The committed final
`reports/EXP-005_REPORT.md` reports **66,521 frames, recall 0.122**. Both are
in the current `reports/` set, so a reader meets the stale story first by
filename order.

**Authoritative:** `reports/EXP-005_REPORT.md` + `EXP-005_AUDIT.md`.

**Resolution:** reclassify `EXP005_ROOT_CAUSE_ANALYSIS.md` as **SUPERSEDED** with
a banner pointing to the final report. This is explicitly directed by C7 of
`reports/PUBLICATION_RECOVERY_PLAN.md` — the project's own newest doc authorizes
it. Keep the file (it is the diagnostic narrative that led to the corrected run);
never cite it. Physical handling in `ARCHIVE_PLAN.md`.

---

## 4. MAJOR contradictions (doc claim vs committed code/artifact)

For every item the resolution is **fix the prose to match the code** — the code
is what ran and produced the results; the design freeze is real, the wording
around it drifted.

### C1 — "SEVERE fatigue is never suppressed by the gate"

**Evidence.** `src/fatigue_fusion.py:196-197` multiplies `raw_score *= reliability`
**unconditionally**, before any severity branch. The SEVERE exemption lives later
in `src/state_manager.py:361-388` (state-level), not at the score level. An
EAR-only maximum is ~0.45, below the SEVERE score threshold 0.75, so a
gate-attenuated score cannot by itself be rescued at the score stage.

**Authoritative:** the code. **Resolution:** reword README:34 / HANDOVER §1 to
"the SEVERE *state* is protected by the state machine (hysteresis), not by a
score-level gate bypass." The safety argument survives; only the level is
corrected. (This is C1 of the recovery plan.)

### C2 — "Variance-based σ²(MAR) speech-jitter filter"

**Evidence.** `src/temporal_analyzer.py:294-298` computes the **mean absolute
first difference** `mean|ΔMAR|`, not a variance. Threshold 0.05.

**Authoritative:** the code. **Resolution:** rename the mechanism throughout
(README:36-38, HANDOVER §1, `paper/main.tex`) to "mean-absolute-jitter MAR
filter" (or state the exact statistic). No behavior changes. (Recovery-plan C2.)

### C3 — "3D EAR"

**Evidence.** `src/detector.py:55-123` documents "pure 2D geometric
calculations"; EAR and MAR are both image-plane 2D. `paper/main.tex:39,46`
describes a 3D EAR.

**Authoritative:** the code. **Resolution:** correct the paper to 2D EAR/MAR.
(Recovery-plan C3.)

### C4 — YawDD listed as evaluated

**Evidence.** `paper/main.tex:22` lists YawDD among evaluation datasets. The
registry shows **no EXP row** ever ran on YawDD; 348 AVI clips sit unused.

**Authoritative:** the registry. **Resolution:** in the paper, move YawDD to
"available but not evaluated (future work)," or remove it from the evaluation
set. No number is affected (there is none). (Recovery-plan C4.)

### C5 — "Operating point held constant across variants"

**Evidence.** `loso_harness.py` `_fpr_at_tpr` (:155-162) selects the FPR at the
**nearest achieved TPR per variant**, and `_fix_operating_point` (:167-174)
returns that per-variant TPR. Realized matched TPRs are 0.80 / 0.7989 / 0.8006 —
close but **not identical**, so the point is not literally held constant on the
V0 threshold.

**Authoritative:** the code + the per-variant CSVs in `experiments/EXP-004_*`.
**Resolution:** reword README:94-97 / HANDOVER §4 to "FPR is read at the matched
TPR (≈0.80) achieved on each variant's own ROC," and report the three realized
TPRs. The negative result is unchanged. (Recovery-plan C5.)

---

## 5. MINOR contradictions (localized, low-risk)

### C9 — Subject-006 ROC-AUC disagrees between report and audit

**Evidence.** `reports/EXP-004_REPORT.md` records subject-006 AUC **0.372733**;
`reports/EXP-004_AUDIT/` records **0.304675** for the same subject. The
aggregate EXP-004 conclusion (negative; V0 highest AUC) is unaffected by which
per-subject figure is correct.

**Authoritative:** whichever traces to `experiments/EXP-004_*/per_subject/`
CSV — the committed artifact is the tiebreaker. **Resolution:** reconcile the two
docs against the per-subject CSV and correct the one that disagrees; add a
one-line note in the report. (Recovery-plan C9.)

### T1 — Test count understated

**Evidence.** README:63 / HANDOVER §5 state "17 unit + 3 smoke tests." The tree
also contains `tests/test_event_metrics.py` with ~65 event-metric tests (the
EXP-005 verification layer), unmentioned.

**Authoritative:** the `tests/` tree. **Resolution:** update the counts to
"17 unit + 3 smoke + ~65 event-metric tests" in README and HANDOVER. Purely a
documentation refresh.

---

## 6. One-authoritative-file-per-topic (quick reference)

| Topic | THE authoritative file | Do not cite |
|---|---|---|
| Design/scope/status | `PROJECT_CONTEXT.md` | any archived planning doc |
| Dataset integrity | `Data/FINAL_PRETRAINING_VERIFICATION_REPORT.md` | the two banner-ed `Data/` reports |
| EXP-002 result | `reports/EXP-002_REPORT.md` (+ dataset/param audits) | phase02 archive docs |
| EXP-003 result | `reports/EXP-003_REPORT.md` | — |
| EXP-004 result | `reports/EXP-004_REPORT.md` (+ `EXP-004_AUDIT/`) | audit_v3.1 archive |
| EXP-005 result | `reports/EXP-005_REPORT.md` (+ `EXP-005_AUDIT.md`) | `EXP005_ROOT_CAUSE_ANALYSIS.md` |
| Measured numbers | `results/measured_results.json` | any doc quoting a number without an EXP row |
| Frozen spec | `reports/IMPLEMENTATION_SPECIFICATION_FROZEN.md` | archived spec drafts |
| Recovery to-do | `reports/PUBLICATION_RECOVERY_PLAN.md` | R1/R2 in isolation |

---

## 7. Cross-reference & structural issues (not content contradictions)

| Issue | Detail | Handling |
|---|---|---|
| Orphaned reviews | R1/R2/recovery-plan reachable from no index | Add to `reports/README.md` (see `DOCUMENTATION_CLEANUP.md`) |
| Orphan module | `src/camera_base.py` imported by nobody | Engineer decision: wire in or drop; not a freeze blocker |
| Duplicate tool name | `evaluation/verify_integrity.py` vs `tools/verify_integrity.py` | **Not** a contradiction — different jobs (I1–I6 gate vs dataset SHA-256 dedup). Keep both; note the distinction in `reports/README.md`. |
| Git staleness | HEAD predates the reorg; critical files untracked | Reproducibility issue — see `REPRODUCIBILITY_CHECK.md` |

---

## 8. Resolution summary

- **2 BLOCKERs (K1, K2)** — both are *status/record* problems, fixable by
  documentation edits + one reclassification; no science touched.
- **5 MAJORs (C1–C5)** — all "prose describes code wrongly"; fix the prose.
  These are exactly recovery-plan items C1–C5, so the project's own newest doc
  already prescribes them.
- **2 MINORs (C9, T1)** — reconcile against the committed artifact / test tree.

Every resolution above is a **recommendation**. None is executed in this pass.
The documentation edits are specified in `DOCUMENTATION_CLEANUP.md`; the file
moves/banners in `ARCHIVE_PLAN.md`; the whole ordered runbook in
`CLEANUP_CHECKLIST.md`.
