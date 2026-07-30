# ARCHIVE PLAN

**Project:** Real-Time Driver Drowsiness Detection via Signal-Reliability Gating
**Date:** 2026-07-30
**Purpose:** Stage the exact physical handling of every non-FINAL file — what
moves where, what gets a SUPERSEDED banner, what is removed — as an ordered,
reversible plan. This turns the classes in `FILE_CLASSIFICATION.md` and the
resolutions in `CONTRADICTION_REPORT.md` into concrete file operations.

> **NOTHING IN THIS DOCUMENT IS EXECUTED.** Every row below is a recommendation
> awaiting the professor's / maintainer's approval. The ordered runbook that
> actually performs them (once approved) is `CLEANUP_CHECKLIST.md`.

---

## 1. Principles applied to every operation

1. **Prefer archive over delete.** A file leaves the working tree only if it is
   OS/interpreter cruft or foreign to the project. Everything with any
   provenance value is *moved*, not removed.
2. **Reproducibility is inviolable.** No file that any FINAL report cites —
   code, config, dataset pointer, measured artifact, figure, raw output — is
   moved or deleted. Archiving touches only *superseded prose and scratch*.
3. **Reversibility.** Every move is a `git mv`-style relocation (or a plain
   `mv`), recoverable from history. Banners are additive (prepended text), never
   content deletion. Deletes are limited to regenerable/foreign items.
4. **One authoritative location per topic.** After the plan, each current topic
   has exactly one non-archived home (see `CONTRADICTION_REPORT.md §6`).
5. **Justify everything.** Every move/banner/delete row states why.

---

## 2. Target archive layout (where things land)

The repo already has `reports/archive/` and `docs/archive/`. This plan adds two
small, clearly-named homes and reuses the existing ones:

```
reports/
  archive/
    audit_v3.1/ phase01/ phase02/ phase02_5/        # already here — unchanged
    planning/ reviews/ verification/                 # already here — unchanged
    publication_review_2026-07-30/                   # NEW — the R1/R2 review bundle
    superseded/                                      # NEW — banner-ed dead reports
scratch/                                             # NEW (repo root) — ad-hoc probes
  archive/                                           # holds the 4 root test_*.py probes
```

Rationale for two new folders only: the review bundle and the superseded
root-cause doc do not fit any existing archive sub-theme, and the four root
`test_*.py` probes are code (not reports) so they do not belong under
`reports/`. Keeping the new folders shallow avoids re-disturbing the existing
archive tree.

---

## 3. Operation A — SUPERSEDED banner + relocate (stale root-cause doc)

**File:** `reports/EXP005_ROOT_CAUSE_ANALYSIS.md`
**Action:** (1) prepend a SUPERSEDED banner (exact text below); (2) move to
`reports/archive/superseded/EXP005_ROOT_CAUSE_ANALYSIS.md`.
**Authority:** `reports/PUBLICATION_RECOVERY_PLAN.md` item **C7** explicitly
directs marking it superseded. This is the project's own newest doc — the plan
does not originate the decision, it executes an already-authorized one.
**Justification:** it narrates an *old* EXP-005 run (10,800 frames, recall 0.0,
16/16 GT episodes missed) that is contradicted by the committed final
`reports/EXP-005_REPORT.md` (66,521 frames, recall 0.122, audited ACCEPT).
Leaving it un-bannered in the current `reports/` set means a reader hits the
dead story first by filename order (see `CONTRADICTION_REPORT.md` K2).
**Why keep it at all:** it is the diagnostic narrative that led to the corrected
run — genuine provenance. Archive, never cite.

Banner to prepend (verbatim):

```markdown
> ⚠️ **SUPERSEDED (2026-07-30).** This document describes an *early, superseded*
> EXP-005 run (10,800 frames, recall 0.0). It does **not** reflect the final,
> audited EXP-005 result. The authoritative EXP-005 record is
> [`reports/EXP-005_REPORT.md`](../../EXP-005_REPORT.md) with audit
> [`reports/EXP-005_AUDIT.md`](../../EXP-005_AUDIT.md) (66,521 frames, recall
> 0.122, verdict ACCEPT). Retained for provenance only — **do not cite**.
> Directed by `PUBLICATION_RECOVERY_PLAN.md` item C7.
```

> The relative link depth (`../../`) assumes the file lands in
> `reports/archive/superseded/`. If the maintainer prefers to keep it in place
> (banner only, no move), use `./` instead — the banner is the essential part;
> the move is the tidiness step.

---

## 4. Operation B — Bundle the publication-review layer

**Files (move as a set):**

| From | To |
|---|---|
| `reports/INDEPENDENT_SCIENTIFIC_REVIEW.md` | `reports/archive/publication_review_2026-07-30/INDEPENDENT_SCIENTIFIC_REVIEW.md` |
| `reports/PUBLICATION_READINESS_ASSESSMENT.md` | `reports/archive/publication_review_2026-07-30/PUBLICATION_READINESS_ASSESSMENT.md` |
| `Prompt_1_Independent_Scientific_Research_Review.md` | `reports/archive/publication_review_2026-07-30/Prompt_1_Independent_Scientific_Research_Review.md` |
| `Prompt_2_Independent_Publication_Readiness_Assessment.md` | `reports/archive/publication_review_2026-07-30/Prompt_2_Independent_Publication_Readiness_Assessment.md` |

**Action:** move the four files into one dated bundle; add a short
`README.md` in that folder listing the input→output→plan chain.
**Justification:** R1 and R2 are genuine review *outputs*, and the two Prompt
files are their read-only *inputs*. All four have already been **consumed** by
`reports/PUBLICATION_RECOVERY_PLAN.md` (the C1–C13 checklist folds R1+R2 into a
single action list). They are currently reachable from no index
(`CONTRADICTION_REPORT.md §7`), so bundling them dated keeps the chain intact
without presenting four loose review docs as "current reports."

**Explicitly NOT moved:** `reports/PUBLICATION_RECOVERY_PLAN.md` stays
top-level in `reports/` — it is classified **FINAL (action doc)**: the newest
link in the chain and the authoritative to-do list the professor will actually
work from.

Bundle README to create (`reports/archive/publication_review_2026-07-30/README.md`):

```markdown
# Publication Review Bundle — 2026-07-30

Input → output → plan chain, archived together for provenance.

| Stage | File | Role |
|---|---|---|
| Input  | Prompt_1_Independent_Scientific_Research_Review.md | Task prompt that produced R1 |
| Output | INDEPENDENT_SCIENTIFIC_REVIEW.md | R1 — "Scientifically Weak" |
| Input  | Prompt_2_Independent_Publication_Readiness_Assessment.md | Task prompt that produced R2 |
| Output | PUBLICATION_READINESS_ASSESSMENT.md | R2 — "Weak Reject" |
| Plan   | ../../PUBLICATION_RECOVERY_PLAN.md | **Current** — consumes R1+R2 into C1–C13 |

These reviews are **historical inputs**. The live, authoritative action list is
`reports/PUBLICATION_RECOVERY_PLAN.md`. Do not cite R1/R2 in isolation.
```

---

## 5. Operation C — Relocate root-level scratch probes

**Files (move as a set):**

| From | To |
|---|---|
| `test_pipeline.py` | `scratch/archive/test_pipeline.py` |
| `test_pose.py` | `scratch/archive/test_pose.py` |
| `test_variance.py` | `scratch/archive/test_variance.py` |
| `test_webcam.py` | `scratch/archive/test_webcam.py` |

**Action:** move the four ad-hoc probes out of the repo root into
`scratch/archive/`; add a one-line `scratch/README.md` marking the folder as
non-authoritative developer probes.
**Justification:** these are ad-hoc developer probes, **not** part of the test
framework (the real gates are `tests/test_suite.py`, `tests/smoke_test.py`,
`tests/test_event_metrics.py`). `IMPLEMENTATION_LOG.md` records them as "left
untouched, low risk." Their `test_*.py` names in the repo root are a navigation
hazard — a reader (or a test runner doing discovery) can mistake them for the
suite. Moving de-clutters root and removes the naming collision.
**Not deleted:** `test_variance.py` contains a dead-duplicate local
`EMASmoother`; deleting it would be safe, but "prefer archive over delete"
governs. Kept in `scratch/archive/`.

> **Caution flag (test discovery):** if any CI or `pytest`/`unittest` discovery
> globs `test_*.py` from the repo root, moving these *changes what gets
> collected*. This is a **desired** correction (they were never real tests), but
> the maintainer should confirm no CI config points discovery at the root before
> executing. Verified target discovery path is `tests/` (see README getting-started).

---

## 6. Operation D — Remove cruft (regenerable / foreign, justified)

Only OS/interpreter cruft and one foreign download are removed. Each is
regenerable or was never part of the project.

| Item | Count / size | Justification for removal | Reversible? |
|---|---|---|---|
| `.DS_Store` | 5 files | macOS Finder metadata; zero project value; should be gitignored | Regenerated by Finder; harmless |
| `__pycache__/` | 5 dirs | Python bytecode cache; regenerated on next run | Fully regenerable |
| `benchmark/` | 1 empty dir (0 B) | Referenced by nothing; the latency harness lives in `evaluation/benchmark_nthan_yawdd.py` | Trivially re-creatable |
| `gitlab-vscode-extension-main/` | 8.9 MB | Third-party VS Code extension download; foreign to the pipeline, gitignored, referenced by nothing | Re-downloadable from source |

**Add to `.gitignore` (if not already):** `.DS_Store`, `__pycache__/`,
`*.pyc`, `gitlab-vscode-extension-main/`, `.venv/`. This prevents the cruft from
returning and keeps the foreign/regenerable items out of the archive bundle.

---

## 7. Operation E — `.venv/` is kept, not touched

**File:** `.venv/` (1.7 GB)
**Action:** **none** at freeze. Keep in place; ensure it is gitignored; exclude
from any archive *bundle* (tarball) that gets shipped to the professor.
**Justification:** it is a regenerable Python 3.11 virtualenv
(`pip install -r requirements.txt`), not junk and not a DELETE candidate. It
stays for the working environment. It is only *excluded from the shipped
archive* because it is large and rebuildable — that is a packaging choice, not a
deletion. Called out explicitly so no one "cleans" it by mistake.

---

## 8. What this plan does NOT move or delete (reproducibility firewall)

The following are **FINAL** and are never touched by any operation above. Listed
so the guarantee is explicit and auditable.

| Preserved | Why untouchable |
|---|---|
| `results/measured_results.json` (+ 2 `.png`) | Canonical figure/number source for every report |
| `experiments/EXP-002_*`, `EXP-003_*`, `EXP-004_*`, `EXP-005_*` (62 MB) | Raw committed evidence — the reproducibility backbone |
| `models/*.tflite` (3) | Quantization outputs cited by EXP-003 |
| `checkpoints/*.keras` (4) | Training checkpoints backing EXP-002 |
| `tensorboard/EXP-002_*` | Training-curve evidence for EXP-002 |
| `src/` (all real modules), `evaluation/` (all 8), canonical `tools/` | The pipeline + harness that produced the results |
| `tests/` (all 3 suites) | The verification gates |
| All 8 FINAL experiment reports + governing docs | Authoritative record |
| `results/measured_results.json.pre_exp004.bak` | Meaningful pre-EXP-004 snapshot; kept (ACTIVE), not cited |
| `src/camera_base.py` | ACTIVE-orphan; engineer decision, **not** an archive/delete target at freeze |

> If executing any operation would touch a row in this table, **stop** — the
> operation is mis-specified. This table is the guardrail.

---

## 9. Full operation summary (execution order)

| # | Op | Kind | Items | Reversible |
|---|---|---|---|---|
| A | Banner + move stale root-cause doc | banner + move | 1 | Yes (git history) |
| B | Bundle publication-review layer | move (4) + new README | 4 + 1 | Yes |
| C | Relocate root scratch probes | move | 4 + new README | Yes |
| D | Remove cruft | delete | 5 `.DS_Store` + 5 `__pycache__` + `benchmark/` + `gitlab-vscode-extension-main/` | Regenerable/foreign |
| E | Keep `.venv/` | none | — | n/a |

**Net effect on the working tree:**
- Repo root loses 4 scratch `test_*.py` + 2 Prompt files + (optionally) the
  bannered doc's clutter → root shows only the 7 governing docs + real dirs.
- `reports/` current set drops from 14 → 11 (root-cause doc, R1, R2 relocated);
  every remaining current report is authoritative.
- No FINAL artifact, no code, no dataset, no figure is moved or deleted.

---

## 10. Execution note

None of the above is executed in this pass. The ordered, copy-pasteable command
sequence (with the `git mv` / `mkdir` / banner-prepend steps and a verification
checkpoint after each) is in `CLEANUP_CHECKLIST.md`, to be run only after the
professor approves this plan. The documentation-only prose edits that pair with
Operation A (the EXP-005 status flip across the six nav docs) are specified
separately in `DOCUMENTATION_CLEANUP.md`.





