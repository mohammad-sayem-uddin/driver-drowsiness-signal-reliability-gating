# `reports/archive/` — Historical & Process Documents

**Archived:** 2026-07-29

> ⚠️ **Everything in this folder is HISTORICAL.** These documents were accurate
> at the time they were written but **predate the completed experiments
> (EXP-001 … EXP-004)** and do **not** describe the current state of the project.
> They are kept for provenance and thesis narrative — *how* and *why* the
> repository evolved — and must **not** be cited as current status or claims.
>
> For current information see [`../README.md`](../README.md), the root
> [`../../PROJECT_CONTEXT.md`](../../PROJECT_CONTEXT.md), and the measured
> `EXP-002/003/004` reports in [`../`](../).

Common ways these files are now **out of date**: they were written when the
system was called "v3.1", when the CNN model file was missing, when *no*
accuracy/benchmark results existed, and when the reliability-gate hypothesis was
still unfalsified. EXP-004 has since produced a **negative result** for that
hypothesis at frame level — none of the documents below reflect that.

---

## What is here

### `audit_v3.1/` — pre-experiment audit series (Stages 1–11, July 2026)
The `01_…` through `12_…` numbered deep-audit of the v3.1 system (project audit,
architecture, implementation status, literature review, related-work matrix,
research-gap analysis, project-vs-literature, novelty audit, reviewer
simulation, publication readiness, improvement roadmap, final verdict), plus
`exhaustive_literature_review_and_gaps.md`. Point-in-time analyses; the
architecture descriptions are broadly still valid, the readiness/roadmap
conclusions are overtaken.

### `reviews/` — simulated peer review & editorial artifacts
`editor_decision.md`, `red_team_review.md`, `reviewer_comments.md`,
`rejection_reasons.md`, `fatal_flaws.md`. Simulated reviewer/editor feedback used
to harden the work. Their central premise — "zero empirical benchmarks / unbacked
accuracy" — is exactly what EXP-002/003/004 were run to address.

### `planning/` — roadmaps, phase directives, and pre-training status
`MASTER_RESEARCH_EXECUTION_ROADMAP.md`, `PHASE_01_REPOSITORY_STABILIZATION.md`
(a role/instruction prompt), `PROJECT_COMPLETION_AND_ARCHITECTURE_UPGRADE_REPORT.md`
(⚠️ dated 2026-07-28 but still says "training deferred / benchmarking pending" —
now false), `conference_publication_plan.md`,
`conference_publication_viability_and_action_plan.md`, `fix_everything.md`.

### `phase01/` — repository-stabilization reports (v3.1 baseline)
Bug fixes, code cleanup, config reference, dependency review, documentation
audit, performance review (a pre-EXP-001 latency profile), repository-health,
testing report, and the `baseline_v1.md` (an earlier freeze superseded by
`../IMPLEMENTATION_SPECIFICATION_FROZEN.md`).

### `phase02/` — data-foundation & benchmark-infrastructure reports
Dataset selection/review/cards/statistics/quality/integrity, the subject-split
report, preprocessing, benchmark preparation, and the phase-02 completion
summary. Describes the dataset pipeline that later fed EXP-002.

### `phase02_5/` — learned-reliability framing
`architecture_upgrade_summary.md` and `learned_reliability_framework.md`: the
log-space temperature-scaling formulation shown to reduce to the weighted
geometric mean. This is the gate machinery EXP-004 later tested.

### `verification/` — independent fact-check committee (Stages 1–10, July 2026)
A verification pass over the `audit_v3.1/` series (citation integrity, reference
accuracy, missing literature, gap validation, novelty re-scoring, threat papers,
implementation verification, experimental verification, publication verification,
master fact-check). Valuable precisely because it flags errors in the older
reports. (The *current* EXP-002 dataset verification was promoted out of here to
[`../EXP-002_DATASET_VERIFICATION.md`](../EXP-002_DATASET_VERIFICATION.md).)
