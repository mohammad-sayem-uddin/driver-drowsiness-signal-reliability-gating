# `docs/archive/` — Historical Documents

**Archived:** 2026-07-29

> ⚠️ **Historical only.** These files predate the completed experiments and the
> frozen implementation spec. Do not read them as current status. For current
> information start at the root [`../../README.md`](../../README.md) and
> [`../../PROJECT_CONTEXT.md`](../../PROJECT_CONTEXT.md).

The rest of `docs/` was cleared out during the 2026-07-29 cleanup: it contained
eight **empty (0-byte) placeholder files** (`AI_RESEARCH_CONSTITUTION.md`,
`RESEARCH_DIRECTOR.md`, `CONTINUOUS_RESEARCH_AGENT.md`,
`LITERATURE_REVIEW_STRATEGY.md`, `IMPLEMENTATION_STRATEGY.md`,
`PROJECT_VS_LITERATURE_COMPARISON.md`, `PUBLICATION_READINESS_AUDIT.md`,
`RESEARCH_EXECUTION_MASTER_PLAN.md`), which were deleted as they held no content.

## What is here

| File | What it is | Why archived |
|---|---|---|
| `old_setup_guide_README.md` | An early macOS environment/setup guide that described a much smaller project layout (`detector.py` + `main.py` + `utils/` only). | Contradicts the current architecture (12+ `src/` modules, evaluation harness, experiments). Superseded by the root [`../../README.md`](../../README.md). |
| `research_notes.md` | A large (~210 KB) working research journal / brainstorming log, including early paper-abstract drafts with unmeasured placeholder numbers (e.g. "~9.5K params", "<0.5 ms", "X% / Y FPS"). | Working notes, not a source of truth; contains pre-measurement estimates now superseded by logged experiments. Kept for narrative provenance. |
| `TECHNICAL_AUDIT_REPORT.md` | A large (~80 KB) point-in-time technical audit of an earlier system revision. | Predates the frozen spec and the measured experiments. |
