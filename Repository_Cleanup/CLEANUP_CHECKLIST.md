# CLEANUP CHECKLIST

**Project:** Real-Time Driver Drowsiness Detection via Signal-Reliability Gating
**Date:** 2026-07-30
**Purpose:** The ordered, copy-pasteable runbook that executes the plan in the
other eight documents — with a verification checkpoint after every step so the
operator can stop safely at any point.

> **Do not run this yet.** Every command below is **staged for approval**. This
> is the only document that recommends destructive/irreversible actions, and it
> runs **only after you say so**. Run from the repository root:
> `/Users/sayemuddin/Desktop/Driver Drowsiness`. Do the steps **in order** —
> the git snapshot in Step 0 is what makes every later step reversible.

---

## Step 0 — Safety snapshot (do this first, always)

Create a branch and a full working-tree snapshot so every later step is
recoverable with a single `git` command.

```bash
git status                                  # confirm you recognize the state
git checkout -b cleanup/publication-freeze  # never work on main
git add -A
git commit -m "snapshot: pre-cleanup working tree (reversible restore point)"
```

**Checkpoint:** `git log --oneline -1` shows the snapshot commit; `git status`
is clean. If anything later goes wrong: `git reset --hard <snapshot-sha>`.

> This commit also closes reproducibility gap **G-2** (`REPRODUCIBILITY_CHECK.md`
> §2): it brings `evaluation/`, `experiments/`, `results/`, `checkpoints/`, and
> the untracked docs into the tracked repository. This is the single most
> important step for the professor handover.

---

## Step A — Banner + archive the superseded EXP-005 root-cause doc

Ref: `ARCHIVE_PLAN.md` §3, `CONTRADICTION_REPORT.md` K2.

1. Prepend the verbatim SUPERSEDED banner (from `ARCHIVE_PLAN.md` §3) to
   `reports/EXP005_ROOT_CAUSE_ANALYSIS.md`, citing recovery-plan item C7 and
   pointing to `reports/EXP-005_REPORT.md`.
2. Move it:

```bash
mkdir -p reports/archive/superseded
git mv reports/EXP005_ROOT_CAUSE_ANALYSIS.md reports/archive/superseded/
```

**Checkpoint:** the file exists only under `reports/archive/superseded/`, opens
with the banner, and no current doc links to its old path
(`grep -rl "EXP005_ROOT_CAUSE_ANALYSIS" --include="*.md" .` returns only archive
refs).

---

## Step B — Bundle the publication-review inputs

Ref: `ARCHIVE_PLAN.md` §4.

```bash
mkdir -p reports/archive/publication_review_2026-07-30
git mv reports/INDEPENDENT_SCIENTIFIC_REVIEW.md      reports/archive/publication_review_2026-07-30/
git mv reports/PUBLICATION_READINESS_ASSESSMENT.md   reports/archive/publication_review_2026-07-30/
git mv Prompt_1*.md Prompt_2*.md                     reports/archive/publication_review_2026-07-30/
```

Then write `reports/archive/publication_review_2026-07-30/README.md` explaining
that R1 + R2 and their two input prompts were folded into
`reports/PUBLICATION_RECOVERY_PLAN.md`, which supersedes them.

**Checkpoint:** `reports/PUBLICATION_RECOVERY_PLAN.md` remains at top level;
R1/R2/prompts live only in the dated folder; the folder README exists.

---

## Step C — Move root scratch probes out of the way

Ref: `ARCHIVE_PLAN.md` §5, `FOLDER_STRUCTURE.md` §4.

```bash
mkdir -p scratch/archive
git mv test_*.py scratch/archive/     # the 4 root-level probe scripts only
```

Write `scratch/README.md`: "Ad-hoc dev probes. Non-authoritative. The real
suites are in `tests/`. Nothing here is cited."

**Checkpoint:** `pytest` default discovery no longer collects the moved probes
alongside `tests/`; the real suites in `tests/` still run and pass. Confirm no
root `test_*.py` remains: `ls test_*.py 2>/dev/null` prints nothing.

---

## Step D — Remove junk and the foreign download

Ref: `ARCHIVE_PLAN.md` §6, `FOLDER_STRUCTURE.md` §5.

First make these ignorable so they never return, then remove them:

1. Update `.gitignore` to include `.DS_Store`, `__pycache__/`, `.venv/`, and the
   foreign extension path.
2. Remove:

```bash
find . -name .DS_Store -delete
find . -type d -name __pycache__ -exec rm -rf {} +
rmdir benchmark 2>/dev/null                 # only if truly empty
rm -rf gitlab-vscode-extension-main         # foreign download, 8.9 MB, not evidence
```

**Checkpoint:** `find . -name .DS_Store` and `find . -type d -name __pycache__`
return nothing; `benchmark/` and `gitlab-vscode-extension-main/` are gone;
`git status` shows only intended changes. **`.venv/` is NOT deleted** — it is
gitignored and kept (`ARCHIVE_PLAN.md` §7).

---

## Step E — Apply the documentation edits

Ref: `DOCUMENTATION_CLEANUP.md` (all sections).

1. §1 status-flip: EXP-005 planned → done in README, HANDOVER, paper,
   EXPERIMENT_REGISTRY, AGENT_MEMORY, IMPLEMENTATION_LOG.
2. §2 accuracy: C1 (2D not 3D), C2 (mean |ΔMAR| not variance), C3 (state-level
   SEVERE guard), C4 (YawDD not evaluated), C5 (per-variant TPR).
3. §3 test count: add the event-metric suite — **use the live count** from
   `tests/`, not "~65".
4. §4 C9: read subject-006 AUC from the committed EXP-004 per-subject CSV and
   correct the disagreeing doc to match. If the CSV is missing, **stop** and
   record it as gap G-1 instead of guessing.

**Checkpoint:** grep the corrected phrases to confirm no stale claim survives —
e.g. `grep -rn "never suppressed\|variance\|3D EAR\|next step" --include="*.md"
--include="*.tex" .` returns only intended/quoted matches.

---

## Step F — Verify, then commit

```bash
python -m pytest tests/                      # real suites still green
python evaluation/verify_integrity.py        # I1–I6 all pass
git add -A
git status                                    # review every change once
git commit -m "chore: repository cleanup & publication freeze (archive + doc fixes)"
```

**Checkpoint:** tests pass, integrity gate passes, `git status` clean. Do **not**
push or merge to `main` — leave the `cleanup/publication-freeze` branch for
review unless you are explicitly asked to push.

---

## Order & reversibility summary

| Step | Action | Reversible by |
|---|---|---|
| 0 | Branch + snapshot commit (also closes G-2) | `git reset --hard <snapshot>` |
| A | Banner + archive superseded EXP-005 doc | `git mv` back |
| B | Bundle R1/R2/prompts | `git mv` back |
| C | Move root scratch probes | `git mv` back |
| D | Delete junk + foreign download | restore from snapshot commit |
| E | Documentation edits | `git checkout <snapshot> -- <file>` |
| F | Verify + commit on branch | branch is discardable |

Every step is a `git`-tracked change on a branch off a clean snapshot, so the
entire cleanup reverts with one command. **Nothing runs until you approve.**


