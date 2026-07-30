# `reports/` — Index

**Last organized:** 2026-07-29

This directory holds the **frozen engineering contract** and the **measured
experiment reports**. Everything here is *current* (reflects the completed
experiments EXP-001 … EXP-004). Point-in-time planning, audit, review, and
phase-history documents have been moved to [`archive/`](archive/) — see the
[archive index](archive/README.md) for what lives there and why.

> **Truth policy (unchanged):** no performance number is citable without an
> `EXP-###` row in [`../EXPERIMENT_REGISTRY.md`](../EXPERIMENT_REGISTRY.md) and a
> committed artifact in [`../results/`](../results/) or
> [`../experiments/`](../experiments/).

---

## Current documents (source of truth)

| File | Role | Status |
|---|---|---|
| [IMPLEMENTATION_SPECIFICATION_FROZEN.md](IMPLEMENTATION_SPECIFICATION_FROZEN.md) | The frozen engineering contract (research design, module contract, protocol, integrity invariants). Do not edit except its Appendix A change-log. | 🔒 FROZEN |
| [EXP-002_REPORT.md](EXP-002_REPORT.md) | MicroEyeNet baseline training on subject-disjoint MRL (val acc 0.9402 / F1 0.9262). | ✅ measured |
| [EXP-002_PARAMETER_AUDIT.md](EXP-002_PARAMETER_AUDIT.md) | Reconciles the measured **19,745** parameter count against the spec's older "~9.5K" prose. | ✅ measured |
| [EXP-002_DATASET_VERIFICATION.md](EXP-002_DATASET_VERIFICATION.md) | Independent verification that the MRL subject-disjoint split is leak-free. | ✅ measured |
| [EXP-003_REPORT.md](EXP-003_REPORT.md) | Float16 & INT8 TFLite quantization (INT8 25.55 KB, −0.026% F1). | ✅ measured |
| [EXP-004_REPORT.md](EXP-004_REPORT.md) | LOSO ablation V0–V4 on NTHU-DDD. **Negative/null result:** the reliability gate does not reduce FPR@matched-TPR. | ✅ measured |
| [EXP-004_AUDIT/](EXP-004_AUDIT/) | Independent post-hoc scientific re-audit of EXP-004 (recomputed metrics, DeLong/bootstrap significance, subject-006 inversion, 8 figures). | ✅ audit |

## Reading order for a new reviewer

1. [`../README.md`](../README.md) — project overview and current status.
2. [`../PROJECT_CONTEXT.md`](../PROJECT_CONTEXT.md) — single source of truth.
3. [IMPLEMENTATION_SPECIFICATION_FROZEN.md](IMPLEMENTATION_SPECIFICATION_FROZEN.md) — the frozen contract.
4. [`../EXPERIMENT_REGISTRY.md`](../EXPERIMENT_REGISTRY.md) — what has actually been measured.
5. The `EXP-002/003/004` reports above, then [EXP-004_AUDIT/](EXP-004_AUDIT/).

## Archived material

Historical planning, audit, peer-review-simulation, and phase-completion
documents are in [`archive/`](archive/). They record *how* the project reached
its current state and are retained for provenance, **but they predate the
completed experiments and must not be read as current claims.** Several of them
(e.g. "no benchmarks exist yet", "CNN model file missing", "training pending")
were true when written and are now superseded.
