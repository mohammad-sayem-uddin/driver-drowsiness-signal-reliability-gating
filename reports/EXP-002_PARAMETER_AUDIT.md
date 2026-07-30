# EXP-002 — Parameter Count Forensic Audit

**Audit ID:** EXP-002-AUDIT
**Date:** 2026-07-28
**Scope:** Determine why the measured MicroEyeNet parameter count is **19,745**
while the frozen specification's prose states **"~9.5K params"**.
**Constraint:** No model redesign, no architecture change, no retraining, no
source-code modification. Every conclusion is backed by measured repository
evidence.

> Method note: the model was rebuilt live from
> `tools/train_exp002_microeyenet.build_microeyenet()` in the project `.venv`
> (TF 2.17.1). The live rebuild reproduced **19,745** parameters exactly,
> byte-for-byte consistent with the committed artifact
> `experiments/EXP-002_microeyenet_baseline/model_summary.txt`. No numbers in
> this report are estimated.

---

## Step 1 — Actual Model (`model.summary()`, verbatim)

Reproduced live and identical to the committed `model_summary.txt`:

```
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ conv2d (Conv2D)                 │ (None, 24, 24, 8)      │            80 │
│ max_pooling2d (MaxPooling2D)    │ (None, 12, 12, 8)      │             0 │
│ conv2d_1 (Conv2D)               │ (None, 12, 12, 16)     │         1,168 │
│ max_pooling2d_1 (MaxPooling2D)  │ (None, 6, 6, 16)       │             0 │
│ flatten (Flatten)               │ (None, 576)            │             0 │
│ dense (Dense)                   │ (None, 32)             │        18,464 │
│ dropout (Dropout)               │ (None, 32)             │             0 │
│ dense_1 (Dense)                 │ (None, 1)              │            33 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params:        19,745 (77.13 KB)
 Trainable params:    19,745 (77.13 KB)
 Non-trainable params:     0 (0.00 B)
```

Measured trainable-weight shapes (from `layer.get_weights()`):

| Layer | Type | Weight shapes | Params |
|-------|------|---------------|--------|
| conv2d | Conv2D | `[(3,3,1,8), (8,)]` | 80 |
| max_pooling2d | MaxPooling2D | `[]` | 0 |
| conv2d_1 | Conv2D | `[(3,3,8,16), (16,)]` | 1,168 |
| max_pooling2d_1 | MaxPooling2D | `[]` | 0 |
| flatten | Flatten | `[]` | 0 |
| dense | Dense | `[(576,32), (32,)]` | 18,464 |
| dropout | Dropout | `[]` | 0 |
| dense_1 | Dense | `[(32,1), (1,)]` | 33 |

- **Total parameters:** 19,745
- **Trainable parameters:** 19,745
- **Non-trainable parameters:** 0

Measured per-layer config spot-checks: `conv2d` relu/use_bias=True, `conv2d_1`
relu/use_bias=True, `dense` relu/use_bias=True, `dropout` rate=0.3, `dense_1`
sigmoid/use_bias=True. Model `input_shape = (None, 24, 24, 1)`. **No BatchNorm
layer exists.**

---

## Step 2 — Layer-by-Layer Comparison vs Frozen Specification

Frozen source: `CNN_IMPLEMENTATION_SPECIFICATION.md` Part 2 "Frozen architecture
table" (lines 93–103), which the spec states "matches `src/cnn_validator.py`
docstring and `tools/train_cnn.py` Keras build."

| # | Frozen spec layer | Spec output shape | Implemented output shape | Verdict |
|---|-------------------|-------------------|--------------------------|---------|
| 0 | Input 24×24×1 grayscale | (24,24,1) | (24,24,1) | ✅ Matches |
| 1 | Conv2D 8, 3×3, `padding='same'`, ReLU | (24,24,8) | (24,24,8) | ✅ Matches |
| 2 | MaxPool2D 2×2 | (12,12,8) | (12,12,8) | ✅ Matches |
| 3 | Conv2D 16, 3×3, `padding='same'`, ReLU | (12,12,16) | (12,12,16) | ✅ Matches |
| 4 | MaxPool2D 2×2 | (6,6,16) | (6,6,16) | ✅ Matches |
| 5 | Flatten → **(576,)** ("6·6·16 = 576") | (576,) | (576,) | ✅ Matches |
| 6 | Dense 32, ReLU | (32,) | (32,) | ✅ Matches |
| 7 | Dropout rate=0.3 | (32,) | (32,) | ✅ Matches |
| 8 | Dense 1, Sigmoid | (1,) | (1,) | ✅ Matches |

**Every layer matches the frozen architecture table, including the flatten
width of 576.** No layer mismatch exists.

The only conflicting statement is the spec's separate prose line 118:
"**Parameter budget:** ~9.5K params (target; exact count to be logged from the
built model in EXP-002)." This prose disagrees with the spec's *own* table
(576-wide flatten). It is explicitly labelled a *target* whose *exact count is
to be logged from the built model* — i.e. the spec itself defers to the measured
count, which is now 19,745.

---

## Step 3 — Manual Parameter Recalculation (independent of TensorFlow)

Conv2D params = `(kernel_h · kernel_w · in_channels · filters) + filters`
Dense params = `(in_features · units) + units`
Pooling / Flatten / Dropout are parameter-free.

| Layer | Formula | Weights | Biases | Layer total |
|-------|---------|---------|--------|-------------|
| Conv2D #1 | 3·3·1·8 + 8 | 72 | 8 | **80** |
| MaxPool #1 | — | 0 | 0 | 0 |
| Conv2D #2 | 3·3·8·16 + 16 | 1,152 | 16 | **1,168** |
| MaxPool #2 | — | 0 | 0 | 0 |
| Flatten | — | 0 | 0 | 0 |
| Dense(32) | 576·32 + 32 | 18,432 | 32 | **18,464** |
| Dropout | — | 0 | 0 | 0 |
| Dense(1) | 32·1 + 1 | 32 | 1 | **33** |

- Manual **total trainable** = 80 + 1,168 + 18,464 + 33 = **19,745**
- Manual **total non-trainable** = **0**
- Manual **grand total** = **19,745**

**Manual (19,745) == TensorFlow (19,745). Exact agreement.** The count is
dominated by the `Dense(32)` layer: 18,464 / 19,745 = **93.5%** of all
parameters, driven entirely by its 576-wide input.

---

## Step 4 — Explanation of the Difference (19,745 vs ~9.5K)

The entire discrepancy lives in the **flatten width feeding Dense(32)**, which
is a direct consequence of **`padding='same'`** on both conv blocks.

- With `padding='same'` (the FROZEN and implemented choice), spatial size is
  preserved through conv and only halved by pooling: 24 → 24 →(pool) 12 → 12
  →(pool) 6. Flatten = **6·6·16 = 576** → Dense = 576·32 + 32 = **18,464** →
  total **19,745**. ✅ This matches the spec's architecture table (line 100).

- With `padding='valid'` (Keras default; **not** the frozen choice), each 3×3
  conv trims a 1-pixel border: 24 →(conv) 22 →(pool) 11 →(conv) 9 →(pool) 4.
  Flatten = **4·4·16 = 256** → Dense = 256·32 + 32 = **8,224** → total
  **9,505**. This was reproduced live and equals the documented "~9.5K".

**Measured conclusion:** the "~9.5K" figure is the parameter count of a
`padding='valid'` model. The implemented (and frozen-table) model uses
`padding='same'`, giving a 576-wide bottleneck and 19,745 params. The estimate
was computed against the wrong (default) padding; the implementation is faithful
to the frozen table.

Ruled out, with evidence:

- **Additional filters?** No — measured Conv shapes are exactly 8 then 16
  (`(3,3,1,8)`, `(3,3,8,16)`).
- **Larger Dense layer?** No — Dense is exactly 32 units (`(576,32)`); the width
  comes from the *input* (576), not extra units.
- **Different input size?** No — `input_shape=(None,24,24,1)`, 24×24×1 grayscale.
- **Duplicated layer?** No — exactly 2 Conv + 2 Pool + Flatten + 2 Dense +
  1 Dropout, matching the table.
- **BatchNorm?** No — zero non-trainable params; no BN layer present.
- **Bias handling?** Standard — every Conv/Dense has `use_bias=True` and biases
  are already included in the manual math.
- **Implementation bug?** No — implementation matches the frozen architecture
  table line-for-line; the number that is wrong is a prose *estimate*, not code.

---

## Step 5 — Frozen-Decision Verification

Each frozen decision confirmed against measured model state:

| Frozen decision | Measured evidence | Status |
|-----------------|-------------------|--------|
| 24×24 grayscale input | `input_shape=(None,24,24,1)` | ✅ Intact |
| Exactly two convolution blocks | conv2d(8) + conv2d_1(16), 2 pools | ✅ Intact |
| ReLU activations (convs + dense) | conv2d/conv2d_1/dense activation=relu | ✅ Intact |
| No BatchNorm | 0 non-trainable params; no BN layer | ✅ Intact |
| Dropout = 0.3 | dropout layer rate=0.3 | ✅ Intact |
| Dense(32) | dense weights `(576,32)` → 32 units | ✅ Intact |
| Dense(1) | dense_1 weights `(32,1)` → 1 unit | ✅ Intact |
| Sigmoid output | dense_1 activation=sigmoid | ✅ Intact |
| `padding='same'` (both convs) | conv output shapes preserved 24→24, 12→12 | ✅ Intact |

**All frozen architectural decisions remain intact.**

---

## Step 6 — Root Cause (choose exactly one)

**B. The implementation is correct; the "~9.5K parameters" written in the
specification was an approximate estimate that should be corrected.**

Evidence:
1. The implemented model matches the frozen architecture **table** line-for-line
   (Step 2), including the table's own stated flatten width of **576**
   ("6·6·16 = 576", spec line 100).
2. Manual and TensorFlow parameter counts agree exactly at **19,745** (Step 3).
3. The "~9.5K" is reproducibly the `padding='valid'` count (**9,505**), whereas
   both the spec table and the code mandate `padding='same'` (Step 4).
4. The spec itself flags the figure as a *"target; exact count to be logged from
   the built model in EXP-002"* (line 118) — it defers to the measured value.
5. All three canonical sources agree with the implementation: `tools/train_cnn.py`
   (lines 81–88, `padding='same'`), `tools/train_exp002_microeyenet.py`
   (lines 76–83, `padding='same'`), and the spec architecture table.

No implementation bug was found. No code change is warranted.

---

## Step 7 — Recommendation (documentation update only — NOT applied)

Two prose locations carry the stale "~9.5K" estimate. They should be corrected
to the measured **19,745** (with a short note that the earlier figure assumed
`padding='valid'`). **No code and no frozen decision changes.** These are
documentation edits only and are *reported, not made*, per the audit constraint:

1. `CNN_IMPLEMENTATION_SPECIFICATION.md`
   - **Line 118** — "**Parameter budget:** ~9.5K params (target; …)" →
     record the measured **19,745** params (dominated by the 576→32 dense
     bottleneck under `padding='same'`); optionally note that ~9.5K corresponds
     to a `padding='valid'` variant, which is **not** the frozen design.
   - **Line 77** — the "~9.5K params" aside in the model-selection rationale →
     update to 19,745 for consistency.
   - **Line 334** — "the ~9.5K-param INT8 model has a small footprint" →
     update the parameter figure (footprint claim itself remains NOT MEASURED).

2. `src/cnn_validator.py`
   - **Line 26** (docstring) — "Total: ~9.5K parameters" → "Total: 19,745
     parameters (measured, EXP-002)". Docstring/comment only; no logic touched.

> These are optional consistency edits. Because the audit rule forbids code
> changes absent a confirmed bug — and none exists — they are left for explicit
> approval and are **not** applied here.

---

## Readiness for EXP-003

EXP-003 (Float16/INT8 quantization + TFLite export) depends on the architecture
being correct and stable, not on the prose parameter figure. The architecture is
verified faithful to the frozen table, the trained weights and metrics artifacts
are intact, and `models/eye_state_model.tflite` (26,488 bytes) is untouched.
**Nothing blocks EXP-003.** The documentation fixes above are non-blocking
housekeeping.

---

## Final Verdict

**IMPLEMENTATION VERIFIED — READY FOR EXP-003**

The 19,745-parameter count is correct and matches both the frozen architecture
table and an independent manual calculation. The "~9.5K" in the specification
prose is an inaccurate estimate (a `padding='valid'` count) and should be
corrected in documentation only. No architecture, methodology, or code change is
required or made.
