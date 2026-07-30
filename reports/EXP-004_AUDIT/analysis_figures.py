#!/usr/bin/env python3
"""
EXP-004 audit — extended analysis (Parts 2b/3/4) + publication figures (Part 6).
READ-ONLY on released artifacts: reads experiments/EXP-004_loso/scores/*.csv and
reports/EXP-004_AUDIT/data/*.json; writes only into reports/EXP-004_AUDIT/{data,figures}/.
Does NOT touch frozen code, models, thresholds, or rerun the experiment.
"""
import csv, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(42)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCOREDIR = os.path.join(ROOT, "experiments/EXP-004_loso/scores")
OUT = os.path.join(ROOT, "reports/EXP-004_AUDIT")
DATA = os.path.join(OUT, "data")
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)
VARIANTS = {"V0": "V0_baseline", "V1": "V1_speech_filter", "V2": "V2_reliability_gate",
            "V3": "V3_full", "V4": "V4_full_cnn"}
SUBJECTS = ["001", "002", "005", "006"]
COL = {"V0": "#1f77b4", "V1": "#ff7f0e", "V2": "#2ca02c", "V3": "#d62728", "V4": "#9467bd"}

def load(v):
    subs, sc, lab = [], [], []
    with open(os.path.join(SCOREDIR, VARIANTS[v] + ".csv")) as fh:
        r = csv.reader(fh); next(r)
        for s, x, l in r:
            subs.append(s); sc.append(float(x)); lab.append(int(l))
    return np.array(subs), np.array(sc, float), np.array(lab, int)

D = {v: load(v) for v in VARIANTS}
subs0, sc0, lab0 = D["V0"]
boot = json.load(open(os.path.join(DATA, "bootstrap_auc.json")))
delong = json.load(open(os.path.join(DATA, "delong.json")))
recomp = json.load(open(os.path.join(DATA, "recomputed_metrics.json")))
FIXED = recomp["fixed_threshold"]

def roc_curve(score, label):
    thr = np.unique(score)[::-1]
    P = (label == 1).sum(); N = (label == 0).sum()
    order = np.argsort(-score, kind="mergesort")
    ss = score[order]; yy = label[order]
    tp = np.cumsum(yy == 1); fp = np.cumsum(yy == 0)
    idx = np.searchsorted(-ss, -thr, side="right") - 1
    tpr = np.concatenate([[0.0], tp[idx] / P])
    fpr = np.concatenate([[0.0], fp[idx] / N])
    return fpr, tpr

def auc_mw(score, label):
    order = np.argsort(score, kind="mergesort"); s = score[order]; y = label[order]
    ranks = np.empty(len(s)); i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]: j += 1
        ranks[i:j + 1] = (i + j) / 2.0 + 1.0; i = j + 1
    npos = int((y == 1).sum()); nneg = int((y == 0).sum())
    return (ranks[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)

extra = {}

# ── Part 2: calibration by score-decile (empirical drowsy rate per bin) ───────
def calibration(score, label, nbin=10):
    # rank-based deciles because of heavy zero-inflation
    order = np.argsort(score, kind="mergesort")
    edges = np.linspace(0, len(score), nbin + 1).astype(int)
    xs, ys, ns = [], [], []
    for k in range(nbin):
        idx = order[edges[k]:edges[k + 1]]
        xs.append(float(score[idx].mean()))
        ys.append(float(label[idx].mean()))
        ns.append(int(len(idx)))
    return xs, ys, ns
cal = {}
for v in ["V0", "V2", "V3"]:
    _, sc, lb = D[v]
    xs, ys, ns = calibration(sc, lb)
    cal[v] = dict(mean_score=xs, emp_drowsy_rate=ys, n=ns)
extra["calibration_decile"] = cal

# ── Part 3: subject-006 deep dive ─────────────────────────────────────────────
sub_stats = {}
for v in ["V0", "V2", "V3"]:
    subs, sc, lb = D[v]
    sub_stats[v] = {}
    for s in SUBJECTS + ["_others"]:
        m = (subs == s) if s != "_others" else (subs != "006")
        scm, lbm = sc[m], lb[m]
        pos, neg = scm[lbm == 1], scm[lbm == 0]
        sub_stats[v][s] = dict(
            n=int(m.sum()), n_pos=int((lbm == 1).sum()), n_neg=int((lbm == 0).sum()),
            prevalence=float((lbm == 1).mean()),
            mean_pos=float(pos.mean()), mean_neg=float(neg.mean()),
            median_pos=float(np.median(pos)), median_neg=float(np.median(neg)),
            zero_pos=float((pos == 0).mean()), zero_neg=float((neg == 0).mean()),
            sep=float(pos.mean() - neg.mean()), auc=float(auc_mw(scm, lbm)))
extra["subject_stats"] = sub_stats

# ── Part 4: reliability-gate paired per-frame effect (V0 → V2) ────────────────
_, scV0, _ = D["V0"]; _, scV2, _ = D["V2"]
assert len(scV0) == len(scV2)
delta = scV2 - scV0                      # gate attenuates multiplicatively → <=0 mostly
atten = scV0 - scV2                       # amount removed
nz = scV0 > 0
gate = dict(
    frames_changed=int((scV2 != scV0).sum()),
    frac_changed=float((scV2 != scV0).mean()),
    frac_attenuated_of_nonzero=float((scV2 < scV0)[nz].mean()),
    mean_atten_pos=float(atten[lab0 == 1].mean()),
    mean_atten_neg=float(atten[lab0 == 0].mean()),
    mean_atten_pos_nz=float(atten[(lab0 == 1) & nz].mean()),
    mean_atten_neg_nz=float(atten[(lab0 == 0) & nz].mean()),
    # separation (pos-mean minus neg-mean) before/after
    sep_V0=float(scV0[lab0 == 1].mean() - scV0[lab0 == 0].mean()),
    sep_V2=float(scV2[lab0 == 1].mean() - scV2[lab0 == 0].mean()),
)
extra["gate_effect"] = gate

json.dump(extra, open(os.path.join(DATA, "extended_analysis.json"), "w"), indent=1)
print("=== gate effect ===")
for k, val in gate.items(): print(f"  {k}: {val}")
print("=== subject 006 vs others (V0) ===")
for s in ["006", "_others"]:
    st = sub_stats["V0"][s]
    print(f"  {s}: prev={st['prevalence']:.3f} mean_pos={st['mean_pos']:.4f} "
          f"mean_neg={st['mean_neg']:.4f} sep={st['sep']:+.4f} auc={st['auc']:.4f}")

# ══════════════════════ FIGURES ══════════════════════════════════════════════
plt.rcParams.update({"font.size": 11, "figure.dpi": 140, "savefig.dpi": 200,
                     "axes.grid": True, "grid.alpha": 0.3})

# Fig 1 — ROC overlay with fixed operating point
fig, ax = plt.subplots(figsize=(6.2, 6))
for v in ["V0", "V1", "V2", "V3"]:
    _, sc, lb = D[v]
    fpr, tpr = roc_curve(sc, lb)
    ax.plot(fpr, tpr, color=COL[v], lw=1.8,
            label=f"{v}  AUC={boot[v]['auc']:.3f} [{boot[v]['lo']:.3f},{boot[v]['hi']:.3f}]")
# operating point on V0
fpr0, tpr0 = roc_curve(sc0, lab0)
k = int(np.argmin(np.abs(tpr0 - 0.80)))
ax.scatter([fpr0[k]], [tpr0[k]], color="k", zorder=5, s=45)
ax.annotate(f"matched TPR=0.80\nFPR={fpr0[k]:.3f}", (fpr0[k], tpr0[k]),
            textcoords="offset points", xytext=(10, -28), fontsize=9)
ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("EXP-004 LOSO ROC by ablation variant (V4≡V3, omitted)")
ax.legend(loc="lower right", fontsize=8.5); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig1_roc_overlay.png")); plt.close(fig)

# Fig 2 — AUC forest plot with bootstrap CIs
fig, ax = plt.subplots(figsize=(6.4, 3.6))
vs = ["V0", "V2", "V1", "V3"]
y = np.arange(len(vs))[::-1]
for yi, v in zip(y, vs):
    lo, hi, a = boot[v]["lo"], boot[v]["hi"], boot[v]["auc"]
    ax.plot([lo, hi], [yi, yi], color=COL[v], lw=2.5)
    ax.scatter([a], [yi], color=COL[v], s=55, zorder=5)
    ax.text(hi + 0.001, yi, f"{a:.4f} [{lo:.4f}, {hi:.4f}]", va="center", fontsize=8.5)
ax.axvline(0.5, color="grey", ls=":", lw=1, label="chance")
ax.set_yticks(y); ax.set_yticklabels(vs)
ax.set_xlabel("ROC-AUC (subject-stratified bootstrap, B=2000)")
ax.set_title("Overall AUC with 95% CIs — separated but overlapping bands")
ax.set_xlim(0.49, 0.645)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig2_auc_forest.png")); plt.close(fig)

# Fig 3 — per-subject AUC grouped bars (highlight 006 below chance)
fig, ax = plt.subplots(figsize=(7.2, 4.2))
w = 0.2
for i, v in enumerate(["V0", "V1", "V2", "V3"]):
    vals = [recomp["per_subject"][v][s]["auc"] for s in SUBJECTS]
    ax.bar(np.arange(len(SUBJECTS)) + (i - 1.5) * w, vals, w, color=COL[v], label=v)
ax.axhline(0.5, color="k", ls="--", lw=1)
ax.text(3.3, 0.51, "chance", fontsize=8)
ax.set_xticks(range(len(SUBJECTS))); ax.set_xticklabels([f"subj {s}" for s in SUBJECTS])
ax.set_ylabel("ROC-AUC"); ax.set_title("Per-subject AUC — subject 006 collapses below chance")
ax.legend(ncol=4, fontsize=9); ax.set_ylim(0, 0.8)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig3_per_subject_auc.png")); plt.close(fig)

# Fig 4 — zero-inflation pos vs neg per variant
zi = json.load(open(os.path.join(DATA, "zero_inflation.json")))
fig, ax = plt.subplots(figsize=(7, 4))
vs = ["V0", "V1", "V2", "V3"]
x = np.arange(len(vs))
ax.bar(x - 0.2, [zi[v]["zero_pos"] for v in vs], 0.4, label="drowsy (pos)", color="#d62728")
ax.bar(x + 0.2, [zi[v]["zero_neg"] for v in vs], 0.4, label="not-drowsy (neg)", color="#1f77b4")
ax.set_xticks(x); ax.set_xticklabels(vs)
ax.set_ylabel("fraction of frames scoring exactly 0")
ax.set_title("Zero-inflation by class — negatives are more zero-inflated")
ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig4_zero_inflation.png")); plt.close(fig)

# Fig 5 — subject 006 score distributions vs pooled others (V0), log-y
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
bins = np.linspace(0, 0.47, 40)
for ax, who, title in [(axes[0], "006", "Subject 006 (LOSO fold)"),
                       (axes[1], "_others", "Pooled subjects 001/002/005")]:
    m = (subs0 == "006") if who == "006" else (subs0 != "006")
    ax.hist(sc0[m & (lab0 == 0)], bins=bins, alpha=0.6, label="not-drowsy", color="#1f77b4")
    ax.hist(sc0[m & (lab0 == 1)], bins=bins, alpha=0.6, label="drowsy", color="#d62728")
    st = sub_stats["V0"][who]
    ax.set_yscale("log"); ax.set_xlabel("fatigue_score (V0)")
    ax.set_title(f"{title}\nsep(pos-neg)={st['sep']:+.4f}  AUC={st['auc']:.3f}")
    ax.legend()
axes[0].set_ylabel("frame count (log)")
fig.suptitle("Fig 5 — Score separation is INVERTED for subject 006", y=1.02)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig5_subject006_dist.png"),
                                bbox_inches="tight"); plt.close(fig)

# Fig 6 — reliability gate: class-conditional mean score, V0 vs V2
fig, ax = plt.subplots(figsize=(6.4, 4))
groups = ["not-drowsy", "drowsy"]
v0m = [sc0[lab0 == 0].mean(), sc0[lab0 == 1].mean()]
v2m = [scV2[lab0 == 0].mean(), scV2[lab0 == 1].mean()]
x = np.arange(2)
ax.bar(x - 0.2, v0m, 0.4, label="V0 (no gate)", color="#1f77b4")
ax.bar(x + 0.2, v2m, 0.4, label="V2 (gate on)", color="#2ca02c")
for xi, (a, b) in enumerate(zip(v0m, v2m)):
    ax.text(xi - 0.2, a + 0.001, f"{a:.4f}", ha="center", fontsize=8)
    ax.text(xi + 0.2, b + 0.001, f"{b:.4f}", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(groups)
ax.set_ylabel("mean fatigue_score")
ax.set_title(f"Reliability gate attenuates BOTH classes → separation "
             f"{gate['sep_V0']:.4f}→{gate['sep_V2']:.4f}")
ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig6_gate_effect.png")); plt.close(fig)

# Fig 7 — calibration (rank-decile empirical drowsy rate)
fig, ax = plt.subplots(figsize=(6, 5))
for v in ["V0", "V2", "V3"]:
    c = cal[v]
    ax.plot(c["mean_score"], c["emp_drowsy_rate"], "o-", color=COL[v], label=v, lw=1.6)
ax.axhline(float((lab0 == 1).mean()), color="grey", ls=":", label="base prevalence")
ax.set_xlabel("mean fatigue_score in rank-decile")
ax.set_ylabel("empirical drowsy rate")
ax.set_title("Score reliability — monotone but weak, compressed near 0")
ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig7_calibration.png")); plt.close(fig)

# Fig 8 — confusion matrices at fixed V0 operating threshold
fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
for ax, v in zip(axes, ["V0", "V1", "V2", "V3"]):
    o = recomp["overall"][v]
    cm = np.array([[o["tn"], o["fp"]], [o["fn"], o["tp"]]])
    ax.imshow(cm, cmap="Blues")
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, f"{val}", ha="center", va="center",
                color="white" if val > cm.max() * 0.5 else "black", fontsize=10)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["pred N", "pred D"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["true N", "true D"])
    ax.set_title(f"{v}  acc={o['acc']:.3f}\nTPR={o['rec']:.3f} FPR={o['fpr']:.3f}")
    ax.grid(False)
fig.suptitle("Fig 8 — Confusion matrices at the fixed V0 operating threshold", y=1.05)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig8_confusion.png"),
                                bbox_inches="tight"); plt.close(fig)

print("\nFigures written to", FIG)
print(os.listdir(FIG))
