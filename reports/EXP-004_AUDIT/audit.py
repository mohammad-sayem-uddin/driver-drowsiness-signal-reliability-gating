#!/usr/bin/env python3
"""
EXP-004 independent scientific audit.
Recomputes every reported metric from the RAW score CSVs (no frozen code reused
for the recomputation, so this is a genuine cross-check), then runs the deep
statistical analyses. Read-only: touches nothing outside reports/EXP-004_AUDIT/.
"""
import csv, json, os, math
import numpy as np

np.random.seed(42)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCOREDIR = os.path.join(ROOT, "experiments/EXP-004_loso/scores")
OUT = os.path.join(ROOT, "reports/EXP-004_AUDIT")
DATA = os.path.join(OUT, "data")
VARIANTS = {
    "V0": "V0_baseline", "V1": "V1_speech_filter", "V2": "V2_reliability_gate",
    "V3": "V3_full", "V4": "V4_full_cnn",
}

def load(v):
    subs, sc, lab = [], [], []
    with open(os.path.join(SCOREDIR, VARIANTS[v] + ".csv")) as fh:
        r = csv.reader(fh); next(r)
        for s, x, l in r:
            subs.append(s); sc.append(float(x)); lab.append(int(l))
    return np.array(subs), np.array(sc, float), np.array(lab, int)

D = {v: load(v) for v in VARIANTS}
subs0, _, lab0 = D["V0"]

# ── ROC / AUC (independent reimplementation) ─────────────────────────────────
def roc_auc(score, label):
    # rank-based AUC (Mann-Whitney), independent of the trapezoid impl
    order = np.argsort(score, kind="mergesort")
    s = score[order]; y = label[order]
    ranks = np.empty(len(s), float)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    n_pos = int((y == 1).sum()); n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0: return float("nan")
    sum_pos = ranks[y == 1].sum()
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

def roc_curve_trapz(score, label):
    thr = np.unique(score)[::-1]
    P = (label == 1).sum(); N = (label == 0).sum()
    order = np.argsort(-score, kind="mergesort")
    ss = score[order]; yy = label[order]
    tp = np.cumsum(yy == 1); fp = np.cumsum(yy == 0)
    # dedup at unique thresholds (last index of each)
    idx = np.searchsorted(-ss, -thr, side="right") - 1
    tpr = np.concatenate([[0.0], tp[idx] / P])
    fpr = np.concatenate([[0.0], fp[idx] / N])
    auc = float(np.trapz(tpr[np.argsort(fpr)], fpr[np.argsort(fpr)]))
    return fpr, tpr, auc

def pr_auc(score, label):
    order = np.argsort(-score, kind="mergesort")
    yy = label[order]
    tp = np.cumsum(yy == 1); fp = np.cumsum(yy == 0)
    P = (label == 1).sum()
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / P
    # prepend recall 0
    rec = np.concatenate([[0.0], rec]); prec = np.concatenate([[1.0], prec])
    return float(np.trapz(prec, rec))

def fpr_at_matched_tpr(score, label, target=0.80):
    fpr, tpr, _ = roc_curve_trapz(score, label)
    k = int(np.argmin(np.abs(tpr - target)))
    return float(fpr[k]), float(tpr[k])

def cm_at_threshold(score, label, thr):
    pred = (score >= thr).astype(int)
    tp = int(((pred == 1) & (label == 1)).sum())
    fp = int(((pred == 1) & (label == 0)).sum())
    tn = int(((pred == 0) & (label == 0)).sum())
    fn = int(((pred == 0) & (label == 1)).sum())
    return tp, fp, tn, fn

def metrics_from_cm(tp, fp, tn, fn):
    n = tp + fp + tn + fn
    acc = (tp + tn) / n
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return acc, prec, rec, spec, f1, fpr

# fixed operating threshold: replicate frozen logic on V0.
# frozen _roc_curve thresholds = unique(scores) descending; op fixes on TPR≈0.80.
# The extended layer then thresholds at the V0 score achieving TPR≈0.80.
sc0, lb0 = D["V0"][1], D["V0"][2]
thr_desc = np.unique(sc0)[::-1]
P0 = (lb0 == 1).sum()
# TPR at each descending threshold (>=)
order = np.argsort(-sc0, kind="mergesort"); yy = lb0[order]; ss = sc0[order]
tp_cum = np.cumsum(yy == 1)
idx = np.searchsorted(-ss, -thr_desc, side="right") - 1
tpr_thr = tp_cum[idx] / P0
kfix = int(np.argmin(np.abs(tpr_thr - 0.80)))
FIXED_THR = float(thr_desc[kfix])
print(f"[op] fixed V0 score threshold = {FIXED_THR:.3e}  (V0 TPR there = {tpr_thr[kfix]:.6f})")

# ── PART 1: recompute overall + per-subject, compare to reported ─────────────
audit = {"fixed_threshold": FIXED_THR, "overall": {}, "per_subject": {}}
overall_rows = []
for v in VARIANTS:
    _, sc, lb = D[v]
    auc = roc_auc(sc, lb); _, _, auc_tr = roc_curve_trapz(sc, lb)
    prauc = pr_auc(sc, lb)
    fmt, mt = fpr_at_matched_tpr(sc, lb)
    tp, fp, tn, fn = cm_at_threshold(sc, lb, FIXED_THR)
    acc, prec, rec, spec, f1, fpr = metrics_from_cm(tp, fp, tn, fn)
    audit["overall"][v] = dict(auc_mannwhitney=auc, auc_trapz=auc_tr, pr_auc=prauc,
        fpr_at_matched_tpr=fmt, matched_tpr=mt, acc=acc, prec=prec, rec=rec,
        spec=spec, f1=f1, fpr=fpr, tp=tp, fp=fp, tn=tn, fn=fn)
    overall_rows.append((v, auc_tr, prauc, acc, prec, rec, spec, f1, fpr, fmt, tp, fp, tn, fn))

for v in VARIANTS:
    audit["per_subject"][v] = {}
    subs, sc, lb = D[v]
    for s in ["001", "002", "005", "006"]:
        m = subs == s
        auc = roc_auc(sc[m], lb[m])
        prauc = pr_auc(sc[m], lb[m])
        tp, fp, tn, fn = cm_at_threshold(sc[m], lb[m], FIXED_THR)
        acc, prec, rec, spec, f1, fpr = metrics_from_cm(tp, fp, tn, fn)
        audit["per_subject"][v][s] = dict(n=int(m.sum()), auc=auc, pr_auc=prauc,
            acc=acc, rec=rec, spec=spec, fpr=fpr, tp=tp, fp=fp, tn=tn, fn=fn)

json.dump(audit, open(os.path.join(DATA, "recomputed_metrics.json"), "w"), indent=1)

# compare to reported per_variant_metrics.csv
rep = {}
with open(os.path.join(ROOT, "experiments/EXP-004_loso/per_variant_metrics.csv")) as fh:
    for row in csv.DictReader(fh):
        rep[row["variant"]] = row
print("\n=== PART 1: OVERALL recompute vs reported (max abs residual per field) ===")
maxres = 0.0
for v in VARIANTS:
    a = audit["overall"][v]; r = rep[v]
    checks = {
        "roc_auc": (a["auc_trapz"], float(r["roc_auc"])),
        "pr_auc": (a["pr_auc"], float(r["pr_auc"])),
        "acc": (a["acc"], float(r["accuracy"])),
        "prec": (a["prec"], float(r["precision"])),
        "rec": (a["rec"], float(r["recall_tpr"])),
        "spec": (a["spec"], float(r["specificity"])),
        "f1": (a["f1"], float(r["f1"])),
        "fpr_at_matched_tpr": (a["fpr_at_matched_tpr"], float(r["fpr_at_matched_tpr"])),
    }
    resid = {k: abs(x - y) for k, (x, y) in checks.items()}
    cm_ok = (a["tp"], a["fp"], a["tn"], a["fn"]) == (int(r["tp"]), int(r["fp"]), int(r["tn"]), int(r["fn"]))
    mr = max(resid.values()); maxres = max(maxres, mr)
    print(f" {v}: max|Δ|={mr:.2e}  CM_exact_match={cm_ok}  auc_MW_vs_trapz|Δ|={abs(a['auc_mannwhitney']-a['auc_trapz']):.2e}")
print(f" OVERALL MAX RESIDUAL (all variants, all fields): {maxres:.3e}")

# compare per-subject
reps = {}
with open(os.path.join(ROOT, "experiments/EXP-004_loso/per_subject_metrics.csv")) as fh:
    for row in csv.DictReader(fh):
        reps[(row["variant"], row["subject"])] = row
print("\n=== PART 1: PER-SUBJECT recompute vs reported ===")
psmax = 0.0; cmfail = 0
for v in VARIANTS:
    for s in ["001", "002", "005", "006"]:
        a = audit["per_subject"][v][s]; r = reps[(v, s)]
        d = max(abs(a["auc"] - float(r["roc_auc"])), abs(a["acc"] - float(r["accuracy"])),
                abs(a["rec"] - float(r["recall_tpr"])), abs(a["spec"] - float(r["specificity"])),
                abs(a["pr_auc"] - float(r["pr_auc"])))
        if (a["tp"], a["fp"], a["tn"], a["fn"]) != (int(r["tp"]), int(r["fp"]), int(r["tn"]), int(r["fn"])):
            cmfail += 1
        psmax = max(psmax, d)
print(f" PER-SUBJECT MAX RESIDUAL: {psmax:.3e}   CM mismatches: {cmfail}/20")

# ── PART 2a: zero-inflation & score distribution ─────────────────────────────
print("\n=== PART 2: zero-inflation & distribution ===")
zi = {}
for v in VARIANTS:
    _, sc, lb = D[v]
    z_all = float((sc == 0).mean())
    z_neg = float((sc[lb == 0] == 0).mean())
    z_pos = float((sc[lb == 1] == 0).mean())
    zi[v] = dict(zero_all=z_all, zero_neg=z_neg, zero_pos=z_pos,
                 mean=float(sc.mean()), median=float(np.median(sc)),
                 p95=float(np.percentile(sc, 95)), max=float(sc.max()),
                 mean_pos=float(sc[lb == 1].mean()), mean_neg=float(sc[lb == 0].mean()))
    print(f" {v}: zero_all={z_all:.4f} zero_neg={z_neg:.4f} zero_pos={z_pos:.4f} "
          f"mean={sc.mean():.4f} p95={np.percentile(sc,95):.4f} max={sc.max():.4f}")
json.dump(zi, open(os.path.join(DATA, "zero_inflation.json"), "w"), indent=1)

# ── PART 2b: DeLong test for correlated AUCs (paired) ────────────────────────
def delong_var_cov(scores_list, label):
    # Fast DeLong (Sun & Xu 2014) for k correlated AUCs on same labels
    pos = label == 1; neg = label == 0
    m = int(pos.sum()); n = int(neg.sum()); k = len(scores_list)
    def midrank(x):
        J = np.argsort(x, kind="mergesort"); Z = x[J]
        N = len(x); T = np.zeros(N)
        i = 0
        while i < N:
            j = i
            while j < N and Z[j] == Z[i]: j += 1
            T[i:j] = 0.5 * (i + j - 1) + 1
            i = j
        out = np.empty(N); out[J] = T
        return out
    tx = np.empty((k, m)); ty = np.empty((k, n)); tz = np.empty((k, m + n))
    aucs = np.empty(k)
    for r in range(k):
        sc = scores_list[r]
        X = sc[pos]; Y = sc[neg]
        tx[r] = midrank(X); ty[r] = midrank(Y); tz[r] = midrank(np.concatenate([X, Y]))
        aucs[r] = (tz[r, :m].sum() - m * (m + 1) / 2.0) / (m * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01); sy = np.cov(v10)
    if k == 1: sx = np.array([[float(sx)]]); sy = np.array([[float(sy)]])
    S = sx / m + sy / n
    return aucs, S

def delong_pvalue(scoreA, scoreB, label):
    aucs, S = delong_var_cov([scoreA, scoreB], label)
    L = np.array([1.0, -1.0])
    var = L @ S @ L
    if var <= 0: return aucs, 1.0, 0.0
    z = (aucs[0] - aucs[1]) / math.sqrt(var)
    from math import erf
    p = 2 * (1 - 0.5 * (1 + erf(abs(z) / math.sqrt(2))))
    return aucs, float(p), float(z)

print("\n=== PART 2: DeLong paired AUC tests (overall) ===")
delong = {}
for a, b in [("V0", "V2"), ("V0", "V1"), ("V0", "V3"), ("V2", "V3"), ("V1", "V3")]:
    aucs, p, z = delong_pvalue(D[a][1], D[b][1], lb0)
    delong[f"{a}_vs_{b}"] = dict(auc_a=float(aucs[0]), auc_b=float(aucs[1]),
                                 delta=float(aucs[0] - aucs[1]), z=z, p=p)
    print(f" {a} vs {b}: AUC {aucs[0]:.5f} vs {aucs[1]:.5f}  Δ={aucs[0]-aucs[1]:+.5f}  z={z:+.3f}  p={p:.3e}")
json.dump(delong, open(os.path.join(DATA, "delong.json"), "w"), indent=1)

# ── PART 2c: stratified bootstrap CI for AUC (per variant) ───────────────────
print("\n=== PART 2: bootstrap 95% CI for ROC-AUC (2000 resamples, subject-stratified) ===")
B = 2000
pos_idx = np.where(lab0 == 1)[0]; neg_idx = np.where(lab0 == 0)[0]
boot = {}
for v in VARIANTS:
    sc = D[v][1]
    aucs = np.empty(B)
    for b in range(B):
        pi = np.random.choice(pos_idx, len(pos_idx), replace=True)
        ni = np.random.choice(neg_idx, len(neg_idx), replace=True)
        ii = np.concatenate([pi, ni])
        aucs[b] = roc_auc(sc[ii], lab0[ii])
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    boot[v] = dict(auc=roc_auc(sc, lab0), lo=float(lo), hi=float(hi), se=float(aucs.std()))
    print(f" {v}: AUC={boot[v]['auc']:.5f}  95%CI=[{lo:.5f}, {hi:.5f}]  SE={aucs.std():.5f}")
json.dump(boot, open(os.path.join(DATA, "bootstrap_auc.json"), "w"), indent=1)

# paired bootstrap for ΔAUC(V0-V2) and (V0-V3)
print("\n=== PART 2: paired bootstrap ΔAUC 95% CI ===")
pb = {}
for a, b in [("V0", "V2"), ("V0", "V3"), ("V0", "V1")]:
    sca, scb = D[a][1], D[b][1]
    d = np.empty(B)
    for i in range(B):
        pi = np.random.choice(pos_idx, len(pos_idx), replace=True)
        ni = np.random.choice(neg_idx, len(neg_idx), replace=True)
        ii = np.concatenate([pi, ni])
        d[i] = roc_auc(sca[ii], lab0[ii]) - roc_auc(scb[ii], lab0[ii])
    lo, hi = np.percentile(d, [2.5, 97.5])
    pb[f"{a}_minus_{b}"] = dict(mean=float(d.mean()), lo=float(lo), hi=float(hi),
                                frac_gt0=float((d > 0).mean()))
    print(f" ΔAUC({a}-{b}): {d.mean():+.5f}  95%CI=[{lo:+.5f},{hi:+.5f}]  P(Δ>0)={(d>0).mean():.3f}")
json.dump(pb, open(os.path.join(DATA, "paired_bootstrap.json"), "w"), indent=1)

# ── PART 2d: McNemar on fixed-threshold predictions (paired) ─────────────────
print("\n=== PART 2: McNemar (fixed-threshold correctness) ===")
def preds(v): return (D[v][1] >= FIXED_THR).astype(int)
correct = {v: (preds(v) == lb0).astype(int) for v in VARIANTS}
mcn = {}
for a, b in [("V0", "V2"), ("V0", "V1"), ("V0", "V3"), ("V2", "V3")]:
    ca, cb = correct[a], correct[b]
    b01 = int(((ca == 1) & (cb == 0)).sum())  # a right, b wrong
    b10 = int(((ca == 0) & (cb == 1)).sum())
    n = b01 + b10
    stat = (abs(b01 - b10) - 1) ** 2 / n if n else 0.0
    from math import erf
    p = math.exp(-stat / 2) if n else 1.0  # chi2 1df survival approx
    # exact-ish via chi2 survival
    p = 1 - (erf(math.sqrt(stat / 2)) if stat > 0 else 0.0)
    mcn[f"{a}_vs_{b}"] = dict(a_right_b_wrong=b01, a_wrong_b_right=b10, chi2=stat, p=p)
    print(f" {a} vs {b}: a✓b✗={b01}  a✗b✓={b10}  χ²={stat:.2f}  p={p:.3e}")
json.dump(mcn, open(os.path.join(DATA, "mcnemar.json"), "w"), indent=1)

print("\nDONE part1-2")
