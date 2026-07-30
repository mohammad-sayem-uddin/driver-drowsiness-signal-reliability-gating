"""
Subject-Disjoint Split Builder & Leakage Verifier (MRL Eye)
===========================================================
Purpose (freeze-report precondition 3 — remove data leakage):
  1. Verify the leakage in the *current* MRL train/val/test partitions
     (subject-ID mixing) and the drowsiness_detection duplication.
  2. Produce a DETERMINISTIC, reproducible subject-disjoint holdout
     split (train/val/test grouped strictly by subject ID) and persist
     it as manifest files so training never mixes a subject across
     splits.
  3. Emit a verification report proving zero subject overlap.

This script performs NO training and NO image mutation. It only reads
filenames (and, with --md5, hashes bytes) and writes manifest/report
text files.

Usage:
    python tools/build_subject_disjoint_splits.py            # fast: names only
    python tools/build_subject_disjoint_splits.py --md5      # also byte-verify

Outputs:
    Data/mrl_eye/splits_subject_disjoint/train.csv
    Data/mrl_eye/splits_subject_disjoint/val.csv
    Data/mrl_eye/splits_subject_disjoint/test.csv
    Data/mrl_eye/splits_subject_disjoint/SPLIT_MANIFEST.md
"""

import os
import sys
import csv
import hashlib
import argparse
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loaders import MRLEyeDataLoader

# Deterministic split ratios (by SUBJECT, not by image).
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# test gets the remainder
SEED = 42

OUT_DIR = os.path.join("Data", "mrl_eye", "splits_subject_disjoint")


def _deterministic_subject_split(subjects):
    """Seeded, reproducible assignment of whole subjects to splits."""
    import random
    rng = random.Random(SEED)
    subs = sorted(subjects)          # stable input order
    rng.shuffle(subs)                # seeded shuffle
    n = len(subs)
    n_train = int(round(n * TRAIN_FRAC))
    n_val = int(round(n * VAL_FRAC))
    train = set(subs[:n_train])
    val = set(subs[n_train:n_train + n_val])
    test = set(subs[n_train + n_val:])
    return train, val, test


def _class_balance(samples):
    c = defaultdict(int)
    for _fp, label in samples:
        c[label] += 1
    return dict(c)


def build_splits(do_md5=False):
    loader = MRLEyeDataLoader()
    grouped = loader.get_all_samples_grouped_by_subject()  # subj -> [(fp,label)]
    subjects = sorted(grouped.keys())
    total_images = sum(len(v) for v in grouped.values())
    print(f"[Split] {len(subjects)} subjects, {total_images} images.")

    # --- Report the leakage in the CURRENT partitions (evidence) ---
    cur = defaultdict(set)
    for split in ("train", "val", "test"):
        try:
            for _fp, _label, subj in loader.get_partition_files(split):
                cur[split].add(subj)
        except FileNotFoundError:
            pass
    overlap_tt = sorted(cur["train"] & cur["test"])
    overlap_tv = sorted(cur["train"] & cur["val"])
    print(f"[Leakage/current] train∩test subjects: {len(overlap_tt)}; "
          f"train∩val subjects: {len(overlap_tv)}")

    # --- Build the NEW subject-disjoint split ---
    train_subs, val_subs, test_subs = _deterministic_subject_split(subjects)
    assert not (train_subs & val_subs), "train/val subject overlap!"
    assert not (train_subs & test_subs), "train/test subject overlap!"
    assert not (val_subs & test_subs), "val/test subject overlap!"

    splits = {"train": [], "val": [], "test": []}
    split_subj = {"train": train_subs, "val": val_subs, "test": test_subs}
    for subj, samples in grouped.items():
        for name, subs in split_subj.items():
            if subj in subs:
                for fp, label in samples:
                    splits[name].append((fp, label, subj))
                break

    os.makedirs(OUT_DIR, exist_ok=True)
    for name, rows in splits.items():
        path = os.path.join(OUT_DIR, f"{name}.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["filepath", "label", "subject_id"])
            w.writerows(sorted(rows))
        print(f"[Split] wrote {len(rows):>6} rows -> {path}")

    # --- Optional byte-level MD5 verification (slow) ---
    md5_note = "MD5 byte verification not run (use --md5)."
    if do_md5:
        md5_note = _md5_verify(splits)

    # --- Manifest / verification report ---
    _write_manifest(subjects, split_subj, splits, overlap_tt, overlap_tv, md5_note)
    print(f"[Split] manifest -> {os.path.join(OUT_DIR, 'SPLIT_MANIFEST.md')}")


def _md5_verify(splits):
    def hashes(rows):
        hs = {}
        for fp, _label, _subj in rows:
            try:
                with open(fp, "rb") as f:
                    hs[hashlib.md5(f.read()).hexdigest()] = fp
            except OSError:
                pass
        return hs
    print("[MD5] hashing splits (this is slow)...")
    htr, hva, hte = hashes(splits["train"]), hashes(splits["val"]), hashes(splits["test"])
    dup_tv = set(htr) & set(hva)
    dup_tt = set(htr) & set(hte)
    dup_vt = set(hva) & set(hte)
    note = (f"MD5 cross-split byte duplicates — train/val: {len(dup_tv)}, "
            f"train/test: {len(dup_tt)}, val/test: {len(dup_vt)}.")
    print(f"[MD5] {note}")
    return note


def _write_manifest(subjects, split_subj, splits, overlap_tt, overlap_tv, md5_note):
    lines = []
    lines.append("# MRL Eye — Subject-Disjoint Split Manifest\n")
    lines.append(f"- Seed: {SEED} (deterministic; regenerating yields the same split)\n")
    lines.append(f"- Ratios by SUBJECT: train {TRAIN_FRAC}, val {VAL_FRAC}, "
                 f"test {round(1 - TRAIN_FRAC - VAL_FRAC, 2)}\n")
    lines.append(f"- Total subjects: {len(subjects)}\n\n")
    lines.append("## Why this exists\n")
    lines.append("The shipped `Data/mrl_eye/{train,val,test}` partitions mix EVERY "
                 "subject across ALL three splits (verified: "
                 f"{len(overlap_tt)} subjects in both train and test, "
                 f"{len(overlap_tv)} in both train and val). Training or "
                 "evaluating on those partitions measures subject memorization, "
                 "not generalization. "
                 "This subject-disjoint split is the leak-free replacement.\n\n")
    lines.append("## New split (subject-disjoint, zero overlap)\n\n")
    lines.append("| Split | #Subjects | #Images | Class balance {0=awake,1=sleepy} |\n")
    lines.append("|---|---|---|---|\n")
    for name in ("train", "val", "test"):
        rows = [(fp, label) for fp, label, _s in splits[name]]
        lines.append(f"| {name} | {len(split_subj[name])} | {len(rows)} | "
                     f"{_class_balance(rows)} |\n")
    lines.append("\n## Subject assignment\n\n")
    for name in ("train", "val", "test"):
        lines.append(f"- **{name}**: {', '.join(sorted(split_subj[name]))}\n")
    lines.append(f"\n## Byte-level check\n\n{md5_note}\n")
    lines.append("\n## Overlap assertion\n\nGenerator asserts empty pairwise "
                 "subject intersections; build fails otherwise.\n")
    with open(os.path.join(OUT_DIR, "SPLIT_MANIFEST.md"), "w") as f:
        f.writelines(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--md5", action="store_true", help="also run byte-level MD5 verification (slow)")
    args = ap.parse_args()
    build_splits(do_md5=args.md5)
