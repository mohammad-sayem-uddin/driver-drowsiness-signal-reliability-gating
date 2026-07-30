"""
EXP-002 — MicroEyeNet Baseline Training Runner (FROZEN spec executor)
=====================================================================
Implements the frozen CNN training recipe EXACTLY as specified in
CNN_IMPLEMENTATION_SPECIFICATION.md (Part 4 augmentation, Part 5 training,
Part 6 validation) and IMPLEMENTATION_SPECIFICATION_FROZEN.md.

This is an ENGINEERING EXECUTOR, not a redesign:
  * Architecture is the frozen MicroEyeNet (reused from tools/train_cnn.py).
  * Data is loaded via the subject-disjoint manifest accessor
    (MRLEyeDataLoader.get_subject_disjoint_files, repointed in EXP-002 Phase 2).
  * Every hyperparameter below is copied verbatim from the frozen Part 5 table.

It DOES NOT export a .tflite (that is EXP-003 quantization) and therefore never
touches the existing models/eye_state_model.tflite asset (integrity I4 +
directive "do not overwrite previous artifacts"). EXP-002 produces Keras
checkpoints and measured metric artifacts only.

Run:  python3 tools/train_exp002_microeyenet.py --execute
Dry:  python3 tools/train_exp002_microeyenet.py            (builds only)
"""

import os
import sys
import json
import time
import random
import argparse

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.train_cnn import load_mrl_dataset_tensors  # single-sourced data path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXP_ID = "EXP-002"
EXP_TAG = "EXP-002_microeyenet_baseline"

# ── FROZEN hyperparameters (CNN_IMPLEMENTATION_SPECIFICATION.md Part 5) ────────
SEED = 42
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
MAX_EPOCHS = 30
CLIPNORM = 1.0
ES_PATIENCE = 5          # EarlyStopping on val_loss, restore best weights
RLROP_FACTOR = 0.5       # ReduceLROnPlateau
RLROP_PATIENCE = 3
DROPOUT = 0.3

CKPT_DIR = os.path.join(ROOT, "checkpoints")
CKPT_TMPL = os.path.join(CKPT_DIR, "microeyenet_epoch{epoch:02d}_valloss{val_loss:.4f}.keras")
TB_DIR = os.path.join(ROOT, "tensorboard", EXP_TAG)
CSV_LOG = os.path.join(ROOT, "logs", "EXP-002_training_log.csv")
EXP_DIR = os.path.join(ROOT, "experiments", EXP_TAG)


# ── Reproducibility ───────────────────────────────────────────────────────────
def set_all_seeds(seed: int = SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    import tensorflow as tf
    tf.random.set_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


# ── Frozen model (identical layers to tools/train_cnn.py) ─────────────────────
def build_microeyenet():
    import tensorflow as tf
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(24, 24, 1)),
        tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(DROPOUT),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])
    opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=CLIPNORM)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model


# ── Frozen augmentation (Part 4; training only, in this order) ─────────────────
def make_augmenter():
    import tensorflow as tf

    def augment(img, label):
        # 1. Horizontal flip p=0.5
        img = tf.cond(tf.random.uniform([]) < 0.5,
                      lambda: tf.image.flip_left_right(img), lambda: img)
        # 2. Brightness jitter ±20% (multiplicative) p=0.5
        img = tf.cond(tf.random.uniform([]) < 0.5,
                      lambda: img * tf.random.uniform([], 0.8, 1.2), lambda: img)
        # 3. Contrast jitter ±15% p=0.3
        img = tf.cond(tf.random.uniform([]) < 0.3,
                      lambda: tf.image.adjust_contrast(img, tf.random.uniform([], 0.85, 1.15)),
                      lambda: img)
        # 4. Small rotation ±8° p=0.3 (approx via 90°-multiple-free affine using tfa-free method:
        #    implement small rotation through tf.image is unavailable; use manual rotate.)
        img = tf.cond(tf.random.uniform([]) < 0.3,
                      lambda: _rotate_small(img), lambda: img)
        # 5. Additive Gaussian noise σ≤0.02 p=0.2
        img = tf.cond(tf.random.uniform([]) < 0.2,
                      lambda: img + tf.random.normal(tf.shape(img), 0.0, 0.02), lambda: img)
        img = tf.clip_by_value(img, 0.0, 1.0)  # keep tensor contract [0,1]
        return img, label

    def _rotate_small(img):
        # Rotate by a random angle in [-8°, +8°] using a rotation matrix + resampling.
        angle = tf.random.uniform([], -8.0, 8.0) * np.pi / 180.0
        return _rotate(img, angle)

    return augment


def _rotate(img, angle):
    """Bilinear rotation about center, zero-fill, for a single (H,W,1) image."""
    import tensorflow as tf
    h = tf.shape(img)[0]
    w = tf.shape(img)[1]
    hf = tf.cast(h, tf.float32)
    wf = tf.cast(w, tf.float32)
    cx = (wf - 1.0) / 2.0
    cy = (hf - 1.0) / 2.0
    ys, xs = tf.meshgrid(tf.range(hf), tf.range(wf), indexing='ij')
    xs = xs - cx
    ys = ys - cy
    cos_a = tf.cos(angle)
    sin_a = tf.sin(angle)
    # inverse map (sample source for each dest pixel)
    src_x = cos_a * xs + sin_a * ys + cx
    src_y = -sin_a * xs + cos_a * ys + cy
    x0 = tf.floor(src_x)
    y0 = tf.floor(src_y)
    x1 = x0 + 1.0
    y1 = y0 + 1.0
    wx = src_x - x0
    wy = src_y - y0

    def gather(px, py):
        pxi = tf.cast(tf.clip_by_value(px, 0.0, wf - 1.0), tf.int32)
        pyi = tf.cast(tf.clip_by_value(py, 0.0, hf - 1.0), tf.int32)
        valid = tf.cast((px >= 0) & (px <= wf - 1) & (py >= 0) & (py <= hf - 1), tf.float32)
        vals = tf.gather_nd(img[..., 0], tf.stack([pyi, pxi], axis=-1))
        return vals * valid

    v00 = gather(x0, y0)
    v01 = gather(x1, y0)
    v10 = gather(x0, y1)
    v11 = gather(x1, y1)
    top = v00 * (1 - wx) + v01 * wx
    bot = v10 * (1 - wx) + v11 * wx
    out = top * (1 - wy) + bot * wy
    return tf.expand_dims(out, -1)


# ── numpy metric helpers (Part 6: trapezoidal AUC, no sklearn dependency) ──────
def _trapz_auc(x, y):
    order = np.argsort(x)
    return float(np.trapz(np.array(y)[order], np.array(x)[order]))


def roc_curve_np(y_true, y_score):
    thr = np.unique(np.concatenate(([0.0, 1.0], np.sort(y_score))))
    thr = np.sort(thr)[::-1]
    P = float(np.sum(y_true == 1))
    N = float(np.sum(y_true == 0))
    tpr, fpr = [], []
    for t in thr:
        pred = (y_score >= t).astype(int)
        tp = float(np.sum((pred == 1) & (y_true == 1)))
        fp = float(np.sum((pred == 1) & (y_true == 0)))
        tpr.append(tp / P if P > 0 else 0.0)
        fpr.append(fp / N if N > 0 else 0.0)
    return np.array(fpr), np.array(tpr), thr


def pr_curve_np(y_true, y_score):
    thr = np.unique(np.concatenate(([0.0, 1.0], np.sort(y_score))))
    thr = np.sort(thr)[::-1]
    P = float(np.sum(y_true == 1))
    prec, rec = [], []
    for t in thr:
        pred = (y_score >= t).astype(int)
        tp = float(np.sum((pred == 1) & (y_true == 1)))
        fp = float(np.sum((pred == 1) & (y_true == 0)))
        prec.append(tp / (tp + fp) if (tp + fp) > 0 else 1.0)
        rec.append(tp / P if P > 0 else 0.0)
    return np.array(prec), np.array(rec), thr


def classification_metrics(y_true, y_score, threshold=0.5):
    y_true = np.asarray(y_true).astype(int).ravel()
    y_score = np.asarray(y_score).astype(float).ravel()
    pred = (y_score >= threshold).astype(int)
    tp = int(np.sum((pred == 1) & (y_true == 1)))
    tn = int(np.sum((pred == 0) & (y_true == 0)))
    fp = int(np.sum((pred == 1) & (y_true == 0)))
    fn = int(np.sum((pred == 0) & (y_true == 1)))
    acc = (tp + tn) / max(1, (tp + tn + fp + fn))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    bal_acc = 0.5 * (recall + specificity)
    fpr_c, tpr_c, _ = roc_curve_np(y_true, y_score)
    roc_auc = _trapz_auc(fpr_c, tpr_c)
    prec_c, rec_c, _ = pr_curve_np(y_true, y_score)
    pr_auc = _trapz_auc(rec_c, prec_c)
    brier = float(np.mean((y_score - y_true) ** 2))
    return {
        "threshold": threshold,
        "accuracy": acc, "balanced_accuracy": bal_acc,
        "precision": precision, "recall": recall, "specificity": specificity,
        "f1": f1, "roc_auc": roc_auc, "pr_auc": pr_auc, "brier": brier,
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


def best_f1_threshold(y_true, y_score):
    y_true = np.asarray(y_true).astype(int).ravel()
    y_score = np.asarray(y_score).astype(float).ravel()
    cand = np.unique(y_score)
    best_t, best_f1 = 0.5, -1.0
    for t in cand:
        m = classification_metrics(y_true, y_score, threshold=float(t))
        if m["f1"] > best_f1:
            best_f1, best_t = m["f1"], float(t)
    return best_t, best_f1


# ── Plotting (matplotlib; measured curves only) ────────────────────────────────
def save_plots(history, val_probs, val_y, test_probs, test_y, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Learning curves
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(history["loss"], label="train")
    ax[0].plot(history["val_loss"], label="val")
    ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(history["accuracy"], label="train")
    ax[1].plot(history["val_accuracy"], label="val")
    ax[1].set_title("Accuracy"); ax[1].set_xlabel("epoch"); ax[1].legend()
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "learning_curves.png"), dpi=130); plt.close(fig)

    # Confusion matrix (test @0.5)
    m = classification_metrics(test_y, test_probs, 0.5)["confusion_matrix"]
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["pred open(0)", "pred closed(1)"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["open(0)", "closed(1)"])
    ax.set_title("Test confusion matrix @0.5")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "confusion_matrix_test.png"), dpi=130); plt.close(fig)

    # ROC (test)
    fpr, tpr, _ = roc_curve_np(np.asarray(test_y).astype(int), np.asarray(test_probs))
    auc = _trapz_auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(fpr, tpr, label=f"AUC={auc:.4f}")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("Test ROC"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "roc_curve_test.png"), dpi=130); plt.close(fig)

    # PR (test)
    prec, rec, _ = pr_curve_np(np.asarray(test_y).astype(int), np.asarray(test_probs))
    pr_auc = _trapz_auc(rec, prec)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(rec, prec, label=f"PR-AUC={pr_auc:.4f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Test PR curve"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "pr_curve_test.png"), dpi=130); plt.close(fig)

    # Reliability diagram (val)
    vy = np.asarray(val_y).astype(int).ravel()
    vp = np.asarray(val_probs).astype(float).ravel()
    bins = np.linspace(0, 1, 11)
    idx = np.digitize(vp, bins) - 1
    xs, ys = [], []
    for b in range(10):
        mask = idx == b
        if np.sum(mask) > 0:
            xs.append(np.mean(vp[mask])); ys.append(np.mean(vy[mask]))
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.plot(xs, ys, "o-")
    ax.set_xlabel("mean predicted P(closed)"); ax.set_ylabel("empirical fraction closed")
    ax.set_title("Val reliability diagram")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "reliability_diagram_val.png"), dpi=130); plt.close(fig)


def main(execute: bool):
    import tensorflow as tf
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(TB_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(CSV_LOG), exist_ok=True)
    os.makedirs(EXP_DIR, exist_ok=True)

    set_all_seeds(SEED)
    model = build_microeyenet()

    # Record model summary + param count
    summary_lines = []
    model.summary(print_fn=lambda s: summary_lines.append(s))
    param_count = int(model.count_params())
    with open(os.path.join(EXP_DIR, "model_summary.txt"), "w") as f:
        f.write("\n".join(summary_lines))
        f.write(f"\n\nTotal parameters: {param_count}\n")
    print(f"[{EXP_ID}] MicroEyeNet built. Params = {param_count}")

    if not execute:
        print(f"[{EXP_ID}] Dry build only (pass --execute to train).")
        return

    # Load data via the subject-disjoint accessor (Phase 2 repoint)
    t0 = time.time()
    X_train, y_train = load_mrl_dataset_tensors(split="train")
    X_val, y_val = load_mrl_dataset_tensors(split="val")
    X_test, y_test = load_mrl_dataset_tensors(split="test")
    load_secs = time.time() - t0
    print(f"[{EXP_ID}] Loaded train={X_train.shape} val={X_val.shape} test={X_test.shape} "
          f"in {load_secs:.1f}s")

    # Frozen policy: report class counts; class imbalance in TRAIN is mild
    # (55/45) → no class weights for the baseline (Part 3 "if material").
    train_counts = {int(c): int(np.sum(y_train == c)) for c in (0, 1)}

    augment = make_augmenter()
    train_ds = (tf.data.Dataset.from_tensor_slices((X_train, y_train))
                .shuffle(len(X_train), seed=SEED, reshuffle_each_iteration=True)
                .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
                .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))
    val_ds = (tf.data.Dataset.from_tensor_slices((X_val, y_val))
              .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=ES_PATIENCE,
                                         restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=RLROP_FACTOR,
                                             patience=RLROP_PATIENCE),
        tf.keras.callbacks.ModelCheckpoint(CKPT_TMPL, monitor="val_loss",
                                           save_best_only=True),
        tf.keras.callbacks.TensorBoard(log_dir=TB_DIR),
        tf.keras.callbacks.CSVLogger(CSV_LOG),
    ]

    t1 = time.time()
    hist = model.fit(train_ds, validation_data=val_ds, epochs=MAX_EPOCHS,
                     callbacks=callbacks, verbose=2)
    train_secs = time.time() - t1

    history = {k: [float(x) for x in v] for k, v in hist.history.items()}
    best_epoch = int(np.argmin(history["val_loss"])) + 1
    stopped_epoch = len(history["val_loss"])
    early_stopped = stopped_epoch < MAX_EPOCHS

    # Evaluate once on val + test (Part 6 test discipline: test touched once)
    val_probs = model.predict(X_val, batch_size=256, verbose=0).ravel()
    test_probs = model.predict(X_test, batch_size=256, verbose=0).ravel()

    val_metrics = classification_metrics(y_val, val_probs, 0.5)
    test_metrics = classification_metrics(y_test, test_probs, 0.5)
    bt_val, bf1_val = best_f1_threshold(y_val, val_probs)
    test_at_valbest = classification_metrics(y_test, test_probs, bt_val)

    # Persist artifacts
    save_plots(history, val_probs, y_val, test_probs, y_test, EXP_DIR)

    training_config = {
        "exp_id": EXP_ID, "seed": SEED, "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE, "max_epochs": MAX_EPOCHS, "clipnorm": CLIPNORM,
        "optimizer": "adam", "loss": "binary_crossentropy", "dropout": DROPOUT,
        "lr_scheduler": {"type": "ReduceLROnPlateau", "factor": RLROP_FACTOR,
                         "patience": RLROP_PATIENCE, "monitor": "val_loss"},
        "early_stopping": {"monitor": "val_loss", "patience": ES_PATIENCE,
                           "restore_best_weights": True},
        "mixed_precision": False, "weight_decay": None,
        "augmentation": "Part4: hflip0.5, bright±20%0.5, contrast±15%0.3, rot±8°0.3, noiseσ0.02 0.2",
        "class_weights": None, "tf_version": tf.__version__,
    }

    result = {
        "exp_id": EXP_ID,
        "param_count": param_count,
        "data_load_seconds": load_secs,
        "training_seconds": train_secs,
        "epochs_run": stopped_epoch, "best_epoch": best_epoch,
        "early_stopped": early_stopped,
        "train_class_counts": train_counts,
        "val_class_counts": {int(c): int(np.sum(y_val == c)) for c in (0, 1)},
        "test_class_counts": {int(c): int(np.sum(y_test == c)) for c in (0, 1)},
        "history": history,
        "val_metrics_at_0.5": val_metrics,
        "test_metrics_at_0.5": test_metrics,
        "val_best_f1_threshold": bt_val, "val_best_f1": bf1_val,
        "test_metrics_at_val_best_threshold": test_at_valbest,
        "training_config": training_config,
    }
    with open(os.path.join(EXP_DIR, "exp002_metrics.json"), "w") as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(EXP_DIR, "training_config.json"), "w") as f:
        json.dump(training_config, f, indent=2)

    print(f"[{EXP_ID}] DONE. epochs_run={stopped_epoch} best_epoch={best_epoch} "
          f"early_stopped={early_stopped} train_time={train_secs:.1f}s")
    print(f"[{EXP_ID}] VAL@0.5  acc={val_metrics['accuracy']:.4f} "
          f"f1={val_metrics['f1']:.4f} roc_auc={val_metrics['roc_auc']:.4f}")
    print(f"[{EXP_ID}] TEST@0.5 acc={test_metrics['accuracy']:.4f} "
          f"f1={test_metrics['f1']:.4f} roc_auc={test_metrics['roc_auc']:.4f} "
          f"pr_auc={test_metrics['pr_auc']:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true", help="run the training loop")
    args = ap.parse_args()
    main(execute=args.execute)
