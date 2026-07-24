#!/usr/bin/env python3
"""
MicroEyeNet Training Pipeline
================================
Trains a tiny CNN for binary eye-state classification (open vs. closed)
and exports it to TFLite for edge deployment.

Architecture (MicroEyeNet):
    Input: 24×24×1 grayscale
    Conv2D(8, 3×3, ReLU) → MaxPool(2×2)
    Conv2D(16, 3×3, ReLU) → MaxPool(2×2)
    Flatten → Dense(32, ReLU) → Dropout(0.3) → Dense(1, Sigmoid)
    Total: ~9.5K parameters

Dataset Structure:
    data/eyes/open/     → eye-open images (24×24 grayscale PNGs)
    data/eyes/closed/   → eye-closed images (24×24 grayscale PNGs)

Output:
    models/eye_state_model.tflite    → Quantized TFLite model
    models/eye_state_model.keras     → Full Keras model (for debugging)

Usage:
    python3 tools/train_eye_cnn.py

Requirements:
    pip install tensorflow  (or tensorflow-cpu for lightweight)
"""

import os
import sys
import numpy as np

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════

DATA_DIR = "data/eyes"
MODEL_DIR = "models"
INPUT_SIZE = 24
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-3
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15


def load_dataset():
    """
    Load eye-crop images from the data directory.

    Returns:
        X: numpy array (N, 24, 24, 1), float32, normalized [0, 1].
        y: numpy array (N,), int — 0=open, 1=closed.
    """
    import cv2

    images = []
    labels = []

    for label_name, label_val in [("open", 0), ("closed", 1)]:
        class_dir = os.path.join(DATA_DIR, label_name)
        if not os.path.isdir(class_dir):
            print(f"[Warning] Directory not found: {class_dir}")
            continue

        files = [f for f in os.listdir(class_dir) if f.endswith(".png")]
        print(f"  {label_name}: {len(files)} images")

        for fname in files:
            fpath = os.path.join(class_dir, fname)
            img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            # Resize if needed
            if img.shape != (INPUT_SIZE, INPUT_SIZE):
                img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE),
                                 interpolation=cv2.INTER_AREA)

            images.append(img)
            labels.append(label_val)

    if len(images) == 0:
        print("[Error] No images found. Run collect_eye_data.py first.")
        sys.exit(1)

    X = np.array(images, dtype=np.float32) / 255.0
    X = X.reshape(-1, INPUT_SIZE, INPUT_SIZE, 1)
    y = np.array(labels, dtype=np.int32)

    return X, y


def build_model():
    """
    Build the MicroEyeNet Keras model.

    Architecture designed for <10K parameters:
        Conv2D(8) → MaxPool → Conv2D(16) → MaxPool → Dense(32) → Dense(1)
    """
    import tensorflow as tf

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(INPUT_SIZE, INPUT_SIZE, 1)),

        # Block 1: 24×24×1 → 22×22×8 → 11×11×8
        tf.keras.layers.Conv2D(8, (3, 3), activation='relu',
                               kernel_initializer='he_uniform',
                               name='conv1'),
        tf.keras.layers.MaxPooling2D((2, 2), name='pool1'),

        # Block 2: 11×11×8 → 9×9×16 → 4×4×16
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu',
                               kernel_initializer='he_uniform',
                               name='conv2'),
        tf.keras.layers.MaxPooling2D((2, 2), name='pool2'),

        # Classifier head: 256 → 32 → 1
        tf.keras.layers.Flatten(name='flatten'),
        tf.keras.layers.Dense(32, activation='relu',
                              kernel_initializer='he_uniform',
                              name='fc1'),
        tf.keras.layers.Dropout(0.3, name='dropout'),
        tf.keras.layers.Dense(1, activation='sigmoid', name='output'),
    ], name='MicroEyeNet')

    return model


def augment_dataset(X, y):
    """
    Apply offline augmentations to increase dataset diversity.

    Augmentations (all suitable for grayscale eye crops):
        - Horizontal flip
        - Brightness jitter (±30%)
        - Small rotation (±10°)
        - Gaussian noise
    """
    import cv2

    augmented_X = list(X)
    augmented_y = list(y)

    for img, label in zip(X, y):
        img_2d = img.reshape(INPUT_SIZE, INPUT_SIZE)

        # 1. Horizontal flip
        flipped = cv2.flip(img_2d, 1)
        augmented_X.append(flipped.reshape(INPUT_SIZE, INPUT_SIZE, 1))
        augmented_y.append(label)

        # 2. Brightness jitter
        factor = np.random.uniform(0.7, 1.3)
        bright = np.clip(img_2d * factor, 0.0, 1.0)
        augmented_X.append(bright.reshape(INPUT_SIZE, INPUT_SIZE, 1).astype(np.float32))
        augmented_y.append(label)

        # 3. Gaussian noise
        noise = np.random.normal(0, 0.05, img_2d.shape).astype(np.float32)
        noisy = np.clip(img_2d + noise, 0.0, 1.0)
        augmented_X.append(noisy.reshape(INPUT_SIZE, INPUT_SIZE, 1))
        augmented_y.append(label)

    return np.array(augmented_X, dtype=np.float32), np.array(augmented_y, dtype=np.int32)


def export_tflite(model, output_path):
    """
    Convert the Keras model to a quantized TFLite model.

    Uses dynamic range quantization for ~4x size reduction
    while maintaining float32 inference accuracy.
    """
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Dynamic range quantization (weights quantized, activations float)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    size_kb = len(tflite_model) / 1024
    print(f"  TFLite model saved: {output_path} ({size_kb:.1f} KB)")


def main():
    print("=" * 60)
    print("  MICROEYENET TRAINING PIPELINE")
    print("=" * 60)

    # ─── Step 1: Load data ──────────────────────────────────────
    print("\n[Step 1] Loading dataset...")
    X, y = load_dataset()
    print(f"  Total: {len(X)} images | Open: {np.sum(y == 0)} | Closed: {np.sum(y == 1)}")

    # ─── Step 2: Augment ────────────────────────────────────────
    print("\n[Step 2] Augmenting dataset...")
    X_aug, y_aug = augment_dataset(X, y)
    print(f"  After augmentation: {len(X_aug)} images")

    # ─── Step 3: Shuffle and split ──────────────────────────────
    print("\n[Step 3] Splitting dataset...")
    indices = np.random.permutation(len(X_aug))
    X_aug = X_aug[indices]
    y_aug = y_aug[indices]

    n = len(X_aug)
    n_test = int(n * TEST_SPLIT)
    n_val = int(n * VALIDATION_SPLIT)
    n_train = n - n_test - n_val

    X_train, y_train = X_aug[:n_train], y_aug[:n_train]
    X_val, y_val = X_aug[n_train:n_train + n_val], y_aug[n_train:n_train + n_val]
    X_test, y_test = X_aug[n_train + n_val:], y_aug[n_train + n_val:]

    print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # ─── Step 4: Build model ────────────────────────────────────
    print("\n[Step 4] Building MicroEyeNet...")

    # Import TF here so the script fails fast if not installed
    try:
        import tensorflow as tf
    except ImportError:
        print("[Error] TensorFlow not installed.")
        print("  pip install tensorflow  (or tensorflow-cpu)")
        sys.exit(1)

    model = build_model()
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )

    # ─── Step 5: Train ──────────────────────────────────────────
    print("\n[Step 5] Training...")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=4,
            min_lr=1e-5,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    # ─── Step 6: Evaluate ───────────────────────────────────────
    print("\n[Step 6] Evaluating on test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}")

    # Confusion matrix
    y_pred_probs = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_pred_probs >= 0.5).astype(int)

    tp = np.sum((y_pred == 1) & (y_test == 1))
    tn = np.sum((y_pred == 0) & (y_test == 0))
    fp = np.sum((y_pred == 1) & (y_test == 0))
    fn = np.sum((y_pred == 0) & (y_test == 1))

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    print(f"\n  Confusion Matrix:")
    print(f"    TP={tp}  FP={fp}")
    print(f"    FN={fn}  TN={tn}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")

    # ─── Step 7: Export ─────────────────────────────────────────
    print("\n[Step 7] Exporting models...")
    os.makedirs(MODEL_DIR, exist_ok=True)

    keras_path = os.path.join(MODEL_DIR, "eye_state_model.keras")
    model.save(keras_path)
    print(f"  Keras model saved: {keras_path}")

    tflite_path = os.path.join(MODEL_DIR, "eye_state_model.tflite")
    export_tflite(model, tflite_path)

    # Parameter count
    total_params = model.count_params()
    print(f"\n  Total parameters: {total_params:,}")
    print(f"  Architecture: MicroEyeNet ({INPUT_SIZE}×{INPUT_SIZE}×1)")

    print(f"\n{'=' * 60}")
    print(f"  Training complete! Model ready for deployment.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
