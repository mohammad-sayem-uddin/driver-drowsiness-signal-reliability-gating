"""
MicroEyeNet Trainer & INT8 TFLite Converter (Real Data Connected)
===================================================================
Trains MicroEyeNet (24x24 single-channel eye ROI model) for binary eye closure classification
on the real MRL Eye Dataset (Data/mrl_eye/) and converts it to INT8 quantized TFLite model format.

Architecture:
    Input: 24x24x1 grayscale ROI
    Conv2D(8, 3x3, ReLU) -> MaxPool(2x2)
    Conv2D(16, 3x3, ReLU) -> MaxPool(2x2)
    Flatten -> Dense(32, ReLU) -> Dropout(0.3) -> Dense(1, Sigmoid)

Outputs:
    models/eye_state_model.tflite   (single INT8-quantized model; this is
    the ONLY artifact produced. A second "micro_eyenet_int8.tflite" was
    removed as a false asset — it was byte-identical to this file, not a
    distinct model, freeze-report precondition 5.)

NOTE (implementation phase, EXP-002): training/val tensors are loaded via
get_subject_disjoint_files(), which reads the leak-free manifests at
Data/mrl_eye/splits_subject_disjoint/{split}.csv (produced by
tools/build_subject_disjoint_splits.py, seed 42, zero subject overlap). This
supersedes the earlier subject-leaky get_partition_files() path so reported CNN
results are not contaminated by subject memorization (precondition 3).
"""

import os
import sys
import numpy as np

# Add project root to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import SystemConfig
from src.data_loaders import MRLEyeDataLoader


def load_mrl_dataset_tensors(split="train", max_samples=None, input_size=24):
    """
    Loads real MRL Eye dataset image tensors for the given split via the
    subject-disjoint manifest (Data/mrl_eye/splits_subject_disjoint/{split}.csv).
    Returns:
        X: np.ndarray of shape (num_samples, 24, 24, 1) normalized float32 in [0, 1]
        y: np.ndarray of shape (num_samples, 1) float32 binary labels (0=awake, 1=sleepy)
    """
    import cv2
    loader = MRLEyeDataLoader()
    sample_files = loader.get_subject_disjoint_files(split)

    if max_samples is not None:
        sample_files = sample_files[:max_samples]

    X_list, y_list = [], []
    for fp, label, _ in sample_files:
        img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        resized = cv2.resize(img, (input_size, input_size), interpolation=cv2.INTER_AREA)
        norm = (resized / 255.0).astype(np.float32)
        X_list.append(np.expand_dims(norm, axis=-1))
        y_list.append(float(label))

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32).reshape(-1, 1)
    return X, y


def train_and_export_tflite(output_path="models/eye_state_model.tflite", execute_training=False):
    """
    Builds MicroEyeNet, connects to Data/mrl_eye loader, and exports quantized TFLite binary.
    If execute_training is False, prepares model structure without executing training loops.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        import tensorflow as tf
        print("[Train CNN] TensorFlow detected. Initializing MicroEyeNet model structure...")

        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(24, 24, 1)),
            tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        print("[Train CNN] MicroEyeNet compiled and ready for Data/mrl_eye/ training.")

        if execute_training:
            X_train, y_train = load_mrl_dataset_tensors(split="train")
            X_val, y_val = load_mrl_dataset_tensors(split="val")
            model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=64, verbose=1)

            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]

            def representative_data_gen():
                for i in range(min(100, len(X_train))):
                    yield [X_train[i:i+1]]

            converter.representative_dataset = representative_data_gen
            tflite_model = converter.convert()

            with open(output_path, 'wb') as f:
                f.write(tflite_model)
            print(f"[Train CNN] TFLite model successfully exported to: {output_path}")

        return True

    except ImportError:
        print("[Train CNN] TensorFlow not detected. Pipeline ready for execution when dependencies are active.")
        return False


if __name__ == "__main__":
    train_and_export_tflite(execute_training=False)
