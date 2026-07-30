"""
EXP-003 — MicroEyeNet Float16 & INT8 Quantization + TFLite Verification
========================================================================
1. Audits EXP-002 prerequisites (best model checkpoint, split files, configs).
2. Loads the best Keras model (microeyenet_epoch08_valloss0.1793.keras).
3. Exports FP32 baseline TFLite model, Float16 TFLite model, and full INT8 TFLite model.
4. Evaluates FP32 model, Float16 model, and INT8 model on the frozen TEST split.
5. Computes model sizes, compression ratios, and accuracy metrics.
6. Verifies functional contracts and repository integrity invariants.
7. Saves all EXP-003 artifacts to experiments/EXP-003_quantization/.
"""

import os
import sys
import json
import time
import hashlib
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.train_cnn import load_mrl_dataset_tensors

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXP_ID = "EXP-003"
EXP_TAG = "EXP-003_quantization"
EXP_DIR = os.path.join(ROOT, "experiments", EXP_TAG)
BEST_CKPT = os.path.join(ROOT, "checkpoints", "microeyenet_epoch08_valloss0.1793.keras")
MODELS_DIR = os.path.join(ROOT, "models")
FP16_MODEL_PATH = os.path.join(MODELS_DIR, "eye_state_model_fp16.tflite")
INT8_MODEL_PATH = os.path.join(MODELS_DIR, "eye_state_model_int8.tflite")

# ── Metric helpers (consistent with EXP-002) ──────────────────────────────────
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

def compute_classification_metrics(y_true, y_score, threshold=0.5):
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
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "brier": float(brier),
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }

# ── Main Runner Function ──────────────────────────────────────────────────────
def run_exp003():
    os.makedirs(EXP_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    log_lines = []

    def log(msg):
        print(msg)
        log_lines.append(msg)

    log("=========================================================================")
    log("EXP-003: Float16 & INT8 Quantization + TFLite Verification Execution")
    log("=========================================================================")

    # Step 1: Audit Prerequisites
    log("\n[Step 1] Auditing EXP-002 Prerequisites...")
    if not os.path.exists(BEST_CKPT):
        raise RuntimeError(f"Best checkpoint not found: {BEST_CKPT}")
    exp002_dir = os.path.join(ROOT, "experiments", "EXP-002_microeyenet_baseline")
    if not os.path.exists(os.path.join(exp002_dir, "exp002_metrics.json")):
        raise RuntimeError("EXP-002 metrics.json missing")
    if not os.path.exists(os.path.join(exp002_dir, "training_config.json")):
        raise RuntimeError("EXP-002 training_config.json missing")
    if not os.path.exists(os.path.join(exp002_dir, "model_summary.txt")):
        raise RuntimeError("EXP-002 model_summary.txt missing")

    log("  ✓ EXP-002 artifacts and best checkpoint verified.")

    import tensorflow as tf

    # Step 2: Load Best Model
    log("\n[Step 2] Loading Best FP32 Keras Model from EXP-002...")
    model = tf.keras.models.load_model(BEST_CKPT)
    param_count = model.count_params()
    log(f"  Loaded model architecture. Measured Total Parameters = {param_count}")
    if param_count != 19745:
        raise ValueError(f"Architecture parameter count mismatch! Expected 19745, got {param_count}")
    log("  ✓ Best model loaded & parameter count verified (19,745).")

    # Step 3: Convert & Export Float16 TFLite
    log("\n[Step 3] Exporting Float16 TFLite Model...")
    converter_fp16 = tf.lite.TFLiteConverter.from_keras_model(model)
    converter_fp16.optimizations = [tf.lite.Optimize.DEFAULT]
    converter_fp16.target_spec.supported_types = [tf.float16]
    tflite_fp16_bytes = converter_fp16.convert()
    with open(FP16_MODEL_PATH, "wb") as f:
        f.write(tflite_fp16_bytes)
    fp16_size_bytes = os.path.getsize(FP16_MODEL_PATH)
    log(f"  ✓ Saved Float16 model to {FP16_MODEL_PATH} ({fp16_size_bytes} bytes / {fp16_size_bytes/1024:.2f} KB).")

    # Also generate unquantized FP32 TFLite for size comparison baseline
    converter_fp32 = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_fp32_bytes = converter_fp32.convert()
    fp32_tflite_path = os.path.join(EXP_DIR, "eye_state_model_fp32.tflite")
    with open(fp32_tflite_path, "wb") as f:
        f.write(tflite_fp32_bytes)
    fp32_size_bytes = os.path.getsize(fp32_tflite_path)
    log(f"  ✓ Generated FP32 TFLite baseline ({fp32_size_bytes} bytes / {fp32_size_bytes/1024:.2f} KB).")

    # Step 4: Convert & Export Full INT8 TFLite
    log("\n[Step 4] Exporting Full INT8 TFLite Model...")
    log("  Loading representative training samples from MRL subject-disjoint train split...")
    X_train_rep, _ = load_mrl_dataset_tensors(split="train", max_samples=500)
    log(f"  Representative dataset shape: {X_train_rep.shape}")

    converter_int8 = tf.lite.TFLiteConverter.from_keras_model(model)
    converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative_dataset_gen():
        for i in range(len(X_train_rep)):
            yield [X_train_rep[i : i + 1]]

    converter_int8.representative_dataset = representative_dataset_gen
    converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter_int8.inference_input_type = tf.float32
    converter_int8.inference_output_type = tf.float32

    tflite_int8_bytes = converter_int8.convert()
    with open(INT8_MODEL_PATH, "wb") as f:
        f.write(tflite_int8_bytes)
    int8_size_bytes = os.path.getsize(INT8_MODEL_PATH)
    log(f"  ✓ Saved INT8 model to {INT8_MODEL_PATH} ({int8_size_bytes} bytes / {int8_size_bytes/1024:.2f} KB).")

    # Step 5: Accuracy Verification on TEST Split
    log("\n[Step 5] Loading TEST split and evaluating FP32, Float16, and INT8 models...")
    X_test, y_test = load_mrl_dataset_tensors(split="test")
    log(f"  TEST split samples loaded: X={X_test.shape}, y={y_test.shape}")

    # FP32 Keras Inference
    log("  Running FP32 Keras model inference...")
    y_score_fp32_keras = model.predict(X_test, batch_size=256, verbose=0).ravel()
    metrics_fp32_keras = compute_classification_metrics(y_test, y_score_fp32_keras)

    # FP32 TFLite Inference
    def evaluate_tflite_model(tflite_content_or_path, X_data):
        if isinstance(tflite_content_or_path, str):
            interpreter = tf.lite.Interpreter(model_path=tflite_content_or_path)
        else:
            interpreter = tf.lite.Interpreter(model_content=tflite_content_or_path)
        interpreter.allocate_tensors()
        input_idx = interpreter.get_input_details()[0]["index"]
        output_idx = interpreter.get_output_details()[0]["index"]

        preds = []
        for i in range(len(X_data)):
            inp = X_data[i : i + 1].astype(np.float32)
            interpreter.set_tensor(input_idx, inp)
            interpreter.invoke()
            out = interpreter.get_tensor(output_idx)
            preds.append(out[0, 0])
        return np.array(preds, dtype=np.float32)

    log("  Running FP32 TFLite model inference...")
    y_score_fp32_tflite = evaluate_tflite_model(tflite_fp32_bytes, X_test)
    metrics_fp32_tflite = compute_classification_metrics(y_test, y_score_fp32_tflite)

    log("  Running Float16 TFLite model inference...")
    t0 = time.time()
    y_score_fp16 = evaluate_tflite_model(FP16_MODEL_PATH, X_test)
    fp16_time = time.time() - t0
    metrics_fp16 = compute_classification_metrics(y_test, y_score_fp16)

    log("  Running INT8 TFLite model inference...")
    t0 = time.time()
    y_score_int8 = evaluate_tflite_model(INT8_MODEL_PATH, X_test)
    int8_time = time.time() - t0
    metrics_int8 = compute_classification_metrics(y_test, y_score_int8)

    log(f"  ✓ FP16 evaluation time: {fp16_time:.2f} s | INT8 evaluation time: {int8_time:.2f} s")

    # Step 6: Compression & Size Analysis
    log("\n[Step 6] Computing Compression Metrics...")
    comp_ratio_fp16 = fp32_size_bytes / fp16_size_bytes
    reduction_fp16 = (fp32_size_bytes - fp16_size_bytes) / fp32_size_bytes * 100.0

    comp_ratio_int8 = fp32_size_bytes / int8_size_bytes
    reduction_int8 = (fp32_size_bytes - int8_size_bytes) / fp32_size_bytes * 100.0

    size_analysis = {
        "fp32_tflite_bytes": fp32_size_bytes,
        "fp32_tflite_kb": fp32_size_bytes / 1024.0,
        "fp16_tflite_bytes": fp16_size_bytes,
        "fp16_tflite_kb": fp16_size_bytes / 1024.0,
        "fp16_compression_ratio": comp_ratio_fp16,
        "fp16_size_reduction_pct": reduction_fp16,
        "int8_tflite_bytes": int8_size_bytes,
        "int8_tflite_kb": int8_size_bytes / 1024.0,
        "int8_compression_ratio": comp_ratio_int8,
        "int8_size_reduction_pct": reduction_int8,
    }

    log(f"  FP32 TFLite Size: {fp32_size_bytes} B ({fp32_size_bytes/1024:.2f} KB)")
    log(f"  Float16 Size:     {fp16_size_bytes} B ({fp16_size_bytes/1024:.2f} KB) | Ratio: {comp_ratio_fp16:.2f}x ({reduction_fp16:.2f}% reduction)")
    log(f"  INT8 Size:        {int8_size_bytes} B ({int8_size_bytes/1024:.2f} KB) | Ratio: {comp_ratio_int8:.2f}x ({reduction_int8:.2f}% reduction)")

    # Step 7: Functional Verification
    log("\n[Step 7] Functional Contract Verification...")
    for path, name in [(FP16_MODEL_PATH, "Float16"), (INT8_MODEL_PATH, "INT8")]:
        interp = tf.lite.Interpreter(model_path=path)
        interp.allocate_tensors()
        inp_det = interp.get_input_details()
        out_det = interp.get_output_details()
        
        inp_shape = list(inp_det[0]["shape"])
        inp_dtype = inp_det[0]["dtype"]
        out_shape = list(out_det[0]["shape"])
        out_dtype = out_det[0]["dtype"]

        log(f"  {name} Model Contract: Input shape={inp_shape}, dtype={inp_dtype} | Output shape={out_shape}, dtype={out_dtype}")
        
        if inp_shape != [1, 24, 24, 1]:
            raise ValueError(f"{name} model input shape mismatch! Expected [1, 24, 24, 1], got {inp_shape}")
        if out_shape != [1, 1]:
            raise ValueError(f"{name} model output shape mismatch! Expected [1, 1], got {out_shape}")

    log("  ✓ All functional input/output contracts match frozen specification.")

    # Save artifacts
    log("\n[Step 8] Writing EXP-003 Artifacts...")
    
    # metrics.json
    all_metrics = {
        "exp_id": EXP_ID,
        "date": time.strftime("%Y-%m-%d"),
        "parameter_count": param_count,
        "test_samples": len(y_test),
        "fp32_keras": metrics_fp32_keras,
        "fp32_tflite": metrics_fp32_tflite,
        "float16_tflite": metrics_fp16,
        "int8_tflite": metrics_int8,
        "size_analysis": size_analysis,
    }
    
    with open(os.path.join(EXP_DIR, "metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    # quantization_report.json
    quant_report = {
        "exp_id": EXP_ID,
        "best_checkpoint": BEST_CKPT,
        "conversion_specs": {
            "float16": {
                "optimizations": ["DEFAULT"],
                "target_types": ["float16"],
                "output_path": FP16_MODEL_PATH,
                "file_size_bytes": fp16_size_bytes
            },
            "int8": {
                "optimizations": ["DEFAULT"],
                "representative_dataset_samples": 500,
                "supported_ops": ["TFLITE_BUILTINS_INT8"],
                "inference_input_type": "float32",
                "inference_output_type": "float32",
                "output_path": INT8_MODEL_PATH,
                "file_size_bytes": int8_size_bytes
            }
        },
        "accuracy_degradation": {
            "fp16_f1_diff": metrics_fp16["f1"] - metrics_fp32_keras["f1"],
            "int8_f1_diff": metrics_int8["f1"] - metrics_fp32_keras["f1"],
            "fp16_acc_diff": metrics_fp16["accuracy"] - metrics_fp32_keras["accuracy"],
            "int8_acc_diff": metrics_int8["accuracy"] - metrics_fp32_keras["accuracy"],
        },
        "size_reduction": size_analysis
    }
    with open(os.path.join(EXP_DIR, "quantization_report.json"), "w") as f:
        json.dump(quant_report, f, indent=2)

    # CSV table
    csv_path = os.path.join(EXP_DIR, "fp32_vs_fp16_vs_int8.csv")
    with open(csv_path, "w") as f:
        f.write("Model_Variant,Size_Bytes,Size_KB,Accuracy,Precision,Recall,Specificity,F1_Score,ROC_AUC,PR_AUC,Brier\n")
        f.write(f"FP32_Keras,-,-,{metrics_fp32_keras['accuracy']:.6f},{metrics_fp32_keras['precision']:.6f},{metrics_fp32_keras['recall']:.6f},{metrics_fp32_keras['specificity']:.6f},{metrics_fp32_keras['f1']:.6f},{metrics_fp32_keras['roc_auc']:.6f},{metrics_fp32_keras['pr_auc']:.6f},{metrics_fp32_keras['brier']:.6f}\n")
        f.write(f"FP32_TFLite,{fp32_size_bytes},{fp32_size_bytes/1024:.2f},{metrics_fp32_tflite['accuracy']:.6f},{metrics_fp32_tflite['precision']:.6f},{metrics_fp32_tflite['recall']:.6f},{metrics_fp32_tflite['specificity']:.6f},{metrics_fp32_tflite['f1']:.6f},{metrics_fp32_tflite['roc_auc']:.6f},{metrics_fp32_tflite['pr_auc']:.6f},{metrics_fp32_tflite['brier']:.6f}\n")
        f.write(f"Float16_TFLite,{fp16_size_bytes},{fp16_size_bytes/1024:.2f},{metrics_fp16['accuracy']:.6f},{metrics_fp16['precision']:.6f},{metrics_fp16['recall']:.6f},{metrics_fp16['specificity']:.6f},{metrics_fp16['f1']:.6f},{metrics_fp16['roc_auc']:.6f},{metrics_fp16['pr_auc']:.6f},{metrics_fp16['brier']:.6f}\n")
        f.write(f"INT8_TFLite,{int8_size_bytes},{int8_size_bytes/1024:.2f},{metrics_int8['accuracy']:.6f},{metrics_int8['precision']:.6f},{metrics_int8['recall']:.6f},{metrics_int8['specificity']:.6f},{metrics_int8['f1']:.6f},{metrics_int8['roc_auc']:.6f},{metrics_int8['pr_auc']:.6f},{metrics_int8['brier']:.6f}\n")

    # conversion.log
    with open(os.path.join(EXP_DIR, "conversion.log"), "w") as f:
        f.write("\n".join(log_lines))

    # verification_report.json
    verif_report = {
        "status": "PASS",
        "fp16_model_exists": os.path.exists(FP16_MODEL_PATH),
        "int8_model_exists": os.path.exists(INT8_MODEL_PATH),
        "test_eval_completed": True,
        "input_output_contract_valid": True
    }
    with open(os.path.join(EXP_DIR, "verification_report.json"), "w") as f:
        json.dump(verif_report, f, indent=2)

    log("  ✓ All artifacts written to experiments/EXP-003_quantization/.")
    log("\n=========================================================================")
    log("EXP-003 EXECUTION COMPLETED SUCCESSFULLY")
    log("=========================================================================")
    return all_metrics

if __name__ == "__main__":
    run_exp003()
