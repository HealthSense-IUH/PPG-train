"""Run Hybrid Model AF detection inference on Huywatch folders (arm and wrist-down).

This script processes each CSV file using the Hybrid Fusion RandomForest model:
1. Estimates the sampling rate from device_millis.
2. Inverts the signal polarity (signal = -df['ir']).
3. Preprocesses the signal (interpolation, band-pass filter, Z-score normalization, resampling to 125 Hz).
4. Segments into 5s windows.
5. Standardizes windows using the final StandardScaler.
6. Extracts 16 deep features from the CNN + BiLSTM model.
7. Extracts 9 HRV features.
8. Concatenates features (25 dimensions).
9. Runs predictions using the trained Hybrid RandomForest model.
10. Saves a summary report.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

# Suppress Keras/TF logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

# Define paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
HYBRID_RF_PATH = PROJECT_ROOT / "models" / "ppg_af_hybrid_rf.joblib"
CNN_MODEL_PATH = PROJECT_ROOT / "models" / "ppg_af_cnn_bilstm.keras"
SCALER_PATH    = PROJECT_ROOT / "models" / "ppg_scaler.joblib"
ARM_DIR        = PROJECT_ROOT / "data" / "raw" / "huywatch" / "arm"
WRIST_DIR      = PROJECT_ROOT / "data" / "raw" / "huywatch" / "wrist-down"
REPORTS_DIR    = PROJECT_ROOT / "reports"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Add src folder to path and import pipeline functions
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from pipeline.ppg_pipeline import (
    build_feature_matrix,
    preprocess_ppg,
    segment_signal,
)

TRAIN_FS = 125  # Hz

def process_file_hybrid(hybrid_rf, feature_extractor, scaler, csv_path: Path, channel: str = "ir", threshold: float = 0.8) -> dict | None:
    """Process a single Huywatch CSV with the Hybrid model."""
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"  Error reading {csv_path.name}: {e}")
        return None

    # Clean headers
    df.columns = [c.strip() for c in df.columns]

    # Verify columns
    required = {"device_millis", channel}
    missing = required - set(df.columns)
    if missing:
        print(f"  Skipping {csv_path.name}: missing columns {missing}.")
        return None

    # Invert polarity
    t_ms = df["device_millis"].values.astype(float)
    signal_raw = -df[channel].values.astype(float)
    
    if len(t_ms) < 100:
        return None

    # Estimate sampling rate
    t_sec = (t_ms - t_ms[0]) / 1000.0
    dt_arr = np.diff(t_sec)
    fs_detected = float(1.0 / np.median(dt_arr))
    duration = t_sec[-1]

    # Preprocess
    try:
        signal_proc = preprocess_ppg(signal_raw, fs=int(round(fs_detected)), target_fs=TRAIN_FS)
    except Exception as e:
        print(f"  Error preprocessing {csv_path.name}: {e}")
        return None

    # Segment
    windows = segment_signal(signal_proc, fs=TRAIN_FS, window_sec=5.0, overlap_sec=2.5)
    if len(windows) == 0:
        return None

    # 1. Scale raw windows
    windows_scaled = scaler.transform(windows)
    windows_nn = np.expand_dims(windows_scaled, axis=-1)

    # 2. Extract deep features
    deep_features = feature_extractor.predict(windows_nn, verbose=0)

    # 3. Extract HRV features
    X_hrv = build_feature_matrix(windows, fs=TRAIN_FS)

    # 4. Concatenate features
    X_hybrid = np.hstack([X_hrv.values, deep_features])

    # 5. Predict
    pred_proba = hybrid_rf.predict_proba(X_hybrid)[:, 1]
    pred_labels = (pred_proba >= threshold).astype(int)

    n_win = len(windows)
    n_af = int((pred_labels == 1).sum())
    af_pct = 100.0 * n_af / n_win if n_win > 0 else 0.0
    mean_prob = float(np.mean(pred_proba)) * 100.0

    return {
        "file_name": csv_path.name,
        "fs_est": round(fs_detected, 1),
        "duration_sec": round(duration, 1),
        "n_windows": n_win,
        "n_AF_windows": n_af,
        "AF_percentage": round(af_pct, 1),
        "mean_AF_probability": round(mean_prob, 1),
        "majority_vote": "AF" if n_af > (n_win / 2) else "Non-AF"
    }

def main() -> None:
    print("=" * 60)
    print("  Huywatch Folders Hybrid AF Inference")
    print("=" * 60)

    # Check models
    if not (HYBRID_RF_PATH.exists() and CNN_MODEL_PATH.exists() and SCALER_PATH.exists()):
        print("Error: Hybrid model files not found. Run train_nn_model.py first.")
        sys.exit(1)

    print("Loading Hybrid model components...")
    hybrid_rf = joblib.load(HYBRID_RF_PATH)
    scaler = joblib.load(SCALER_PATH)
    cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)
    
    # Create feature extractor submodel
    feature_extractor = tf.keras.Model(inputs=cnn_model.input, outputs=cnn_model.get_layer("deep_features").output)
    print("Hybrid model loaded successfully.")

    # Process 'arm' folder
    print("\nProcessing 'arm' folder...")
    arm_results = []
    if ARM_DIR.exists():
        csv_files = sorted(ARM_DIR.glob("*.csv"))
        for csv_path in csv_files:
            print(f"  Processing {csv_path.name}...")
            res = process_file_hybrid(hybrid_rf, feature_extractor, scaler, csv_path, channel="ir")
            if res:
                arm_results.append(res)

    # Process 'wrist-down' folder
    print("\nProcessing 'wrist-down' folder...")
    wrist_results = []
    if WRIST_DIR.exists():
        csv_files = sorted(WRIST_DIR.glob("*.csv"))
        for csv_path in csv_files:
            print(f"  Processing {csv_path.name}...")
            res = process_file_hybrid(hybrid_rf, feature_extractor, scaler, csv_path, channel="ir")
            if res:
                wrist_results.append(res)

    # Generate Report
    report_lines = [
        "============================================================",
        "          HUYWATCH FOLDERS HYBRID AF INFERENCE REPORT",
        "============================================================",
        f"Hybrid Model used : {HYBRID_RF_PATH.name} (HRV + CNN-BiLSTM)",
        f"Channel           : IR",
        f"Threshold         : 0.80 (Safe threshold)",
        "",
        "--- ARM FOLDER RESULTS ---",
    ]
    
    if arm_results:
        df_arm = pd.DataFrame(arm_results)
        report_lines.append(
            df_arm[["file_name", "fs_est", "duration_sec", "n_windows", "n_AF_windows", "AF_percentage", "mean_AF_probability", "majority_vote"]].to_string(index=False)
        )
        avg_af_arm = df_arm["AF_percentage"].mean()
        report_lines.append(f"\nAverage AF percentage in 'arm': {avg_af_arm:.1f}%")
    else:
        report_lines.append("No results in 'arm'.")
    report_lines.append("\n" + "="*60 + "\n")

    report_lines.append("--- WRIST-DOWN FOLDER RESULTS ---")
    if wrist_results:
        df_wrist = pd.DataFrame(wrist_results)
        report_lines.append(
            df_wrist[["file_name", "fs_est", "duration_sec", "n_windows", "n_AF_windows", "AF_percentage", "mean_AF_probability", "majority_vote"]].to_string(index=False)
        )
        avg_af_wrist = df_wrist["AF_percentage"].mean()
        report_lines.append(f"\nAverage AF percentage in 'wrist-down': {avg_af_wrist:.1f}%")
    else:
        report_lines.append("No results in 'wrist-down'.")
    report_lines.append("\n" + "="*60)
    report_text = "\n".join(report_lines)
    # Save report
    report_path = REPORTS_DIR / "hybrid_folders_summary.txt"
    report_path.write_text(report_text, encoding="utf-8")
    print(f"\nSaved Hybrid summary report to: {report_path}")

if __name__ == "__main__":
    main()
