"""Run AF detection inference on all files in Huywatch folders (arm and wrist-down).

This script processes each CSV file in the 'arm' and 'wrist-down' directories:
1. Estimates the sampling rate from device_millis.
2. Inverts the signal polarity (signal = -df['ir']).
3. Preprocesses the signal (interpolation, band-pass filter 0.5-8.0 Hz, Z-score normalization).
4. Resamples the signal to the training sampling rate of 125 Hz.
5. Segments into 5s windows with 2.5s overlap and extracts 9 features.
6. Runs predictions using the trained Random Forest model.
7. Saves a summary report.

Usage
-----
    python code/predict_folders.py
"""

from __future__ import annotations

import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

# Define paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH   = PROJECT_ROOT / "models" / "ppg_af_rf.joblib"
ARM_DIR      = PROJECT_ROOT / "data" / "raw" / "huywatch" / "arm"
WRIST_DIR    = PROJECT_ROOT / "data" / "raw" / "huywatch" / "wrist-down"
REPORTS_DIR  = PROJECT_ROOT / "reports"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Add src folder to path and import pipeline functions
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from pipeline.ppg_pipeline import (
    build_feature_matrix,
    detect_beats,
    preprocess_ppg,
    segment_signal,
)

TRAIN_FS = 125  # Hz - training target sampling rate

def process_file(clf, csv_path: Path, channel: str = "ir", threshold: float = 0.5) -> dict | None:
    """Process a single Huywatch CSV and return a dictionary of summary metrics."""
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"  Error reading {csv_path.name}: {e}")
        return None

    # Standardize column headers (some wrist files have leading spaces in header)
    df.columns = [c.strip() for c in df.columns]

    # Verify required columns
    required = {"device_millis", channel}
    missing = required - set(df.columns)
    if missing:
        print(f"  Skipping {csv_path.name}: missing columns {missing}. Found columns: {list(df.columns)}")
        return None

    # Extract time and raw signal (invert polarity)
    t_ms = df["device_millis"].values.astype(float)
    signal_raw = -df[channel].values.astype(float)
    
    if len(t_ms) < 100:
        print(f"  Skipping {csv_path.name}: signal too short ({len(t_ms)} samples).")
        return None

    # Estimate sampling rate
    t_sec = (t_ms - t_ms[0]) / 1000.0
    dt_arr = np.diff(t_sec)
    median_dt = np.median(dt_arr)
    if median_dt <= 0:
        print(f"  Skipping {csv_path.name}: invalid time intervals.")
        return None
    fs_detected = float(1.0 / median_dt)
    duration = t_sec[-1]

    # Preprocess and resample to 125 Hz
    try:
        # preprocess_ppg handles: interpolation -> bandpass 0.5-8Hz -> zscore -> resampling to 125 Hz
        signal_proc = preprocess_ppg(signal_raw, fs=int(round(fs_detected)), target_fs=TRAIN_FS)
    except Exception as e:
        print(f"  Error preprocessing {csv_path.name}: {e}")
        return None

    # Segment into windows (5s length, 2.5s overlap)
    windows = segment_signal(signal_proc, fs=TRAIN_FS, window_sec=5.0, overlap_sec=2.5)
    if len(windows) == 0:
        print(f"  Skipping {csv_path.name}: signal duration ({duration:.1f}s) too short for windowing.")
        return None

    # Extract features
    X_feat = build_feature_matrix(windows, fs=TRAIN_FS)

    # Predict AF probability
    pred_proba = clf.predict_proba(X_feat)[:, 1]
    pred_labels = (pred_proba >= threshold).astype(int)

    n_win = len(windows)
    n_af = int((pred_labels == 1).sum())
    af_pct = 100.0 * n_af / n_win if n_win > 0 else 0.0
    mean_prob = float(np.mean(pred_proba)) * 100.0

    return {
        "file_name": csv_path.name,
        "n_samples": len(df),
        "duration_sec": round(duration, 1),
        "fs_est": round(fs_detected, 1),
        "n_windows": n_win,
        "n_AF_windows": n_af,
        "AF_percentage": round(af_pct, 1),
        "mean_AF_probability": round(mean_prob, 1),
        "majority_vote": "AF" if n_af > (n_win / 2) else "Non-AF"
    }

def main() -> None:
    print("=" * 60)
    print("  Huywatch Folders AF Inference")
    print("=" * 60)

    # 1. Load trained model
    if not MODEL_PATH.exists():
        print(f"Error: Model file not found at {MODEL_PATH}")
        print("Please train the model first by running: python code/train_af_model.py")
        sys.exit(1)

    print(f"Loading model from {MODEL_PATH}...")
    clf = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")

    # 2. Process 'arm' folder
    print("\nProcessing 'arm' folder...")
    arm_results = []
    if ARM_DIR.exists():
        csv_files = sorted(ARM_DIR.glob("*.csv"))
        print(f"Found {len(csv_files)} CSV files in 'arm'.")
        for csv_path in csv_files:
            print(f"  Processing {csv_path.name}...")
            res = process_file(clf, csv_path, channel="ir")
            if res:
                arm_results.append(res)
    else:
        print(f"Warning: 'arm' directory not found at {ARM_DIR}")

    # 3. Process 'wrist-down' folder
    print("\nProcessing 'wrist-down' folder...")
    wrist_results = []
    if WRIST_DIR.exists():
        csv_files = sorted(WRIST_DIR.glob("*.csv"))
        print(f"Found {len(csv_files)} CSV files in 'wrist-down'.")
        for csv_path in csv_files:
            print(f"  Processing {csv_path.name}...")
            res = process_file(clf, csv_path, channel="ir")
            if res:
                wrist_results.append(res)
    else:
        print(f"Warning: 'wrist-down' directory not found at {WRIST_DIR}")

    # 4. Generate Summary Report
    report_lines = [
        "============================================================",
        "          HUYWATCH FOLDERS AF INFERENCE SUMMARY REPORT",
        "============================================================",
        f"Model used : {MODEL_PATH.name}",
        f"Channel    : IR",
        f"Threshold  : 0.50",
        "",
    ]

    # Arm folder report
    report_lines.append("--- ARM FOLDER RESULTS ---")
    if arm_results:
        df_arm = pd.DataFrame(arm_results)
        report_lines.append(
            df_arm[["file_name", "fs_est", "duration_sec", "n_windows", "n_AF_windows", "AF_percentage", "majority_vote"]].to_string(index=False)
        )
        avg_af_arm = df_arm["AF_percentage"].mean()
        report_lines.append(f"\nAverage AF percentage in 'arm': {avg_af_arm:.1f}%")
    else:
        report_lines.append("No results generated for 'arm' folder.")
    report_lines.append("\n" + "="*60 + "\n")

    # Wrist folder report
    report_lines.append("--- WRIST-DOWN FOLDER RESULTS ---")
    if wrist_results:
        df_wrist = pd.DataFrame(wrist_results)
        report_lines.append(
            df_wrist[["file_name", "fs_est", "duration_sec", "n_windows", "n_AF_windows", "AF_percentage", "majority_vote"]].to_string(index=False)
        )
        avg_af_wrist = df_wrist["AF_percentage"].mean()
        report_lines.append(f"\nAverage AF percentage in 'wrist-down': {avg_af_wrist:.1f}%")
    else:
        report_lines.append("No results generated for 'wrist-down' folder.")
    report_lines.append("\n" + "="*60 + "\n")

    # Comparison summary
    report_lines.append("--- SUMMARY COMPARISON ---")
    if arm_results and wrist_results:
        report_lines.append(f"Arm folder average AF windows %        : {avg_af_arm:.1f}%")
        report_lines.append(f"Wrist-down folder average AF windows % : {avg_af_wrist:.1f}%")
        report_lines.append("")
        if avg_af_arm < 20 and avg_af_wrist < 20:
            report_lines.append("Conclusion: Low risk of AF detected in both postures.")
        else:
            report_lines.append("Conclusion: Higher AF-like rhythm characteristics detected in one or both postures.")
    else:
        report_lines.append("Cannot perform comparison due to missing results in one or both folders.")

    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    # Save report
    report_path = REPORTS_DIR / "folders_summary.txt"
    report_path.write_text(report_text, encoding="utf-8")
    print(f"\nSummary report saved to: {report_path}")

if __name__ == "__main__":
    main()
