"""Evaluate the retrained model on unseen Patient 012 data and export test files.

This script:
1. Loads 012_PPG.mat and 012_ECG.mat (unseen by the model).
2. Performs synchronization, preprocessing, and window feature extraction.
3. Evaluates the trained model (models/ppg_af_rf.joblib) on Patient 012 features.
4. Finds a 100-second session segment of pure AF and pure Normal.
5. Exports these segments as CSV files:
   - outputs/patient_012_af_test.csv
   - outputs/patient_012_normal_test.csv
"""

from __future__ import annotations

import sys
from pathlib import Path
import h5py
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Define paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PPG_MAT_PATH = PROJECT_ROOT / "data" / "012_PPG.mat"
ECG_MAT_PATH = PROJECT_ROOT / "data" / "012_ECG.mat"
MODEL_PATH   = PROJECT_ROOT / "models" / "ppg_af_rf.joblib"
OUTPUTS_DIR  = PROJECT_ROOT / "outputs"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Add code folder to path and import pipeline functions
sys.path.insert(0, str(PROJECT_ROOT / "code"))
from ppg_pipeline import (
    build_feature_matrix,
    preprocess_ppg,
    segment_signal,
)

FS_PPG = 100.0  # Hz
FS_ECG = 500.0  # Hz
FEATURE_COLS = [
    "signal_mean", "signal_std", "signal_range", "signal_energy",
    "peak_count", "ibi_mean", "ibi_std", "ibi_rmssd", "ibi_cv"
]

def time_str_to_seconds(time_str: str) -> float:
    h, m, s = map(int, time_str.strip().split(":"))
    return h * 3600.0 + m * 60.0 + s

def compute_offset_sec(day1: str, time1: str, day2: str, time2: str) -> float:
    d1 = int(day1)
    day2_clean = day2.replace("\x00", "").strip()
    d2 = d1 if not day2_clean else int(day2_clean)
    t1 = time_str_to_seconds(time1)
    t2 = time_str_to_seconds(time2)
    return (d2 - d1) * 86400.0 + (t2 - t1)

def get_hdf5_text(f: h5py.File, ref: h5py.Reference) -> str:
    obj = f[ref]
    arr = obj[:].flatten().astype(int)
    return "".join(chr(c) for c in arr if c != 0)

def main() -> None:
    print("=" * 60)
    print("  Evaluating Model on Unseen Patient 012")
    print("=" * 60)

    # 1. Load Model
    if not MODEL_PATH.exists():
        print(f"Error: Model not found at {MODEL_PATH}")
        sys.exit(1)
    clf = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")

    # 2. Load ECG annotations
    print("\nOpening 012_ECG.mat...")
    with h5py.File(ECG_MAT_PATH, "r") as f_ecg:
        ecg_day = get_hdf5_text(f_ecg, "recording_startday")
        ecg_time = get_hdf5_text(f_ecg, "recording_starttime")
        print(f"  ECG Recording Start: Day {ecg_day}, Time {ecg_time}")
        
        qrs_indices = f_ecg["QRSindex"][:].flatten()
        qrs_times = qrs_indices / FS_ECG
        af_annotations = f_ecg["AF_annotation"][:].flatten()
        print(f"  Total beats: {len(qrs_times)} | AF beats: {int((af_annotations==1).sum())}")

    # 3. Load PPG sessions and extract features + labels
    print("\nOpening 012_PPG.mat...")
    features_list = []
    
    # We will also save raw signals of sessions to extract test CSVs later
    af_session_data = None
    normal_session_data = None

    with h5py.File(PPG_MAT_PATH, "r") as f_ppg:
        n_sessions = len(f_ppg["recording_startday"])
        
        for s_idx in range(n_sessions):
            s_day = get_hdf5_text(f_ppg, f_ppg["recording_startday"][s_idx, 0])
            s_time = get_hdf5_text(f_ppg, f_ppg["recording_starttime"][s_idx, 0])
            
            offset_sec = compute_offset_sec(ecg_day, ecg_time, s_day, s_time)
            
            ref_ppg = f_ppg["PPG_GREEN"][s_idx, 0]
            ppg_raw = f_ppg[ref_ppg][0, :].astype(float)
            duration = len(ppg_raw) / FS_PPG

            try:
                ppg_proc = preprocess_ppg(ppg_raw, fs=int(FS_PPG))
            except Exception:
                continue

            window_sec = 5.0
            overlap_sec = 2.5
            step_sec = window_sec - overlap_sec
            
            windows = segment_signal(ppg_proc, fs=int(FS_PPG), window_sec=window_sec, overlap_sec=overlap_sec)
            n_win = len(windows)
            if n_win == 0:
                continue

            feat_df = build_feature_matrix(windows, fs=int(FS_PPG))
            
            # Label windows
            win_start_times = offset_sec + np.arange(n_win) * step_sec
            win_end_times = win_start_times + window_sec
            
            labels = []
            valid_mask = []
            
            for i in range(n_win):
                t_start = win_start_times[i]
                t_end = win_end_times[i]
                
                left_idx = np.searchsorted(qrs_times, t_start)
                right_idx = np.searchsorted(qrs_times, t_end)
                
                window_beat_labels = af_annotations[left_idx:right_idx]
                
                if len(window_beat_labels) < 3:
                    labels.append(-1)
                    valid_mask.append(False)
                else:
                    unique, counts = np.unique(window_beat_labels, return_counts=True)
                    majority_lbl = unique[np.argmax(counts)]
                    
                    if majority_lbl in [0.0, 1.0]:
                        labels.append(int(majority_lbl))
                        valid_mask.append(True)
                    else:
                        labels.append(-1)
                        valid_mask.append(False)

            valid_mask = np.array(valid_mask)
            if not valid_mask.any():
                continue
                
            feat_df_valid = feat_df.iloc[valid_mask].copy()
            feat_df_valid["label"] = np.array(labels)[valid_mask]
            
            # Count labels to identify good sessions for exporting test CSVs
            n_normal = int((feat_df_valid["label"] == 0).sum())
            n_af = int((feat_df_valid["label"] == 1).sum())
            
            # Save raw session data if it is pure Normal or pure AF
            if n_af == len(feat_df_valid) and len(feat_df_valid) > 200 and af_session_data is None:
                af_session_data = ppg_raw.copy()
                print(f"  -> Found pure AF Session {s_idx} ({n_af} windows).")
            if n_normal == len(feat_df_valid) and len(feat_df_valid) > 200 and normal_session_data is None:
                normal_session_data = ppg_raw.copy()
                print(f"  -> Found pure Normal Session {s_idx} ({n_normal} windows).")

            features_list.append(feat_df_valid)

    # 4. Consolidate & Evaluate
    if not features_list:
        print("Error: No valid features extracted for Patient 012.")
        sys.exit(1)
        
    df_012 = pd.concat(features_list, ignore_index=True)
    X = df_012[FEATURE_COLS]
    y = df_012["label"]
    
    print(f"\nConsolidated Patient 012 Set: {len(df_012)} windows (Normal: {int((y==0).sum())} | AF: {int((y==1).sum())})")
    
    y_pred = clf.predict(X)
    y_proba = clf.predict_proba(X)[:, 1]
    
    acc = clf.score(X, y)
    auc = roc_auc_score(y, y_proba)
    
    print(f"\nPatient 012 Accuracy : {acc:.4f}")
    print(f"Patient 012 ROC-AUC  : {auc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y, y_pred, target_names=["Normal", "AF"]))
    
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print("Confusion Matrix:")
    print(f"              Normal    AF")
    print(f"  Normal  :   {tn:6d}  {fp:6d}")
    print(f"  AF      :   {fn:6d}  {tp:6d}")

    # 5. Export Test CSVs from Patient 012 raw signals
    print("\n" + "=" * 50)
    print("  EXPORTING UNSEEN TEST CSVs FROM PATIENT 012")
    print("=" * 50)

    # Export AF CSV
    if af_session_data is not None:
        n_samples = min(12000, len(af_session_data))
        af_slice = af_session_data[:n_samples]
        device_millis = np.arange(n_samples) * 10
        
        df_af = pd.DataFrame({
            "device_millis": device_millis,
            "red": np.zeros(n_samples, dtype=int),
            "ir": -af_slice.astype(int)  # invert polarity for Huywatch format
        })
        
        path_af = OUTPUTS_DIR / "patient_012_af_test.csv"
        df_af.to_csv(path_af, index=False)
        print(f"  Saved unseen AF test file to     : {path_af}")
    else:
        print("  Warning: Could not find pure AF session to export.")

    # Export Normal CSV
    if normal_session_data is not None:
        n_samples = min(12000, len(normal_session_data))
        normal_slice = normal_session_data[:n_samples]
        device_millis = np.arange(n_samples) * 10
        
        df_normal = pd.DataFrame({
            "device_millis": device_millis,
            "red": np.zeros(n_samples, dtype=int),
            "ir": -normal_slice.astype(int)  # invert polarity for Huywatch format
        })
        
        path_normal = OUTPUTS_DIR / "patient_012_normal_test.csv"
        df_normal.to_csv(path_normal, index=False)
        print(f"  Saved unseen Normal test file to : {path_normal}")
    else:
        print("  Warning: Could not find pure Normal session to export.")

    print("\nDONE. You can now use these two files to test the Streamlit app.")

if __name__ == "__main__":
    main()
