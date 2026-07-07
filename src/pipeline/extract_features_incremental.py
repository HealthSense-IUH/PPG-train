"""Incremental feature extraction script for Vilnius University Zenodo dataset.

This script scans the 'data/' folder for any pair of patient files (e.g. XXX_PPG.mat and XXX_ECG.mat):
1. Detects the Patient ID (e.g., '002').
2. Loads the PPG and ECG files.
3. Synchronizes the PPG sessions to the ECG timeline.
4. Preprocesses the PPG signals at 100 Hz, segments them into 5s windows.
5. Labels the windows using ECG QRSindex and AF_annotation.
6. Extracts 9 features for valid windows.
7. Appends the results to outputs/vilnius_features_consolidated.csv.

Usage
-----
    python code/extract_features_incremental.py
"""

from __future__ import annotations

import sys
from pathlib import Path
import h5py
import numpy as np
import pandas as pd

# Define paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR     = PROJECT_ROOT / "data" / "raw"
OUTPUT_CSV   = PROJECT_ROOT / "data" / "processed" / "vilnius_features_consolidated.csv"

# Add src folder to path and import pipeline functions
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from pipeline.ppg_pipeline import (
    build_feature_matrix,
    preprocess_ppg,
    segment_signal,
)

FS_PPG = 100.0  # Hz - Vilnius PPG sampling rate
FS_ECG = 500.0  # Hz - Vilnius ECG sampling rate

def time_str_to_seconds(time_str: str) -> float:
    """Convert HH:MM:SS to seconds from start of day."""
    h, m, s = map(int, time_str.strip().split(":"))
    return h * 3600.0 + m * 60.0 + s

def compute_offset_sec(day1: str, time1: str, day2: str, time2: str) -> float:
    """Compute seconds offset of day2/time2 relative to day1/time1."""
    d1 = int(day1)
    day2_clean = day2.replace("\x00", "").strip()
    if not day2_clean:
        d2 = d1  # Fallback to the same day as ECG if PPG startday is empty/null
    else:
        d2 = int(day2_clean)
        
    t1 = time_str_to_seconds(time1)
    t2 = time_str_to_seconds(time2)
    
    day_diff = d2 - d1
    return day_diff * 86400.0 + (t2 - t1)

def get_hdf5_text(f: h5py.File, ref: h5py.Reference) -> str:
    """Safely extract string text from a MATLAB HDF5 object reference."""
    obj = f[ref]
    arr = obj[:].flatten().astype(int)
    # Filter out null characters (0 in ASCII)
    chars = [chr(c) for c in arr if c != 0]
    return "".join(chars)

def detect_patient_files() -> list[tuple[Path, Path, str]]:
    """Scan data/ folder and return a list of (ppg_path, ecg_path, patient_id) tuples."""
    if not DATA_DIR.exists():
        return []
        
    ppg_files = sorted(list(DATA_DIR.glob("*_PPG.mat")))
    results = []
    for ppg_path in ppg_files:
        patient_id = ppg_path.name.split("_")[0]
        ecg_path = DATA_DIR / f"{patient_id}_ECG.mat"
        if ecg_path.exists():
            results.append((ppg_path, ecg_path, patient_id))
        else:
            print(f"Warning: Found PPG file {ppg_path.name} but matching ECG file {ecg_path.name} is missing.")
    return results

def main() -> None:
    print("=" * 60)
    print("  Incremental Feature Extraction - Vilnius Dataset")
    print("=" * 60)

    # 1. Detect patient files in data/
    detected_list = detect_patient_files()
    if not detected_list:
        print("Error: Could not find any matching patient files (e.g. XXX_PPG.mat and XXX_ECG.mat) in data/ directory.")
        print(f"Please place the files in: {DATA_DIR}")
        sys.exit(1)
        
    print(f"Detected {len(detected_list)} patients to process.")
    
    for ppg_path, ecg_path, patient_id in detected_list:
        print("\n" + "#" * 60)
        print(f"  PROCESSING PATIENT {patient_id}")
        print("#" * 60)
        
        case_id = int(patient_id)

        # 2. Load ECG annotations
        print("\nOpening ECG file to load QRSindex and AF_annotation...")
        with h5py.File(ecg_path, "r") as f_ecg:
            ecg_day = get_hdf5_text(f_ecg, "recording_startday")
            ecg_time = get_hdf5_text(f_ecg, "recording_starttime")
            print(f"  ECG Recording Start: Day {ecg_day}, Time {ecg_time}")
            
            print("  Loading QRS indices...")
            qrs_indices = f_ecg["QRSindex"][:].flatten()
            qrs_times = qrs_indices / FS_ECG
            
            print("  Loading AF annotations...")
            af_annotations = f_ecg["AF_annotation"][:].flatten()
            print(f"  Total beats: {len(qrs_times)} | AF beats: {int((af_annotations==1).sum())}")

        # 3. Load PPG signals and process
        print("\nOpening PPG file to extract sessions...")
        features_list = []

        with h5py.File(ppg_path, "r") as f_ppg:
            n_sessions = len(f_ppg["recording_startday"])
            print(f"Found {n_sessions} PPG sessions.")

            for s_idx in range(n_sessions):
                s_day = get_hdf5_text(f_ppg, f_ppg["recording_startday"][s_idx, 0])
                s_time = get_hdf5_text(f_ppg, f_ppg["recording_starttime"][s_idx, 0])
                
                offset_sec = compute_offset_sec(ecg_day, ecg_time, s_day, s_time)
                print(f"\nProcessing Session {s_idx:02d}: Start Day={s_day}, Time={s_time} | Offset={offset_sec:.1f}s")
                
                ref_ppg = f_ppg["PPG_GREEN"][s_idx, 0]
                ppg_raw = f_ppg[ref_ppg][0, :].astype(float)
                duration = len(ppg_raw) / FS_PPG
                print(f"  Samples: {len(ppg_raw)} | Duration: {duration:.1f}s ({duration/3600:.2f} hours)")

                try:
                    ppg_proc = preprocess_ppg(ppg_raw, fs=int(FS_PPG))
                except Exception as e:
                    print(f"  Failed preprocessing: {e}. Skipping session.")
                    continue

                window_sec = 5.0
                overlap_sec = 2.5
                step_sec = window_sec - overlap_sec
                
                windows = segment_signal(ppg_proc, fs=int(FS_PPG), window_sec=window_sec, overlap_sec=overlap_sec)
                n_win = len(windows)
                if n_win == 0:
                    print("  No windows generated. Skipping session.")
                    continue

                print(f"  Generated {n_win} windows. Labeling windows and extracting features...")
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
                    print("  No valid normal or AF windows found in this session.")
                    continue
                    
                feat_df_valid = feat_df.iloc[valid_mask].copy()
                feat_df_valid["label"] = np.array(labels)[valid_mask]
                feat_df_valid["session_id"] = s_idx
                feat_df_valid["case_id"] = case_id
                
                n_normal_win = int((feat_df_valid["label"] == 0).sum())
                n_af_win = int((feat_df_valid["label"] == 1).sum())
                print(f"  Valid windows: {len(feat_df_valid)} (Normal: {n_normal_win} | AF: {n_af_win})")

                features_list.append(feat_df_valid)

        # 4. Consolidate and Append to CSV
        if features_list:
            df_all = pd.concat(features_list, ignore_index=True)
            
            # Ensure column ordering
            cols_id = ["case_id", "session_id"]
            cols_features = [c for c in df_all.columns if c not in cols_id + ["label"]]
            df_all = df_all[cols_id + cols_features + ["label"]]
            
            # Check if output consolidated file exists
            file_exists = OUTPUT_CSV.exists()
            
            # Append data to the consolidated CSV
            df_all.to_csv(OUTPUT_CSV, mode="a", header=not file_exists, index=False)
            
            print("\n" + "=" * 60)
            print(f"  EXTRACTION COMPLETE FOR PATIENT {patient_id}")
            print("=" * 60)
            print(f"  Records extracted for patient {patient_id} : {len(df_all)}")
            print(f"  Consolidated CSV path                  : {OUTPUT_CSV}")
            print(f"  You can now safely delete:")
            print(f"    - {ppg_path.name}")
            print(f"    - {ecg_path.name}")
        else:
            print(f"\nError: No valid features extracted for Patient {patient_id}.")
            
    print("\n" + "=" * 60)
    print("  ALL DETECTED PATIENTS PROCESSED SUCCESSFULLY")
    print("=" * 60)

if __name__ == "__main__":
    main()
