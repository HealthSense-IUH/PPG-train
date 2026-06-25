"""Extract features from the Vilnius University Zenodo dataset (Patient 001).

This script:
1. Loads 001_PPG.mat and 001_ECG.mat.
2. Synchronizes the 13 PPG sessions to the continuous ECG timeline.
3. Preprocesses the PPG signals at their native 100 Hz (interpolation, 0.5-8.0 Hz bandpass, z-score).
4. Segments the signals into 5s windows with 2.5s overlap (at 100 Hz).
5. Labels each window using the ECG QRSindex and AF_annotation.
6. Extracts the 9 features for each valid window.
7. Saves the consolidated feature matrix to outputs/vilnius_features_001.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path
import h5py
import numpy as np
import pandas as pd

# Define paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PPG_MAT_PATH = PROJECT_ROOT / "data" / "001_PPG.mat"
ECG_MAT_PATH = PROJECT_ROOT / "data" / "001_ECG.mat"
OUTPUT_CSV   = PROJECT_ROOT / "outputs" / "vilnius_features_001.csv"

# Add code folder to path and import pipeline functions
sys.path.insert(0, str(PROJECT_ROOT / "code"))
from ppg_pipeline import (
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
    d2 = int(day2)
    t1 = time_str_to_seconds(time1)
    t2 = time_str_to_seconds(time2)
    
    day_diff = d2 - d1
    return day_diff * 86400.0 + (t2 - t1)

def get_hdf5_text(f: h5py.File, ref: h5py.Reference) -> str:
    """Safely extract string text from a MATLAB HDF5 object reference."""
    obj = f[ref]
    arr = obj[:].flatten().astype(int)
    return "".join(chr(c) for c in arr)

def main() -> None:
    print("=" * 60)
    print("  Vilnius Dataset Feature Extraction (Patient 001)")
    print("=" * 60)

    # 1. Verify files exist
    if not PPG_MAT_PATH.exists():
        print(f"Error: PPG file not found at {PPG_MAT_PATH}")
        sys.exit(1)
    if not ECG_MAT_PATH.exists():
        print(f"Error: ECG file not found at {ECG_MAT_PATH}")
        sys.exit(1)

    print("Opening 001_ECG.mat to load QRSindex and AF_annotation...")
    with h5py.File(ECG_MAT_PATH, "r") as f_ecg:
        ecg_day = get_hdf5_text(f_ecg, "recording_startday")
        ecg_time = get_hdf5_text(f_ecg, "recording_starttime")
        print(f"  ECG Recording Start: Day {ecg_day}, Time {ecg_time}")
        
        # Load QRS indices and convert to timestamps in seconds from ECG start
        print("  Loading QRS indices...")
        qrs_indices = f_ecg["QRSindex"][:].flatten()
        qrs_times = qrs_indices / FS_ECG
        
        # Load beat-to-beat labels (AF=1, Normal=0, other classes)
        print("  Loading AF annotations...")
        af_annotations = f_ecg["AF_annotation"][:].flatten()
        
        print(f"  Total beats: {len(qrs_times)} | AF beats: {int((af_annotations==1).sum())}")

    print("\nOpening 001_PPG.mat to extract PPG sessions...")
    features_list = []

    with h5py.File(PPG_MAT_PATH, "r") as f_ppg:
        n_sessions = len(f_ppg["recording_startday"])
        print(f"Found {n_sessions} PPG sessions.")

        for s_idx in range(n_sessions):
            s_day = get_hdf5_text(f_ppg, f_ppg["recording_startday"][s_idx, 0])
            s_time = get_hdf5_text(f_ppg, f_ppg["recording_starttime"][s_idx, 0])
            
            # Compute time offset of this PPG session relative to the continuous ECG start
            offset_sec = compute_offset_sec(ecg_day, ecg_time, s_day, s_time)
            print(f"\nProcessing Session {s_idx:02d}: Start Day={s_day}, Time={s_time} | Offset={offset_sec:.1f}s")
            
            # Extract PPG Green channel
            ref_ppg = f_ppg["PPG_GREEN"][s_idx, 0]
            ppg_raw = f_ppg[ref_ppg][0, :].astype(float)
            duration = len(ppg_raw) / FS_PPG
            print(f"  Samples: {len(ppg_raw)} | Duration: {duration:.1f}s ({duration/3600:.2f} hours)")

            # Preprocess signal (interpolate, band-pass 0.5-8Hz, z-score normalize at 100 Hz)
            try:
                ppg_proc = preprocess_ppg(ppg_raw, fs=int(FS_PPG))
            except Exception as e:
                print(f"  Failed preprocessing: {e}. Skipping session.")
                continue

            # Segment into 5.0s windows with 2.5s overlap (step is 2.5s)
            window_sec = 5.0
            overlap_sec = 2.5
            step_sec = window_sec - overlap_sec
            
            windows = segment_signal(ppg_proc, fs=int(FS_PPG), window_sec=window_sec, overlap_sec=overlap_sec)
            n_win = len(windows)
            if n_win == 0:
                print("  No windows generated. Skipping session.")
                continue

            print(f"  Generated {n_win} windows. Labeling windows and extracting features...")

            # Extract features for all windows at once
            feat_df = build_feature_matrix(windows, fs=int(FS_PPG))
            
            # Now, label each window using ECG annotations
            # Calculate absolute timeline (seconds from ECG start) for each window
            win_start_times = offset_sec + np.arange(n_win) * step_sec
            win_end_times = win_start_times + window_sec
            
            labels = []
            valid_mask = []
            
            for i in range(n_win):
                t_start = win_start_times[i]
                t_end = win_end_times[i]
                
                # Find indices of beats falling within this window
                # Use binary search for speed on the large sorted array
                left_idx = np.searchsorted(qrs_times, t_start)
                right_idx = np.searchsorted(qrs_times, t_end)
                
                window_beat_labels = af_annotations[left_idx:right_idx]
                
                if len(window_beat_labels) < 3:
                    # Too few beats in the window (e.g. bradycardia or lead-off), skip
                    labels.append(-1)
                    valid_mask.append(False)
                else:
                    # Find majority label in the window
                    unique, counts = np.unique(window_beat_labels, return_counts=True)
                    majority_lbl = unique[np.argmax(counts)]
                    
                    if majority_lbl in [0.0, 1.0]:  # Only keep Normal (0) or AF (1)
                        labels.append(int(majority_lbl))
                        valid_mask.append(True)
                    else:
                        # Skip if majority is PAC/PVC (2), uncertain (3), or noisy (5)
                        labels.append(-1)
                        valid_mask.append(False)

            # Filter out invalid windows
            valid_mask = np.array(valid_mask)
            if not valid_mask.any():
                print("  No valid normal or AF windows found in this session.")
                continue
                
            feat_df_valid = feat_df.iloc[valid_mask].copy()
            feat_df_valid["label"] = np.array(labels)[valid_mask]
            feat_df_valid["session_id"] = s_idx
            feat_df_valid["case_id"] = 1  # Patient 001
            
            # Print class count for this session
            n_normal_win = int((feat_df_valid["label"] == 0).sum())
            n_af_win = int((feat_df_valid["label"] == 1).sum())
            print(f"  Valid windows: {len(feat_df_valid)} (Normal: {n_normal_win} | AF: {n_af_win})")

            features_list.append(feat_df_valid)

    # 5. Consolidate and Save
    if features_list:
        df_all = pd.concat(features_list, ignore_index=True)
        
        # Ensure column ordering: identifiers first, then features, then label
        cols_id = ["case_id", "session_id"]
        cols_features = [c for c in df_all.columns if c not in cols_id + ["label"]]
        df_all = df_all[cols_id + cols_features + ["label"]]
        
        df_all.to_csv(OUTPUT_CSV, index=False)
        print("\n" + "=" * 60)
        print("  EXTRACTION COMPLETE")
        print("=" * 60)
        print(f"  Total records extracted : {len(df_all)}")
        print(f"  Normal windows (Label 0): {int((df_all['label']==0).sum())}")
        print(f"  AF windows (Label 1)    : {int((df_all['label']==1).sum())}")
        print(f"  Saved features CSV to   : {OUTPUT_CSV}")
    else:
        print("\nError: No valid features extracted from any session.")

if __name__ == "__main__":
    main()
