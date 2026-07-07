"""Run AF detection inference on a Huywatch PPG CSV file.

Huywatch format:  device_millis, red, ir

Usage
-----
    python code/predict_huywatch.py
    python code/predict_huywatch.py --file "archive/huywatch-ppg-20260623-132651 (1).csv"
    python code/predict_huywatch.py --channel red --threshold 0.6

Outputs
-------
    outputs/huywatch_predictions.csv   – per-window predictions
    outputs/huywatch_summary.txt       – overall summary report
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT  = Path(__file__).resolve().parent.parent.parent
DEFAULT_FILE  = PROJECT_ROOT / "data" / "raw" / "huywatch" / "huywatch-ppg-20260623-132651 (1).csv"
MODEL_PATH    = PROJECT_ROOT / "models" / "ppg_af_rf.joblib"
PREDICTIONS_DIR = PROJECT_ROOT / "data" / "predictions"
REPORTS_DIR   = PROJECT_ROOT / "reports"

PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJECT_ROOT / "src"))
from pipeline.ppg_pipeline import (  # noqa: E402
    bandpass_filter,
    build_feature_matrix,
    detect_beats,
    interpolate_invalid_values,
    resample_to_target_fs,
    segment_signal,
    zscore_normalize,
)

TRAIN_FS = 125  # Hz — must match the fs used during model training

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def print_section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def load_huywatch(csv_path: Path, channel: str = "ir") -> tuple[np.ndarray, float]:
    """Load Huywatch CSV and return (signal, fs).

    Parameters
    ----------
    csv_path : path to the Huywatch CSV file
    channel  : 'ir' (recommended) or 'red'

    Returns
    -------
    signal : raw ADC values for the chosen channel
    fs     : estimated sampling rate in Hz
    """
    df = pd.read_csv(csv_path)

    required = {"device_millis", channel}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Expected columns {required} in {csv_path}. "
            f"Found: {list(df.columns)}"
        )

    t_ms    = df["device_millis"].values.astype(float)
    # Invert signal polarity to match training convention (systolic peaks pointing upwards)
    signal  = -df[channel].values.astype(float)
    t_sec   = (t_ms - t_ms[0]) / 1000.0
    dt_arr  = np.diff(t_sec)
    fs      = float(1.0 / np.median(dt_arr))
    return signal, fs, t_sec


def preprocess_huywatch(signal: np.ndarray, fs: float) -> np.ndarray:
    """Preprocess Huywatch signal: interpolate -> bandpass -> zscore.
    
    NOTE: bandpass is applied at the ORIGINAL fs before resampling
    so the filter removes the large DC offset (~192k counts) first.
    The returned signal is bandpass-filtered and z-score normalized.
    """
    clean    = interpolate_invalid_values(signal)
    filtered = bandpass_filter(clean, fs=int(round(fs)), lowcut=0.5, highcut=min(8.0, float(fs)/2 - 0.5))
    return zscore_normalize(filtered)


def summarize_predictions(
    pred_labels: np.ndarray,
    pred_proba:  np.ndarray,
    window_sec:  float,
) -> dict:
    """Compute summary statistics over all window predictions."""
    n_total    = len(pred_labels)
    n_af       = int((pred_labels == 1).sum())
    n_non_af   = int((pred_labels == 0).sum())
    af_pct     = 100.0 * n_af / n_total if n_total > 0 else 0.0
    mean_prob  = float(np.mean(pred_proba)) * 100.0
    max_prob   = float(np.max(pred_proba))  * 100.0

    # Majority vote
    majority = "AF" if n_af > n_non_af else "Non-AF"

    # Consecutive AF windows (longest run)
    max_consecutive = 0
    cur_run = 0
    for lbl in pred_labels:
        if lbl == 1:
            cur_run += 1
            max_consecutive = max(max_consecutive, cur_run)
        else:
            cur_run = 0

    return {
        "n_windows"           : n_total,
        "n_AF_windows"        : n_af,
        "n_NonAF_windows"     : n_non_af,
        "AF_percentage"       : round(af_pct, 2),
        "mean_AF_probability" : round(mean_prob, 2),
        "max_AF_probability"  : round(max_prob, 2),
        "majority_vote"       : majority,
        "max_consecutive_AF_windows" : max_consecutive,
        "max_consecutive_AF_sec"     : round(max_consecutive * window_sec, 1),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:

    csv_path  = Path(args.file)
    channel   = args.channel
    threshold = args.threshold
    window_sec  = args.window_sec
    overlap_sec = args.overlap_sec

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    print_section("1. Loading model")
    if not MODEL_PATH.exists():
        print(f"  ERROR: Model not found at {MODEL_PATH}")
        print("  Run: python code/train_af_model.py")
        sys.exit(1)
    clf = joblib.load(MODEL_PATH)
    print(f"  Model loaded : {MODEL_PATH}")
    print(f"  Type         : {type(clf).__name__}")

    # ------------------------------------------------------------------
    # 2. Load Huywatch data
    # ------------------------------------------------------------------
    print_section("2. Loading Huywatch data")
    if not csv_path.exists():
        print(f"  ERROR: File not found: {csv_path}")
        sys.exit(1)

    signal_raw, fs, t_sec = load_huywatch(csv_path, channel=channel)
    duration = t_sec[-1]

    print(f"  File     : {csv_path.name}")
    print(f"  Channel  : {channel.upper()}")
    print(f"  Samples  : {len(signal_raw)}")
    print(f"  Duration : {duration:.1f} s  ({duration/60:.1f} min)")
    print(f"  fs       : {fs:.1f} Hz")
    print(f"  Signal   : min={signal_raw.min():.0f}  max={signal_raw.max():.0f}  "
          f"mean={signal_raw.mean():.0f}  range={signal_raw.max()-signal_raw.min():.0f}")

    # ------------------------------------------------------------------
    # 3. Preprocess at original fs (bandpass removes DC offset first)
    # ------------------------------------------------------------------
    print_section("3. Preprocessing at original fs")
    signal_proc_src = preprocess_huywatch(signal_raw, fs)
    print(f"  Interpolate invalid values : done")
    print(f"  Bandpass filter (0.5-8 Hz) : done (at {int(round(fs))} Hz)")
    print(f"  Z-score normalize          : done")
    print(f"  Processed range : [{signal_proc_src.min():.3f}, {signal_proc_src.max():.3f}]")

    # ------------------------------------------------------------------
    # 4. Resample preprocessed signal to TRAIN_FS (125 Hz)
    # ------------------------------------------------------------------
    print_section("4. Resampling to training fs")
    fs_src = int(round(fs))
    if fs_src != TRAIN_FS:
        signal_proc = resample_to_target_fs(signal_proc_src, src_fs=fs_src, target_fs=TRAIN_FS)
        print(f"  {fs_src} Hz  ->  {TRAIN_FS} Hz  (ratio {TRAIN_FS}/{fs_src})")
        print(f"  Samples before: {len(signal_proc_src)}  ->  after: {len(signal_proc)}")
        print(f"  Resampling after bandpass avoids DC-offset ringing artifacts.")
    else:
        signal_proc = signal_proc_src
        print(f"  Already at {TRAIN_FS} Hz -- no resampling needed.")
    fs_model = TRAIN_FS

    # Beat detection on resampled+processed signal
    peaks, ibi, _ = detect_beats(signal_proc, fs=fs_model)
    print(f"\n  Beats detected : {len(peaks)}")
    if len(ibi) > 0:
        hr_mean = 60.0 / float(np.mean(ibi))
        hr_min  = 60.0 / float(np.max(ibi))
        hr_max  = 60.0 / float(np.min(ibi))
        print(f"  HR mean        : {hr_mean:.1f} bpm  (range: {hr_min:.1f} - {hr_max:.1f})")
        print(f"  IBI mean       : {np.mean(ibi)*1000:.1f} ms")
        print(f"  IBI std        : {np.std(ibi)*1000:.1f} ms")
        print(f"  IBI CV         : {np.std(ibi)/np.mean(ibi):.4f}")
        rmssd = float(np.sqrt(np.mean(np.diff(ibi)**2))) * 1000 if len(ibi) > 1 else 0.0
        print(f"  RMSSD          : {rmssd:.1f} ms")

    # ------------------------------------------------------------------
    # 5. Segment & extract features (at 125 Hz)
    # ------------------------------------------------------------------
    print_section("5. Segmenting & extracting features")
    windows = segment_signal(
        signal_proc, fs=fs_model, window_sec=window_sec, overlap_sec=overlap_sec
    )
    if len(windows) == 0:
        print(f"  ERROR: Signal too short for windowing.")
        print(f"  Need at least {window_sec}s, got {duration:.1f}s")
        sys.exit(1)

    X_feat = build_feature_matrix(windows, fs=fs_model)
    n_win  = len(windows)
    print(f"  Window size     : {window_sec}s  ({int(round(window_sec * fs_model))} samples at {fs_model}Hz)")
    print(f"  Overlap         : {overlap_sec}s")
    print(f"  Windows created : {n_win}")
    print(f"  Feature matrix  : {X_feat.shape}")

    # ------------------------------------------------------------------
    # 6. Inference
    # ------------------------------------------------------------------
    print_section("6. Running inference")
    pred_proba  = clf.predict_proba(X_feat)[:, 1]
    pred_labels = (pred_proba >= threshold).astype(int)

    print(f"  Threshold       : {threshold:.2f}")
    print(f"  AF windows      : {int((pred_labels==1).sum())} / {n_win}")
    print(f"  Non-AF windows  : {int((pred_labels==0).sum())} / {n_win}")

    # ------------------------------------------------------------------
    # 7. Build per-window output
    # ------------------------------------------------------------------
    step_sec    = window_sec - overlap_sec
    start_times = np.array([i * step_sec for i in range(n_win)])
    end_times   = start_times + window_sec

    df_pred = pd.DataFrame({
        "Window_Index"    : np.arange(n_win),
        "Start_Time_Sec"  : np.round(start_times, 2),
        "End_Time_Sec"    : np.round(end_times,   2),
        "AF_Prediction"   : pred_labels,
        "AF_Probability"  : np.round(pred_proba, 4),
        "Label"           : ["AF" if l == 1 else "Non-AF" for l in pred_labels],
    })

    # ------------------------------------------------------------------
    # 8. Summary
    # ------------------------------------------------------------------
    print_section("7. Summary")
    summary = summarize_predictions(pred_labels, pred_proba, window_sec)

    print(f"  Total windows          : {summary['n_windows']}")
    print(f"  AF windows             : {summary['n_AF_windows']}  ({summary['AF_percentage']:.1f}%)")
    print(f"  Non-AF windows         : {summary['n_NonAF_windows']}")
    print(f"  Mean AF probability    : {summary['mean_AF_probability']:.1f}%")
    print(f"  Max  AF probability    : {summary['max_AF_probability']:.1f}%")
    print(f"  Majority vote          : {summary['majority_vote']}")
    print(f"  Max consecutive AF     : {summary['max_consecutive_AF_windows']} windows  "
          f"({summary['max_consecutive_AF_sec']}s)")
    print()

    # Clinical-style alert
    af_pct = summary["AF_percentage"]
    if af_pct >= 50:
        print("  [!] CAUTION : >50% windows classified as AF")
        print("       -> Possible atrial fibrillation detected.")
        print("       -> Please consult a physician for confirmation.")
    elif af_pct >= 20:
        print("  [~] BORDERLINE : 20-50% windows classified as AF")
        print("       -> Occasional irregular rhythm detected.")
        print("       -> Consider repeat measurement.")
    else:
        print("  [OK] LOW RISK : <20% windows classified as AF")
        print("       -> Mostly regular rhythm in this recording.")

    print()
    print("  NOTE: This is a research prototype, NOT a medical device.")
    print("        Model was trained on MIMIC data (bedside monitor),")
    print("        not validated for Huywatch wearable. Use with caution.")

    # ------------------------------------------------------------------
    # 9. Save outputs
    # ------------------------------------------------------------------
    print_section("8. Saving outputs")

    pred_path = PREDICTIONS_DIR / "huywatch_predictions.csv"
    df_pred.to_csv(pred_path, index=False)
    print(f"  Predictions CSV : {pred_path}")

    # Summary txt
    summary_lines = [
        "Huywatch AF Detection — Inference Summary",
        "=" * 50,
        f"File      : {csv_path.name}",
        f"Channel   : {channel.upper()}",
        f"src fs    : {fs_src} Hz  ->  resampled: {fs_model} Hz",
        f"Duration  : {duration:.1f} s",
        f"Threshold : {threshold:.2f}",
        "",
        f"Windows total     : {summary['n_windows']}",
        f"AF windows        : {summary['n_AF_windows']} ({summary['AF_percentage']:.1f}%)",
        f"Non-AF windows    : {summary['n_NonAF_windows']}",
        f"Mean AF prob      : {summary['mean_AF_probability']:.1f}%",
        f"Max  AF prob      : {summary['max_AF_probability']:.1f}%",
        f"Majority vote     : {summary['majority_vote']}",
        f"Max consec. AF    : {summary['max_consecutive_AF_windows']} windows ({summary['max_consecutive_AF_sec']}s)",
        "",
        "Per-window (first 20):",
        f"  {'Win':>4}  {'Start':>7}  {'End':>7}  {'Label':>7}  {'AF Prob':>8}",
        f"  {'----':>4}  {'-------':>7}  {'-------':>7}  {'-------':>7}  {'--------':>8}",
    ]
    for _, row in df_pred.head(20).iterrows():
        summary_lines.append(
            f"  {int(row['Window_Index']):>4}  {row['Start_Time_Sec']:>7.1f}s "
            f"{row['End_Time_Sec']:>7.1f}s  {row['Label']:>7}  {row['AF_Probability']:>8.4f}"
        )

    summary_path = REPORTS_DIR / "huywatch_summary.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"  Summary TXT     : {summary_path}")

    print_section("DONE")
    print(f"  Predictions : {pred_path}")
    print(f"  Summary     : {summary_path}")
    print(f"  src fs: {fs_src} Hz  ->  resampled to {fs_model} Hz (train fs)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AF inference on Huywatch PPG data")
    parser.add_argument(
        "--file", type=str,
        default=str(DEFAULT_FILE),
        help="Path to Huywatch CSV file (default: archive/huywatch-ppg-...csv)",
    )
    parser.add_argument(
        "--channel", type=str, default="ir", choices=["ir", "red"],
        help="PPG channel to use: 'ir' (recommended) or 'red' (default: ir)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Classification threshold for AF (default: 0.5)",
    )
    parser.add_argument(
        "--window_sec", type=float, default=5.0,
        help="Window length in seconds (default: 5.0)",
    )
    parser.add_argument(
        "--overlap_sec", type=float, default=2.5,
        help="Overlap between windows in seconds (default: 2.5)",
    )
    args = parser.parse_args()
    main(args)
