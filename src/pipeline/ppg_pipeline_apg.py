from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import cheby2, filtfilt

DEFAULT_FS = 125
DEFAULT_BANDPASS = (0.5, 8.0)

def cheby2_bandpass(signal: np.ndarray, fs: int, lowcut: float = 0.5, highcut: float = 8.0, order: int = 4, rs: int = 20) -> np.ndarray:
    """Apply a 4th-order Chebyshev Type II band-pass filter with 20dB stopband attenuation."""
    signal = np.asarray(signal, dtype=float)
    nyquist = 0.5 * fs
    
    # Ensure cuts are within Nyquist limit
    low = lowcut / nyquist
    high = highcut / nyquist
    if high >= 1.0:
        high = 0.99
        
    b, a = cheby2(order, rs, [low, high], btype="bandpass")
    return filtfilt(b, a, signal)

def preprocess_ppg_apg(signal: np.ndarray, fs: int) -> tuple[np.ndarray, np.ndarray]:
    """Preprocess PPG signal using the double Chebyshev 2 filtering from the paper.
    Returns:
        f_ppg: 1st filtered signal (0.5 - 10.0 Hz)
        s2_ppg: 2nd filtered signal (0.5 - 5.0 Hz) used for APG calculation
    """
    signal = np.asarray(signal, dtype=float)
    
    # Interpolate NaNs/Infs
    invalid = ~np.isfinite(signal)
    if invalid.any():
        valid_idx = np.flatnonzero(~invalid)
        if valid_idx.size == 0:
            raise ValueError("Signal contains no finite values.")
        signal = signal.copy()
        signal[invalid] = np.interp(np.flatnonzero(invalid), valid_idx, signal[valid_idx])
        
    # First filter: 0.5 - 10.0 Hz Chebyshev 2
    f_ppg = cheby2_bandpass(signal, fs=fs, lowcut=0.5, highcut=10.0)
    
    # Second filter: 0.5 - 5.0 Hz Chebyshev 2
    s2_ppg = cheby2_bandpass(f_ppg, fs=fs, lowcut=0.5, highcut=5.0)
    
    return f_ppg, s2_ppg

def compute_apg(s2_ppg: np.ndarray, fs: int) -> np.ndarray:
    """Compute the second derivative of the filtered PPG signal (APG)."""
    T = 1.0 / fs
    
    # First derivative S2' using central difference
    s2_prime = np.zeros_like(s2_ppg)
    s2_prime[1:-1] = (s2_ppg[2:] - s2_ppg[:-2]) / (2 * T)
    s2_prime[0] = (s2_ppg[1] - s2_ppg[0]) / T
    s2_prime[-1] = (s2_ppg[-1] - s2_ppg[-2]) / T
    
    # Second derivative APG using central difference on S2'
    apg = np.zeros_like(s2_prime)
    apg[1:-1] = (s2_prime[2:] - s2_prime[:-2]) / (2 * T)
    apg[0] = (s2_prime[1] - s2_prime[0]) / T
    apg[-1] = (s2_prime[-1] - s2_prime[-2]) / T
    
    return apg

def detect_apg_peaks_and_feet(
    signal_raw: np.ndarray, 
    fs: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Detect peaks and feet using the APG second-derivative double moving average (Elgendi).
    Returns:
        systolic_peaks: indices of Systolic Peaks in filtered signal
        foot_peaks: indices of Foot points in filtered signal
        idx_b_list: indices of wave b in APG
        idx_a_list: indices of wave a in APG
    """
    f_ppg, s2_ppg = preprocess_ppg_apg(signal_raw, fs)
    apg = compute_apg(s2_ppg, fs)
    
    # 1. Reverse APG to cancel wave a and highlight wave b
    F = -apg
    F[F < 0] = 0
    
    # 2. Square the signal
    g = F ** 2
    
    # 3. Generate double moving averages
    N1 = max(3, int(round(0.175 * fs)))  # 175 ms window
    N2 = max(5, int(round(1.000 * fs)))  # 1000 ms window
    
    ma_peak = pd.Series(g).rolling(window=N1, center=True, min_periods=1).mean().values
    ma_beat = pd.Series(g).rolling(window=N2, center=True, min_periods=1).mean().values
    
    # Thresholds
    alpha = 0.02 * np.mean(F)
    THR1 = ma_beat + alpha
    
    # Find blocks of interest where ma_peak > THR1
    blocks = ma_peak > THR1
    
    # Group consecutive True values into blocks
    block_starts = []
    block_ends = []
    in_block = False
    
    for i in range(len(blocks)):
        if blocks[i] and not in_block:
            block_starts.append(i)
            in_block = True
        elif not blocks[i] and in_block:
            block_ends.append(i)
            in_block = False
    if in_block:
        block_ends.append(len(blocks))
        
    idx_b_list = []
    
    # Filter blocks based on minimum width THR2 = N1 (175 ms)
    for start, end in zip(block_starts, block_ends):
        width = end - start
        if width >= N1:
            # Find index of wave b (maximum of F[n] inside block)
            idx_b = start + np.argmax(F[start:end])
            idx_b_list.append(idx_b)
            
    idx_b_list = np.array(idx_b_list)
    
    # 4. Trace back to find Systolic Peaks on filtered PPG S2
    # Search within 70-100ms (we use 100ms) to both sides of wave b's peak
    search_w = int(round(0.100 * fs))
    systolic_peaks = []
    for idx_b in idx_b_list:
        start_search = max(0, idx_b - search_w)
        end_search = min(len(s2_ppg), idx_b + search_w + 1)
        idx_peak = start_search + np.argmax(s2_ppg[start_search:end_search])
        systolic_peaks.append(idx_peak)
    systolic_peaks = np.array(systolic_peaks)
    
    # 5. Trace back to find wave a on APG (8 ms to 136 ms before wave b)
    min_k = int(round(0.008 * fs))
    max_k = int(round(0.136 * fs))
    idx_a_list = []
    for idx_b in idx_b_list:
        start_search = max(0, idx_b - max_k)
        end_search = max(0, idx_b - min_k)
        if start_search == end_search:
            idx_a = start_search
        else:
            # Sóng a là đỉnh cao nhất của APG trước sóng b
            idx_a = start_search + np.argmax(apg[start_search:end_search])
        idx_a_list.append(idx_a)
    idx_a_list = np.array(idx_a_list)
    
    # 6. Trace back to find the Foot of the signal on S2 (80ms - 120ms around wave a)
    search_foot_w = int(round(0.120 * fs))
    foot_peaks = []
    for idx_a in idx_a_list:
        start_search = max(0, idx_a - search_foot_w)
        end_search = min(len(s2_ppg), idx_a + search_foot_w + 1)
        # Chân sóng là điểm thấp nhất của S2 xung quanh sóng a
        idx_foot = start_search + np.argmin(s2_ppg[start_search:end_search])
        foot_peaks.append(idx_foot)
    foot_peaks = np.array(foot_peaks)
    
    return systolic_peaks, foot_peaks, idx_b_list, idx_a_list

def segment_and_extract_apg_features(
    signal_raw: np.ndarray,
    fs: int,
    window_sec: float = 5.0,
    overlap_sec: float = 2.5
) -> tuple[np.ndarray, pd.DataFrame]:
    """Segment signal and extract 14 features (9 standard + 5 APG/CT/GT features)."""
    # 1. Detections
    peaks, feet, _, _ = detect_apg_peaks_and_feet(signal_raw, fs)
    f_ppg, s2_ppg = preprocess_ppg_apg(signal_raw, fs)
    
    # Z-score normalize S2 for window features
    s2_mean = np.mean(s2_ppg)
    s2_std = np.std(s2_ppg)
    s2_norm = (s2_ppg - s2_mean) / (s2_std if s2_std > 0 else 1.0)
    
    # 2. Segment signal
    window_len = int(round(window_sec * fs))
    overlap_len = int(round(overlap_sec * fs))
    step = window_len - overlap_len
    
    if len(s2_norm) < window_len:
        return np.empty((0, window_len)), pd.DataFrame()
        
    windows_list = []
    features_list = []
    
    n_win = int((len(s2_norm) - window_len) // step) + 1
    
    for i in range(n_win):
        start_sample = i * step
        end_sample = start_sample + window_len
        
        window = s2_norm[start_sample:end_sample]
        windows_list.append(window)
        
        # Extract features for this window
        # Get peaks falling in this window
        win_peaks = peaks[(peaks >= start_sample) & (peaks < end_sample)] - start_sample
        win_feet = feet[(feet >= start_sample) & (feet < end_sample)] - start_sample
        
        # Calculate IBI
        win_ibi = np.diff(win_peaks) / fs if len(win_peaks) > 1 else np.array([])
        diff_ibi = np.diff(win_ibi) if len(win_ibi) > 1 else np.array([])
        
        # Standard features (9)
        feat_dict = {
            "signal_mean": float(np.mean(window)),
            "signal_std": float(np.std(window)),
            "signal_range": float(np.max(window) - np.min(window)),
            "signal_energy": float(np.mean(window**2)),
            "peak_count": float(len(win_peaks)),
            "ibi_mean": float(np.mean(win_ibi)) if len(win_ibi) else 0.0,
            "ibi_std": float(np.std(win_ibi)) if len(win_ibi) else 0.0,
            "ibi_rmssd": float(np.sqrt(np.mean(diff_ibi**2))) if len(diff_ibi) else 0.0,
            "ibi_cv": float(np.std(win_ibi) / np.mean(win_ibi)) if len(win_ibi) and np.mean(win_ibi) > 0 else 0.0,
        }
        
        # Morphological features (CT and GT) from the PDF paper
        # We match each peak with its preceding foot
        ct_vals = []
        gt_vals = []
        
        # Global indices for calculations
        glob_peaks = peaks[(peaks >= start_sample) & (peaks < end_sample)]
        for p in glob_peaks:
            # Find the closest preceding foot
            prec_feet = feet[feet < p]
            if len(prec_feet) > 0:
                f_prev = prec_feet[-1]
                # Crest Time (CT)
                ct = (p - f_prev) / fs
                # Check physiological sanity (CT usually 0.05s to 0.4s)
                if 0.03 <= ct <= 0.5:
                    ct_vals.append(ct)
                    
                    # GT (diastolic time to the next foot)
                    next_feet = feet[feet > p]
                    if len(next_feet) > 0:
                        f_next = next_feet[0]
                        gt = (f_next - p) / fs
                        if 0.1 <= gt <= 1.5:
                            gt_vals.append(gt)
                            
        # 5 New features
        ct_mean = float(np.mean(ct_vals)) if ct_vals else 0.0
        ct_std = float(np.std(ct_vals)) if len(ct_vals) > 1 else 0.0
        gt_mean = float(np.mean(gt_vals)) if gt_vals else 0.0
        gt_std = float(np.std(gt_vals)) if len(gt_vals) > 1 else 0.0
        
        # Ratio CT/GT
        if ct_mean > 0 and gt_mean > 0:
            ct_gt_ratio = ct_mean / gt_mean
        else:
            ct_gt_ratio = 0.0
            
        feat_dict.update({
            "ct_mean": ct_mean,
            "ct_std": ct_std,
            "gt_mean": gt_mean,
            "gt_std": gt_std,
            "ct_gt_ratio": ct_gt_ratio
        })
        
        features_list.append(feat_dict)
        
    return np.vstack(windows_list), pd.DataFrame(features_list)

def plot_apg_overview(signal_raw: np.ndarray, fs: int):
    """Plot raw PPG, filtered PPG S2 with peaks/feet, and the second derivative APG."""
    f_ppg, s2_ppg = preprocess_ppg_apg(signal_raw, fs)
    apg = compute_apg(s2_ppg, fs)
    peaks, feet, idx_b, idx_a = detect_apg_peaks_and_feet(signal_raw, fs)
    
    # Limit plotting to 10 seconds for clarity
    sec = min(10.0, len(signal_raw) / fs)
    n_samples = int(sec * fs)
    time_axis = np.arange(n_samples) / fs
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    # 1. Raw PPG
    axes[0].plot(time_axis, signal_raw[:n_samples], color="purple", label="Raw PPG", alpha=0.7)
    axes[0].set_title("Raw PPG Signal")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # 2. Filtered S2 with peaks and feet
    axes[1].plot(time_axis, s2_ppg[:n_samples], color="teal", label="Filtered S2 (0.5 - 5Hz)", linewidth=1.2)
    
    win_peaks = peaks[peaks < n_samples]
    win_feet = feet[feet < n_samples]
    axes[1].scatter(win_peaks / fs, s2_ppg[win_peaks], color="red", s=25, label="Systolic Peak", zorder=5)
    axes[1].scatter(win_feet / fs, s2_ppg[win_feet], color="blue", s=25, label="Foot point", zorder=5)
    axes[1].set_title("Processed PPG with Detected Systolic Peaks & Foot Points")
    axes[1].set_ylabel("Amplitude")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # 3. Second Derivative APG
    axes[2].plot(time_axis, apg[:n_samples], color="darkorchid", label="Second Derivative (APG)", linewidth=1.2)
    win_b = idx_b[idx_b < n_samples]
    win_a = idx_a[idx_a < n_samples]
    axes[2].scatter(win_b / fs, apg[win_b], color="red", marker="x", s=30, label="Wave b", zorder=5)
    axes[2].scatter(win_a / fs, apg[win_a], color="blue", marker="o", s=20, label="Wave a", zorder=5)
    axes[2].set_title("Second Derivative (APG) with Wave a and Wave b Peaks")
    axes[2].set_ylabel("APG (d2S/dt2)")
    axes[2].set_xlabel("Time (s)")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    plt.tight_layout()
    return fig, axes
