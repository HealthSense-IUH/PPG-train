from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks


DEFAULT_FS = 125
DEFAULT_BANDPASS = (0.5, 8.0)


def _read_ppg_csv(csv_path: Path, fs_fallback: int = DEFAULT_FS) -> dict[str, Any]:
    """Read a single MIMIC-format CSV (Time, PPG, resp) and return a record dict."""
    frame = pd.read_csv(csv_path)
    if "PPG" not in frame.columns:
        raise KeyError(f"Expected 'PPG' column in {csv_path}.")

    if "Time" in frame.columns and len(frame) > 1:
        dt = float(frame["Time"].iloc[1] - frame["Time"].iloc[0])
        fs = int(round(1.0 / dt)) if dt > 0 else fs_fallback
    else:
        fs = fs_fallback

    return {
        "ppg": {"fs": fs, "v": frame["PPG"].to_numpy(dtype=float)},
        "source_path": str(csv_path),
    }


def load_archive_dataset(archive_root: str | Path, fs_fallback: int = DEFAULT_FS) -> np.ndarray:
    """Load records from data/raw/mimic/af and data/raw/mimic/non-af folder structure.

    Expected layout::

        data/raw/mimic/
          af/
            mimic_perform_af_001_data.csv
            ...
          non-af/
            mimic_perform_non_af_001_data.csv
            ...

    Returns a numpy object array of record dicts with keys:
        ``ppg`` (dict with ``fs`` and ``v``), ``label`` (int), ``source_path`` (str).
    """
    archive_root = Path(archive_root)
    dataset_specs = [
        ("af",     "mimic_perform_af_*_data.csv",     1),
        ("non-af", "mimic_perform_non_af_*_data.csv", 0),
    ]

    records: list[dict[str, Any]] = []
    for folder_name, pattern, label in dataset_specs:
        folder = archive_root / folder_name
        if not folder.exists():
            raise FileNotFoundError(f"Expected folder not found: {folder}")
        for csv_path in sorted(folder.glob(pattern)):
            record = _read_ppg_csv(csv_path, fs_fallback=fs_fallback)
            record["label"] = label
            records.append(record)

    if not records:
        raise FileNotFoundError(
            f"No dataset CSV files found under {archive_root}. "
            "Check that af/ and non-af/ contain *_data.csv files."
        )

    return np.asarray(records, dtype=object)


def load_mimic_perform_csv_dataset(data_root: str | Path, fs_fallback: int = DEFAULT_FS) -> np.ndarray:
    """[Legacy] Load records from the old nested mimic_perform_*_csv folder structure.

    .. deprecated::
        Use :func:`load_archive_dataset` instead, which matches the actual
        ``data/raw/mimic/af`` and ``data/raw/mimic/non-af`` directory layout.
    """
    data_root = Path(data_root)
    dataset_specs = [
        ("mimic_perform_af_csv", "mimic_perform_af_csv", "mimic_perform_af_*_data.csv", 1),
        ("mimic_perform_non_af_csv", "mimic_perform_non_af_csv", "mimic_perform_non_af_*_data.csv", 0),
    ]

    records: list[dict[str, Any]] = []
    for outer_dir, inner_dir, pattern, label in dataset_specs:
        folder = data_root / outer_dir / inner_dir
        for csv_path in sorted(folder.glob(pattern)):
            record = _read_ppg_csv(csv_path, fs_fallback=fs_fallback)
            record["label"] = label
            records.append(record)

    if not records:
        raise FileNotFoundError(f"No dataset CSV files found under {data_root}.")

    return np.asarray(records, dtype=object)


def get_recording_fs(record: Any, default_fs: int = DEFAULT_FS) -> int:
    """Read the per-recording sampling rate, falling back to the project default."""
    if isinstance(record, dict):
        ppg = record.get("ppg", {})
        return int(ppg.get("fs", default_fs))
    ppg = getattr(record, "ppg", None)
    fs = getattr(ppg, "fs", default_fs)
    return int(fs)


def get_recording_signal(record: Any) -> np.ndarray:
    """Extract the raw PPG vector from a dataset record."""
    if isinstance(record, dict):
        ppg = record.get("ppg", {})
        values = ppg.get("v")
        if values is None:
            raise AttributeError("Record does not contain 'ppg.v'.")
        return np.asarray(values, dtype=float).reshape(-1)
    ppg = getattr(record, "ppg", None)
    values = getattr(ppg, "v", None)
    if values is None:
        raise AttributeError("Record does not contain 'ppg.v'.")
    return np.asarray(values, dtype=float).reshape(-1)


def get_recording_label(record: Any, label_attr: str = "label", default: int | None = None) -> int:
    """Read a binary label from a dataset record."""
    if isinstance(record, dict):
        if label_attr not in record:
            if default is None:
                raise AttributeError(f"Record does not contain '{label_attr}'.")
            return int(default)
        return int(np.asarray(record[label_attr]).squeeze())
    if not hasattr(record, label_attr):
        if default is None:
            raise AttributeError(f"Record does not contain '{label_attr}'.")
        return int(default)
    value = np.asarray(getattr(record, label_attr)).squeeze()
    return int(value)


def resample_to_target_fs(
    signal: np.ndarray,
    src_fs: int,
    target_fs: int = DEFAULT_FS,
) -> np.ndarray:
    """Resample a signal from *src_fs* to *target_fs* Hz.

    Uses :func:`scipy.signal.resample_poly` when the up/down ratio can be
    expressed as small integers (exact rational resampling — no aliasing).
    Falls back to linear interpolation via :func:`numpy.interp` for arbitrary
    ratios.

    Parameters
    ----------
    signal    : 1-D array of raw PPG samples
    src_fs    : original sampling rate (Hz)
    target_fs : desired sampling rate (Hz), default 125 Hz (MIMIC train fs)

    Returns
    -------
    resampled signal at *target_fs* Hz

    Examples
    --------
    Resample Huywatch 100 Hz → 125 Hz (ratio 5:4)::

        sig_125 = resample_to_target_fs(sig_100, src_fs=100, target_fs=125)
    """
    signal = np.asarray(signal, dtype=float).reshape(-1)

    if src_fs == target_fs:
        return signal.copy()

    try:
        from math import gcd
        from scipy.signal import resample_poly
        g    = gcd(int(target_fs), int(src_fs))
        up   = int(target_fs) // g
        down = int(src_fs)    // g
        return resample_poly(signal, up, down).astype(float)
    except Exception:
        # Fallback: linear interpolation on a uniform time grid
        n_src    = len(signal)
        duration = n_src / src_fs
        t_src    = np.arange(n_src) / src_fs
        n_dst    = int(round(duration * target_fs))
        t_dst    = np.arange(n_dst) / target_fs
        return np.interp(t_dst, t_src, signal)


def interpolate_invalid_values(signal: np.ndarray) -> np.ndarray:
    """Replace NaN/Inf values using linear interpolation."""
    signal = np.asarray(signal, dtype=float).copy()
    invalid = ~np.isfinite(signal)
    if not invalid.any():
        return signal

    valid_idx = np.flatnonzero(~invalid)
    invalid_idx = np.flatnonzero(invalid)
    if valid_idx.size == 0:
        raise ValueError("Signal contains no finite values to interpolate from.")

    signal[invalid_idx] = np.interp(invalid_idx, valid_idx, signal[valid_idx])
    return signal


def bandpass_filter(signal: np.ndarray, fs: int, lowcut: float = 0.5, highcut: float = 8.0, order: int = 3) -> np.ndarray:
    """Apply the project band-pass filter used for PPG preprocessing."""
    signal = np.asarray(signal, dtype=float)
    b, a = butter(order, [lowcut, highcut], btype="bandpass", fs=fs)
    return filtfilt(b, a, signal)


def zscore_normalize(signal: np.ndarray) -> np.ndarray:
    """Normalize a signal to zero mean and unit variance."""
    signal = np.asarray(signal, dtype=float)
    std = np.std(signal)
    if std == 0:
        return signal - np.mean(signal)
    return (signal - np.mean(signal)) / std


def preprocess_ppg(
    signal: np.ndarray,
    fs: int = DEFAULT_FS,
    lowcut: float = DEFAULT_BANDPASS[0],
    highcut: float = DEFAULT_BANDPASS[1],
    order: int = 3,
    target_fs: int | None = None,
) -> np.ndarray:
    """Preprocess a PPG signal with interpolation, filtering, normalization, and optional resampling.

    NOTE: If target_fs is provided and different from fs, the signal is resampled after
    interpolation, band-pass filtering, and z-score normalization to avoid DC offset/ringing
    artifacts in resampling.
    """
    clean_signal = interpolate_invalid_values(signal)
    if len(clean_signal) < 30:
        raise ValueError(
            f"Signal length ({len(clean_signal)}) is too short for bandpass filtering. "
            f"Must be at least 30 samples."
        )
    
    # Ensure highcut is strictly less than Nyquist frequency (fs / 2)
    nyquist = float(fs) / 2.0
    safe_highcut = highcut
    if safe_highcut >= nyquist:
        safe_highcut = nyquist - 0.5
        if safe_highcut <= lowcut:
            # Fallback if fs is extremely low
            safe_highcut = (lowcut + nyquist) / 2.0
            
    filtered_signal = bandpass_filter(clean_signal, fs=fs, lowcut=lowcut, highcut=safe_highcut, order=order)
    normalized = zscore_normalize(filtered_signal)
    
    if target_fs is not None and int(target_fs) != int(fs):
        return resample_to_target_fs(normalized, src_fs=int(fs), target_fs=int(target_fs))
    return normalized


def detect_beats(
    signal: np.ndarray,
    fs: int = DEFAULT_FS,
    min_peak_height_ratio: float = 0.5,
    min_peak_distance_sec: float = 0.4,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Detect systolic peaks and inter-beat intervals from a processed PPG signal."""
    signal = np.asarray(signal, dtype=float)
    peaks, properties = find_peaks(
        signal,
        height=min_peak_height_ratio * np.max(signal),
        distance=max(1, int(round(min_peak_distance_sec * fs))),
    )
    ibi = np.diff(peaks) / fs
    return peaks, ibi, properties


def segment_signal(
    signal: np.ndarray,
    fs: int = DEFAULT_FS,
    window_sec: float = 5.0,
    overlap_sec: float = 2.5,
) -> np.ndarray:
    """Split a signal into overlapping fixed-length windows."""
    signal = np.asarray(signal, dtype=float).reshape(-1)
    window_len = int(round(window_sec * fs))
    overlap_len = int(round(overlap_sec * fs))
    step = window_len - overlap_len
    if window_len <= 0:
        raise ValueError("window_sec must produce at least one sample.")
    if step <= 0:
        raise ValueError("overlap_sec must be smaller than window_sec.")
    if signal.size < window_len:
        return np.empty((0, window_len), dtype=float)

    windows = [signal[start : start + window_len] for start in range(0, signal.size - window_len + 1, step)]
    return np.vstack(windows)


def check_window_quality(window: np.ndarray, fs: int = DEFAULT_FS) -> bool:
    """Analyze a 5-second normalized PPG window and return True if quality is acceptable."""
    signal = np.asarray(window, dtype=float).reshape(-1)
    
    # 1. Check peak count (physiological bounds)
    # At 40 - 180 bpm, a 5-second window should have between 3 and 15 peaks.
    peaks, ibi, _ = detect_beats(signal, fs=fs)
    if len(peaks) < 3 or len(peaks) > 15:
        return False
        
    # 2. Check physiological limits of IBI
    # Heart rate shouldn't exceed 200 bpm (0.3s IBI) or go below 35 bpm (1.7s IBI)
    if len(ibi) > 0:
        if np.any(ibi < 0.3) or np.any(ibi > 1.7):
            return False
            
    # 3. Spectral Quality Index (SQI)
    # Calculate FFT power spectrum in the band of interest (0.5 - 8.0 Hz)
    n = len(signal)
    fft_vals = np.fft.rfft(signal)
    fft_freqs = np.fft.rfftfreq(n, d=1.0/fs)
    
    # Extract power
    power = np.abs(fft_vals)**2
    
    # Keep only the bandwidth of our filter (0.5 - 8.0 Hz)
    band_mask = (fft_freqs >= 0.5) & (fft_freqs <= 8.0)
    if not np.any(band_mask):
        return False
        
    freqs_band = fft_freqs[band_mask]
    power_band = power[band_mask]
    
    # Find dominant frequency peak in the heartbeat range (0.75 - 2.5 Hz, i.e., 45-150 bpm)
    hr_mask = (freqs_band >= 0.75) & (freqs_band <= 2.5)
    if not np.any(hr_mask):
        return False
        
    idx_peak = np.argmax(power_band[hr_mask])
    peak_freq = freqs_band[hr_mask][idx_peak]
    
    # Define a narrow band around the peak (peak_freq +- 0.25 Hz)
    peak_band_mask = (freqs_band >= (peak_freq - 0.25)) & (freqs_band <= (peak_freq + 0.25))
    
    power_total = np.sum(power_band)
    power_peak = np.sum(power_band[peak_band_mask])
    
    if power_total == 0:
        return False
        
    sqi_spectral = power_peak / power_total
    
    # If the dominant peak has less than 35% of the total band power, it is likely noise.
    if sqi_spectral < 0.35:
        return False
        
    return True


def extract_window_features(window: np.ndarray, fs: int = DEFAULT_FS) -> dict[str, float]:
    """Summarize one PPG window with rhythm and signal statistics."""
    signal = np.asarray(window, dtype=float).reshape(-1)
    peaks, ibi, _ = detect_beats(signal, fs=fs)
    diff_ibi = np.diff(ibi) if len(ibi) > 1 else np.array([])
    return {
        "signal_mean": float(np.mean(signal)),
        "signal_std": float(np.std(signal)),
        "signal_range": float(np.max(signal) - np.min(signal)),
        "signal_energy": float(np.mean(signal**2)),
        "peak_count": float(len(peaks)),
        "ibi_mean": float(np.mean(ibi)) if len(ibi) else 0.0,
        "ibi_std": float(np.std(ibi)) if len(ibi) else 0.0,
        "ibi_rmssd": float(np.sqrt(np.mean(diff_ibi**2))) if len(diff_ibi) else 0.0,
        "ibi_cv": float(np.std(ibi) / np.mean(ibi)) if len(ibi) and np.mean(ibi) > 0 else 0.0,
    }


def build_feature_matrix(windows: np.ndarray, fs: int = DEFAULT_FS) -> pd.DataFrame:
    """Convert segmented windows into a tabular feature matrix."""
    windows = np.asarray(windows)
    if windows.ndim != 2 or len(windows) == 0:
        raise ValueError("windows must be a non-empty 2D array.")
    return pd.DataFrame([extract_window_features(window, fs=fs) for window in windows])


def build_window_dataset(
    records: Any,
    *,
    fs_fallback: int = DEFAULT_FS,
    window_sec: float = 5.0,
    overlap_sec: float = 2.5,
    label_attr: str = "label",
    default_label: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Preprocess and segment every record into a model-ready window dataset."""
    x_windows: list[np.ndarray] = []
    y_windows: list[int] = []

    for record in np.atleast_1d(records):
        fs = get_recording_fs(record, default_fs=fs_fallback)
        raw_signal = get_recording_signal(record)
        processed_signal = preprocess_ppg(raw_signal, fs=fs)
        windows = segment_signal(processed_signal, fs=fs, window_sec=window_sec, overlap_sec=overlap_sec)
        if windows.size == 0:
            continue

        label = get_recording_label(record, label_attr=label_attr, default=default_label)
        x_windows.extend(windows)
        y_windows.extend([label] * len(windows))

    if not x_windows:
        return np.empty((0, 0), dtype=float), np.empty((0,), dtype=int)

    return np.asarray(x_windows, dtype=float), np.asarray(y_windows, dtype=int)


def plot_signal_overview(
    raw_signal: np.ndarray,
    processed_signal: np.ndarray,
    fs: int,
    peaks: np.ndarray | None = None,
):
    """Plot raw and processed PPG, optionally with detected peaks."""
    raw_signal = np.asarray(raw_signal).reshape(-1)
    processed_signal = np.asarray(processed_signal).reshape(-1)
    time_axis = np.arange(raw_signal.size) / fs

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(time_axis, raw_signal, label="Raw PPG", alpha=0.5)
    ax.plot(time_axis[: processed_signal.size], processed_signal, label="Processed PPG", linewidth=1.2)
    if peaks is not None and len(peaks) > 0:
        ax.scatter(peaks / fs, processed_signal[peaks], color="red", s=20, label="Detected peaks")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("PPG Signal Overview")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig, ax


def plot_windows(windows: np.ndarray, max_windows: int = 3):
    """Plot a few segmented windows for quick visual inspection."""
    windows = np.asarray(windows)
    if windows.ndim != 2 or len(windows) == 0:
        raise ValueError("windows must be a non-empty 2D array.")

    count = min(max_windows, len(windows))
    fig, axes = plt.subplots(count, 1, figsize=(12, 3 * count), sharex=False)
    if count == 1:
        axes = [axes]

    for idx, axis in enumerate(axes):
        axis.plot(windows[idx])
        axis.set_title(f"Window {idx + 1}")
        axis.set_xlabel("Samples")
        axis.set_ylabel("Normalized PPG")
        axis.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig, axes
