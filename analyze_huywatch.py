import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

df = pd.read_csv("archive/huywatch-ppg-20260623-132651 (1).csv")
t_ms = df["device_millis"].values
ir = df["ir"].values.astype(float)
red = df["red"].values.astype(float)

# Timing
t_sec = (t_ms - t_ms[0]) / 1000.0
dt_arr = np.diff(t_sec)
fs = 1.0 / np.median(dt_arr)
print(f"Estimated fs : {fs:.1f} Hz")
print(f"dt jitter (std): {np.std(dt_arr)*1000:.2f} ms")
print(f"Max gap      : {np.max(dt_arr)*1000:.1f} ms")
print(f"Duration     : {t_sec[-1]:.1f} s")
print(f"N samples    : {len(df)}")
print()

# Bandpass 0.5-8Hz
def bandpass(sig, fs, lo=0.5, hi=8.0, order=3):
    b, a = butter(order, [lo, hi], btype="bandpass", fs=fs)
    return filtfilt(b, a, sig)

ir_bp = bandpass(ir, fs)
red_bp = bandpass(red, fs)

# Normalize
def zscore(s):
    return (s - s.mean()) / s.std()

ir_z = zscore(ir_bp)
red_z = zscore(red_bp)

# Peak detection
min_dist = int(0.4 * fs)
peaks_ir, _ = find_peaks(ir_z, height=0.5 * np.max(ir_z), distance=min_dist)
peaks_red, _ = find_peaks(red_z, height=0.5 * np.max(red_z), distance=min_dist)

print("=== After bandpass + zscore ===")
print(f"IR  peaks found : {len(peaks_ir)}")
print(f"RED peaks found : {len(peaks_red)}")
print()

if len(peaks_ir) > 1:
    ibi_ir = np.diff(peaks_ir) / fs
    hr_ir = 60.0 / ibi_ir
    rmssd = np.sqrt(np.mean(np.diff(ibi_ir)**2)) * 1000
    pnn50 = np.sum(np.abs(np.diff(ibi_ir)) > 0.05) / len(np.diff(ibi_ir)) * 100
    print("=== IR Inter-beat Intervals ===")
    print(f"  IBI mean  : {ibi_ir.mean():.3f} s")
    print(f"  IBI std   : {ibi_ir.std():.3f} s")
    print(f"  IBI CV    : {ibi_ir.std()/ibi_ir.mean():.3f}")
    print(f"  HR mean   : {hr_ir.mean():.1f} bpm")
    print(f"  HR range  : {hr_ir.min():.1f} - {hr_ir.max():.1f} bpm")
    print(f"  RMSSD     : {rmssd:.1f} ms")
    print(f"  pNN50     : {pnn50:.1f}%")
    print()

if len(peaks_red) > 1:
    ibi_red = np.diff(peaks_red) / fs
    hr_red = 60.0 / ibi_red
    print("=== RED Inter-beat Intervals ===")
    print(f"  IBI mean  : {ibi_red.mean():.3f} s")
    print(f"  IBI std   : {ibi_red.std():.3f} s")
    print(f"  IBI CV    : {ibi_red.std()/ibi_red.mean():.3f}")
    print(f"  HR mean   : {hr_red.mean():.1f} bpm")
    print(f"  HR range  : {hr_red.min():.1f} - {hr_red.max():.1f} bpm")
    print()

# Polarity check
ir_z_inv = -ir_z
peaks_inv, _ = find_peaks(ir_z_inv, height=0.5 * np.max(ir_z_inv), distance=min_dist)
print("=== Peak polarity check (IR) ===")
print(f"  Normal peaks  : {len(peaks_ir)}")
print(f"  Inverted peaks: {len(peaks_inv)}")
if len(peaks_ir) >= len(peaks_inv):
    print("  -> NORMAL polarity (peaks up)")
else:
    print("  -> INVERTED signal (peaks are troughs, need to flip!)")

# Perfusion Index
DC_ir = ir.mean()
AC_ir = ir_bp.std() * 2
pi = AC_ir / DC_ir * 100
print()
print("=== Perfusion Index (PI) ===")
print(f"  PI (IR) = {pi:.4f}%")
print("  Typical: 0.02% (poor) to 20% (excellent)")
if pi < 0.2:
    assessment = "POOR - weak pulse or sensor contact issue"
elif pi < 1.0:
    assessment = "FAIR - acceptable for monitoring"
elif pi < 5.0:
    assessment = "GOOD"
else:
    assessment = "EXCELLENT"
print(f"  Assessment: {assessment}")

# SpO2 estimate (R ratio)
AC_red = red_bp.std() * 2
DC_red = red.mean()
R = (AC_red / DC_red) / (AC_ir / DC_ir)
spo2_est = 110 - 25 * R  # empirical formula
print()
print("=== SpO2 estimate (rough) ===")
print(f"  R ratio = {R:.4f}")
print(f"  SpO2 ~  {spo2_est:.1f}% (rough estimate, not calibrated)")

# Noise floor check: std in flat region between beats
print()
print("=== Signal-to-Noise ===")
print(f"  IR  AC amplitude (std*2) : {AC_ir:.1f} counts")
print(f"  RED AC amplitude (std*2) : {AC_red:.1f} counts")
print(f"  IR  DC                   : {DC_ir:.0f} counts")
print(f"  RED DC                   : {DC_red:.0f} counts")

# Window-level SNR
window_snr = []
wlen = int(5 * fs)  # 5s window
for start in range(0, len(ir) - wlen, wlen):
    seg = ir_bp[start:start+wlen]
    if len(peaks_ir[(peaks_ir >= start) & (peaks_ir < start+wlen)]) > 1:
        pks = peaks_ir[(peaks_ir >= start) & (peaks_ir < start+wlen)] - start
        signal_power = np.max(seg[pks]) - np.min(seg)
        # noise: residual after peaks
        noise_floor = np.std(seg)
        window_snr.append(signal_power / (noise_floor + 1e-9))

if window_snr:
    print(f"  Window SNR (avg): {np.mean(window_snr):.1f}")
    print(f"  Window SNR (min): {np.min(window_snr):.1f}")
    print(f"  Window SNR (max): {np.max(window_snr):.1f}")
