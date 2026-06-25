import io
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="PPG APG AF Classifier", layout="wide")

# Add code folder to path and import pipeline functions
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ppg_pipeline_apg import (
    segment_and_extract_apg_features,
    detect_apg_peaks_and_feet,
    preprocess_ppg_apg,
)
from ppg_pipeline import check_window_quality

TRAIN_FS = 125  # Hz — must match the fs used during model training

# -----------------------------
# Custom CSS for Premium Dark Theme
# -----------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Main App styling */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%) !important;
        font-family: 'Inter', sans-serif !important;
        color: #f1f5f9 !important;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #030712 !important;
        border-right: 1px solid #312e81 !important;
    }
    section[data-testid="stSidebar"] h2 {
        color: #f8fafc !important;
    }

    /* Headings styling */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif !important;
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    
    /* Main title gradient */
    h1 {
        background: linear-gradient(90deg, #38bdf8 0%, #a78bfa 100%) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        font-size: 2.5rem !important;
        padding-bottom: 0.5rem !important;
    }

    /* Markdown paragraph text visibility */
    .stMarkdown p {
        color: #cbd5e1 !important;
        font-size: 1rem !important;
    }

    /* File uploader and dropzone */
    div[data-testid="stFileUploader"] {
        background-color: #1e1b4b !important;
        border: 1px dashed #6366f1 !important;
        border-radius: 8px !important;
        padding: 10px !important;
    }

    /* Metrics and labels */
    div[data-testid="stWidgetLabel"] p {
        color: #f8fafc !important;
        font-weight: 500 !important;
    }
    
    /* Notification styling overrides */
    div[data-testid="stNotification"] {
        background-color: #0f172a !important;
        border: 1px solid #4338ca !important;
        border-radius: 8px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Load trained model
# -----------------------------
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODEL_DIR / "ppg_af_rf_apg.joblib"

model = None
model_load_error = None
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    model_load_error = f"Random Forest APG model not found at {MODEL_PATH}. Run train_apg_model.py first."

# -----------------------------
# Sidebar for settings
# -----------------------------
st.sidebar.header("Segmentation Settings")
window_sec  = st.sidebar.slider("Window length (sec)", 1, 20, 5, key="window_slider")
overlap_max = float(window_sec) - 0.5
overlap_sec = st.sidebar.slider(
    "Overlap length (sec)", 0.0, overlap_max,
    min(2.5, overlap_max - 0.5),
    step=0.5,
    key="overlap_slider",
)

st.sidebar.header("Tuning Settings (MIMIC compensation)")
prob_threshold = st.sidebar.slider(
    "Ngưỡng xác suất AF (Decision Threshold)",
    0.50, 0.95, 0.80, 0.05,
    key="prob_threshold_slider",
    help="Ngưỡng xác suất tối thiểu của mô hình để phân loại là AF. Tăng lên giúp giảm báo sai do nhiễu."
)

use_voting = st.sidebar.checkbox(
    "Bộ lọc tích lũy (Voting Filter)",
    value=True,
    key="use_voting_checkbox",
    help="Chỉ báo AF nếu nhịp bất thường kéo dài liên tục qua nhiều cửa sổ."
)

if use_voting:
    vote_window = st.sidebar.slider(
        "Cửa sổ lọc (Voting Window Size)",
        3, 15, 5, 2,
        key="vote_window_slider",
        help="Số lượng cửa sổ sạch liên tiếp tham gia biểu quyết."
    )
    vote_ratio = st.sidebar.slider(
        "Tỷ lệ đồng thuận (Vote Ratio)",
        0.50, 1.00, 0.60, 0.10,
        key="vote_ratio_slider",
        help="Tỷ lệ cửa sổ bị AF tối thiểu trong nhóm để kích hoạt cảnh báo."
    )
else:
    vote_window = 1
    vote_ratio = 1.0

# -----------------------------
# Main UI
# -----------------------------
st.title("PPG APG + Morphological AF Classifier")
st.write("Triển khai mô hình Random Forest chẩn đoán AF từ **14 đặc trưng** kết hợp phân tích hình thái đạo hàm bậc hai (APG, CT, GT) theo tài liệu ICSET 2023.")

if model_load_error:
    st.error(model_load_error)
    st.stop()

uploaded_file = st.file_uploader("Choose a PPG file", type=["csv"], key="ppg_file_uploader")

if uploaded_file is not None:
    # ------------------------------------------------------------------
    # Load & detect file format
    # ------------------------------------------------------------------
    df = pd.read_csv(uploaded_file)
    cols = [c.lower() for c in df.columns]

    # Detect format and extract PPG signal + sampling rate
    channel_choice = None
    if "ppg" in [c for c in df.columns]:
        # MIMIC format: Time, PPG, resp
        ppg_raw = df["PPG"].values.astype(float)
        if "Time" in df.columns and len(df) > 1:
            dt = float(df["Time"].iloc[1] - df["Time"].iloc[0])
            fs_detected = int(round(1.0 / dt)) if dt > 0 else 125
        else:
            fs_detected = 125
        format_label = "MIMIC (Time, PPG, resp)"

    elif "ppg" in cols and "ppg" not in [c for c in df.columns]:
        # merged dataset: time, ppg, resp, status (lowercase)
        ppg_col = df.columns[[c.lower() == "ppg" for c in df.columns]][0]
        ppg_raw = df[ppg_col].values.astype(float)
        fs_detected = 125
        format_label = "Merged dataset (lowercase ppg)"

    elif "device_millis" in df.columns and ("ir" in df.columns or "red" in df.columns):
        # Huywatch format: device_millis, red, ir
        available_channels = [c for c in ["ir", "red"] if c in df.columns]
        channel_choice = st.sidebar.selectbox(
            "Huywatch channel",
            options=available_channels,
            index=0,
            key="channel_select",
        )
        ppg_raw = -df[channel_choice].values.astype(float)
        t_ms = df["device_millis"].values
        t_sec = (t_ms - t_ms[0]) / 1000.0
        dt_arr = np.diff(t_sec)
        fs_detected = int(round(1.0 / float(np.median(dt_arr))))
        format_label = f"Huywatch (channel: {channel_choice}, fs≈{fs_detected} Hz - Đã nghịch đảo)"

    else:
        # Unknown: fallback to first numeric column
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            st.error("Không tìm thấy cột số nào trong file CSV.")
            st.stop()
        ppg_raw = df[numeric_cols[0]].values.astype(float)
        fs_detected = 125
        format_label = f"Unknown (dùng cột '{numeric_cols[0]}', fs mặc định 125 Hz)"

    st.info(
        f"Format: **{format_label}** | "
        f"src fs = **{fs_detected} Hz** "
        f"{'-> resampled to **125 Hz** (train fs)' if fs_detected != TRAIN_FS else '(train fs)'} | "
        f"{len(ppg_raw)} samples"
    )

    # ------------------------------------------------------------------
    # Preprocess and Resample to TRAIN_FS (125 Hz)
    # ------------------------------------------------------------------
    fs_use = TRAIN_FS
    # If original fs is not 125 Hz, resample raw signal first
    if fs_detected != fs_use:
        from ppg_pipeline import resample_to_target_fs
        ppg_resampled = resample_to_target_fs(ppg_raw, src_fs=fs_detected, target_fs=fs_use)
    else:
        ppg_resampled = ppg_raw
        
    try:
        # Preprocess using APG double filtering (Chebyshev 0.5-10Hz & 0.5-5Hz)
        f_ppg, s2_ppg = preprocess_ppg_apg(ppg_resampled, fs=fs_use)
    except ValueError as e:
        st.error(f"Lỗi xử lý tín hiệu PPG: {e}")
        st.stop()

    # Display raw PPG
    st.subheader("Raw PPG Signal Overview")
    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.plot(ppg_raw, color="purple")
    ax.set_title(f"Raw PPG Signal (fs={fs_detected} Hz)")
    ax.set_xlabel("Samples")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    # Detect peaks and feet using APG second-derivative method
    peaks, feet, _, _ = detect_apg_peaks_and_feet(ppg_resampled, fs=fs_use)

    # Display filtered S2 PPG with detected peaks/feet
    st.subheader("Processed PPG Signal (APG Double Chebyshev Filtered)")
    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.plot(s2_ppg, color="teal", linewidth=1.1, label="Filtered PPG S2")
    if len(peaks) > 0:
        ax.scatter(peaks, s2_ppg[peaks], color="crimson", s=20, label="Systolic Peak (Đỉnh)", zorder=5)
    if len(feet) > 0:
        ax.scatter(feet, s2_ppg[feet], color="dodgerblue", s=20, label="Foot point (Chân sóng)", zorder=5)
    ax.legend()
    ax.set_title("Filtered S2 PPG with Systolic Peaks & Foot Points")
    ax.set_xlabel("Samples")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    # Estimate physiological stats
    if len(peaks) > 1:
        ibi = np.diff(peaks) / fs_use
        hr = 60.0 / float(np.mean(ibi))
        st.caption(
            f"Detected {len(peaks)} beats | {len(ibi)} inter-beat intervals | "
            f"Estimated HR ≈ {hr:.1f} bpm"
        )
    else:
        st.caption(f"Detected {len(peaks)} beats | no inter-beat intervals computed")

    # ------------------------------------------------------------------
    # Segment and Extract 14 Features
    # ------------------------------------------------------------------
    windows, X_features = segment_and_extract_apg_features(
        ppg_resampled, 
        fs=fs_use, 
        window_sec=window_sec, 
        overlap_sec=overlap_sec
    )
    
    if len(windows) == 0:
        st.error("PPG signal too short for segmentation.")
        st.stop()

    # Check quality of each window
    qualities = np.array([check_window_quality(w, fs=fs_use) for w in windows])
    n_total = len(windows)
    n_clean = int(np.sum(qualities))
    n_noisy = n_total - n_clean

    # Add quality indicators for visualization in DataFrame
    X_features["quality_ok"] = qualities

    st.subheader("Extracted 14 Features (including CT and GT)")
    st.dataframe(X_features.head(min(5, len(X_features))))

    # Predict with Random Forest
    predictions = model.predict_proba(X_features.drop(columns=["quality_ok"]))[:, 1]
    
    # Apply custom decision threshold
    pred_labels_raw = (predictions >= prob_threshold).astype(int)

    # Apply moving majority vote filter (on clean windows only)
    clean_indices = np.where(qualities)[0]
    pred_labels_smoothed = np.zeros(n_total, dtype=int)
    
    if len(clean_indices) > 0:
        clean_preds_raw = pred_labels_raw[clean_indices]
        clean_preds_smoothed = np.zeros(len(clean_indices), dtype=int)
        
        for idx in range(len(clean_indices)):
            half_w = vote_window // 2
            start_idx = max(0, idx - half_w)
            end_idx = min(len(clean_indices), idx + half_w + 1)
            
            local_block = clean_preds_raw[start_idx:end_idx]
            af_ratio = np.mean(local_block)
            
            if af_ratio >= vote_ratio:
                clean_preds_smoothed[idx] = 1
            else:
                clean_preds_smoothed[idx] = 0
                
        for i, idx in enumerate(clean_indices):
            pred_labels_smoothed[idx] = clean_preds_smoothed[i]
    
    adjusted_labels_raw = np.array([pred_labels_raw[i] if qualities[i] else -1 for i in range(n_total)])
    adjusted_labels_smoothed = np.array([pred_labels_smoothed[i] if qualities[i] else -1 for i in range(n_total)])

    # Summary and Stats
    if n_clean > 0:
        raw_af_percentage = 100 * np.mean(pred_labels_raw[qualities])
        smoothed_af_percentage = 100 * np.mean(pred_labels_smoothed[qualities])
        clean_mean_probability = 100 * np.mean(predictions[qualities])
        
        st.success(f"Phân tích hoàn tất (Mô hình APG): Phát hiện {n_clean}/{n_total} cửa sổ sạch ({n_noisy} cửa sổ bị bỏ qua do nhiễu).")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Raw AF % (Thresholded)", value=f"{raw_af_percentage:.2f}%")
        with col2:
            st.metric(label="Smoothed AF % (Voting)", value=f"{smoothed_af_percentage:.2f}%")
            
        st.markdown(f"### Xác suất AF trung bình: {clean_mean_probability:.2f}%")
        
        if use_voting and raw_af_percentage > 0 and smoothed_af_percentage == 0:
            st.info("💡 **Bộ lọc tích lũy** đã thành công triệt tiêu các cảnh báo dương tính giả ngắn hạn xuất hiện rải rác!")
        elif smoothed_af_percentage > 20:
            st.warning("⚠️ **Cảnh báo:** Phát hiện cơn rung tâm nhĩ (AF) kéo dài liên tục qua mô hình APG.")
    else:
        st.warning("Cảnh báo: Toàn bộ cửa sổ đều bị nhận diện là Nhiễu. Vui lòng giữ yên tay và đo lại.")

    # ------------------------------------------------------------------
    # NEW FEATURE: Poincaré, Crest Time (CT), and GT Visualizations (Figure 11)
    # ------------------------------------------------------------------
    st.subheader("Trực quan hóa hình thái chu kỳ nhịp tim (ICSET 2023)")
    
    # Calculate global CT, GT and Poincaré
    ct_vals = []
    gt_vals = []
    ibi_vals = []
    
    for p in peaks:
        prec_feet = feet[feet < p]
        if len(prec_feet) > 0:
            f_prev = prec_feet[-1]
            ct = (p - f_prev) / fs_use
            if 0.03 <= ct <= 0.5:
                ct_vals.append(ct)
                next_feet = feet[feet > p]
                if len(next_feet) > 0:
                    f_next = next_feet[0]
                    gt = (f_next - p) / fs_use
                    if 0.1 <= gt <= 1.5:
                        gt_vals.append(gt)
                        ibi_vals.append(ct + gt)
                        
    if len(ibi_vals) > 2:
        # Compute differences for Poincaré plot
        ibi_arr = np.array(ibi_vals)
        diff_ibi = np.diff(ibi_arr)
        
        fig_mor, axes_mor = plt.subplots(1, 3, figsize=(15, 4.5))
        
        # 1. Poincaré plot of successive differences (Delta IBI^n vs Delta IBI^n-1)
        axes_mor[0].scatter(diff_ibi[:-1], diff_ibi[1:], color="blue", alpha=0.6, edgecolors="white", s=40)
        axes_mor[0].axhline(0, color="gray", linestyle=":", alpha=0.5)
        axes_mor[0].axvline(0, color="gray", linestyle=":", alpha=0.5)
        axes_mor[0].set_title("Poincaré Plot of successive diffs")
        axes_mor[0].set_xlabel("ΔIBI^n-1 (seconds)")
        axes_mor[0].set_ylabel("ΔIBI^n (seconds)")
        axes_mor[0].grid(True, alpha=0.3)
        
        # 2. Crest Time (CT) over beat cycles
        axes_mor[1].plot(np.arange(len(ct_vals)), np.array(ct_vals) * 1000.0, color="crimson", marker="o", markersize=4, linewidth=1)
        axes_mor[1].set_title("Crest Time (CT) per cycle")
        axes_mor[1].set_xlabel("Beats")
        axes_mor[1].set_ylabel("CT (ms)")
        axes_mor[1].grid(True, alpha=0.3)
        
        # 3. GT (diastolic time) over beat cycles
        axes_mor[2].plot(np.arange(len(gt_vals)), np.array(gt_vals) * 1000.0, color="dodgerblue", marker="s", markersize=4, linewidth=1)
        axes_mor[2].set_title("Diastolic time (GT) per cycle")
        axes_mor[2].set_xlabel("Beats")
        axes_mor[2].set_ylabel("GT (ms)")
        axes_mor[2].grid(True, alpha=0.3)
        
        st.pyplot(fig_mor)
        st.caption("Biểu đồ phân tích nhịp tim Poincaré (trái) và diễn biến biến động hai pha Crest Time (CT - giữa), GT (phải) qua các nhịp đập.")
    else:
        st.info("Không đủ số nhịp tim sạch được phát hiện để vẽ biểu đồ Poincaré, CT và GT.")

    # Plot example windows
    st.subheader("Example PPG Windows")
    n_windows = min(3, len(windows))
    fig, axs = plt.subplots(n_windows, 1, figsize=(10, 3 * n_windows))

    if n_windows == 1:
        axs = [axs]

    for i in range(n_windows):
        if not qualities[i]:
            color = "orange"
            status_text = "Nhiễu chuyển động (Bỏ qua)"
        else:
            raw_text = "AF" if pred_labels_raw[i] == 1 else "Normal"
            smooth_text = "AF" if pred_labels_smoothed[i] == 1 else "Normal"
            color = "red" if pred_labels_smoothed[i] == 1 else "green"
            status_text = f"Tức thời: {raw_text} | Sau lọc: {smooth_text} (p={predictions[i]:.3f})"
            
        axs[i].plot(windows[i], color=color)
        axs[i].set_title(f"Window {i + 1} - {status_text}")
        axs[i].set_xlabel("Samples")
        axs[i].set_ylabel("Amplitude")
        axs[i].grid(True)

    st.pyplot(fig)

    # Download predictions CSV
    pred_status_raw = []
    pred_status_smooth = []
    for i in range(n_total):
        if not qualities[i]:
            pred_status_raw.append("Noisy/Movement")
            pred_status_smooth.append("Noisy/Movement")
        else:
            pred_status_raw.append("AF" if pred_labels_raw[i] == 1 else "Normal")
            pred_status_smooth.append("AF" if pred_labels_smoothed[i] == 1 else "Normal")

    df_pred = pd.DataFrame(
        {
            "Window_Index": np.arange(n_total),
            "Signal_Quality": ["Clean" if q else "Noisy" for q in qualities],
            "AF_Probability": predictions,
            "Raw_Prediction": pred_status_raw,
            "Smoothed_Prediction": pred_status_smooth,
        }
    )
    csv_buffer = io.StringIO()
    df_pred.to_csv(csv_buffer, index=False)
    csv_bytes = csv_buffer.getvalue().encode()
    st.download_button(
        label="Download Predictions CSV",
        data=csv_bytes,
        file_name="ppg_af_predictions_apg.csv",
        mime="text/csv",
        key="download_csv",
    )
    st.success("Predictions ready for download!")
