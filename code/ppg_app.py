import io
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="PPG AF Classifier", layout="wide")

# Add code folder to path and import pipeline functions
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ppg_pipeline import (
    build_feature_matrix,
    detect_beats,
    preprocess_ppg,
    resample_to_target_fs,
    segment_signal,
    check_window_quality,
)

TRAIN_FS = 125  # Hz — must match the fs used during model training (125 Hz for MIMIC model)

# -----------------------------
# Custom CSS for Premium Dark Theme
# -----------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Main App styling */
    .stApp {
        background: linear-gradient(135deg, #0b0f19 0%, #111827 100%) !important;
        font-family: 'Inter', sans-serif !important;
        color: #e5e7eb !important;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #030712 !important;
        border-right: 1px solid #1f2937 !important;
    }
    section[data-testid="stSidebar"] h2 {
        color: #f9fafb !important;
    }

    /* Headings styling */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif !important;
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    
    /* Main title gradient */
    h1 {
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 100%) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        font-size: 2.5rem !important;
        padding-bottom: 0.5rem !important;
    }

    /* Markdown paragraph text visibility */
    .stMarkdown p {
        color: #d1d5db !important;
        font-size: 1rem !important;
    }

    /* File uploader and dropzone */
    div[data-testid="stFileUploader"] {
        background-color: #1f2937 !important;
        border: 1px dashed #4b5563 !important;
        border-radius: 8px !important;
        padding: 10px !important;
    }

    /* Metrics and labels */
    div[data-testid="stWidgetLabel"] p {
        color: #f3f4f6 !important;
        font-weight: 500 !important;
    }
    
    /* Notification styling overrides */
    div[data-testid="stNotification"] {
        background-color: #111827 !important;
        border: 1px solid #374151 !important;
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
MODEL_PATH = MODEL_DIR / "ppg_af_rf.joblib"

model = None
model_load_error = None
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    model_load_error = f"Random Forest model not found at {MODEL_PATH}."

# -----------------------------
# Sidebar for settings
# -----------------------------
st.sidebar.header("Segmentation Settings")
window_sec  = st.sidebar.slider("Window length (sec)", 1, 20, 5, key="window_slider")
# overlap must be strictly less than window
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
st.title("PPG Atrial Fibrillation Classifier")
st.write("Upload a PPG `.csv` file to predict AF vs Normal rhythm using the feature-based Random Forest model.")

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
        # Invert signal polarity to match training convention (systolic peaks pointing upwards)
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
    # Preprocess at original fs to remove DC offset, then resample to TRAIN_FS
    try:
        ppg_proc = preprocess_ppg(ppg_raw, fs=fs_detected, target_fs=fs_use)
    except ValueError as e:
        st.error(f"Lỗi xử lý tín hiệu PPG: {e}")
        st.stop()

    # Display raw PPG (original, before preprocessing & resampling)
    st.subheader("Raw PPG Signal Overview")
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(ppg_raw, color="purple")
    ax.set_title(f"Raw PPG Signal (fs={fs_detected} Hz)")
    ax.set_xlabel("Samples")
    ax.set_ylabel("Amplitude")
    ax.grid(True)
    st.pyplot(fig)

    # Detect beats on the preprocessed resampled signal
    peak_locs, ibi, _ = detect_beats(ppg_proc, fs=fs_use)

    st.subheader("Processed PPG Signal")
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(ppg_proc, color="teal", linewidth=1.1)
    if len(peak_locs) > 0:
        ax.scatter(peak_locs, ppg_proc[peak_locs], color="crimson", s=15, label="Peaks")
        ax.legend()
    ax.set_title("Filtered + Normalized PPG")
    ax.set_xlabel("Samples")
    ax.set_ylabel("Amplitude")
    ax.grid(True)
    st.pyplot(fig)

    st.caption(
        f"Detected {len(peak_locs)} beats | {len(ibi)} inter-beat intervals | "
        f"HR ≈ {60.0/float(np.mean(ibi)):.1f} bpm" if len(ibi) > 0 else
        f"Detected {len(peak_locs)} beats | no inter-beat intervals computed"
    )

    # Segment and featurize at TRAIN_FS
    windows = segment_signal(ppg_proc, fs=fs_use, window_sec=window_sec, overlap_sec=overlap_sec)
    if len(windows) == 0:
        st.error("PPG signal too short for segmentation.")
        st.stop()

    # Check quality of each window using the new check_window_quality function
    qualities = np.array([check_window_quality(w, fs=fs_use) for w in windows])
    n_total = len(windows)
    n_clean = int(np.sum(qualities))
    n_noisy = n_total - n_clean

    X_features = build_feature_matrix(windows, fs=fs_use)
    # Add quality indicators for visualization in DataFrame
    X_features["quality_ok"] = qualities

    st.subheader("Extracted Window Features")
    st.dataframe(X_features.head(min(5, len(X_features))))

    # Predict with Random Forest (excluding the helper column)
    predictions = model.predict_proba(X_features.drop(columns=["quality_ok"]))[:, 1]
    
    # 1. Apply custom decision threshold
    pred_labels_raw = (predictions >= prob_threshold).astype(int)

    # 2. Apply moving majority vote filter (on clean windows only)
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
    
    # Adjusted labels: 1 = AF, 0 = Normal, -1 = Noisy/Movement
    adjusted_labels_raw = np.array([pred_labels_raw[i] if qualities[i] else -1 for i in range(n_total)])
    adjusted_labels_smoothed = np.array([pred_labels_smoothed[i] if qualities[i] else -1 for i in range(n_total)])

    # Summary and Stats
    if n_clean > 0:
        raw_af_percentage = 100 * np.mean(pred_labels_raw[qualities])
        smoothed_af_percentage = 100 * np.mean(pred_labels_smoothed[qualities])
        clean_mean_probability = 100 * np.mean(predictions[qualities])
        
        st.success(f"Phân tích hoàn tất: Phát hiện {n_clean}/{n_total} cửa sổ tín hiệu sạch ({n_noisy} cửa sổ bị bỏ qua do nhiễu hoặc chuyển động mạnh).")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                label="Tỷ lệ dự đoán AF tức thời (Raw AF %)",
                value=f"{raw_af_percentage:.2f}%",
                help="Tính trên ngưỡng xác suất tùy chỉnh, chưa qua bộ lọc bỏ phiếu."
            )
        with col2:
            st.metric(
                label="Tỷ lệ AF sau khi lọc tích lũy (Smoothed AF %)",
                value=f"{smoothed_af_percentage:.2f}%",
                help="Đã lọc các tín hiệu bất thường ngắt quãng ngắn bằng phương pháp đa số biểu quyết."
            )
            
        st.markdown(f"### Xác suất AF trung bình (các cửa sổ sạch): {clean_mean_probability:.2f}%")
        
        if use_voting and raw_af_percentage > 0 and smoothed_af_percentage == 0:
            st.info("💡 **Bộ lọc tích lũy** đã thành công triệt tiêu các cảnh báo dương tính giả ngắn hạn xuất hiện rải rác!")
        elif smoothed_af_percentage > 20:
            st.warning("⚠️ **Chú ý:** Phát hiện cơn rung tâm nhĩ (AF) kéo dài liên tục qua bộ lọc tích lũy. Nên nghỉ ngơi và kiểm tra lại.")
    else:
        st.warning(f"Cảnh báo: Toàn bộ {n_total} cửa sổ đều bị nhận diện là Nhiễu hoặc Chuyển động mạnh. Không thể đưa ra dự đoán AF tin cậy. Vui lòng giữ yên tay và đo lại.")

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

    # Download predictions
    # Save adjusted prediction text
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
        file_name="ppg_af_predictions.csv",
        mime="text/csv",
        key="download_csv",
    )
    st.success("Predictions ready for download!")
