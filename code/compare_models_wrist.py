import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from ppg_pipeline import (
    preprocess_ppg,
    segment_signal,
    build_feature_matrix,
    check_window_quality,
    resample_to_target_fs
)
from ppg_pipeline_apg import (
    preprocess_ppg_apg,
    segment_and_extract_apg_features
)

OLD_MODEL_PATH = PROJECT_ROOT / "models" / "ppg_af_rf.joblib"
NEW_MODEL_PATH = PROJECT_ROOT / "models" / "ppg_af_rf_apg.joblib"
WRIST_DIR = PROJECT_ROOT / "wrist-down"

def load_huywatch(csv_path: Path, channel: str = "ir") -> tuple[np.ndarray, float]:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns] # Clean leading/trailing spaces
    t_ms = df["device_millis"].values.astype(float)
    signal = -df[channel].values.astype(float)
    t_sec = (t_ms - t_ms[0]) / 1000.0
    dt_arr = np.diff(t_sec)
    fs = float(1.0 / np.median(dt_arr))
    return signal, fs

def run_old_model(clf, signal_raw, fs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the old model and return (probabilities, qualities, raw_windows)."""
    # 1. Preprocess and resample to 125Hz
    signal_proc = preprocess_ppg(signal_raw, fs=int(round(fs)), target_fs=125)
    
    # 2. Segment
    windows = segment_signal(signal_proc, fs=125, window_sec=5.0, overlap_sec=2.5)
    if len(windows) == 0:
        return np.array([]), np.array([]), np.array([])
        
    # 3. Quality check
    qualities = np.array([check_window_quality(w, fs=125) for w in windows])
    
    # 4. Extract features
    X_feat = build_feature_matrix(windows, fs=125)
    
    # 5. Predict
    probs = clf.predict_proba(X_feat)[:, 1]
    return probs, qualities, windows

def run_new_model(clf, signal_raw, fs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the new APG model and return (probabilities, qualities, raw_windows)."""
    # 1. Resample raw signal to 125Hz
    fs_src = int(round(fs))
    if fs_src != 125:
        ppg_resampled = resample_to_target_fs(signal_raw, src_fs=fs_src, target_fs=125)
    else:
        ppg_resampled = signal_raw
        
    # 2. Segment and extract features (uses Chebyshev 2 double filtering internally)
    windows, X_feat = segment_and_extract_apg_features(
        ppg_resampled, 
        fs=125, 
        window_sec=5.0, 
        overlap_sec=2.5
    )
    if len(windows) == 0:
        return np.array([]), np.array([]), np.array([])
        
    # 3. Quality check (using standard check_window_quality on S2 windows)
    # Note: segment_and_extract_apg_features returns normalized S2 windows
    qualities = np.array([check_window_quality(w, fs=125) for w in windows])
    
    # 4. Predict (model expects 14 features)
    probs = clf.predict_proba(X_feat)[:, 1]
    return probs, qualities, windows

def main():
    print("Loading models...")
    old_clf = joblib.load(OLD_MODEL_PATH)
    new_clf = joblib.load(NEW_MODEL_PATH)
    print("Models loaded successfully.")
    
    csv_files = sorted(WRIST_DIR.glob("*.csv"))
    print(f"Found {len(csv_files)} CSV files in wrist-down folder.")
    
    comparison_data = []
    
    for csv_path in csv_files:
        print(f"\nProcessing {csv_path.name}...")
        signal_raw, fs = load_huywatch(csv_path)
        
        # Run Old Model
        probs_old, qual_old, _ = run_old_model(old_clf, signal_raw, fs)
        # Run New Model
        probs_new, qual_new, _ = run_new_model(new_clf, signal_raw, fs)
        
        n_win = len(probs_old)
        n_clean_old = int(np.sum(qual_old))
        n_clean_new = int(np.sum(qual_new))
        
        # Predictions at threshold 0.5 (only on clean windows)
        clean_idx_old = np.where(qual_old)[0]
        clean_idx_new = np.where(qual_new)[0]
        
        # Old model metrics (Thresh 0.5 and 0.8)
        if len(clean_idx_old) > 0:
            probs_clean_old = probs_old[clean_idx_old]
            af_count_old_50 = int((probs_clean_old >= 0.5).sum())
            af_count_old_80 = int((probs_clean_old >= 0.8).sum())
            mean_prob_old = float(np.mean(probs_clean_old)) * 100
        else:
            af_count_old_50 = af_count_old_80 = 0
            mean_prob_old = 0.0
            
        # New model metrics (Thresh 0.5 and 0.8)
        if len(clean_idx_new) > 0:
            probs_clean_new = probs_new[clean_idx_new]
            af_count_new_50 = int((probs_clean_new >= 0.5).sum())
            af_count_new_80 = int((probs_clean_new >= 0.8).sum())
            mean_prob_new = float(np.mean(probs_clean_new)) * 100
        else:
            af_count_new_50 = af_count_new_80 = 0
            mean_prob_new = 0.0
            
        comparison_data.append({
            "File": csv_path.name,
            "Total_Win": n_win,
            "Clean_Win_Old": n_clean_old,
            "Clean_Win_New": n_clean_new,
            "Old_Mean_Prob": mean_prob_old,
            "Old_AF_50": af_count_old_50,
            "Old_AF_80": af_count_old_80,
            "New_Mean_Prob": mean_prob_new,
            "New_AF_50": af_count_new_50,
            "New_AF_80": af_count_new_80
        })
        
        print(f"  Old Model (Clean={n_clean_old}/{n_win}): Mean AF Prob = {mean_prob_old:.1f}%, AF Windows (>=0.5) = {af_count_old_50}, (>=0.8) = {af_count_old_80}")
        print(f"  New Model (Clean={n_clean_new}/{n_win}): Mean AF Prob = {mean_prob_new:.1f}%, AF Windows (>=0.5) = {af_count_new_50}, (>=0.8) = {af_count_new_80}")
        
    df_comp = pd.DataFrame(comparison_data)
    
    print("\n" + "="*80)
    print("                          SUMMARY COMPARISON REPORT")
    print("="*80)
    print(df_comp.to_string(index=False))
    print("="*80)
    
    # Save comparison report to markdown
    report_lines = [
        "# Báo cáo so sánh mô hình cũ và mô hình mới (APG)",
        "",
        "Báo cáo này so sánh kết quả dự đoán của **Mô hình cũ (9 đặc trưng)** và **Mô hình mới (14 đặc trưng hình thái APG)** trên tập dữ liệu đo ở cổ tay khi xuôi tay (**wrist-down**).",
        "",
        "## Kết quả đo đạc chi tiết",
        "",
        "| File | Tổng số Cửa sổ | Cửa sổ sạch (Cũ/Mới) | Xác suất AF TB (Cũ) | AF Windows >= 0.5 (Cũ) | AF Windows >= 0.8 (Cũ) | Xác suất AF TB (Mới) | AF Windows >= 0.5 (Mới) | AF Windows >= 0.8 (Mới) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]
    for r in comparison_data:
        report_lines.append(
            f"| {r['File']} | {r['Total_Win']} | {r['Clean_Win_Old']} / {r['Clean_Win_New']} | {r['Old_Mean_Prob']:.1f}% | {r['Old_AF_50']} | {r['Old_AF_80']} | {r['New_Mean_Prob']:.1f}% | {r['New_AF_50']} | {r['New_AF_80']} |"
        )
        
    report_lines.append("\n## Nhận xét chi tiết")
    
    # Analyze false alarm rate
    avg_prob_old = df_comp["Old_Mean_Prob"].mean()
    avg_prob_new = df_comp["New_Mean_Prob"].mean()
    total_af_old_50 = df_comp["Old_AF_50"].sum()
    total_af_new_50 = df_comp["New_AF_50"].sum()
    total_af_old_80 = df_comp["Old_AF_80"].sum()
    total_af_new_80 = df_comp["New_AF_80"].sum()
    
    report_lines.append(f"- **Tỷ lệ báo động giả (False Alarm Rate) ở ngưỡng mặc định 0.5**:")
    report_lines.append(f"  - Mô hình cũ dự đoán tổng cộng **{total_af_old_50}** cửa sổ là AF với xác suất AF trung bình là **{avg_prob_old:.1f}%**.")
    report_lines.append(f"  - Mô hình mới dự đoán tổng cộng **{total_af_new_50}** cửa sổ là AF với xác suất AF trung bình là **{avg_prob_new:.1f}%**.")
    report_lines.append(f"- **Tỷ lệ báo động giả ở ngưỡng an toàn 0.8**:")
    report_lines.append(f"  - Mô hình cũ dự đoán tổng cộng **{total_af_old_80}** cửa sổ là AF.")
    report_lines.append(f"  - Mô hình mới dự đoán tổng cộng **{total_af_new_80}** cửa sổ là AF.")
    
    # Save to file
    out_path = PROJECT_ROOT / "outputs" / "model_comparison_wrist_down.md"
    out_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\nSaved markdown report to: {out_path}")

if __name__ == "__main__":
    main()
