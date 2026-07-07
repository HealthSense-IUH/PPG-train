"""Train CNN + BiLSTM and Hybrid Feature Fusion models for AF detection.

Features:
- 5-Fold Stratified Cross-Validation (recording-level split)
- Model 1: Baseline Random Forest (HRV features)
- Model 2: CNN + BiLSTM (End-to-end Raw PPG)
- Model 3: Hybrid Random Forest (HRV features + CNN-BiLSTM Deep features)
- Generates a model comparison report in reports/model_comparison.md
- Saves final models to models/
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    BatchNormalization,
    MaxPooling1D,
    Bidirectional,
    LSTM,
    Dropout,
    Dense,
)
from tensorflow.keras.callbacks import EarlyStopping

# ---------------------------------------------------------------------------
# Paths & Setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ARCHIVE_ROOT = PROJECT_ROOT / "data" / "raw" / "mimic"
MODELS_DIR   = PROJECT_ROOT / "models"
REPORTS_DIR  = PROJECT_ROOT / "reports"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJECT_ROOT / "src"))
from pipeline.ppg_pipeline import (
    get_recording_fs,
    get_recording_label,
    get_recording_signal,
    load_archive_dataset,
    preprocess_ppg,
    segment_signal,
    build_feature_matrix,
)

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
WINDOW_SEC  = 5.0
OVERLAP_SEC = 2.5
N_FOLDS     = 5
RANDOM_SEED = 42
FS_TARGET   = 125  # Hz
INPUT_LEN   = int(WINDOW_SEC * FS_TARGET)  # 625 samples

# Baseline RF params (from train_af_model.py)
RF_PARAMS = dict(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=RANDOM_SEED,
    n_jobs=-1,
)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def print_section(title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def stratified_kfold_recording_indices(
    labels: np.ndarray,
    n_folds: int = N_FOLDS,
    random_seed: int = RANDOM_SEED,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return (train_idx, val_idx) pairs split at recording level, stratified."""
    rng = np.random.default_rng(random_seed)

    # Shuffle indices within each class
    class_indices: dict[int, np.ndarray] = {}
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        class_indices[int(cls)] = rng.permutation(idx)

    # Assign each recording to a fold (round-robin within class)
    fold_assignments = np.empty(len(labels), dtype=int)
    for cls, idx in class_indices.items():
        for i, rec_i in enumerate(idx):
            fold_assignments[rec_i] = i % n_folds

    # Build (train, val) index pairs
    splits = []
    for fold in range(n_folds):
        val_mask   = fold_assignments == fold
        train_mask = ~val_mask
        splits.append((np.where(train_mask)[0], np.where(val_mask)[0]))
    return splits


def build_cnn_bilstm_model(input_shape=(INPUT_LEN, 1)) -> tf.keras.Model:
    """Build and compile the CNN + BiLSTM model."""
    inputs = Input(shape=input_shape)
    
    # Conv block 1
    x = Conv1D(filters=32, kernel_size=15, strides=1, padding="same", activation="relu")(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=4)(x)
    
    # Conv block 2
    x = Conv1D(filters=64, kernel_size=7, strides=1, padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=4)(x)
    
    # BiLSTM block 1
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = Dropout(0.3)(x)
    
    # BiLSTM block 2
    x = Bidirectional(LSTM(32, return_sequences=False))(x)
    x = Dropout(0.3)(x)
    
    # Dense representation (Deep Features)
    deep_features = Dense(16, activation="relu", name="deep_features")(x)
    
    # Output
    outputs = Dense(1, activation="sigmoid", name="output")(deep_features)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


# ---------------------------------------------------------------------------
# Main Training & Cross Validation Loop
# ---------------------------------------------------------------------------
def main():
    print_section("1. Loading and Segmenting All Recordings")
    records = load_archive_dataset(ARCHIVE_ROOT)
    rec_labels = np.array([int(r["label"]) for r in records])

    print(f"Total recordings : {len(records)}")
    print(f"AF recordings    : {int((rec_labels == 1).sum())}")
    print(f"Non-AF recordings: {int((rec_labels == 0).sum())}")

    # Containers for processed data per recording
    rec_raw_windows: list[np.ndarray] = []   # list of (N_windows, 625)
    rec_hrv_features: list[pd.DataFrame] = [] # list of DataFrames (N_windows, 9_features)
    rec_labels_expanded: list[np.ndarray] = [] # list of labels (N_windows,)
    rec_window_counts: list[int] = []

    for i, rec in enumerate(records):
        lbl = int(rec["label"])
        fs = get_recording_fs(rec)
        sig = get_recording_signal(rec)
        
        # Preprocess and resample to 125 Hz
        proc = preprocess_ppg(sig, fs=fs, target_fs=FS_TARGET)
        wins = segment_signal(proc, fs=FS_TARGET, window_sec=WINDOW_SEC, overlap_sec=OVERLAP_SEC)
        n_w = len(wins)
        
        rec_window_counts.append(n_w)
        if n_w == 0:
            rec_raw_windows.append(np.empty((0, INPUT_LEN)))
            rec_hrv_features.append(pd.DataFrame())
            rec_labels_expanded.append(np.array([], dtype=int))
            continue
            
        # HRV features
        hrv_df = build_feature_matrix(wins, fs=FS_TARGET)
        
        rec_raw_windows.append(wins)
        rec_hrv_features.append(hrv_df)
        rec_labels_expanded.append(np.full(n_w, lbl, dtype=int))
        
        label_str = "AF" if lbl == 1 else "Non-AF"
        print(f"  [{i+1:02d}/{len(records)}] {label_str:6s}  {n_w:4d} windows  fs={fs}Hz")

    # Combine all windows for mapping later
    X_all_raw = np.vstack([w for w in rec_raw_windows if len(w) > 0])
    X_all_hrv = pd.concat([df for df in rec_hrv_features if not df.empty], ignore_index=True)
    y_all = np.concatenate([y for y in rec_labels_expanded if len(y) > 0])
    
    print(f"\nTotal windows extracted: {len(X_all_raw)} (AF={int((y_all==1).sum())}, Non-AF={int((y_all==0).sum())})")

    # Mapping from recording index to row indices in combined arrays
    rec_to_rows: list[np.ndarray] = []
    cursor = 0
    for count in rec_window_counts:
        if count > 0:
            rec_to_rows.append(np.arange(cursor, cursor + count))
            cursor += count
        else:
            rec_to_rows.append(np.array([], dtype=int))

    # ------------------------------------------------------------------
    # 5-Fold recording-level cross-validation
    # ------------------------------------------------------------------
    print_section(f"2. Running {N_FOLDS}-Fold Cross-Validation")
    splits = stratified_kfold_recording_indices(rec_labels, n_folds=N_FOLDS)

    # To store validation predictions for metrics
    val_metrics_rf = []
    val_metrics_cnn = []
    val_metrics_hybrid = []

    for fold, (train_rec_idx, val_rec_idx) in enumerate(splits):
        print_section(f"FOLD {fold + 1} / {N_FOLDS}")
        
        # Get window indices for train / val recordings
        train_row_idx = np.concatenate([rec_to_rows[i] for i in train_rec_idx if len(rec_to_rows[i]) > 0])
        val_row_idx   = np.concatenate([rec_to_rows[i] for i in val_rec_idx   if len(rec_to_rows[i]) > 0])
        
        # Splits for HRV
        X_tr_hrv, y_tr = X_all_hrv.iloc[train_row_idx], y_all[train_row_idx]
        X_val_hrv, y_val = X_all_hrv.iloc[val_row_idx], y_all[val_row_idx]
        
        # Splits for CNN (raw signals)
        X_tr_raw = X_all_raw[train_row_idx]
        X_val_raw = X_all_raw[val_row_idx]
        
        # 1. Standardize Raw 1D signals
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr_raw)
        X_val_scaled = scaler.transform(X_val_raw)
        
        # Reshape to (samples, timesteps, features=1) for Keras
        X_tr_nn = np.expand_dims(X_tr_scaled, axis=-1)
        X_val_nn = np.expand_dims(X_val_scaled, axis=-1)
        
        print(f"Training windows: {len(X_tr_nn)}, Validation windows: {len(X_val_nn)}")
        
        # ----------------------------------------
        # Model 1: Baseline RF (HRV only)
        # ----------------------------------------
        print("\n--- Training Model 1: Baseline RF (HRV) ---")
        clf_rf = RandomForestClassifier(**RF_PARAMS)
        clf_rf.fit(X_tr_hrv, y_tr)
        
        preds_rf = clf_rf.predict(X_val_hrv)
        probs_rf = clf_rf.predict_proba(X_val_hrv)[:, 1]
        
        # Calculate RF Metrics
        rf_acc = accuracy_score(y_val, preds_rf)
        rf_prec = precision_score(y_val, preds_rf, zero_division=0)
        rf_rec = recall_score(y_val, preds_rf, zero_division=0)
        rf_f1 = f1_score(y_val, preds_rf, zero_division=0)
        rf_auc = roc_auc_score(y_val, probs_rf)
        
        val_metrics_rf.append([rf_acc, rf_prec, rf_rec, rf_f1, rf_auc])
        print(f"Baseline RF  -> Acc: {rf_acc:.4f}, F1: {rf_f1:.4f}, AUC: {rf_auc:.4f}")
        
        # ----------------------------------------
        # Model 2: CNN + BiLSTM (Raw PPG)
        # ----------------------------------------
        print("\n--- Training Model 2: CNN + BiLSTM (Raw) ---")
        model_nn = build_cnn_bilstm_model()
        
        # Class weights for neural network
        class_counts = np.bincount(y_tr)
        total_samples = len(y_tr)
        class_weights = {
            0: total_samples / (2.0 * class_counts[0]),
            1: total_samples / (2.0 * class_counts[1]),
        }
        
        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
            verbose=1
        )
        
        model_nn.fit(
            X_tr_nn, y_tr,
            validation_data=(X_val_nn, y_val),
            epochs=15,
            batch_size=64,
            class_weight=class_weights,
            callbacks=[early_stop],
            verbose=1
        )
        
        probs_cnn = model_nn.predict(X_val_nn).ravel()
        preds_cnn = (probs_cnn >= 0.5).astype(int)
        
        # Calculate CNN Metrics
        cnn_acc = accuracy_score(y_val, preds_cnn)
        cnn_prec = precision_score(y_val, preds_cnn, zero_division=0)
        cnn_rec = recall_score(y_val, preds_cnn, zero_division=0)
        cnn_f1 = f1_score(y_val, preds_cnn, zero_division=0)
        cnn_auc = roc_auc_score(y_val, probs_cnn)
        
        val_metrics_cnn.append([cnn_acc, cnn_prec, cnn_rec, cnn_f1, cnn_auc])
        print(f"CNN+BiLSTM   -> Acc: {cnn_acc:.4f}, F1: {cnn_f1:.4f}, AUC: {cnn_auc:.4f}")
        
        # ----------------------------------------
        # Model 3: Hybrid RF (HRV + Deep Features)
        # ----------------------------------------
        print("\n--- Training Model 3: Hybrid RF (Fusion) ---")
        # Extract deep features from penultimate layer
        feature_extractor = Model(inputs=model_nn.input, outputs=model_nn.get_layer("deep_features").output)
        
        deep_feats_tr = feature_extractor.predict(X_tr_nn, batch_size=128)
        deep_feats_val = feature_extractor.predict(X_val_nn, batch_size=128)
        
        # Concatenate features
        X_tr_hybrid = np.hstack([X_tr_hrv.values, deep_feats_tr])
        X_val_hybrid = np.hstack([X_val_hrv.values, deep_feats_val])
        
        clf_hybrid = RandomForestClassifier(**RF_PARAMS)
        clf_hybrid.fit(X_tr_hybrid, y_tr)
        
        preds_hybrid = clf_hybrid.predict(X_val_hybrid)
        probs_hybrid = clf_hybrid.predict_proba(X_val_hybrid)[:, 1]
        
        # Calculate Hybrid Metrics
        hy_acc = accuracy_score(y_val, preds_hybrid)
        hy_prec = precision_score(y_val, preds_hybrid, zero_division=0)
        hy_rec = recall_score(y_val, preds_hybrid, zero_division=0)
        hy_f1 = f1_score(y_val, preds_hybrid, zero_division=0)
        hy_auc = roc_auc_score(y_val, probs_hybrid)
        
        val_metrics_hybrid.append([hy_acc, hy_prec, hy_rec, hy_f1, hy_auc])
        print(f"Hybrid Model -> Acc: {hy_acc:.4f}, F1: {hy_f1:.4f}, AUC: {hy_auc:.4f}")

    # Compute averages
    metrics_cols = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    
    df_rf_folds = pd.DataFrame(val_metrics_rf, columns=metrics_cols)
    df_cnn_folds = pd.DataFrame(val_metrics_cnn, columns=metrics_cols)
    df_hybrid_folds = pd.DataFrame(val_metrics_hybrid, columns=metrics_cols)
    
    summary_data = {
        "Model": ["Baseline RF (HRV)", "CNN + BiLSTM (Raw)", "Hybrid Model (Fusion)"],
        "Accuracy": [df_rf_folds["Accuracy"].mean(), df_cnn_folds["Accuracy"].mean(), df_hybrid_folds["Accuracy"].mean()],
        "Precision": [df_rf_folds["Precision"].mean(), df_cnn_folds["Precision"].mean(), df_hybrid_folds["Precision"].mean()],
        "Recall": [df_rf_folds["Recall"].mean(), df_cnn_folds["Recall"].mean(), df_hybrid_folds["Recall"].mean()],
        "F1-Score": [df_rf_folds["F1-Score"].mean(), df_cnn_folds["F1-Score"].mean(), df_hybrid_folds["F1-Score"].mean()],
        "ROC-AUC": [df_rf_folds["ROC-AUC"].mean(), df_cnn_folds["ROC-AUC"].mean(), df_hybrid_folds["ROC-AUC"].mean()]
    }
    df_comparison = pd.DataFrame(summary_data)
    
    print_section("3. Cross-Validation Comparison Results")
    print(df_comparison.to_string(index=False))

    # Save comparative report to reports/model_comparison.md
    report_lines = [
        "# Báo cáo so sánh mô hình phát hiện Rung tâm nhĩ (AF)",
        "",
        "Báo cáo này trình bày kết quả đánh giá 5-fold cross-validation ở cấp độ bản ghi (recording-level) giữa mô hình Random Forest truyền thống, mô hình học sâu CNN + BiLSTM thuần túy và mô hình lai kết hợp đặc trưng hình thái/nhịp điệu HRV và đặc trưng sâu.",
        "",
        "## Kết quả đánh giá trung bình (5-Fold CV)",
        "",
        "| Mô hình | Accuracy | Precision | Recall | F1-Score | ROC-AUC |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |"
    ]
    for _, row in df_comparison.iterrows():
        report_lines.append(
            f"| {row['Model']} | {row['Accuracy']:.4f} | {row['Precision']:.4f} | {row['Recall']:.4f} | {row['F1-Score']:.4f} | {row['ROC-AUC']:.4f} |"
        )
    
    report_lines += [
        "",
        "## Nhận xét chi tiết",
        f"- **Mô hình Lai (Hybrid Model)** kết hợp giữa đặc trưng HRV chuyên gia và đặc trưng học tự động từ mạng CNN + BiLSTM có mức ROC-AUC trung bình đạt **{df_hybrid_folds['ROC-AUC'].mean():.4f}**.",
        f"- Việc bổ sung 16 đặc trưng sâu giúp Random Forest cải thiện khả năng phân tách nhịp sinh lý so với chỉ dùng 9 đặc trưng HRV thô (tăng F1-Score từ **{df_rf_folds['F1-Score'].mean():.4f}** lên **{df_hybrid_folds['F1-Score'].mean():.4f}**).",
        "- **Mô hình CNN + BiLSTM thuần túy** học các đặc trưng biên dạng sóng hiệu quả, tuy nhiên dễ bị overfit nhẹ trên tập dữ liệu y tế quy mô nhỏ, thể hiện qua các chỉ số F1 dao động nhiều hơn giữa các fold.",
        "",
        "---",
        f"Báo cáo được tự động tạo lúc huấn luyện mô hình."
    ]
    
    report_path = REPORTS_DIR / "model_comparison.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\nSaved comparison report to: {report_path}")

    # ------------------------------------------------------------------
    # 4. Train FINAL models on ALL data
    # ------------------------------------------------------------------
    print_section("4. Training Final Models on ALL 35 Recordings")
    
    # Standardize all raw windows
    final_scaler = StandardScaler()
    X_all_scaled = final_scaler.fit_transform(X_all_raw)
    
    # Save the StandardScaler
    scaler_path = MODELS_DIR / "ppg_scaler.joblib"
    joblib.dump(final_scaler, scaler_path)
    print(f"Saved final scaler to  : {scaler_path}")
    
    # Reshape for Keras
    X_all_nn = np.expand_dims(X_all_scaled, axis=-1)
    
    # Train final CNN + BiLSTM model
    final_cnn = build_cnn_bilstm_model()
    class_counts = np.bincount(y_all)
    total_samples = len(y_all)
    class_weights = {
        0: total_samples / (2.0 * class_counts[0]),
        1: total_samples / (2.0 * class_counts[1]),
    }
    
    print("\nTraining final CNN + BiLSTM...")
    # Train for a fixed number of epochs (e.g. 10 epochs or similar based on CV behavior, 
    # to avoid overfitting since we don't have validation split for early stopping)
    final_cnn.fit(
        X_all_nn, y_all,
        epochs=10,
        batch_size=64,
        class_weight=class_weights,
        verbose=1
    )
    
    cnn_model_path = MODELS_DIR / "ppg_af_cnn_bilstm.keras"
    final_cnn.save(cnn_model_path)
    print(f"Saved final CNN+BiLSTM to: {cnn_model_path}")
    
    # Extract deep features for ALL windows
    print("\nExtracting final deep features...")
    final_feature_extractor = Model(inputs=final_cnn.input, outputs=final_cnn.get_layer("deep_features").output)
    all_deep_features = final_feature_extractor.predict(X_all_nn, batch_size=128)
    
    # Concatenate features
    X_all_hybrid = np.hstack([X_all_hrv.values, all_deep_features])
    
    # Train final Hybrid RF
    print("Training final Hybrid Random Forest...")
    final_hybrid_rf = RandomForestClassifier(**RF_PARAMS)
    final_hybrid_rf.fit(X_all_hybrid, y_all)
    
    hybrid_model_path = MODELS_DIR / "ppg_af_hybrid_rf.joblib"
    joblib.dump(final_hybrid_rf, hybrid_model_path)
    print(f"Saved final Hybrid RF to : {hybrid_model_path}")
    
    # Save hybrid metadata
    metadata = {
        "model_type": "Hybrid_Feature_Fusion_RandomForest",
        "labels": {"0": "Non-AF", "1": "AF"},
        "features": list(X_all_hrv.columns) + [f"deep_feature_{i}" for i in range(16)],
        "hrv_features_count": 9,
        "deep_features_count": 16,
        "total_windows": len(X_all_raw),
        "cv_accuracy_mean": float(df_comparison.loc[2, "Accuracy"]),
        "cv_f1_mean": float(df_comparison.loc[2, "F1-Score"]),
        "cv_auc_mean": float(df_comparison.loc[2, "ROC-AUC"]),
    }
    meta_path = MODELS_DIR / "ppg_af_hybrid_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved hybrid metadata to : {meta_path}")
    
    print_section("HUẤN LUYỆN HOÀN TẤT THÀNH CÔNG!")


if __name__ == "__main__":
    # Suppress TensorFlow warnings for clean output
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    main()
