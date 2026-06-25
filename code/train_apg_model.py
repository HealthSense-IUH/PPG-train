import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold

# Define paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARCHIVE_ROOT = PROJECT_ROOT / "archive"
MODELS_DIR   = PROJECT_ROOT / "models"
REPORTS_DIR  = PROJECT_ROOT / "reports"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Add code folder to path and import pipeline functions
sys.path.insert(0, str(PROJECT_ROOT / "code"))
from ppg_pipeline import load_archive_dataset, get_recording_fs, get_recording_signal, get_recording_label
from ppg_pipeline_apg import segment_and_extract_apg_features

# Hyperparameters
WINDOW_SEC  = 5.0
OVERLAP_SEC = 2.5
N_FOLDS     = 5
RANDOM_SEED = 42

RF_PARAMS = dict(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=RANDOM_SEED,
    n_jobs=-1,
)

FEATURE_COLS = [
    "signal_mean",
    "signal_std",
    "signal_range",
    "signal_energy",
    "peak_count",
    "ibi_mean",
    "ibi_std",
    "ibi_rmssd",
    "ibi_cv",
    "ct_mean",
    "ct_std",
    "gt_mean",
    "gt_std",
    "ct_gt_ratio"
]

def build_features_for_records_apg(recs: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    """Preprocess, segment, and extract 14 features for a list of recordings."""
    dfs: list[pd.DataFrame] = []
    ys:  list[np.ndarray]   = []
    target_fs = 125
    
    for i, rec in enumerate(recs):
        fs = get_recording_fs(rec)
        sig = get_recording_signal(rec)
        lbl = get_recording_label(rec)
        
        # If the recording fs is not 125 Hz, we resample it first
        if fs != target_fs:
            from ppg_pipeline import resample_to_target_fs
            sig = resample_to_target_fs(sig, src_fs=fs, target_fs=target_fs)
            fs_use = target_fs
        else:
            fs_use = fs
            
        try:
            # Segment and extract the 14 features (9 old + 5 new)
            _, feat_df = segment_and_extract_apg_features(
                sig, 
                fs=fs_use, 
                window_sec=WINDOW_SEC, 
                overlap_sec=OVERLAP_SEC
            )
            
            if not feat_df.empty:
                dfs.append(feat_df)
                ys.append(np.full(len(feat_df), lbl, dtype=int))
                print(f"  [{i+1:02d}/{len(recs):02d}] Label={lbl} | Extracted {len(feat_df)} windows")
            else:
                print(f"  [{i+1:02d}/{len(recs):02d}] Label={lbl} | Warning: No windows extracted")
        except Exception as e:
            print(f"  [{i+1:02d}/{len(recs):02d}] Label={lbl} | Error processing: {e}")
            
    if not dfs:
        raise ValueError("No features extracted from any recordings.")
        
    return pd.concat(dfs, ignore_index=True), np.concatenate(ys)

def main():
    print("=" * 60)
    print("  Training APG + Morphological Model on MIMIC PPG Dataset")
    print("=" * 60)

    # 1. Load MIMIC recordings
    print("\n[Step 1] Loading MIMIC recordings from archive...")
    recs = load_archive_dataset(ARCHIVE_ROOT)
    print(f"Loaded {len(recs)} recordings.")

    # 2. Extract features (all 14 features)
    print("\n[Step 2] Extracting 14 features using APG peak detection...")
    X_df, y = build_features_for_records_apg(recs)
    X = X_df[FEATURE_COLS].values
    print(f"Feature matrix shape: {X.shape} | Labels shape: {y.shape}")
    print(f"Class balance: Normal (0) = {(y==0).sum()} | AF (1) = {(y==1).sum()}")

    # 3. 5-Fold cross-validation (recording-level split)
    print("\n[Step 3] Running 5-Fold recording-level cross-validation...")
    
    rec_labels = np.array([get_recording_label(r) for r in recs])
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    
    cv_accs = []
    cv_aucs = []
    
    # We will accumulate all fold predictions to generate a global confusion matrix
    all_y_true = []
    all_y_pred = []
    
    for fold, (train_rec_idx, val_rec_idx) in enumerate(skf.split(np.zeros(len(recs)), rec_labels)):
        print(f"\n--- Fold {fold+1}/{N_FOLDS} ---")
        
        # Build features for training and validation splits
        train_recs = recs[train_rec_idx]
        val_recs = recs[val_rec_idx]
        
        print("  Extracting training features...")
        X_train_df, y_train = build_features_for_records_apg(train_recs)
        X_train = X_train_df[FEATURE_COLS].values
        
        print("  Extracting validation features...")
        X_val_df, y_val = build_features_for_records_apg(val_recs)
        X_val = X_val_df[FEATURE_COLS].values
        
        # Train Random Forest
        clf = RandomForestClassifier(**RF_PARAMS)
        clf.fit(X_train, y_train)
        
        # Predict on validation fold
        preds = clf.predict(X_val)
        probs = clf.predict_proba(X_val)[:, 1]
        
        acc = np.mean(preds == y_val)
        auc = roc_auc_score(y_val, probs)
        
        cv_accs.append(acc)
        cv_aucs.append(auc)
        all_y_true.extend(y_val)
        all_y_pred.extend(preds)
        
        print(f"  Fold {fold+1} Metrics -> Accuracy: {acc:.4f} | ROC-AUC: {auc:.4f}")
        
    print("\n" + "="*50)
    print("  Cross-Validation Summary (5 folds):")
    print(f"    Accuracy : {np.mean(cv_accs):.4f} +/- {np.std(cv_accs):.4f}")
    print(f"    ROC-AUC  : {np.mean(cv_aucs):.4f} +/- {np.std(cv_aucs):.4f}")
    print("="*50)

    # 4. Train final model on ALL recordings
    print("\n[Step 4] Training final Random Forest model on all recordings...")
    clf_final = RandomForestClassifier(**RF_PARAMS)
    clf_final.fit(X, y)
    print("Final model trained successfully.")

    # 5. Extract Feature Importances
    importances = clf_final.feature_importances_
    df_imp = pd.DataFrame({
        "Feature": FEATURE_COLS,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)
    
    print("\nFeature Importances (Final Model):")
    for idx, row in df_imp.iterrows():
        bar = "#" * int(round(row['Importance'] * 50))
        print(f"  {row['Feature']:<15} : {row['Importance']:.4f}  {bar}")

    # 6. Save model and metadata
    final_model_path = MODELS_DIR / "ppg_af_rf_apg.joblib"
    metadata_path = MODELS_DIR / "ppg_af_rf_apg_metadata.json"
    
    joblib.dump(clf_final, final_model_path)
    
    metadata = {
        "fs": 125,
        "window_sec": WINDOW_SEC,
        "overlap_sec": OVERLAP_SEC,
        "features": FEATURE_COLS,
        "cv_accuracy": float(np.mean(cv_accs)),
        "cv_accuracy_std": float(np.std(cv_accs)),
        "cv_roc_auc": float(np.mean(cv_aucs))
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
        
    # Save reports
    df_imp.to_csv(REPORTS_DIR / "rf_apg_feature_importance.csv", index=False)
    
    # Save confusion matrix
    cm = confusion_matrix(all_y_true, all_y_pred)
    pd.DataFrame(cm, columns=["Pred_Normal", "Pred_AF"], index=["True_Normal", "True_AF"]).to_csv(
        REPORTS_DIR / "rf_apg_cv_confusion_matrix.csv"
    )
    
    print("\n" + "=" * 60)
    print(f"  MIMIC APG Model Saved   : {final_model_path}")
    print(f"  Metadata Saved           : {metadata_path}")
    print(f"  Feature Importance CSV  : {REPORTS_DIR / 'rf_apg_feature_importance.csv'}")
    print("=" * 60)

if __name__ == "__main__":
    main()
