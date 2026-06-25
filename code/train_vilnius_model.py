"""Retrain the Random Forest model using the consolidated Vilnius smartwatch-based dataset.

This script:
1. Loads the consolidated features from outputs/vilnius_features_consolidated.csv.
2. Performs a patient-level train-test split (Group Split by case_id) to avoid data leakage.
   - Train cases: 1, 2, 3, 4, 5, 6, 7, 8
   - Test cases: 9, 10
3. Trains a RandomForestClassifier with balanced class weights.
4. Evaluates the model on test cases (accuracy, precision, recall, F1, ROC-AUC, confusion matrix).
5. Saves the new model to models/ppg_af_rf.joblib (overwriting the old MIMIC model)
   and saves metadata to models/ppg_af_rf_metadata.json.
"""

from __future__ import annotations

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Define paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONSOLIDATED_CSV = PROJECT_ROOT / "outputs" / "vilnius_features_consolidated.csv"
MODEL_DIR        = PROJECT_ROOT / "models"
MODEL_PATH       = MODEL_DIR / "ppg_af_rf.joblib"
METADATA_PATH    = MODEL_DIR / "ppg_af_rf_metadata.json"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Hyperparameters
RF_PARAMS = dict(
    n_estimators=100,        # 100 trees is sufficient and keeps model size under control
    max_depth=15,            # Limit depth to avoid overfitting and keep memory usage reasonable
    min_samples_leaf=10,     # Stop splitting early to generalize better
    class_weight="balanced", # Handle class imbalance (Normal >> AF)
    random_state=42,
    n_jobs=-1,               # Use all available CPU cores
)

# Feature names (must match the exact ones used in training and app)
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
]

def main() -> None:
    print("=" * 60)
    print("  Retraining AF Model on Vilnius Smartwatch PPG Dataset")
    print("=" * 60)

    # 1. Load consolidated feature matrix
    if not CONSOLIDATED_CSV.exists():
        print(f"Error: Consolidated features file not found at {CONSOLIDATED_CSV}")
        return
        
    print(f"Loading feature matrix from {CONSOLIDATED_CSV}...")
    df = pd.read_csv(CONSOLIDATED_CSV)
    print(f"Loaded {len(df)} records.")
    print("Class Balance:")
    print(df["label"].value_counts())

    # 2. Patient-level train-test split (Group Split)
    # We use cases 1 to 8 for training, and cases 9 and 10 for testing
    train_cases = [1, 2, 3, 4, 5, 6, 7, 8]
    test_cases = [9, 10]
    
    print(f"\nPerforming patient-level split:")
    print(f"  Training Patients : {train_cases}")
    print(f"  Testing Patients  : {test_cases}")

    train_mask = df["case_id"].isin(train_cases)
    test_mask = df["case_id"].isin(test_cases)

    X_train = df.loc[train_mask, FEATURE_COLS]
    y_train = df.loc[train_mask, "label"]

    X_test = df.loc[test_mask, FEATURE_COLS]
    y_test = df.loc[test_mask, "label"]

    print(f"\nTrain Set size: {len(X_train)} (Normal: {int((y_train==0).sum())} | AF: {int((y_train==1).sum())})")
    print(f"Test Set size : {len(X_test)} (Normal: {int((y_test==0).sum())} | AF: {int((y_test==1).sum())})")

    # 3. Train RandomForest model
    print(f"\nTraining RandomForest model with parameters:\n{json.dumps(RF_PARAMS, indent=2)}...")
    clf = RandomForestClassifier(**RF_PARAMS)
    clf.fit(X_train, y_train)
    print("Training completed successfully.")

    # 4. Evaluate model
    print("\nEvaluating model on Test Patients (Case 9 & 10)...")
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    # Metrics
    acc = clf.score(X_test, y_test)
    auc = roc_auc_score(y_test, y_proba)
    
    print(f"\nTest Accuracy : {acc:.4f}")
    print(f"Test ROC-AUC  : {auc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "AF"]))
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print("Confusion Matrix:")
    print(f"              Normal    AF")
    print(f"  Normal  :   {tn:6d}  {fp:6d}")
    print(f"  AF      :   {fn:6d}  {tp:6d}")

    # Feature Importances
    importances = pd.Series(clf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print("\nFeature Importances:")
    for feat, imp in importances.items():
        bar = "#" * int(imp * 50)
        print(f"  {feat:<15} : {imp:.4f}  {bar}")

    # 5. Save model and metadata
    print(f"\nSaving model to {MODEL_PATH}...")
    joblib.dump(clf, MODEL_PATH)
    print("Model saved.")

    metadata = {
        "model_type": "RandomForestClassifier",
        "dataset": "Vilnius University Wrist PPG Dataset (Patients 001-008)",
        "train_patients": train_cases,
        "test_patients": test_cases,
        "sampling_rate_hz": 100.0,
        "features": FEATURE_COLS,
        "class_mapping": {"0": "Normal", "1": "AF"},
        "hyperparameters": {k: str(v) for k, v in RF_PARAMS.items()}
    }
    
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {METADATA_PATH}")
    print("\nDONE. The Streamlit app is now ready to use this newly trained model.")

if __name__ == "__main__":
    main()
