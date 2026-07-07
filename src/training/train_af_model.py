"""Train a Random Forest classifier for AF detection from MIMIC PPG recordings.

Strategy
--------
* **Evaluation**  : 5-fold recording-level cross-validation (stratified by label)
  to get a fair, low-variance performance estimate across ALL 35 recordings.
* **Final model** : trained on ALL 35 recordings so every piece of data
  contributes to the deployed model.

Usage
-----
    python code/train_af_model.py

Outputs
-------
    models/ppg_af_rf.joblib              – final model trained on ALL recordings
    models/ppg_af_rf_metadata.json       – window/feature config
    reports/rf_cv_evaluation.txt         – per-fold + averaged CV metrics
    reports/rf_cv_confusion_matrix.csv   – aggregated CV confusion matrix
    reports/rf_feature_importance.csv    – feature importances from final model
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ARCHIVE_ROOT = PROJECT_ROOT / "data" / "raw" / "mimic"
MODELS_DIR   = PROJECT_ROOT / "models"
REPORTS_DIR  = PROJECT_ROOT / "reports"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJECT_ROOT / "src"))
from pipeline.ppg_pipeline import (  # noqa: E402
    build_feature_matrix,
    get_recording_fs,
    get_recording_label,
    get_recording_signal,
    load_archive_dataset,
    preprocess_ppg,
    segment_signal,
)

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def print_section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def build_features_for_records(recs: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    """Preprocess, segment, and extract features for a list of recordings.
    Uses the per-recording sampling rate detected from the Time column,
    and resamples to the training target sampling rate of 125 Hz.
    """
    dfs: list[pd.DataFrame] = []
    ys:  list[np.ndarray]   = []
    target_fs = 125
    for rec in recs:
        fs        = get_recording_fs(rec)
        sig       = get_recording_signal(rec)
        lbl       = get_recording_label(rec)
        processed = preprocess_ppg(sig, fs=fs, target_fs=target_fs)
        windows   = segment_signal(
            processed, fs=target_fs, window_sec=WINDOW_SEC, overlap_sec=OVERLAP_SEC
        )
        if len(windows) == 0:
            continue
        feat_df = build_feature_matrix(windows, fs=target_fs)
        dfs.append(feat_df)
        ys.append(np.full(len(windows), lbl, dtype=int))
    return pd.concat(dfs, ignore_index=True), np.concatenate(ys)


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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:

    # ------------------------------------------------------------------
    # 1. Load ALL recordings
    # ------------------------------------------------------------------
    print_section("1. Loading ALL recordings")
    records = load_archive_dataset(ARCHIVE_ROOT)
    rec_labels = np.array([int(r["label"]) for r in records])

    n_af     = int((rec_labels == 1).sum())
    n_non_af = int((rec_labels == 0).sum())
    print(f"  Total recordings : {len(records)}")
    print(f"  AF recordings    : {n_af}")
    print(f"  Non-AF recordings: {n_non_af}")

    # ------------------------------------------------------------------
    # 2. Extract features for ALL recordings (once, reused in each fold)
    # ------------------------------------------------------------------
    print_section("2. Preprocessing & feature extraction (all recordings)")
    print(f"  window_sec  = {WINDOW_SEC}s")
    print(f"  overlap_sec = {OVERLAP_SEC}s")

    all_feat_dfs: list[pd.DataFrame] = []
    all_labels:   list[np.ndarray]   = []
    rec_window_counts: list[int]      = []   # how many windows each recording produced

    for i, rec in enumerate(records):
        lbl  = int(rec["label"])
        fs   = get_recording_fs(rec)
        sig  = get_recording_signal(rec)
        proc = preprocess_ppg(sig, fs=fs)
        wins = segment_signal(proc, fs=fs, window_sec=WINDOW_SEC, overlap_sec=OVERLAP_SEC)
        n_w  = len(wins)
        rec_window_counts.append(n_w)
        if n_w == 0:
            continue
        feat_df = build_feature_matrix(wins, fs=fs)
        all_feat_dfs.append(feat_df)
        all_labels.append(np.full(n_w, lbl, dtype=int))
        label_str = "AF" if lbl == 1 else "Non-AF"
        print(f"  [{i+1:02d}/{len(records)}] {label_str:6s}  {n_w:4d} windows  fs={fs}Hz", flush=True)

    X_all = pd.concat(all_feat_dfs, ignore_index=True)
    y_all = np.concatenate(all_labels)
    print(f"\n  Total windows : {len(X_all)}  "
          f"(AF={int((y_all==1).sum())}, Non-AF={int((y_all==0).sum())})")

    # Build a mapping from recording index → window row indices in X_all
    rec_to_rows: list[np.ndarray] = []
    cursor = 0
    for count in rec_window_counts:
        if count > 0:
            rec_to_rows.append(np.arange(cursor, cursor + count))
            cursor += count
        else:
            rec_to_rows.append(np.array([], dtype=int))

    # ------------------------------------------------------------------
    # 3. 5-Fold recording-level cross-validation
    # ------------------------------------------------------------------
    print_section(f"3. {N_FOLDS}-Fold recording-level cross-validation")

    splits = stratified_kfold_recording_indices(rec_labels, n_folds=N_FOLDS)

    fold_metrics: list[dict] = []
    cm_total = np.zeros((2, 2), dtype=int)

    for fold, (train_rec_idx, val_rec_idx) in enumerate(splits):
        # Gather window indices for train / val recordings
        train_row_idx = np.concatenate([rec_to_rows[i] for i in train_rec_idx if len(rec_to_rows[i]) > 0])
        val_row_idx   = np.concatenate([rec_to_rows[i] for i in val_rec_idx   if len(rec_to_rows[i]) > 0])

        X_tr, y_tr = X_all.iloc[train_row_idx], y_all[train_row_idx]
        X_val, y_val = X_all.iloc[val_row_idx], y_all[val_row_idx]

        val_af     = int((y_val == 1).sum())
        val_non_af = int((y_val == 0).sum())
        val_recs   = len(val_rec_idx)

        clf_fold = RandomForestClassifier(**RF_PARAMS)
        clf_fold.fit(X_tr, y_tr)

        y_pred  = clf_fold.predict(X_val)
        y_proba = clf_fold.predict_proba(X_val)[:, 1]

        acc = float(np.mean(y_pred == y_val))
        auc = float(roc_auc_score(y_val, y_proba))
        cm  = confusion_matrix(y_val, y_pred)
        cm_total += cm

        tn, fp, fn, tp = cm.ravel()
        recall_af     = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        recall_non_af = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1_af         = 2*tp / (2*tp + fp + fn) if (2*tp + fp + fn) > 0 else 0.0

        fold_metrics.append({
            "fold": fold + 1,
            "val_recs": val_recs, "val_AF": val_af, "val_NonAF": val_non_af,
            "accuracy": acc, "roc_auc": auc,
            "recall_AF": recall_af, "recall_NonAF": recall_non_af, "f1_AF": f1_af,
        })

        print(f"\n  Fold {fold+1}/{N_FOLDS}  "
              f"(val: {val_recs} recordings, AF={val_af} wins, Non-AF={val_non_af} wins)")
        print(f"    Accuracy  : {acc:.4f}")
        print(f"    ROC-AUC   : {auc:.4f}")
        print(f"    AF  Recall: {recall_af:.4f}")
        print(f"    Non-AF Rec: {recall_non_af:.4f}")
        print(f"    AF  F1    : {f1_af:.4f}")

    # Aggregate CV results
    mean_acc     = float(np.mean([m["accuracy"]    for m in fold_metrics]))
    std_acc      = float(np.std( [m["accuracy"]    for m in fold_metrics]))
    mean_auc     = float(np.mean([m["roc_auc"]     for m in fold_metrics]))
    std_auc      = float(np.std( [m["roc_auc"]     for m in fold_metrics]))
    mean_rec_af  = float(np.mean([m["recall_AF"]   for m in fold_metrics]))
    mean_rec_naf = float(np.mean([m["recall_NonAF"]for m in fold_metrics]))
    mean_f1_af   = float(np.mean([m["f1_AF"]       for m in fold_metrics]))

    print(f"\n  {'-'*50}")
    print(f"  CV Summary ({N_FOLDS} folds):")
    print(f"    Accuracy     : {mean_acc:.4f} +/- {std_acc:.4f}")
    print(f"    ROC-AUC      : {mean_auc:.4f} +/- {std_auc:.4f}")
    print(f"    AF  Recall   : {mean_rec_af:.4f}")
    print(f"    Non-AF Recall: {mean_rec_naf:.4f}")
    print(f"    AF  F1-score : {mean_f1_af:.4f}")
    print(f"  {'-'*50}")

    # CV confusion matrix (aggregated)
    tn, fp, fn, tp = cm_total.ravel()
    print(f"\n  Aggregated CV Confusion Matrix:")
    print(f"              Non-AF    AF")
    print(f"  Non-AF  :   {tn:6d}  {fp:6d}")
    print(f"  AF      :   {fn:6d}  {tp:6d}")

    # ------------------------------------------------------------------
    # 4. Train FINAL model on ALL recordings
    # ------------------------------------------------------------------
    print_section("4. Training FINAL model on ALL recordings")
    print(f"  Using all {len(records)} recordings, {len(X_all)} windows")

    clf_final = RandomForestClassifier(**RF_PARAMS)
    clf_final.fit(X_all, y_all)
    print("  Final model training complete.")

    # Feature importances
    importances = pd.Series(clf_final.feature_importances_, index=X_all.columns)
    importances = importances.sort_values(ascending=False)
    print()
    print("  Feature importances (final model):")
    for feat, imp in importances.items():
        bar = "#" * int(imp * 50)
        print(f"    {feat:<20s} {imp:.4f}  {bar}")

    # ------------------------------------------------------------------
    # 5. Save model & reports
    # ------------------------------------------------------------------
    print_section("5. Saving model & reports")

    # Model
    model_path = MODELS_DIR / "ppg_af_rf.joblib"
    joblib.dump(clf_final, model_path)
    print(f"  Model saved        : {model_path}")

    # Metadata
    metadata = {
        "model_type"        : "RandomForestClassifier",
        "labels"            : {"0": "Non-AF", "1": "AF"},
        "train_mode"        : "full_data_all_recordings",
        "n_recordings"      : int(len(records)),
        "n_AF_recordings"   : int(n_af),
        "n_NonAF_recordings": int(n_non_af),
        "total_windows"     : int(len(X_all)),
        "window_sec"        : WINDOW_SEC,
        "overlap_sec"       : OVERLAP_SEC,
        "bandpass"          : [0.5, 8.0],
        "features"          : list(X_all.columns),
        "n_estimators"      : RF_PARAMS["n_estimators"],
        "cv_folds"          : N_FOLDS,
        "cv_accuracy_mean"  : round(mean_acc, 4),
        "cv_accuracy_std"   : round(std_acc, 4),
        "cv_roc_auc_mean"   : round(mean_auc, 4),
        "cv_roc_auc_std"    : round(std_auc, 4),
        "cv_recall_AF"      : round(mean_rec_af, 4),
        "cv_recall_NonAF"   : round(mean_rec_naf, 4),
        "cv_f1_AF"          : round(mean_f1_af, 4),
    }
    meta_path = MODELS_DIR / "ppg_af_rf_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"  Metadata saved     : {meta_path}")

    # CV evaluation report
    cv_report_lines = [
        "Random Forest AF Detection — Cross-Validation Report",
        "=" * 60,
        f"Strategy    : {N_FOLDS}-fold recording-level CV + full-data final model",
        f"Recordings  : {len(records)} total (AF={n_af}, Non-AF={n_non_af})",
        f"Total windows: {len(X_all)}",
        "",
        "Per-fold results:",
        f"  {'Fold':>4}  {'Val recs':>8}  {'Acc':>7}  {'AUC':>7}  {'AF Rec':>7}  {'NAF Rec':>7}  {'AF F1':>7}",
        f"  {'----':>4}  {'--------':>8}  {'-------':>7}  {'-------':>7}  {'-------':>7}  {'-------':>7}  {'-------':>7}",
    ]
    for m in fold_metrics:
        cv_report_lines.append(
            f"  {m['fold']:>4}  {m['val_recs']:>8}  {m['accuracy']:>7.4f}  "
            f"{m['roc_auc']:>7.4f}  {m['recall_AF']:>7.4f}  "
            f"{m['recall_NonAF']:>7.4f}  {m['f1_AF']:>7.4f}"
        )
    cv_report_lines += [
        f"  {'----':>4}  {'--------':>8}  {'-------':>7}  {'-------':>7}  {'-------':>7}  {'-------':>7}  {'-------':>7}",
        f"  {'MEAN':>4}  {'':>8}  {mean_acc:>7.4f}  {mean_auc:>7.4f}  "
        f"{mean_rec_af:>7.4f}  {mean_rec_naf:>7.4f}  {mean_f1_af:>7.4f}",
        f"  {'STD':>4}  {'':>8}  {std_acc:>7.4f}  {std_auc:>7.4f}",
        "",
        "Aggregated CV Confusion Matrix (all folds combined):",
        f"              Non-AF    AF",
        f"  Non-AF  :   {tn:6d}  {fp:6d}",
        f"  AF      :   {fn:6d}  {tp:6d}",
        "",
        "Feature Importances (final model trained on all data):",
    ]
    for feat, imp in importances.items():
        cv_report_lines.append(f"  {feat:<20s} {imp:.4f}")

    report_path = REPORTS_DIR / "rf_cv_evaluation.txt"
    report_path.write_text("\n".join(cv_report_lines), encoding="utf-8")
    print(f"  CV report saved    : {report_path}")

    # Aggregated CV confusion matrix CSV
    cm_df = pd.DataFrame(
        cm_total,
        index=["Actual Non-AF", "Actual AF"],
        columns=["Pred Non-AF", "Pred AF"],
    )
    cm_path = REPORTS_DIR / "rf_cv_confusion_matrix.csv"
    cm_df.to_csv(cm_path)
    print(f"  CV conf. matrix    : {cm_path}")

    # Feature importance CSV
    fi_path = REPORTS_DIR / "rf_feature_importance.csv"
    importances.reset_index().rename(
        columns={"index": "feature", 0: "importance"}
    ).to_csv(fi_path, index=False)
    print(f"  Feature importance : {fi_path}")

    print_section("DONE")
    print(f"  Final model        : {model_path}")
    print(f"  CV Accuracy        : {mean_acc:.4f} +/- {std_acc:.4f}")
    print(f"  CV ROC-AUC         : {mean_auc:.4f} +/- {std_auc:.4f}")
    print(f"  CV AF  Recall      : {mean_rec_af:.4f}")
    print(f"  CV Non-AF Recall   : {mean_rec_naf:.4f}")
    print(f"  Run Streamlit app  : streamlit run src/app/ppg_app.py")


if __name__ == "__main__":
    main()

