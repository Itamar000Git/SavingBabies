import os
import re
import glob
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)

from xgboost import XGBClassifier


# ============================================================
# CONFIG
# ============================================================

CSV_DIR = "csv_output"
DATASET_DIR = "dataset"
PH_FILE = "dataset/ph_levels.csv"

FS = 4
PH_DANGER_THRESHOLD = 7.10
TARGET_RECALL = 0.60

RANDOM_STATE = 42
PRINT_EVERY = 50

# Minimum threshold, to avoid a useless threshold like 0.01
MIN_THRESHOLD = 0.05
MAX_THRESHOLD = 0.95

# Multi-window feature extraction
# None means full recording.
WINDOWS = {
    "last5": 5 * 60 * FS,
    "last10": 10 * 60 * FS,
    "last20": 20 * 60 * FS,
    "full": None,
}

# Web app artifacts
WEIGHTS_DIR = Path("fetal_health_web_app") / "backend" / "weights"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = WEIGHTS_DIR / "xgboost_model.joblib"
STATS_PATH = WEIGHTS_DIR / "xgboost_stats.json"


# ============================================================
# REAL-TIME SAFE METADATA
# ============================================================

META_FEATURES = [
    ("gestational_weeks", "Gest. weeks"),
    ("mother_age", "Age"),
    ("gravidity", "Gravidity"),
    ("parity", "Parity"),
    ("diabetes", "Diabetes"),
    ("hypertension", "Hypertension"),
    ("preeclampsia", "Preeclampsia"),
    ("liq_praecox", "Liq. praecox"),
    ("pyrexia", "Pyrexia"),
    ("meconium", "Meconium"),
    ("presentation", "Presentation"),
    ("induced", "Induced"),
]

META_FIELD_NAMES = [name for name, _ in META_FEATURES]


# ============================================================
# LOAD LABELS
# ============================================================

ph_df = pd.read_csv(PH_FILE)
ph_df["record_id"] = ph_df["record_id"].astype(str)
ph_dict = dict(zip(ph_df["record_id"], ph_df["pH"]))


# ============================================================
# CLEANING HELPERS
# ============================================================

def last_window_or_pad(arr, seq_len, pad):
    arr = arr.astype(np.float32)

    if len(arr) >= seq_len:
        return arr[-seq_len:]

    diff = seq_len - len(arr)
    return np.pad(arr, (0, diff), constant_values=pad)


def take_window(arr, window_len, pad_value):
    """
    If window_len is None, return the full array.
    Otherwise return last window_len samples, padded if needed.
    """
    arr = arr.astype(np.float32)

    if window_len is None:
        return arr

    return last_window_or_pad(arr, window_len, pad_value)


def clean_fhr(fhr):
    """
    FHR cleaning:
    - 0, negative values, NaN are invalid.
    - interpolate invalid values.
    - fallback to 140 bpm.
    """
    fhr = np.asarray(fhr, dtype=np.float32)

    return (
        pd.Series(fhr)
        .mask((fhr <= 0) | np.isnan(fhr), np.nan)
        .interpolate()
        .fillna(140)
        .values.astype(np.float32)
    )


def clean_uc(uc):
    """
    UC cleaning:
    - zeros can be real.
    - only NaN values are interpolated/fillna.
    """
    uc = np.asarray(uc, dtype=np.float32)

    return (
        pd.Series(uc)
        .interpolate()
        .fillna(0)
        .values.astype(np.float32)
    )


# ============================================================
# METADATA PARSER
# ============================================================

def extract_number_from_text(text):
    match = re.search(r"-?\d+(\.\d+)?", str(text))
    if not match:
        return np.nan
    return float(match.group(0))


def read_realtime_metadata(record_id):
    """
    Reads dataset/{record_id}.hea and extracts only real-time-safe metadata.

    Missing fields are returned as np.nan.
    XGBoost can handle missing values.
    """
    hea_path = Path(DATASET_DIR) / f"{record_id}.hea"

    values = {field_name: np.nan for field_name in META_FIELD_NAMES}

    if not hea_path.exists():
        return values

    try:
        lines = hea_path.read_text(encoding="utf-8", errors="ignore").splitlines()

        for field_name, hea_label in META_FEATURES:
            prefix = f"#{hea_label}"

            for line in lines:
                clean_line = line.strip()

                if clean_line.startswith(prefix):
                    raw_value = clean_line[len(prefix):].strip()
                    values[field_name] = extract_number_from_text(raw_value)
                    break

    except Exception as e:
        print(f"Warning: failed to parse metadata for record {record_id}: {e}")

    return values


# ============================================================
# CTG FEATURE HELPERS
# ============================================================

def accelerations(fhr, baseline):
    return int(np.sum(fhr > baseline + 15))


def decelerations(fhr, baseline):
    return int(np.sum(fhr < baseline - 15))


def stv(fhr):
    if len(fhr) < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(fhr))))


def ltv(fhr):
    window = 120
    vals = []

    for i in range(0, len(fhr) - window, window):
        vals.append(np.std(fhr[i:i + window]))

    return float(np.mean(vals)) if vals else float(np.std(fhr))


def safe_slope(fhr):
    if len(fhr) < 2:
        return 0.0

    t = np.arange(len(fhr))

    try:
        return float(np.polyfit(t, fhr, 1)[0])
    except Exception:
        return 0.0


def extract_features_for_window(fhr_clean, uc_clean, fhr_valid_mask, prefix):
    """
    Extract CTG summary features from one specific window.
    Prefix is used to avoid name collisions:
    last5_fhr_mean, last10_fhr_mean, full_fhr_mean, etc.
    """
    feats = {}

    if len(fhr_clean) == 0:
        # Should not happen, but keep it safe.
        for name in [
            "fhr_mean", "fhr_std", "fhr_min", "fhr_max", "fhr_range",
            "fhr_p05", "fhr_p95", "fhr_slope", "stv", "ltv",
            "accelerations", "decelerations", "deceleration_ratio",
            "acceleration_ratio", "uc_mean", "uc_std", "uc_max",
            "uc_activity_ratio", "missing_signal_pct", "valid_fhr_ratio",
        ]:
            feats[f"{prefix}_{name}"] = 0.0
        return feats

    baseline = np.median(fhr_clean)

    missing_signal_pct = float(100.0 * np.mean(fhr_valid_mask == 0))
    valid_fhr_ratio = float(np.mean(fhr_valid_mask == 1))

    acc_count = accelerations(fhr_clean, baseline)
    dec_count = decelerations(fhr_clean, baseline)

    n = max(len(fhr_clean), 1)

    feats[f"{prefix}_fhr_mean"] = float(np.mean(fhr_clean))
    feats[f"{prefix}_fhr_std"] = float(np.std(fhr_clean))
    feats[f"{prefix}_fhr_min"] = float(np.min(fhr_clean))
    feats[f"{prefix}_fhr_max"] = float(np.max(fhr_clean))
    feats[f"{prefix}_fhr_range"] = float(np.max(fhr_clean) - np.min(fhr_clean))

    feats[f"{prefix}_fhr_p05"] = float(np.percentile(fhr_clean, 5))
    feats[f"{prefix}_fhr_p95"] = float(np.percentile(fhr_clean, 95))

    feats[f"{prefix}_fhr_slope"] = safe_slope(fhr_clean)

    feats[f"{prefix}_stv"] = stv(fhr_clean)
    feats[f"{prefix}_ltv"] = ltv(fhr_clean)

    feats[f"{prefix}_accelerations"] = int(acc_count)
    feats[f"{prefix}_decelerations"] = int(dec_count)

    feats[f"{prefix}_acceleration_ratio"] = float(acc_count / n)
    feats[f"{prefix}_deceleration_ratio"] = float(dec_count / n)

    feats[f"{prefix}_uc_mean"] = float(np.mean(uc_clean))
    feats[f"{prefix}_uc_std"] = float(np.std(uc_clean))
    feats[f"{prefix}_uc_max"] = float(np.max(uc_clean))
    feats[f"{prefix}_uc_activity_ratio"] = float(np.mean(uc_clean > 0))

    feats[f"{prefix}_missing_signal_pct"] = missing_signal_pct
    feats[f"{prefix}_valid_fhr_ratio"] = valid_fhr_ratio

    return feats


def extract_multi_window_features(fhr_raw, uc_raw):
    """
    Main CTG feature extraction.

    Instead of using only the last 5 minutes, extract features from:
    - last 5 minutes
    - last 10 minutes
    - last 20 minutes
    - full recording
    """
    fhr_raw = np.asarray(fhr_raw, dtype=np.float32)
    uc_raw = np.asarray(uc_raw, dtype=np.float32)

    fhr_valid_mask_raw = np.ones_like(fhr_raw, dtype=np.float32)
    fhr_valid_mask_raw[(fhr_raw <= 0) | np.isnan(fhr_raw)] = 0.0

    fhr_clean_full = clean_fhr(fhr_raw)
    uc_clean_full = clean_uc(uc_raw)

    all_features = {}

    for prefix, window_len in WINDOWS.items():
        if window_len is None:
            fhr_w = fhr_clean_full
            uc_w = uc_clean_full
            mask_w = fhr_valid_mask_raw
        else:
            fhr_w = take_window(fhr_clean_full, window_len, 140)
            uc_w = take_window(uc_clean_full, window_len, 0)
            mask_w = take_window(fhr_valid_mask_raw, window_len, 0)

        window_features = extract_features_for_window(
            fhr_clean=fhr_w,
            uc_clean=uc_w,
            fhr_valid_mask=mask_w,
            prefix=prefix,
        )

        all_features.update(window_features)

    return all_features


# ============================================================
# THRESHOLD SELECTION
# ============================================================

def choose_threshold_for_recall(y_true, y_prob, target_recall=0.60):
    """
    Choose threshold using VALIDATION set.

    Scan thresholds and pick one that:
    - reaches recall >= target_recall for Danger
    - among those, maximizes Danger precision

    If no threshold reaches target recall, choose threshold with max recall.

    Note:
    The minimum threshold is intentionally not below MIN_THRESHOLD,
    to avoid classifying nearly everything as Danger.
    """
    thresholds = np.linspace(MIN_THRESHOLD, MAX_THRESHOLD, 91)

    best_thr = 0.5
    best_precision = -1.0
    best_recall = -1.0

    best_recall_thr = 0.5
    best_recall_value = -1.0

    for thr in thresholds:
        pred = (np.array(y_prob) >= thr).astype(int)

        rec = recall_score(y_true, pred, pos_label=1, zero_division=0)
        pre = precision_score(y_true, pred, pos_label=1, zero_division=0)

        if rec > best_recall_value:
            best_recall_value = rec
            best_recall_thr = thr

        if rec >= target_recall and pre > best_precision:
            best_precision = pre
            best_recall = rec
            best_thr = thr

    if best_precision < 0:
        pred = (np.array(y_prob) >= best_recall_thr).astype(int)
        return (
            float(best_recall_thr),
            float(best_recall_value),
            float(precision_score(y_true, pred, pos_label=1, zero_division=0)),
        )

    return float(best_thr), float(best_recall), float(best_precision)


def print_danger_only_evaluation(y_true, y_prob, threshold, title):
    pred = (np.array(y_prob) >= threshold).astype(int)

    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(y_true, pred)
    danger_precision = precision_score(y_true, pred, pos_label=1, zero_division=0)
    danger_recall = recall_score(y_true, pred, pos_label=1, zero_division=0)
    danger_f1 = f1_score(y_true, pred, pos_label=1, zero_division=0)

    pr_auc = average_precision_score(y_true, y_prob)

    if len(set(y_true)) == 2:
        roc_auc = roc_auc_score(y_true, y_prob)
    else:
        roc_auc = None

    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    print(f"Decision threshold: {threshold:.3f}")
    print()

    print("Confusion Matrix:")
    print(f"TN = {tn} | FP = {fp}")
    print(f"FN = {fn} | TP = {tp}")
    print()

    print("Danger Metrics Only:")
    print(f"Accuracy:          {accuracy:.3f}")
    print(f"Danger Precision:  {danger_precision:.3f}")
    print(f"Danger Recall:     {danger_recall:.3f}")
    print(f"Danger F1-score:   {danger_f1:.3f}")
    print()

    print("Medical Risk View:")
    print(f"False Negatives - Danger predicted as Normal: {fn}")
    print(f"False Positives - Normal predicted as Danger: {fp}")
    print()

    print(f"PR-AUC:            {pr_auc:.3f}")

    if roc_auc is not None:
        print(f"ROC-AUC:           {roc_auc:.3f}")
    else:
        print("ROC-AUC:           not defined")

    print("=" * 60)

    return {
        "accuracy": float(accuracy),
        "danger_precision": float(danger_precision),
        "danger_recall": float(danger_recall),
        "danger_f1": float(danger_f1),
        "false_negatives": int(fn),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "true_positives": int(tp),
        "pr_auc": float(pr_auc),
        "roc_auc": None if roc_auc is None else float(roc_auc),
    }


def print_threshold_table(y_true, y_prob):
    """
    Analysis only.
    Helps understand whether a useful threshold exists.
    """
    print("\nThreshold analysis on TEST set:")
    print("thr   | TP | FP | FN | TN | Recall | Precision")
    print("-" * 55)

    for thr in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        pred = (np.array(y_prob) >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()

        rec = recall_score(y_true, pred, pos_label=1, zero_division=0)
        pre = precision_score(y_true, pred, pos_label=1, zero_division=0)

        print(f"{thr:0.2f} | {tp:2d} | {fp:2d} | {fn:2d} | {tn:2d} | {rec:0.3f}  | {pre:0.3f}")


# ============================================================
# LOAD DATA
# ============================================================

print("Loading recordings...")

files = glob.glob(os.path.join(CSV_DIR, "*.csv"))
rows = []

for i, file in enumerate(files):
    record_id = os.path.splitext(os.path.basename(file))[0]

    if record_id not in ph_dict:
        continue

    try:
        df = pd.read_csv(file)

        if "FHR" not in df.columns or "UC" not in df.columns:
            continue

        fhr_raw = df["FHR"].values
        uc_raw = df["UC"].values

        feats = extract_multi_window_features(fhr_raw, uc_raw)

        meta_feats = read_realtime_metadata(record_id)
        feats.update(meta_feats)

        ph_val = float(ph_dict[record_id])
        label = 1 if ph_val < PH_DANGER_THRESHOLD else 0

        feats["record_id"] = record_id
        feats["y"] = label

        rows.append(feats)

        if i % PRINT_EVERY == 0:
            print("processed", i)

    except Exception as e:
        print(f"Failed to process {file}: {e}")

feat_df = pd.DataFrame(rows)

if len(feat_df) == 0:
    raise RuntimeError("No valid records were loaded. Check csv_output and dataset/ph_levels.csv.")

print("Dataset:", len(feat_df))
print("Class counts (Normal=0, Danger=1):", np.bincount(feat_df["y"]))

print("\nWindows used:")
for name, length in WINDOWS.items():
    if length is None:
        print(f"- {name}: full recording")
    else:
        print(f"- {name}: {length / FS / 60:.1f} minutes")

print("\nMetadata fields used:")
for field in META_FIELD_NAMES:
    print(f"- {field}")


# ============================================================
# SPLIT: TRAIN / VALIDATION / TEST
# ============================================================

X = feat_df.drop(columns=["y", "record_id"])
y = feat_df["y"].astype(int)

feature_names = list(X.columns)

X_temp, X_test, y_temp, y_test = train_test_split(
    X,
    y,
    test_size=0.15,
    stratify=y,
    random_state=RANDOM_STATE,
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp,
    y_temp,
    test_size=0.15,
    stratify=y_temp,
    random_state=RANDOM_STATE,
)

print("\nSplit summary:")
print(f"Train class counts: {np.bincount(y_train)}")
print(f"Val class counts:   {np.bincount(y_val)}")
print(f"Test class counts:  {np.bincount(y_test)}")


# ============================================================
# CLASS IMBALANCE
# ============================================================

scale_pos_weight = np.sum(y_train == 0) / max(np.sum(y_train == 1), 1)

print("\nClass weighting:")
print(f"scale_pos_weight = {scale_pos_weight:.3f}")


# ============================================================
# MODEL
# ============================================================

print("\nTraining XGBoost multi-window model...")

model = XGBClassifier(
    n_estimators=900,
    max_depth=5,
    learning_rate=0.02,
    subsample=0.9,
    colsample_bytree=0.9,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    n_jobs=-1,
    random_state=RANDOM_STATE,
)

model.fit(X_train, y_train)


# ============================================================
# THRESHOLD SELECTION ON VALIDATION
# ============================================================

val_prob = model.predict_proba(X_val)[:, 1]

best_threshold, val_recall, val_precision = choose_threshold_for_recall(
    y_val,
    val_prob,
    TARGET_RECALL,
)

print("\n--- Threshold selection on VALIDATION ---")
print(f"Target Danger Recall: {TARGET_RECALL:.2f}")
print(f"Chosen threshold:     {best_threshold:.3f}")
print(f"Val Danger Recall:    {val_recall:.3f}")
print(f"Val Danger Precision: {val_precision:.3f}")


# ============================================================
# TEST EVALUATION - DANGER ONLY
# ============================================================

test_prob = model.predict_proba(X_test)[:, 1]

test_metrics = print_danger_only_evaluation(
    y_true=y_test,
    y_prob=test_prob,
    threshold=best_threshold,
    title="TEST RESULTS SUMMARY - XGBoost Multi-Window CTG + Metadata",
)

print_threshold_table(y_test, test_prob)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = model.feature_importances_
importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importance,
}).sort_values("importance", ascending=False)

print("\nTop 20 Feature Importances:")
for _, row in importance_df.head(20).iterrows():
    print(f"{row['feature']}: {row['importance']:.4f}")


# ============================================================
# SAVE ARTIFACTS FOR WEB APP
# ============================================================

joblib.dump(model, MODEL_PATH)

stats = {
    "model_type": "XGBoostMultiWindowCTGWithMetadata",

    "feature_names": feature_names,

    "windows": {
        name: None if length is None else int(length)
        for name, length in WINDOWS.items()
    },

    "window_descriptions": {
        name: "full recording" if length is None else f"{length / FS / 60:.1f} minutes"
        for name, length in WINDOWS.items()
    },

    "metadata_fields": META_FIELD_NAMES,

    "fs": FS,

    "threshold": float(best_threshold),
    "min_threshold": float(MIN_THRESHOLD),
    "target_recall": float(TARGET_RECALL),
    "ph_danger_threshold": float(PH_DANGER_THRESHOLD),

    "scale_pos_weight": float(scale_pos_weight),

    "test_metrics": test_metrics,

    "top_feature_importances": importance_df.head(20).to_dict(orient="records"),

    "metadata_leakage_excluded_fields": [
        "pH",
        "BDecf",
        "pCO2",
        "BE",
        "Apgar1",
        "Apgar5",
        "NICU days",
        "Seizures",
        "HIE",
        "Intubation",
        "Main diag.",
        "Other diag.",
        "Sig2Birth",
        "Deliv. type",
        "Weight(g)",
    ],
}

with open(STATS_PATH, "w", encoding="utf-8") as f:
    json.dump(stats, f, indent=2)

print("\nSaved artifacts for web app:")
print(f"Model: {MODEL_PATH}")
print(f"Stats: {STATS_PATH}")

print("\nImportant:")
print("To use this model in the web app, update/add an XGBoost adapter that:")
print("1. Extracts the same multi-window CTG features from the uploaded CSV.")
print("2. Reads the same metadata fields from the matching .hea file.")
print("3. Orders features exactly according to xgboost_stats.json['feature_names'].")
print("4. Loads xgboost_model.joblib and applies the saved threshold.")