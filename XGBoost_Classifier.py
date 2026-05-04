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

SEQ_LEN = 1200                 # 5 minutes at 4Hz
FS = 4

PH_DANGER_THRESHOLD = 7.10     # label: Danger if pH < 7.10
TARGET_RECALL = 0.70           # choose threshold on validation to reach this Danger recall

RANDOM_STATE = 42
PRINT_EVERY = 50

# Web app artifacts
WEIGHTS_DIR = Path("fetal_health_web_app") / "backend" / "weights"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = WEIGHTS_DIR / "xgboost_model.joblib"
STATS_PATH = WEIGHTS_DIR / "xgboost_stats.json"


# ============================================================
# REAL-TIME SAFE METADATA
# ============================================================
# Use only fields that are reasonably available before/during labor.
# Do NOT use outcome/leakage fields:
# pH, BDecf, pCO2, BE, Apgar1, Apgar5, NICU days, Seizures,
# HIE, Intubation, Main diag., Other diag., Sig2Birth, Deliv. type,
# and real birth Weight(g).

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


def clean_fhr(fhr):
    return (
        pd.Series(fhr)
        .replace(0, np.nan)
        .mask(lambda s: s < 0, np.nan)
        .interpolate()
        .fillna(140)
        .values.astype(np.float32)
    )


def clean_uc(uc):
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
    XGBoost can handle missing values, and we also save which metadata fields were used.
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
# CTG FEATURES
# ============================================================

def accelerations(fhr, baseline):
    return np.sum(fhr > baseline + 15)


def decelerations(fhr, baseline):
    return np.sum(fhr < baseline - 15)


def stv(fhr):
    return np.mean(np.abs(np.diff(fhr)))


def ltv(fhr):
    window = 120
    vals = []

    for i in range(0, len(fhr) - window, window):
        vals.append(np.std(fhr[i:i + window]))

    return np.mean(vals) if vals else np.std(fhr)


def extract_features(fhr, uc):
    baseline = np.median(fhr)
    t = np.arange(len(fhr))

    feats = {}

    # FHR summary
    feats["fhr_mean"] = float(np.mean(fhr))
    feats["fhr_std"] = float(np.std(fhr))
    feats["fhr_min"] = float(np.min(fhr))
    feats["fhr_max"] = float(np.max(fhr))
    feats["fhr_p05"] = float(np.percentile(fhr, 5))
    feats["fhr_p95"] = float(np.percentile(fhr, 95))

    # Trend
    feats["fhr_slope"] = float(np.polyfit(t, fhr, 1)[0])

    # Variability
    feats["stv"] = float(stv(fhr))
    feats["ltv"] = float(ltv(fhr))

    # Simple event-like counts
    feats["accelerations"] = int(accelerations(fhr, baseline))
    feats["decelerations"] = int(decelerations(fhr, baseline))

    # UC summary
    feats["uc_mean"] = float(np.mean(uc))
    feats["uc_std"] = float(np.std(uc))
    feats["uc_max"] = float(np.max(uc))

    return feats


# ============================================================
# THRESHOLD SELECTION
# ============================================================

def choose_threshold_for_recall(y_true, y_prob, target_recall=0.70):
    """
    Choose threshold using VALIDATION set.

    We scan thresholds and pick one that:
    - reaches recall >= target_recall for Danger
    - among those, maximizes Danger precision

    If no threshold reaches target recall, choose threshold with max recall.
    """
    thresholds = np.linspace(0.01, 0.95, 95)

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

        fhr = clean_fhr(df["FHR"].values)
        uc = clean_uc(df["UC"].values)

        fhr = last_window_or_pad(fhr, SEQ_LEN, 140)
        uc = last_window_or_pad(uc, SEQ_LEN, 0)

        feats = extract_features(fhr, uc)

        # Add clinical metadata known in real time
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

print("\nTraining XGBoost model...")

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
    title="TEST RESULTS SUMMARY - XGBoost CTG + Metadata",
)


# ============================================================
# OPTIONAL: FEATURE IMPORTANCE TOP 15
# ============================================================

importance = model.feature_importances_
importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importance,
}).sort_values("importance", ascending=False)

print("\nTop 15 Feature Importances:")
for _, row in importance_df.head(15).iterrows():
    print(f"{row['feature']}: {row['importance']:.4f}")


# ============================================================
# SAVE ARTIFACTS FOR WEB APP
# ============================================================

joblib.dump(model, MODEL_PATH)

stats = {
    "model_type": "XGBoostCTGWithMetadata",

    "feature_names": feature_names,
    "ctg_features": [
        "fhr_mean",
        "fhr_std",
        "fhr_min",
        "fhr_max",
        "fhr_p05",
        "fhr_p95",
        "fhr_slope",
        "stv",
        "ltv",
        "accelerations",
        "decelerations",
        "uc_mean",
        "uc_std",
        "uc_max",
    ],
    "metadata_fields": META_FIELD_NAMES,

    "seq_len": SEQ_LEN,
    "fs": FS,

    "threshold": float(best_threshold),
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
print("To use this model in the web app, add an XGBoost adapter that:")
print("1. Extracts the same CTG features from the uploaded CSV.")
print("2. Reads the same metadata fields from the matching .hea file.")
print("3. Orders features exactly according to xgboost_stats.json['feature_names'].")
print("4. Loads xgboost_model.joblib and applies the saved threshold.")