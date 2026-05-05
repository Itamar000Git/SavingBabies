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

# Assumption:
# This script is located in the project root:
# SavingBabies/XGBoost_Classifier_logistic.py
PROJECT_ROOT = Path(__file__).resolve().parent

CSV_DIR = PROJECT_ROOT / "csv_output"
DATASET_DIR = PROJECT_ROOT / "dataset"
INFO_DIR = DATASET_DIR / "info"
PH_FILE = DATASET_DIR / "ph_levels.csv"

FS = 4
PH_DANGER_THRESHOLD = 7.05

RANDOM_STATE = 42
PRINT_EVERY = 50

TARGET_RECALL = 0.60

MIN_THRESHOLD = 0.05
MAX_THRESHOLD = 0.95

# Multi-window feature extraction.
# None means full recording.
WINDOWS = {
    "last5": 5 * 60 * FS,
    "last10": 10 * 60 * FS,
    "last20": 20 * 60 * FS,
    "full": None,
}

# Web app artifacts.
WEIGHTS_DIR = PROJECT_ROOT / "fetal_health_web_app" / "backend" / "weights"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = WEIGHTS_DIR / "xgboost_model.joblib"
STATS_PATH = WEIGHTS_DIR / "xgboost_stats.json"


# ============================================================
# METADATA FEATURES FROM dataset/info
# ============================================================
# These are fields that can be considered patient / clinical context.
# Do NOT include outcome/leakage fields like pH, Apgar, NICU, HIE, etc.

METADATA_FEATURES = [
    ("meta_gestational_weeks", "Gest. weeks"),
    ("meta_mother_age", "Age"),
    ("meta_gravidity", "Gravidity"),
    ("meta_parity", "Parity"),
    ("meta_diabetes", "Diabetes"),
    ("meta_hypertension", "Hypertension"),
    ("meta_preeclampsia", "Preeclampsia"),
    ("meta_liq_praecox", "Liq. praecox"),
    ("meta_pyrexia", "Pyrexia"),
    ("meta_meconium", "Meconium"),
    ("meta_presentation", "Presentation"),
    ("meta_induced", "Induced"),

    # Labor / delivery-process descriptors.
    # Use only if you consider them available at the decision time.
    ("meta_i_stage", "I.stage"),
    ("meta_no_progress", "NoProgress"),
    ("meta_ck_kp", "CK/KP"),
    ("meta_ii_stage", "II.stage"),
]

METADATA_FIELD_NAMES = [name for name, _ in METADATA_FEATURES]

INCLUDE_METADATA_MISSING_INDICATORS = True


# ============================================================
# LOAD LABELS
# ============================================================

if not PH_FILE.exists():
    raise FileNotFoundError(f"Could not find pH file: {PH_FILE}")

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
    - invalid values are interpolated.
    - fallback value is 140 bpm.
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
    """
    Extract first numeric value from text.
    Supports integers, floats, and negative values.
    """
    match = re.search(r"-?\d+(\.\d+)?", str(text))
    if not match:
        return np.nan

    return float(match.group(0))


def find_info_file(record_id):
    """
    Find metadata/info file for a record inside dataset/info.

    Supports common possibilities:
    - dataset/info/1001.hea
    - dataset/info/1001.txt
    - dataset/info/1001.info
    - dataset/info/1001
    - any file containing record_id in its name
    """
    candidates = [
        INFO_DIR / f"{record_id}.hea",
        INFO_DIR / f"{record_id}.txt",
        INFO_DIR / f"{record_id}.info",
        INFO_DIR / f"{record_id}",
    ]

    for path in candidates:
        if path.exists():
            return path

    matches = list(INFO_DIR.glob(f"*{record_id}*"))
    if matches:
        return matches[0]

    return None


def read_patient_metadata(record_id):
    """
    Reads metadata from dataset/info/{record_id}.*.

    Missing values are returned as np.nan.
    XGBoost can handle NaN values.

    If INCLUDE_METADATA_MISSING_INDICATORS is True,
    for each metadata value we also add:
    <field_name>_missing = 1 if missing else 0
    """
    values = {field_name: np.nan for field_name in METADATA_FIELD_NAMES}

    info_path = find_info_file(record_id)

    if info_path is None:
        result = values.copy()

        if INCLUDE_METADATA_MISSING_INDICATORS:
            for field_name in METADATA_FIELD_NAMES:
                result[f"{field_name}_missing"] = 1.0

        return result

    try:
        lines = info_path.read_text(encoding="utf-8", errors="ignore").splitlines()

        for field_name, label_in_file in METADATA_FEATURES:
            prefix = f"#{label_in_file}"

            for line in lines:
                clean_line = line.strip()

                if clean_line.startswith(prefix):
                    raw_value = clean_line[len(prefix):].strip()
                    values[field_name] = extract_number_from_text(raw_value)
                    break

    except Exception as e:
        print(f"Warning: failed to parse metadata for record {record_id}: {e}")

    result = values.copy()

    if INCLUDE_METADATA_MISSING_INDICATORS:
        for field_name in METADATA_FIELD_NAMES:
            result[f"{field_name}_missing"] = 1.0 if pd.isna(values[field_name]) else 0.0

    return result


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

# ============================================================
# CLINICAL EVENT DETECTION
# ============================================================

def detect_accelerations_clinical(fhr, fs=4):
    if len(fhr) == 0:
        return []

    baseline = np.median(fhr)
    thr = baseline + 15
    min_len = int(15 * fs)

    res, start = [], None
    for i in range(len(fhr)):
        if fhr[i] >= thr:
            if start is None:
                start = i
        else:
            if start is not None:
                if i - start >= min_len:
                    res.append((start, i))
                start = None
    return res


def detect_decelerations_clinical(fhr, uc=None, fs=4):
    if len(fhr) == 0:
        return []

    baseline = np.median(fhr)
    thr = baseline - 15
    min_len = int(15 * fs)

    res, start = [], None
    for i in range(len(fhr)):
        if fhr[i] <= thr:
            if start is None:
                start = i
        else:
            if start is not None:
                end = i
                if end - start >= min_len:
                    seg = fhr[start:end]
                    depth = baseline - np.min(seg)
                    duration = (end - start) / fs

                    nadir = start + np.argmin(seg)
                    onset = (nadir - start) / fs

                    if duration >= 120:
                        typ = "prolonged"
                    elif onset < 30:
                        typ = "variable"
                    else:
                        typ = "unspecified"

                    if uc is not None:
                        uc_seg = uc[start:end]
                        if len(uc_seg) > 0:
                            uc_peak = np.argmax(uc_seg) + start
                            typ = "late" if nadir > uc_peak else "early"

                    res.append({
                        "duration": duration,
                        "depth": depth,
                        "type": typ
                    })
                start = None
    return res


# ============================================================
# CLINICAL FEATURES
# ============================================================

def clinical_features(fhr, uc, fs=4):
    if len(fhr) == 0:

        return {
            "dec_count":0,"dec_late":0,"dec_variable":0,"dec_prolonged":0,
            "dec_avg_depth":0,"dec_max_depth":0,"dec_avg_duration":0,
            "dec_time_ratio":0,"acc_count":0,"acc_rate":0,
            "reactive":0,"nonreactive":0
        }



    decs = detect_decelerations_clinical(fhr, uc, fs)
    accs = detect_accelerations_clinical(fhr, fs)

    total_t = len(fhr) / fs

    feats = {}
    feats["dec_count"] = len(decs)
    feats["dec_late"] = sum(d["type"]=="late" for d in decs)
    feats["dec_variable"] = sum(d["type"]=="variable" for d in decs)
    feats["dec_prolonged"] = sum(d["type"]=="prolonged" for d in decs)

    if decs:
        feats["dec_avg_depth"] = float(np.mean([d["depth"] for d in decs]))
        feats["dec_max_depth"] = float(np.max([d["depth"] for d in decs]))
        feats["dec_avg_duration"] = float(np.mean([d["duration"] for d in decs]))
    else:
        feats["dec_avg_depth"] = 0.0
        feats["dec_max_depth"] = 0.0
        feats["dec_avg_duration"] = 0.0

    feats["dec_time_ratio"] = float(sum(d["duration"] for d in decs) / total_t)

    feats["acc_count"] = len(accs)
    feats["acc_rate"] = float(len(accs) / (total_t / 60))

    feats["reactive"] = int(len(accs) >= 2)
    feats["nonreactive"] = int(len(accs) < 2)

    return feats
def extract_features_for_window(fhr_clean, uc_clean, fhr_valid_mask, prefix):
    feats = {}

    # =========================================================
    # EMPTY CASE
    # =========================================================
    if len(fhr_clean) == 0:
        for name in [
            "fhr_mean",
            "fhr_std",
            "fhr_min",
            "fhr_max",
            "fhr_range",
            "fhr_p05",
            "fhr_p95",
            "fhr_slope",
            "stv",
            "ltv",
            "accelerations",
            "decelerations",
            "acceleration_ratio",
            "deceleration_ratio",
            "uc_mean",
            "uc_std",
            "uc_max",
            "uc_activity_ratio",
            "missing_signal_pct",
            "valid_fhr_ratio",
        ]:
            feats[f"{prefix}_{name}"] = 0.0

        # ✅ רק אפסים — בלי קריאה לפונקציה
        for k in [
            "dec_count", "dec_late", "dec_variable", "dec_prolonged",
            "dec_avg_depth", "dec_max_depth", "dec_avg_duration",
            "dec_time_ratio", "acc_count", "acc_rate",
            "reactive", "nonreactive"
        ]:
            feats[f"{prefix}_{k}"] = 0.0

        return feats

    # =========================================================
    # NORMAL CASE
    # =========================================================

    baseline = np.median(fhr_clean)

    missing_signal_pct = float(100.0 * np.mean(fhr_valid_mask == 0))
    valid_fhr_ratio = float(np.mean(fhr_valid_mask == 1))

    acc_count = accelerations(fhr_clean, baseline)
    dec_count = decelerations(fhr_clean, baseline)

    n = max(len(fhr_clean), 1)

    # FHR summary
    feats[f"{prefix}_fhr_mean"] = float(np.mean(fhr_clean))
    feats[f"{prefix}_fhr_std"] = float(np.std(fhr_clean))
    feats[f"{prefix}_fhr_min"] = float(np.min(fhr_clean))
    feats[f"{prefix}_fhr_max"] = float(np.max(fhr_clean))
    feats[f"{prefix}_fhr_range"] = float(np.max(fhr_clean) - np.min(fhr_clean))

    feats[f"{prefix}_fhr_p05"] = float(np.percentile(fhr_clean, 5))
    feats[f"{prefix}_fhr_p95"] = float(np.percentile(fhr_clean, 95))
    feats[f"{prefix}_fhr_slope"] = safe_slope(fhr_clean)

    # Variability
    feats[f"{prefix}_stv"] = stv(fhr_clean)
    feats[f"{prefix}_ltv"] = ltv(fhr_clean)

    # Events
    feats[f"{prefix}_accelerations"] = int(acc_count)
    feats[f"{prefix}_decelerations"] = int(dec_count)
    feats[f"{prefix}_acceleration_ratio"] = float(acc_count / n)
    feats[f"{prefix}_deceleration_ratio"] = float(dec_count / n)

    # UC
    feats[f"{prefix}_uc_mean"] = float(np.mean(uc_clean))
    feats[f"{prefix}_uc_std"] = float(np.std(uc_clean))
    feats[f"{prefix}_uc_max"] = float(np.max(uc_clean))
    feats[f"{prefix}_uc_activity_ratio"] = float(np.mean(uc_clean > 0))

    # Signal quality
    feats[f"{prefix}_missing_signal_pct"] = missing_signal_pct
    feats[f"{prefix}_valid_fhr_ratio"] = valid_fhr_ratio

    # =========================================================
    # ADD CLINICAL FEATURES ✅ (רק כאן!)
    # =========================================================
    clinical = clinical_features(fhr_clean, uc_clean)

    for k, v in clinical.items():
        feats[f"{prefix}_{k}"] = v

    return feats

def extract_ctg_multi_window_features(fhr_raw, uc_raw):
    """
    Extract features from CTG signals:
    - FHR
    - UC
    - FHR validity / missing signal mask
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

        all_features.update(
            extract_features_for_window(
                fhr_clean=fhr_w,
                uc_clean=uc_w,
                fhr_valid_mask=mask_w,
                prefix=prefix,
            )
        )

    return all_features


# ============================================================
# THRESHOLD SELECTION
# ============================================================

def choose_threshold_for_recall(y_true, y_prob, target_recall=0.60):
    """
    Choose classification threshold using VALIDATION set.

    The model itself is a binary classifier.
    The probability is used only to choose the classification threshold.
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

# ============================================================
# AUGMENTATION (ADD HERE)
# ============================================================

def augment_minority_class(X, y, factor=4, random_state=42):
    np.random.seed(random_state)

    X = np.array(X)
    y = np.array(y)

    X_normal = X[y == 0]
    X_danger = X[y == 1]

    if len(X_danger) == 0:
        return X, y

    X_danger_aug = np.repeat(X_danger, factor, axis=0)
    y_danger_aug = np.ones(len(X_danger_aug), dtype=int)

    X_new = np.vstack([X_normal, X_danger_aug])
    y_new = np.concatenate([np.zeros(len(X_normal)), y_danger_aug])

    idx = np.random.permutation(len(X_new))
    return X_new[idx], y_new[idx].astype(int)
def print_threshold_table(y_true, y_prob):
    """
    Analysis only.
    Shows how classification changes for different thresholds.
    """
    print("\nThreshold analysis on TEST set:")
    print("thr   | TP | FP | FN | TN | Recall | Precision")
    print("-" * 55)

    for thr in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        pred = (np.array(y_prob) >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()

        rec = recall_score(y_true, pred, pos_label=1, zero_division=0)
        pre = precision_score(y_true, pred, pos_label=1, zero_division=0)

        print(
            f"{thr:0.2f} | "
            f"{tp:2d} | {fp:2d} | {fn:2d} | {tn:2d} | "
            f"{rec:0.3f}  | {pre:0.3f}"
        )


# ============================================================
# LOAD DATA
# ============================================================

print("Loading recordings...")

files = glob.glob(str(CSV_DIR / "*.csv"))
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

        feats = extract_ctg_multi_window_features(fhr_raw, uc_raw)

        metadata_feats = read_patient_metadata(record_id)
        feats.update(metadata_feats)

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
    raise RuntimeError("No valid records were loaded. Check csv_output, dataset/info, and dataset/ph_levels.csv.")

print("Dataset:", len(feat_df))
print("Class counts (Normal=0, Danger=1):", np.bincount(feat_df["y"]))

print("\nInput data used by this model:")
print("- FHR signal")
print("- UC signal")
print("- FHR missing-signal mask")
print("- Patient / clinical metadata from dataset/info")

print("\nMetadata directory:")
print(INFO_DIR)

print("\nMetadata fields used:")
for field in METADATA_FIELD_NAMES:
    print(f"- {field}")

if INCLUDE_METADATA_MISSING_INDICATORS:
    print("- missing indicators for each metadata field")

print("\nWindows used:")
for name, length in WINDOWS.items():
    if length is None:
        print(f"- {name}: full recording")
    else:
        print(f"- {name}: {length / FS / 60:.1f} minutes")


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
# =========================
# APPLY AUGMENTATION HERE
# =========================
X_train, y_train = augment_minority_class(X_train, y_train, factor=4)

print("\n--- Augmentation Applied ---")
print("Train class counts after augmentation:", np.bincount(y_train.astype(int)))

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

print("\nTraining XGBoost CTG + metadata classification model...")

model = XGBClassifier(
    objective="binary:logistic",
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
    title="TEST RESULTS SUMMARY - XGBoost CTG + Metadata Classification",
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

print("\nTop 25 Feature Importances:")
for _, row in importance_df.head(25).iterrows():
    print(f"{row['feature']}: {row['importance']:.4f}")


# ============================================================
# SAVE ARTIFACTS FOR WEB APP
# ============================================================

joblib.dump(model, MODEL_PATH)

stats = {
    "model_type": "XGBoostCTGAndMetadataClassification",

    "input_type": "ctg_plus_metadata",
    "uses_patient_metadata": True,
    "metadata_source": str(INFO_DIR),

    "feature_names": feature_names,

    "metadata_fields": METADATA_FIELD_NAMES,
    "metadata_missing_indicators": bool(INCLUDE_METADATA_MISSING_INDICATORS),

    "windows": {
        name: None if length is None else int(length)
        for name, length in WINDOWS.items()
    },

    "window_descriptions": {
        name: "full recording" if length is None else f"{length / FS / 60:.1f} minutes"
        for name, length in WINDOWS.items()
    },

    "fs": FS,

    "threshold": float(best_threshold),
    "min_threshold": float(MIN_THRESHOLD),
    "target_recall": float(TARGET_RECALL),
    "ph_danger_threshold": float(PH_DANGER_THRESHOLD),

    "scale_pos_weight": float(scale_pos_weight),

    "test_metrics": test_metrics,

    "top_feature_importances": importance_df.head(25).to_dict(orient="records"),

    "classification_labels": {
        "0": "Normal",
        "1": "Danger",
    },

    "label_definition": {
        "Danger": f"pH < {PH_DANGER_THRESHOLD}",
        "Normal": f"pH >= {PH_DANGER_THRESHOLD}",
    },

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
        "dbID",
    ],

    "note": (
        "This XGBoost model is a binary classification model. "
        "It uses CTG multi-window features from FHR and UC, plus patient/clinical metadata "
        "loaded from dataset/info. pH is used only to create the training label and is not used as input."
    ),
}

with open(STATS_PATH, "w", encoding="utf-8") as f:
    json.dump(stats, f, indent=2)

print("\nSaved artifacts for web app:")
print(f"Model: {MODEL_PATH}")
print(f"Stats: {STATS_PATH}")

print("\nImportant:")
print("To use this model in the web app, update/add an XGBoost adapter that:")
print("1. Extracts the same CTG multi-window features from the uploaded CSV.")
print("2. Reads metadata from dataset/info for the matching record ID.")
print("3. Adds missing indicators exactly as in training.")
print("4. Orders features exactly according to xgboost_stats.json['feature_names'].")
print("5. Loads xgboost_model.joblib and applies the saved threshold.")







def print_ctg_analysis(fhr, uc, prediction, probability):
    decs = detect_decelerations_clinical(fhr, uc)
    accs = detect_accelerations_clinical(fhr)

    print("\n======================================")
    print("         FETAL STATUS ANALYSIS")
    print("======================================\n")

    # 🔴 החלטת מודל
    status = "⚠️ DANGER" if prediction == 1 else "✅ NORMAL"
    print(f"Prediction: {status}")
    print(f"Confidence: {probability:.3f}\n")

    # 🔵 סיכום מהיר
    print("Key Findings:")

    if len(decs) > 0:
        late = sum(d["type"] == "late" for d in decs)
        prolonged = sum(d["type"] == "prolonged" for d in decs)

        if late > 0:
            print(f"- {late} late decelerations detected")
        if prolonged > 0:
            print(f"- {prolonged} prolonged decelerations")

    if len(accs) < 2:
        print("- Non-reactive (low accelerations)")

    print()

    # 🟡 האטות
    print("Decelerations:")
    print(f"- Total: {len(decs)}")
    print(f"- Late: {sum(d['type']=='late' for d in decs)}")
    print(f"- Variable: {sum(d['type']=='variable' for d in decs)}")
    print(f"- Prolonged: {sum(d['type']=='prolonged' for d in decs)}")

    if len(decs) > 0:
        print("\nDetails:")
        for i, d in enumerate(decs[:5]):  # רק 5 ראשונים
            print(f"  #{i+1} | duration={d['duration']:.1f}s | depth={d['depth']:.1f} | type={d['type']}")

    print()

    # 🟢 האצות
    print("Accelerations:")
    print(f"- Count: {len(accs)}")
    print(f"- Reactive: {'Yes' if len(accs) >= 2 else 'No'}\n")

    print("======================================\n")