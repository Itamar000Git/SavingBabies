import os
import re
import glob
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold
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

PROJECT_ROOT = Path(__file__).resolve().parent

CSV_DIR = PROJECT_ROOT / "csv_output"
DATASET_DIR = PROJECT_ROOT / "dataset"
INFO_DIR = DATASET_DIR / "info"
PH_FILE = DATASET_DIR / "ph_levels_with_be_risk_filled.csv"

FS = 4
RANDOM_STATE = 42
TARGET_RECALL = 0.5
MIN_THRESHOLD = 0.02
MAX_THRESHOLD = 0.95

WINDOWS = {
    "last5": 5 * 60 * FS,
    "last10": 10 * 60 * FS,
    "last20": 20 * 60 * FS,
    "full": None,
}

WEIGHTS_DIR = PROJECT_ROOT / "fetal_health_web_app" / "backend" / "weights"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = WEIGHTS_DIR / "xgboost_model_riskBE.joblib"
STATS_PATH = WEIGHTS_DIR / "xgboost_stats_riskBE.json"

# ============================================================
# METADATA FEATURES FROM dataset/info
# ============================================================
# BE אינו נמצא כאן כדי למנוע זליגת נתונים!

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
    ("meta_i_stage", "I.stage"),
    ("meta_no_progress", "NoProgress"),
    ("meta_ck_kp", "CK/KP"),
    ("meta_ii_stage", "II.stage"),
]

METADATA_FIELD_NAMES = [name for name, _ in METADATA_FEATURES]
INCLUDE_METADATA_MISSING_INDICATORS = True

# ============================================================
# LOAD LABELS (riskBE)
# ============================================================

if not PH_FILE.exists():
    raise FileNotFoundError(f"Could not find label file: {PH_FILE}")

ph_df = pd.read_csv(PH_FILE)
ph_df["record_id"] = ph_df["record_id"].astype(str)
risk_dict = dict(zip(ph_df["record_id"], ph_df["riskBE"]))

# ============================================================
# SIGNAL & METADATA HELPERS
# ============================================================

def last_window_or_pad(arr, seq_len, pad):
    arr = arr.astype(np.float32)
    if len(arr) >= seq_len: return arr[-seq_len:]
    return np.pad(arr, (0, seq_len - len(arr)), constant_values=pad)

def take_window(arr, window_len, pad_value):
    arr = arr.astype(np.float32)
    if window_len is None: return arr
    return last_window_or_pad(arr, window_len, pad_value)

def clean_fhr(fhr):
    fhr = np.asarray(fhr, dtype=np.float32)
    return pd.Series(fhr).mask((fhr <= 0) | np.isnan(fhr), np.nan).interpolate().fillna(140).values.astype(np.float32)

def clean_uc(uc):
    uc = np.asarray(uc, dtype=np.float32)
    return pd.Series(uc).interpolate().fillna(0).values.astype(np.float32)

def extract_number_from_text(text):
    match = re.search(r"-?\d+(\.\d+)?", str(text))
    return float(match.group(0)) if match else np.nan

def find_info_file(record_id):
    candidates = [
        INFO_DIR / f"{record_id}.hea", INFO_DIR / f"{record_id}.txt",
        INFO_DIR / f"{record_id}.info", INFO_DIR / f"{record_id}",
    ]
    for path in candidates:
        if path.exists(): return path
    matches = list(INFO_DIR.glob(f"*{record_id}*"))
    return matches[0] if matches else None

def read_patient_metadata(record_id):
    values = {field_name: np.nan for field_name in METADATA_FIELD_NAMES}
    info_path = find_info_file(record_id)

    if info_path is None:
        result = values.copy()
        if INCLUDE_METADATA_MISSING_INDICATORS:
            for field_name in METADATA_FIELD_NAMES: result[f"{field_name}_missing"] = 1.0
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
    except Exception:
        pass # Silently ignore parsing errors for individual files

    result = values.copy()
    if INCLUDE_METADATA_MISSING_INDICATORS:
        for field_name in METADATA_FIELD_NAMES:
            result[f"{field_name}_missing"] = 1.0 if pd.isna(values[field_name]) else 0.0
    return result

# ============================================================
# CTG & CLINICAL FEATURE HELPERS
# ============================================================

def accelerations(fhr, baseline): return int(np.sum(fhr > baseline + 15))
def decelerations(fhr, baseline): return int(np.sum(fhr < baseline - 15))

def stv(fhr):
    return float(np.mean(np.abs(np.diff(fhr)))) if len(fhr) >= 2 else 0.0

def ltv(fhr):
    window = 120
    vals = [np.std(fhr[i:i + window]) for i in range(0, len(fhr) - window, window)]
    return float(np.mean(vals)) if vals else float(np.std(fhr))

def safe_slope(fhr):
    if len(fhr) < 2: return 0.0
    t = np.arange(len(fhr))
    try: return float(np.polyfit(t, fhr, 1)[0])
    except Exception: return 0.0

def detect_accelerations_clinical(fhr, fs=4):
    if len(fhr) == 0: return []
    baseline = np.median(fhr)
    thr = baseline + 15
    min_len = int(15 * fs)

    res, start = [], None
    for i in range(len(fhr)):
        if fhr[i] >= thr:
            if start is None: start = i
        else:
            if start is not None:
                if i - start >= min_len: res.append((start, i))
                start = None
    return res

def detect_decelerations_clinical(fhr, uc=None, fs=4):
    if len(fhr) == 0: return []
    baseline = np.median(fhr)
    thr = baseline - 15
    min_len = int(15 * fs)

    res, start = [], None
    for i in range(len(fhr)):
        if fhr[i] <= thr:
            if start is None: start = i
        else:
            if start is not None:
                end = i
                if end - start >= min_len:
                    seg = fhr[start:end]
                    depth = baseline - np.min(seg)
                    duration = (end - start) / fs
                    nadir = start + np.argmin(seg)
                    onset = (nadir - start) / fs

                    if duration >= 120: typ = "prolonged"
                    elif onset < 30: typ = "variable"
                    else: typ = "unspecified"

                    if uc is not None:
                        uc_seg = uc[start:end]
                        if len(uc_seg) > 0:
                            uc_peak = np.argmax(uc_seg) + start
                            typ = "late" if nadir > uc_peak else "early"

                    res.append({"duration": duration, "depth": depth, "type": typ})
                start = None
    return res

def clinical_features(fhr, uc, fs=4):
    empty_feats = {
        "dec_count":0, "dec_late":0, "dec_variable":0, "dec_prolonged":0,
        "dec_avg_depth":0.0, "dec_max_depth":0.0, "dec_avg_duration":0.0,
        "dec_time_ratio":0.0, "acc_count":0, "acc_rate":0.0, "reactive":0, "nonreactive":0
    }
    if len(fhr) == 0: return empty_feats

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
    if len(fhr_clean) == 0:
        for name in ["fhr_mean", "fhr_std", "fhr_min", "fhr_max", "fhr_range", "fhr_p05", "fhr_p95", "fhr_slope", "stv", "ltv", "accelerations", "decelerations", "acceleration_ratio", "deceleration_ratio", "uc_mean", "uc_std", "uc_max", "uc_activity_ratio", "missing_signal_pct", "valid_fhr_ratio"]:
            feats[f"{prefix}_{name}"] = 0.0
        for k in ["dec_count", "dec_late", "dec_variable", "dec_prolonged", "dec_avg_depth", "dec_max_depth", "dec_avg_duration", "dec_time_ratio", "acc_count", "acc_rate", "reactive", "nonreactive"]:
            feats[f"{prefix}_{k}"] = 0.0
        return feats

    baseline = np.median(fhr_clean)
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
    
    acc_count = accelerations(fhr_clean, baseline)
    dec_count = decelerations(fhr_clean, baseline)
    feats[f"{prefix}_accelerations"] = int(acc_count)
    feats[f"{prefix}_decelerations"] = int(dec_count)
    feats[f"{prefix}_acceleration_ratio"] = float(acc_count / n)
    feats[f"{prefix}_deceleration_ratio"] = float(dec_count / n)
    
    feats[f"{prefix}_uc_mean"] = float(np.mean(uc_clean))
    feats[f"{prefix}_uc_std"] = float(np.std(uc_clean))
    feats[f"{prefix}_uc_max"] = float(np.max(uc_clean))
    feats[f"{prefix}_uc_activity_ratio"] = float(np.mean(uc_clean > 0))
    feats[f"{prefix}_missing_signal_pct"] = float(100.0 * np.mean(fhr_valid_mask == 0))
    feats[f"{prefix}_valid_fhr_ratio"] = float(np.mean(fhr_valid_mask == 1))

    clinical = clinical_features(fhr_clean, uc_clean)
    for k, v in clinical.items(): feats[f"{prefix}_{k}"] = v

    return feats

def extract_ctg_multi_window_features(fhr_raw, uc_raw):
    fhr_raw, uc_raw = np.asarray(fhr_raw, dtype=np.float32), np.asarray(uc_raw, dtype=np.float32)
    fhr_valid_mask_raw = np.ones_like(fhr_raw, dtype=np.float32)
    fhr_valid_mask_raw[(fhr_raw <= 0) | np.isnan(fhr_raw)] = 0.0

    fhr_clean_full = clean_fhr(fhr_raw)
    uc_clean_full = clean_uc(uc_raw)

    all_features = {}
    for prefix, window_len in WINDOWS.items():
        if window_len is None:
            fhr_w, uc_w, mask_w = fhr_clean_full, uc_clean_full, fhr_valid_mask_raw
        else:
            fhr_w = take_window(fhr_clean_full, window_len, 140)
            uc_w = take_window(uc_clean_full, window_len, 0)
            mask_w = take_window(fhr_valid_mask_raw, window_len, 0)

        all_features.update(extract_features_for_window(fhr_w, uc_w, mask_w, prefix))

    return all_features

# ============================================================
# AUGMENTATION
# ============================================================

def augment_minority_class(X, y, factor=6, noise_std_multiplier=0.01, random_state=42):
    rng = np.random.default_rng(random_state)
    X, y = np.array(X, dtype=np.float64), np.array(y)
    X_normal, X_danger = X[y == 0], X[y == 1]

    if len(X_danger) == 0:
        return X, y

    feature_stds = np.std(X_danger, axis=0)
    feature_stds = np.where(feature_stds == 0, 1e-6, feature_stds)

    augmented_copies = []
    for _ in range(factor - 1):
        noise = rng.normal(loc=0.0, scale=noise_std_multiplier * feature_stds, size=X_danger.shape)
        augmented_copies.append(X_danger + noise)

    X_danger_aug = np.vstack([X_danger] + augmented_copies)
    y_danger_aug = np.ones(len(X_danger_aug), dtype=int)

    X_new = np.vstack([X_normal, X_danger_aug])
    y_new = np.concatenate([np.zeros(len(X_normal)), y_danger_aug])

    idx = rng.permutation(len(X_new))
    return X_new[idx], y_new[idx].astype(int)

# ============================================================
# EVALUATION HELPERS
# ============================================================

def choose_threshold_for_recall(y_true, y_prob, target_recall=0.60):
    thresholds = np.linspace(MIN_THRESHOLD, MAX_THRESHOLD, 91)
    best_thr, best_precision, best_recall = 0.5, -1.0, -1.0
    best_recall_thr, best_recall_value = 0.5, -1.0

    for thr in thresholds:
        pred = (np.array(y_prob) >= thr).astype(int)
        rec = recall_score(y_true, pred, pos_label=1, zero_division=0)
        pre = precision_score(y_true, pred, pos_label=1, zero_division=0)

        if rec > best_recall_value:
            best_recall_value, best_recall_thr = rec, thr

        if rec >= target_recall and pre > best_precision:
            best_precision, best_recall, best_thr = pre, rec, thr

    if best_precision < 0:
        pred = (np.array(y_prob) >= best_recall_thr).astype(int)
        return float(best_recall_thr), float(best_recall_value), float(precision_score(y_true, pred, pos_label=1, zero_division=0))

    return float(best_thr), float(best_recall), float(best_precision)

def choose_threshold_cv(X_tr, y_tr, model_params, target_recall=TARGET_RECALL, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    X_tr, y_tr = np.array(X_tr), np.array(y_tr)
    oof_probs = np.zeros(len(y_tr))

    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_tr, y_tr)):
        n_neg = int(np.sum(y_tr[tr_idx] == 0))
        n_pos = int(np.sum(y_tr[tr_idx] == 1))
        fold_spw = n_neg / max(n_pos, 1)
        X_fa, y_fa = augment_minority_class(X_tr[tr_idx], y_tr[tr_idx], factor=6, random_state=RANDOM_STATE + fold_idx)
        fm = XGBClassifier(**{**model_params, "scale_pos_weight": fold_spw})
        fm.fit(X_fa, y_fa, verbose=False)
        oof_probs[val_idx] = fm.predict_proba(X_tr[val_idx])[:, 1]

    thr, rec, pre = choose_threshold_for_recall(y_tr, oof_probs, target_recall)
    print(f"    CV threshold: thr={thr:.3f}, recall={rec:.3f}, precision={pre:.3f}")
    return thr

def evaluate_and_print(y_true, y_prob, threshold, title):
    pred = (np.array(y_prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    
    metrics = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, pred, pos_label=1, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(set(y_true)) == 2 else None,
        "fn": int(fn), "fp": int(fp), "tn": int(tn), "tp": int(tp)
    }

    print("\n" + "=" * 55)
    print(title)
    print("=" * 55)
    print(f"Threshold: {threshold:.3f}")
    print(f"Accuracy:  {metrics['accuracy']:.3f}")
    print(f"Recall:    {metrics['recall']:.3f}  (Identified {tp} out of {tp+fn})")
    print(f"Precision: {metrics['precision']:.3f}  (True alarms: {tp} | False alarms: {fp})")
    print(f"F1-Score:  {metrics['f1']:.3f}")
    print(f"False Negatives (Missed Danger): {fn}")
    print(f"False Positives (False Alarms):  {fp}")
    print(f"PR-AUC:    {metrics['pr_auc']:.3f}")
    print("=" * 55)
    return metrics

# ============================================================
# PIPELINE EXECUTION
# ============================================================

print("[1] Loading recordings and extracting features (this may take a moment)...")

files = glob.glob(str(CSV_DIR / "*.csv"))
rows = []

for file in files:
    record_id = os.path.splitext(os.path.basename(file))[0]
    if record_id not in risk_dict: continue

    try:
        df = pd.read_csv(file)
        if "FHR" not in df.columns or "UC" not in df.columns: continue

        feats = extract_ctg_multi_window_features(df["FHR"].values, df["UC"].values)
        feats.update(read_patient_metadata(record_id))
        
        feats["record_id"] = record_id
        feats["y"] = int(risk_dict[record_id]) # Using riskBE label
        rows.append(feats)
    except Exception:
        pass # Silently skip problematic files

feat_df = pd.DataFrame(rows)
if len(feat_df) == 0: raise RuntimeError("No valid records were loaded.")

print(f"    Loaded {len(feat_df)} valid records.")
print(f"    Class distribution (Normal=0, Danger BE=1): {np.bincount(feat_df['y'])}")

# --- Split ---
X = feat_df.drop(columns=["y", "record_id"])
y = feat_df["y"].astype(int)
feature_names = list(X.columns)

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=RANDOM_STATE)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.15, stratify=y_temp, random_state=RANDOM_STATE)

# --- Augment with Gaussian jitter; keep X_train/y_train clean for CV and eval_set ---
X_train_aug, y_train_aug = augment_minority_class(X_train, y_train, factor=6)
print(f"    After augmentation: {np.bincount(y_train_aug.astype(int))}")

# --- scale_pos_weight from POST-augmentation distribution with 2x recall emphasis ---
# Using post-augmentation ratio avoids double-counting the imbalance on top of augmentation
n_neg_aug = int(np.sum(y_train_aug == 0))
n_pos_aug = int(np.sum(y_train_aug == 1))
scale_pos_weight = (n_neg_aug / max(n_pos_aug, 1)) * 2.5
print(f"    scale_pos_weight (post-aug × 2.5): {scale_pos_weight:.2f}")

# --- Train ---
print("\n[2] Training XGBoost Classifier...")
model = XGBClassifier(
    objective="binary:logistic",
    n_estimators=600,
    max_depth=4,
    learning_rate=0.02,
    subsample=0.9,
    colsample_bytree=0.9,
    scale_pos_weight=scale_pos_weight,
    max_delta_step=1,
    min_child_weight=3,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    eval_metric="logloss",
    early_stopping_rounds=50,
    n_jobs=-1,
    random_state=RANDOM_STATE,
)
model.fit(
    X_train_aug, y_train_aug,
    eval_set=[(np.array(X_val), np.array(y_val))],
    verbose=False,
)
print(f"    Best iteration: {model.best_iteration}")

# --- Threshold selection via StratifiedKFold CV on original training data ---
print("\n[3] Selecting threshold via StratifiedKFold CV...")
cv_params = dict(
    objective="binary:logistic",
    n_estimators=model.best_iteration if (hasattr(model, "best_iteration") and model.best_iteration) else 300,
    max_depth=4, learning_rate=0.02, subsample=0.9, colsample_bytree=0.9,
    max_delta_step=1, min_child_weight=3, gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
    eval_metric="logloss", n_jobs=-1, random_state=RANDOM_STATE,
)
best_threshold = choose_threshold_cv(X_train, y_train, cv_params, target_recall=TARGET_RECALL)
print(f"    Selected threshold: {best_threshold:.3f}")

print("\n[4] Evaluating on Test Set...")
test_prob = model.predict_proba(X_test)[:, 1]
test_metrics = evaluate_and_print(y_test, test_prob, best_threshold, "TEST RESULTS SUMMARY - XGBoost (Predict riskBE)")

# --- Save ---
print("\n[5] Saving Artifacts...")
joblib.dump(model, MODEL_PATH)

stats = {
    "model_type": "XGBoostCTGAndMetadataClassification_riskBE",
    "feature_names": feature_names,
    "metadata_fields": METADATA_FIELD_NAMES,
    "metadata_missing_indicators": bool(INCLUDE_METADATA_MISSING_INDICATORS),
    "windows": {name: None if length is None else int(length) for name, length in WINDOWS.items()},
    "fs": FS,
    "threshold": float(best_threshold),
    "scale_pos_weight": float(scale_pos_weight),
    "test_metrics": test_metrics,
}

with open(STATS_PATH, "w", encoding="utf-8") as f:
    json.dump(stats, f, indent=2)

print(f"    Model saved to: {MODEL_PATH}")
print(f"    Stats saved to: {STATS_PATH}")
print("\nProcess Complete.")