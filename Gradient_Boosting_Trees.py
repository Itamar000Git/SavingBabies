import os
import glob
import time
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
    roc_auc_score
)

from xgboost import XGBClassifier

# ============================================================
# CONFIG
# ============================================================

CSV_DIR = "csv_output"
PH_FILE = "dataset/ph_levels.csv"

FS = 4
SEQ_LEN = 1200
PH_DANGER_THRESHOLD = 7.10

RANDOM_STATE = 42
THRESHOLD = 0.25

PRINT_EVERY = 50

# ============================================================
# LOAD LABELS
# ============================================================

ph_df = pd.read_csv(PH_FILE)
ph_df["record_id"] = ph_df["record_id"].astype(str)
ph_dict = dict(zip(ph_df["record_id"], ph_df["pH"]))

# ============================================================
# CLEANING
# ============================================================

def last_window_or_pad(arr, seq_len, pad_value):

    arr = arr.astype(np.float32)

    if len(arr) >= seq_len:
        return arr[-seq_len:]

    pad = seq_len - len(arr)
    return np.pad(arr, (0, pad), constant_values=pad_value)

def clean_fhr_and_mask(fhr):

    fhr = fhr.astype(np.float32)

    mask = np.ones_like(fhr)
    mask[(fhr == 0) | np.isnan(fhr)] = 0

    fhr_clean = (
        pd.Series(fhr)
        .replace(0, np.nan)
        .interpolate()
        .fillna(140)
        .values
    )

    return fhr_clean, mask

def clean_uc(uc):

    return (
        pd.Series(uc)
        .interpolate()
        .fillna(0)
        .values
    )

# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(fhr, uc, mask):

    t = np.arange(len(fhr))

    feats = {}

    feats["valid_ratio"] = np.mean(mask)

    feats["fhr_mean"] = np.mean(fhr)
    feats["fhr_std"] = np.std(fhr)
    feats["fhr_min"] = np.min(fhr)
    feats["fhr_max"] = np.max(fhr)

    feats["fhr_p05"] = np.percentile(fhr, 5)
    feats["fhr_p95"] = np.percentile(fhr, 95)

    feats["fhr_median"] = np.median(fhr)

    feats["fhr_slope"] = np.polyfit(t, fhr, 1)[0]

    feats["fhr_energy"] = np.mean(fhr**2)

    feats["fhr_diff_std"] = np.std(np.diff(fhr))

    feats["uc_mean"] = np.mean(uc)
    feats["uc_std"] = np.std(uc)
    feats["uc_max"] = np.max(uc)

    feats["uc_energy"] = np.mean(uc**2)

    return feats

# ============================================================
# LOAD DATA
# ============================================================

print("Loading recordings...")

csv_files = glob.glob(os.path.join(CSV_DIR, "*.csv"))

rows = []

start = time.time()

for i, file in enumerate(csv_files):

    record_id = os.path.splitext(os.path.basename(file))[0]

    if record_id not in ph_dict:
        continue

    df = pd.read_csv(file)

    if "FHR" not in df.columns or "UC" not in df.columns:
        continue

    fhr_clean, mask = clean_fhr_and_mask(df["FHR"].values)
    uc_clean = clean_uc(df["UC"].values)

    fhr = last_window_or_pad(fhr_clean, SEQ_LEN, 140)
    uc = last_window_or_pad(uc_clean, SEQ_LEN, 0)
    mask = last_window_or_pad(mask, SEQ_LEN, 0)

    feats = extract_features(fhr, uc, mask)

    ph_val = ph_dict[record_id]
    label = 1 if ph_val < PH_DANGER_THRESHOLD else 0

    feats["y"] = label

    rows.append(feats)

    if i % PRINT_EVERY == 0:
        print("processed", i)

feat_df = pd.DataFrame(rows)

print("\nDataset size:", len(feat_df))
print("Class distribution:", np.bincount(feat_df["y"]))

# ============================================================
# SPLIT
# ============================================================

X = feat_df.drop(columns=["y"])
y = feat_df["y"]

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y,
    test_size=0.15,
    stratify=y,
    random_state=RANDOM_STATE
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=0.15,
    stratify=y_temp,
    random_state=RANDOM_STATE
)

# ============================================================
# CLASS IMBALANCE HANDLING
# ============================================================

normal = np.sum(y_train == 0)
danger = np.sum(y_train == 1)

scale_pos_weight = normal / danger

print("scale_pos_weight =", scale_pos_weight)

# ============================================================
# MODEL
# ============================================================

print("\nTraining XGBoost...")

model = XGBClassifier(

    n_estimators=700,
    max_depth=6,
    learning_rate=0.03,

    subsample=0.8,
    colsample_bytree=0.8,

    scale_pos_weight=scale_pos_weight,

    eval_metric="logloss",

    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbosity=1
)

model.fit(X_train, y_train)

# ============================================================
# EVALUATION
# ============================================================

def evaluate(name, Xs, ys):

    prob = model.predict_proba(Xs)[:,1]

    pred = (prob >= THRESHOLD).astype(int)

    print("\n====", name, "====")

    print(confusion_matrix(ys, pred))

    print(classification_report(
        ys,
        pred,
        target_names=["Normal","Danger"],
        zero_division=0
    ))

    print("PR-AUC:", average_precision_score(ys, prob))

    if len(np.unique(ys)) == 2:
        print("ROC-AUC:", roc_auc_score(ys, prob))

evaluate("VAL", X_val, y_val)
evaluate("TEST", X_test, y_test)