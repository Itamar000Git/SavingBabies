import os
import glob
import numpy as np
import pandas as pd
import time

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

SEQ_LEN = 1200
PH_DANGER_THRESHOLD = 7.10

THRESHOLD = 0.05
RANDOM_STATE = 42

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
        .interpolate()
        .fillna(140)
        .values
    )


def clean_uc(uc):

    return (
        pd.Series(uc)
        .interpolate()
        .fillna(0)
        .values
    )

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

    for i in range(0, len(fhr)-window, window):
        vals.append(np.std(fhr[i:i+window]))

    return np.mean(vals) if vals else np.std(fhr)

# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(fhr, uc):

    baseline = np.median(fhr)
    t = np.arange(len(fhr))

    feats = {}

    feats["mean"] = np.mean(fhr)
    feats["std"] = np.std(fhr)

    feats["min"] = np.min(fhr)
    feats["max"] = np.max(fhr)

    feats["p05"] = np.percentile(fhr,5)
    feats["p95"] = np.percentile(fhr,95)

    feats["slope"] = np.polyfit(t,fhr,1)[0]

    feats["stv"] = stv(fhr)
    feats["ltv"] = ltv(fhr)

    feats["accelerations"] = accelerations(fhr,baseline)
    feats["decelerations"] = decelerations(fhr,baseline)

    feats["uc_mean"] = np.mean(uc)
    feats["uc_std"] = np.std(uc)
    feats["uc_max"] = np.max(uc)

    return feats


# ============================================================
# LOAD DATA
# ============================================================

print("Loading recordings...")

files = glob.glob(os.path.join(CSV_DIR,"*.csv"))

rows = []

for i,file in enumerate(files):

    record_id = os.path.splitext(os.path.basename(file))[0]

    if record_id not in ph_dict:
        continue

    df = pd.read_csv(file)

    if "FHR" not in df.columns:
        continue

    fhr = clean_fhr(df["FHR"].values)
    uc = clean_uc(df["UC"].values)

    fhr = last_window_or_pad(fhr,SEQ_LEN,140)
    uc = last_window_or_pad(uc,SEQ_LEN,0)

    feats = extract_features(fhr,uc)

    ph_val = ph_dict[record_id]

    label = 1 if ph_val < PH_DANGER_THRESHOLD else 0

    feats["y"] = label

    rows.append(feats)

    if i % PRINT_EVERY == 0:
        print("processed",i)

feat_df = pd.DataFrame(rows)

print("Dataset:",len(feat_df))
print("Class:",np.bincount(feat_df["y"]))

# ============================================================
# SPLIT
# ============================================================

X = feat_df.drop(columns=["y"])
y = feat_df["y"]

X_train,X_test,y_train,y_test = train_test_split(
    X,y,
    test_size=0.2,
    stratify=y,
    random_state=RANDOM_STATE
)

# ============================================================
# CLASS IMBALANCE
# ============================================================

scale_pos_weight = np.sum(y_train==0)/np.sum(y_train==1)

print("scale_pos_weight =",scale_pos_weight)

# ============================================================
# MODEL
# ============================================================

print("Training model...")

model = XGBClassifier(

    n_estimators=900,
    max_depth=5,

    learning_rate=0.02,

    subsample=0.9,
    colsample_bytree=0.9,

    scale_pos_weight=scale_pos_weight,

    eval_metric="logloss",

    n_jobs=-1,
    random_state=RANDOM_STATE
)

model.fit(X_train,y_train)

# ============================================================
# EVALUATION
# ============================================================

prob = model.predict_proba(X_test)[:,1]

pred = (prob >= THRESHOLD).astype(int)

print("\nConfusion Matrix")

print(confusion_matrix(y_test,pred))

print("\nClassification")

print(classification_report(
    y_test,
    pred,
    target_names=["Normal","Danger"],
    zero_division=0
))

print("PR-AUC:",average_precision_score(y_test,prob))
print("ROC-AUC:",roc_auc_score(y_test,prob))