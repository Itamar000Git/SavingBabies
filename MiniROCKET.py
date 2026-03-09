import os
import glob
import time
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
    roc_auc_score
)

# =========================
# CONFIG
# =========================
CSV_DIR = "csv_output"
PH_FILE = "dataset/ph_levels.csv"

FS = 4
SEQ_LEN = 1200              # <-- CHANGED: 1200 samples = 5 minutes @ 4Hz

PH_DANGER_THRESHOLD = 7.10
RANDOM_STATE = 42

# ROCKET-like parameters
N_KERNELS = 4000
KERNEL_MIN = 7
KERNEL_MAX = 51
DILATION_MAX = 64
USE_MASK_CHANNEL = True

# Progress printing
PRINT_EVERY_LOAD = 50
PRINT_EVERY_FEATS = 5        # <-- CHANGED: print more often during feature extraction

# =========================
# Load labels
# =========================
ph_df = pd.read_csv(PH_FILE)
ph_df["record_id"] = ph_df["record_id"].astype(str)
ph_dict = dict(zip(ph_df["record_id"], ph_df["pH"]))

# =========================
# Helpers
# =========================
def last_window_or_pad(arr: np.ndarray, seq_len: int) -> np.ndarray:
    n = len(arr)
    if n >= seq_len:
        return arr[-seq_len:]
    if n == 0:
        return np.zeros(seq_len, dtype=np.float32)
    pad = seq_len - n
    return np.pad(arr, (0, pad), mode="edge").astype(np.float32)

def clean_fhr_and_mask(fhr_raw: np.ndarray):
    fhr_raw = fhr_raw.astype(np.float32)
    mask = np.ones_like(fhr_raw, dtype=np.float32)
    mask[(fhr_raw == 0) | np.isnan(fhr_raw)] = 0.0

    fhr_clean = (
        pd.Series(fhr_raw)
          .replace(0, np.nan)
          .interpolate(method="linear")
          .fillna(140)
          .values.astype(np.float32)
    )
    return fhr_clean, mask

def clean_uc(uc_raw: np.ndarray) -> np.ndarray:
    uc_raw = uc_raw.astype(np.float32)
    return (
        pd.Series(uc_raw)
          .interpolate(method="linear")
          .fillna(0)
          .values.astype(np.float32)
    )

# =========================
# Load dataset
# =========================
print("Loading CTG CSV files and building windows...")

csv_files = glob.glob(os.path.join(CSV_DIR, "*.csv"))
X_list, y_list = [], []

t0 = time.time()
kept = 0
skipped_no_label = 0
skipped_bad_cols = 0
failed = 0

for i, file in enumerate(csv_files, start=1):
    record_id = os.path.splitext(os.path.basename(file))[0]

    if record_id not in ph_dict:
        skipped_no_label += 1
        continue

    try:
        df = pd.read_csv(file)
        if "FHR" not in df.columns or "UC" not in df.columns:
            skipped_bad_cols += 1
            continue

        fhr_clean, fhr_mask = clean_fhr_and_mask(df["FHR"].values)
        uc_clean = clean_uc(df["UC"].values)

        fhr_win = last_window_or_pad(fhr_clean, SEQ_LEN)
        uc_win  = last_window_or_pad(uc_clean, SEQ_LEN)
        mask_win = last_window_or_pad(fhr_mask, SEQ_LEN)

        if USE_MASK_CHANNEL:
            x = np.stack([fhr_win, uc_win, mask_win], axis=1)  # (L, 3)
        else:
            x = np.stack([fhr_win, uc_win], axis=1)            # (L, 2)

        ph_val = float(ph_dict[record_id])
        y = 1 if ph_val < PH_DANGER_THRESHOLD else 0

        X_list.append(x)
        y_list.append(y)
        kept += 1

    except Exception as e:
        failed += 1

    if i % PRINT_EVERY_LOAD == 0:
        elapsed = time.time() - t0
        print(f"[LOAD] processed_files={i}/{len(csv_files)} | kept={kept} | "
              f"skipped_no_label={skipped_no_label} | skipped_bad_cols={skipped_bad_cols} | "
              f"failed={failed} | elapsed={elapsed:.1f}s")

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.int64)

print(f"\nLoaded {len(X)} records.")
print(f"Class counts (Normal=0, Danger=1): {np.bincount(y)}")

# =========================
# Split
# =========================
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=RANDOM_STATE, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.15, random_state=RANDOM_STATE, stratify=y_temp
)

# =========================
# Normalize train-only
# =========================
train_mean = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0)
train_std  = X_train.reshape(-1, X_train.shape[-1]).std(axis=0) + 1e-8

def normalize(X_arr):
    return (X_arr - train_mean) / train_std

X_train_n = normalize(X_train)
X_val_n   = normalize(X_val)
X_test_n  = normalize(X_test)

# =========================
# ROCKET-like kernels
# =========================
rng = np.random.default_rng(RANDOM_STATE)

def generate_random_kernel():
    length = int(rng.integers(KERNEL_MIN, KERNEL_MAX + 1))
    weights = rng.normal(0, 1, size=length).astype(np.float32)
    bias = float(rng.uniform(-1, 1))
    dilation = int(2 ** rng.integers(0, int(np.log2(DILATION_MAX)) + 1))
    return weights, bias, dilation

KERNELS = [generate_random_kernel() for _ in range(N_KERNELS)]

def conv1d_valid_dilated(x: np.ndarray, w: np.ndarray, bias: float, dilation: int) -> np.ndarray:
    L = x.shape[0]
    K = w.shape[0]
    out_len = L - (K - 1) * dilation
    if out_len <= 0:
        return np.array([], dtype=np.float32)

    out = np.empty(out_len, dtype=np.float32)
    for i in range(out_len):
        seg = x[i : i + dilation * K : dilation]
        out[i] = (seg * w).sum() + bias
    return out

def rocket_features_one_record(X_rec: np.ndarray) -> np.ndarray:
    L, C = X_rec.shape
    feats = []
    for (w, b, d) in KERNELS:
        for c in range(C):
            sig = X_rec[:, c]
            conv_out = conv1d_valid_dilated(sig, w, b, d)
            if conv_out.size == 0:
                feats.append(0.0)  # max
                feats.append(0.0)  # ppv
            else:
                feats.append(float(conv_out.max()))
                feats.append(float((conv_out > 0).mean()))
    return np.array(feats, dtype=np.float32)

def rocket_transform(X_arr: np.ndarray, name: str) -> np.ndarray:
    """
    Transform dataset to ROCKET-like feature matrix with frequent progress prints.
    Prints: processed/N, elapsed, rate, ETA
    """
    N = X_arr.shape[0]
    out = []
    t_start = time.time()

    for i in range(N):
        out.append(rocket_features_one_record(X_arr[i]))

        if (i + 1) % PRINT_EVERY_FEATS == 0 or (i + 1) == N:
            elapsed = time.time() - t_start
            rate = (i + 1) / max(elapsed, 1e-9)
            eta = (N - (i + 1)) / max(rate, 1e-9)
            print(f"[FEATS-{name}] {i+1}/{N} | elapsed={elapsed:.1f}s | "
                  f"rec/s={rate:.2f} | ETA={eta:.1f}s")

    return np.vstack(out)

# =========================
# Feature extraction (NOW WITH PRINTS)
# =========================
print("\nExtracting ROCKET-like features (with progress)...")
F_train = rocket_transform(X_train_n, "train")
F_val   = rocket_transform(X_val_n, "val")
F_test  = rocket_transform(X_test_n, "test")

print(f"\nFeature shapes: train={F_train.shape}, val={F_val.shape}, test={F_test.shape}")

# =========================
# Train classifier
# =========================
clf = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    n_jobs=-1
)

print("\nTraining Logistic Regression...")
clf.fit(F_train, y_train)

val_prob = clf.predict_proba(F_val)[:, 1]
test_prob = clf.predict_proba(F_test)[:, 1]

val_pred = (val_prob >= 0.5).astype(int)
test_pred = (test_prob >= 0.5).astype(int)

print("\n--- VALIDATION RESULTS ---")
print("Confusion Matrix [[TN, FP],[FN, TP]]:")
print(confusion_matrix(y_val, val_pred))
print(classification_report(y_val, val_pred, target_names=["Normal", "Danger"], zero_division=0))
print(f"Val PR-AUC:  {average_precision_score(y_val, val_prob):.4f}")
if len(set(y_val)) == 2:
    print(f"Val ROC-AUC: {roc_auc_score(y_val, val_prob):.4f}")

print("\n--- TEST RESULTS ---")
print("Confusion Matrix [[TN, FP],[FN, TP]]:")
print(confusion_matrix(y_test, test_pred))
print(classification_report(y_test, test_pred, target_names=["Normal", "Danger"], zero_division=0))
print(f"Test PR-AUC:  {average_precision_score(y_test, test_prob):.4f}")
if len(set(y_test)) == 2:
    print(f"Test ROC-AUC: {roc_auc_score(y_test, test_prob):.4f}")