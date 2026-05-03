"""
MiniROCKET training script.

Run from SavingBabies/ project root:
    python fetal_health_web_app/backend/training/train_minirocket.py

Outputs to fetal_health_web_app/backend/weights/:
    minirocket_model.joblib  — fitted LogisticRegression
    minirocket_stats.json    — train_mean, train_std
"""
from __future__ import annotations
import glob
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(_HERE, "..", "weights")
CSV_DIR = os.getenv("CSV_DIR", "csv_output")
PH_FILE = os.getenv("PH_FILE", "dataset/ph_levels.csv")

# ── Hyperparameters (keep in sync with minirocket_adapter.py) ─────────────────
FS = 4
SEQ_LEN = 1200       # 5 min @ 4 Hz
PH_DANGER_THRESHOLD = 7.10
N_KERNELS = 4000
KERNEL_MIN = 7
KERNEL_MAX = 51
DILATION_MAX = 64
RANDOM_STATE = 42


def load_dataset():
    ph_df = pd.read_csv(PH_FILE)
    ph_df["record_id"] = ph_df["record_id"].astype(str)
    ph_dict = dict(zip(ph_df["record_id"], ph_df["pH"]))

    X_list, y_list = [], []
    for file in glob.glob(os.path.join(CSV_DIR, "*.csv")):
        record_id = os.path.basename(file).split(".")[0]
        if record_id not in ph_dict:
            continue
        try:
            df = pd.read_csv(file)
            if "FHR" not in df.columns or "UC" not in df.columns:
                continue
            fhr_raw = df["FHR"].values.astype(np.float32)
            uc_raw = df["UC"].values.astype(np.float32)
            mask = np.ones_like(fhr_raw)
            mask[(fhr_raw == 0) | np.isnan(fhr_raw)] = 0.0
            fhr_clean = (
                pd.Series(fhr_raw).replace(0, np.nan)
                .interpolate(method="linear").fillna(140).values.astype(np.float32)
            )
            uc_clean = (
                pd.Series(uc_raw).interpolate(method="linear")
                .fillna(0).values.astype(np.float32)
            )
            n = len(fhr_clean)
            if n >= SEQ_LEN:
                fhr_fix, uc_fix, mask_fix = fhr_clean[-SEQ_LEN:], uc_clean[-SEQ_LEN:], mask[-SEQ_LEN:]
            else:
                pad = SEQ_LEN - n
                fhr_fix = np.pad(fhr_clean, (0, pad), mode="edge")
                uc_fix = np.pad(uc_clean, (0, pad), mode="edge")
                mask_fix = np.concatenate([mask, np.zeros(pad, dtype=np.float32)])
            X_list.append(np.stack([fhr_fix, uc_fix, mask_fix], axis=1))
            y_list.append(1 if float(ph_dict[record_id]) < PH_DANGER_THRESHOLD else 0)
        except Exception as e:
            print(f"  skip {file}: {e}")

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int64)


def generate_kernels(n_kernels, rng):
    kernels = []
    for _ in range(n_kernels):
        length = int(rng.integers(KERNEL_MIN, KERNEL_MAX + 1))
        weights = rng.normal(0, 1, size=length).astype(np.float32)
        bias = float(rng.uniform(-1, 1))
        dilation = int(2 ** rng.integers(0, int(np.log2(DILATION_MAX)) + 1))
        kernels.append((weights, bias, dilation))
    return kernels


def conv1d_dilated(x, w, bias, dilation):
    L, K = x.shape[0], w.shape[0]
    out_len = L - (K - 1) * dilation
    if out_len <= 0:
        return np.array([], dtype=np.float32)
    out = np.empty(out_len, dtype=np.float32)
    for i in range(out_len):
        out[i] = (x[i: i + dilation * K: dilation] * w).sum() + bias
    return out


def rocket_features_one(x_rec, kernels):
    C = x_rec.shape[1]
    feats: list[float] = []
    for (w, b, d) in kernels:
        for c in range(C):
            conv_out = conv1d_dilated(x_rec[:, c], w, b, d)
            if conv_out.size == 0:
                feats.extend([0.0, 0.0])
            else:
                feats.append(float(conv_out.max()))
                feats.append(float((conv_out > 0).mean()))
    return np.array(feats, dtype=np.float32)


def rocket_transform(X_arr, kernels, name):
    N = X_arr.shape[0]
    out = []
    t0 = time.time()
    for i in range(N):
        out.append(rocket_features_one(X_arr[i], kernels))
        if (i + 1) % 20 == 0 or (i + 1) == N:
            elapsed = time.time() - t0
            print(f"  [{name}] {i+1}/{N} | {elapsed:.0f}s elapsed")
    return np.vstack(out)


if __name__ == "__main__":
    print(f"Loading dataset from {CSV_DIR}/ ...")
    X, y = load_dataset()
    print(f"Loaded {len(X)} records. Class counts: {np.bincount(y)}")

    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=RANDOM_STATE, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.15, random_state=RANDOM_STATE, stratify=y_temp)

    train_mean = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0)
    train_std = X_train.reshape(-1, X_train.shape[-1]).std(axis=0) + 1e-8
    X_train_n = (X_train - train_mean) / train_std
    X_val_n = (X_val - train_mean) / train_std
    X_test_n = (X_test - train_mean) / train_std

    print("\nGenerating ROCKET kernels (seed=42) ...")
    rng = np.random.default_rng(RANDOM_STATE)
    kernels = generate_kernels(N_KERNELS, rng)

    print("\nExtracting features ...")
    F_train = rocket_transform(X_train_n, kernels, "train")
    F_val = rocket_transform(X_val_n, kernels, "val")
    F_test = rocket_transform(X_test_n, kernels, "test")
    print(f"Feature shape: {F_train.shape}")

    print("\nTraining LogisticRegression ...")
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=-1)
    clf.fit(F_train, y_train)

    test_prob = clf.predict_proba(F_test)[:, 1]
    test_pred = (test_prob >= 0.5).astype(int)

    print("\n--- TEST RESULTS ---")
    print(confusion_matrix(y_test, test_pred))
    print(classification_report(y_test, test_pred, target_names=["Normal", "Danger"], zero_division=0))
    print(f"PR-AUC:  {average_precision_score(y_test, test_prob):.4f}")
    if len(set(y_test)) == 2:
        print(f"ROC-AUC: {roc_auc_score(y_test, test_prob):.4f}")

    # ── Save weights ────────────────────────────────────────────────────────────
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    out_model = os.path.join(WEIGHTS_DIR, "minirocket_model.joblib")
    out_stats = os.path.join(WEIGHTS_DIR, "minirocket_stats.json")
    joblib.dump(clf, out_model)
    with open(out_stats, "w") as f:
        json.dump(
            {"train_mean": train_mean.tolist(), "train_std": train_std.tolist()},
            f,
            indent=2,
        )
    print(f"\nSaved: {out_model}")
    print(f"Saved: {out_stats}")
    print("Done. Restart the FastAPI server to load the new weights.")
