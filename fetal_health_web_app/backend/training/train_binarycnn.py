"""
BinaryCNN training script.

Run from SavingBabies/ project root:
    python fetal_health_web_app/backend/training/train_binarycnn.py

Outputs to fetal_health_web_app/backend/weights/:
    binarycnn_model.pt       — model state dict
    binarycnn_stats.json     — train_mean, train_std, threshold
"""
from __future__ import annotations
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(_HERE, "..", "weights")
CSV_DIR = os.getenv("CSV_DIR", "csv_output")
PH_FILE = os.getenv("PH_FILE", "dataset/ph_levels.csv")

# ── Hyperparameters (keep in sync with binarycnn_adapter.py SEQ_LEN) ──────────
FS = 4
SEQ_LEN = 4800
BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-3
PH_DANGER_THRESHOLD = 7.10
TARGET_RECALL = 0.70
POS_WEIGHT_MULT = 1.0
PATIENCE = 6
MIN_DELTA_AP = 1e-3
RANDOM_STATE = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Model definition (must match binarycnn_adapter.py _CNNBinaryCTG) ──────────
class CNNBinaryCTG(nn.Module):
    def __init__(self, in_ch: int = 3):
        super().__init__()
        self.fe = nn.Sequential(
            nn.Conv1d(in_ch, 32, kernel_size=15, padding=7),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=15, padding=7),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=11, padding=5),
            nn.BatchNorm1d(128), nn.ReLU(),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.fe(x)
        x = self.gap(x).squeeze(-1)
        return self.head(x).squeeze(-1)


class CTGDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def make_fixed_length(fhr, uc, mask, seq_len):
    n = len(fhr)
    if n >= seq_len:
        return fhr[-seq_len:], uc[-seq_len:], mask[-seq_len:]
    pad = seq_len - n
    return (
        np.pad(fhr, (0, pad), mode="edge"),
        np.pad(uc, (0, pad), mode="edge"),
        np.concatenate([mask, np.zeros(pad, dtype=np.float32)]),
    )


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
            fhr_fix, uc_fix, mask_fix = make_fixed_length(fhr_clean, uc_clean, mask, SEQ_LEN)
            X_list.append(np.stack([fhr_fix, uc_fix, mask_fix], axis=1))
            y_list.append(1 if float(ph_dict[record_id]) < PH_DANGER_THRESHOLD else 0)
        except Exception as e:
            print(f"  skip {file}: {e}")

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int64)


def get_probs(model, loader):
    model.eval()
    yt, yp = [], []
    with torch.no_grad():
        for xb, yb in loader:
            logit = model(xb.to(DEVICE))
            prob = torch.sigmoid(logit).cpu().numpy().reshape(-1)
            yt.extend(yb.numpy().astype(int).tolist())
            yp.extend(prob.tolist())
    return yt, yp


def choose_threshold(y_true, y_prob, target_recall=0.70):
    best_thr, best_prec, best_rec = 0.5, -1.0, -1.0
    best_rec_thr, best_rec_val = 0.5, -1.0
    for thr in np.linspace(0.05, 0.95, 19):
        preds = [1 if p >= thr else 0 for p in y_prob]
        rec = recall_score(y_true, preds, zero_division=0)
        pre = precision_score(y_true, preds, zero_division=0)
        if rec > best_rec_val:
            best_rec_val, best_rec_thr = rec, thr
        if rec >= target_recall and pre > best_prec:
            best_prec, best_rec, best_thr = pre, rec, thr
    if best_prec < 0:
        return best_rec_thr
    return best_thr


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

    train_loader = DataLoader(CTGDataset(X_train_n, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(CTGDataset(X_val_n, y_val), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(CTGDataset(X_test_n, y_test), batch_size=1, shuffle=False)

    model = CNNBinaryCTG(in_ch=3).to(DEVICE)
    neg, pos = int((y_train == 0).sum()), int((y_train == 1).sum())
    pos_weight = torch.tensor([neg / max(pos, 1) * POS_WEIGHT_MULT]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_ap, patience_counter = -1.0, 0
    _tmp_path = os.path.join(WEIGHTS_DIR, "_tmp_best.pt")
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    print("\nTraining ...")
    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()

        val_true, val_prob = get_probs(model, val_loader)
        val_ap = average_precision_score(val_true, val_prob)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS} | Val PR-AUC: {val_ap:.4f}")

        if val_ap > best_ap + MIN_DELTA_AP:
            best_ap, patience_counter = val_ap, 0
            torch.save(model.state_dict(), _tmp_path)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stop at epoch {epoch+1}. Best Val PR-AUC: {best_ap:.4f}")
                break

    model.load_state_dict(torch.load(_tmp_path, map_location=DEVICE))
    model.eval()

    val_true, val_prob = get_probs(model, val_loader)
    best_thr = choose_threshold(val_true, val_prob, TARGET_RECALL)
    print(f"\nChosen threshold: {best_thr:.2f}")

    test_true, test_prob = get_probs(model, test_loader)
    test_pred = [1 if p >= best_thr else 0 for p in test_prob]
    print("\n--- TEST RESULTS ---")
    print(confusion_matrix(test_true, test_pred))
    print(classification_report(test_true, test_pred, target_names=["Normal", "Danger"], zero_division=0))
    print(f"PR-AUC: {average_precision_score(test_true, test_prob):.4f}")
    if len(set(test_true)) == 2:
        print(f"ROC-AUC: {roc_auc_score(test_true, test_prob):.4f}")

    # ── Save weights ────────────────────────────────────────────────────────────
    out_model = os.path.join(WEIGHTS_DIR, "binarycnn_model.pt")
    out_stats = os.path.join(WEIGHTS_DIR, "binarycnn_stats.json")
    torch.save(model.state_dict(), out_model)
    with open(out_stats, "w") as f:
        json.dump(
            {
                "train_mean": train_mean.tolist(),
                "train_std": train_std.tolist(),
                "threshold": float(best_thr),
            },
            f,
            indent=2,
        )
    os.remove(_tmp_path)
    print(f"\nSaved: {out_model}")
    print(f"Saved: {out_stats}")
    print("Done. Restart the FastAPI server to load the new weights.")
