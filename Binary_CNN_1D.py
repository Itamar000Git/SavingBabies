import os
import glob
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,   # PR-AUC (Average Precision)
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)

# ============================================================
# CONFIG
# ============================================================

csv_dir = "csv_output"
ph_file_path = "dataset/ph_levels.csv"

FS = 4
SEQ_LEN = 4800                 # 20 minutes @ 4Hz  (change if needed)
BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-3

PH_DANGER_THRESHOLD = 7.10     # label: Danger if pH < 7.10

# --- Change #2: choose threshold to reach target Recall ---
TARGET_RECALL = 0.70           # we want at least this recall for Danger on VALIDATION

# --- Change #3: knob to push recall vs precision ---
# pos_weight = (neg/pos) * POS_WEIGHT_MULT
# bigger -> more recall (usually) but more false alarms
# smaller -> fewer false alarms but recall can drop
POS_WEIGHT_MULT = 1.0

# --- Change #1: early stopping based on PR-AUC (not val loss) ---
PATIENCE = 6
MIN_DELTA_AP = 1e-3            # minimal AP improvement to be considered "better"
BEST_MODEL_PATH = "best_cnn_binary_by_ap.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 1) LOAD pH LABELS
# ============================================================

ph_df = pd.read_csv(ph_file_path)
ph_df["record_id"] = ph_df["record_id"].astype(str)
ph_dict = dict(zip(ph_df["record_id"], ph_df["pH"]))


# ============================================================
# 2) LOAD CTG SEQUENCES + BUILD MASK CHANNEL
# ============================================================
# X will be (N, SEQ_LEN, 3) with channels:
#   [FHR, UC, FHR_mask]
# FHR_mask = 1 if FHR was real, 0 if dropout/missing

print("Loading real data from CSV files...")

csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
X_list, y_list = [], []

def make_fixed_length(fhr, uc, mask, seq_len):
    """
    Make sequence length fixed:
    - If longer: keep LAST seq_len samples
    - If shorter: pad at the END (edge padding for signals, zeros for mask)
    """
    n = len(fhr)
    if n >= seq_len:
        return fhr[-seq_len:], uc[-seq_len:], mask[-seq_len:]
    pad = seq_len - n
    fhr_pad = np.pad(fhr, (0, pad), mode="edge")
    uc_pad  = np.pad(uc,  (0, pad), mode="edge")
    mask_pad = np.concatenate([mask, np.zeros(pad, dtype=np.float32)])
    return fhr_pad, uc_pad, mask_pad

for file in csv_files:
    record_id = os.path.basename(file).split(".")[0]
    if record_id not in ph_dict:
        continue

    try:
        df = pd.read_csv(file)
        if "FHR" not in df.columns or "UC" not in df.columns:
            continue

        fhr_raw = df["FHR"].values.astype(np.float32)
        uc_raw  = df["UC"].values.astype(np.float32)

        # Mask for real FHR points
        mask = np.ones_like(fhr_raw, dtype=np.float32)
        mask[(fhr_raw == 0) | np.isnan(fhr_raw)] = 0.0

        # Clean FHR: 0 -> NaN -> interpolate -> fill baseline
        fhr_clean = (
            pd.Series(fhr_raw)
            .replace(0, np.nan)
            .interpolate(method="linear")
            .fillna(140)
            .values.astype(np.float32)
        )

        # Clean UC: keep zeros (zeros can be real), only fill NaN
        uc_clean = (
            pd.Series(uc_raw)
            .interpolate(method="linear")
            .fillna(0)
            .values.astype(np.float32)
        )

        fhr_fix, uc_fix, mask_fix = make_fixed_length(fhr_clean, uc_clean, mask, SEQ_LEN)
        seq = np.stack([fhr_fix, uc_fix, mask_fix], axis=1)  # (SEQ_LEN, 3)

        ph_val = float(ph_dict[record_id])
        y = 1 if ph_val < PH_DANGER_THRESHOLD else 0

        X_list.append(seq)
        y_list.append(y)

    except Exception as e:
        print(f"Failed to process {file}: {e}")

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.int64)

print(f"Successfully loaded {len(X)} records.")
print(f"Class counts (Normal=0, Danger=1): {np.bincount(y)}")


# ============================================================
# 3) SPLIT (train/val/test) WITH STRATIFY
# ============================================================

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.15, random_state=42, stratify=y_temp
)


# ============================================================
# 4) NORMALIZE USING TRAIN STATS ONLY
# ============================================================

train_mean = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0)
train_std  = X_train.reshape(-1, X_train.shape[-1]).std(axis=0) + 1e-8

def normalize(X_arr):
    return (X_arr - train_mean) / train_std

X_train_n = normalize(X_train)
X_val_n   = normalize(X_val)
X_test_n  = normalize(X_test)


# ============================================================
# 5) DATASET / DATALOADER
# ============================================================

class CTGBinaryDataset(Dataset):
    """
    X: (SEQ_LEN, 3) float
    y: float 0.0/1.0 because BCEWithLogitsLoss expects float targets
    """
    def __init__(self, X_arr, y_arr):
        self.X = torch.tensor(X_arr, dtype=torch.float32)
        self.y = torch.tensor(y_arr, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(CTGBinaryDataset(X_train_n, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(CTGBinaryDataset(X_val_n, y_val), batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(CTGBinaryDataset(X_test_n, y_test), batch_size=1, shuffle=False)


# ============================================================
# 6) CNN-ONLY MODEL
# ============================================================

class CNNBinaryCTG(nn.Module):
    """
    CNN-only model. Output: 1 logit (before sigmoid).
    sigmoid(logit) = probability of Danger.
    """
    def __init__(self, in_ch=3):
        super().__init__()
        self.fe = nn.Sequential(
            nn.Conv1d(in_ch, 32, kernel_size=15, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=15, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=11, padding=5),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = x.transpose(1, 2)         # (B, L, C) -> (B, C, L)
        x = self.fe(x)                # (B, 128, L')
        x = self.gap(x).squeeze(-1)   # (B, 128)
        logit = self.head(x).squeeze(-1)  # (B,)
        return logit

model = CNNBinaryCTG(in_ch=3).to(DEVICE)


# ============================================================
# 7) LOSS (with pos_weight) + OPTIMIZER
# ============================================================
# Change #3: pos_weight has a multiplier you can tune.

neg = int((y_train == 0).sum())
pos = int((y_train == 1).sum())
base_pos_weight = (neg / max(pos, 1))
pos_weight = torch.tensor([base_pos_weight * POS_WEIGHT_MULT], dtype=torch.float32).to(DEVICE)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=LR)


# ============================================================
# 8) HELPERS: get probabilities + choose best threshold on VAL
# ============================================================

def get_probs_and_labels(loader):
    """
    Run model on a loader and return:
      y_true: list of 0/1
      y_prob: list of probabilities (sigmoid output)
    """
    model.eval()
    yt, yp = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            logit = model(xb)
            prob = torch.sigmoid(logit).cpu().numpy().reshape(-1)
            yt.extend(yb.numpy().astype(int).tolist())
            yp.extend(prob.tolist())
    return yt, yp

def choose_threshold_for_recall(y_true, y_prob, target_recall=0.70):
    """
    Change #2:
    Choose threshold automatically using VALIDATION set.

    We scan thresholds and pick one that:
      - achieves recall >= target_recall (for Danger class)
      - among those, we choose the one with the highest precision
    If none reaches target_recall, we choose the threshold with max recall.
    """
    thresholds = np.linspace(0.05, 0.95, 19)
    best_thr = 0.5
    best_prec = -1.0
    best_rec = -1.0

    # Track best recall in case we cannot reach target
    best_rec_thr = 0.5
    best_rec_val = -1.0

    for thr in thresholds:
        preds = [1 if p >= thr else 0 for p in y_prob]
        rec = recall_score(y_true, preds, zero_division=0)
        pre = precision_score(y_true, preds, zero_division=0)

        if rec > best_rec_val:
            best_rec_val = rec
            best_rec_thr = thr

        if rec >= target_recall:
            # among thresholds that meet recall target, maximize precision
            if pre > best_prec:
                best_prec = pre
                best_rec = rec
                best_thr = thr

    if best_prec < 0:
        # could not reach recall target
        return best_rec_thr, best_rec_val, precision_score(y_true, [1 if p >= best_rec_thr else 0 for p in y_prob], zero_division=0)
    else:
        return best_thr, best_rec, best_prec


# ============================================================
# 9) TRAINING LOOP with Early Stopping by PR-AUC (Change #1)
# ============================================================

print("\n--- Starting Binary CNN Training (Early Stop by PR-AUC) ---")

best_ap = -1.0
patience_counter = 0

for epoch in range(EPOCHS):
    # ----- TRAIN -----
    model.train()
    train_loss_sum = 0.0

    for xb, yb in train_loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item()

    train_loss = train_loss_sum / max(len(train_loader), 1)

    # ----- VALIDATION: compute PR-AUC (Average Precision) -----
    val_true, val_prob = get_probs_and_labels(val_loader)
    val_ap = average_precision_score(val_true, val_prob)

    # (Optional) also compute val loss for logging
    model.eval()
    val_loss_sum = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            logits = model(xb)
            val_loss_sum += criterion(logits, yb).item()
    val_loss = val_loss_sum / max(len(val_loader), 1)

    if (epoch + 1) % 2 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val PR-AUC: {val_ap:.4f}")

    # ----- Early stopping on PR-AUC -----
    if val_ap > best_ap + MIN_DELTA_AP:
        best_ap = val_ap
        patience_counter = 0
        torch.save(model.state_dict(), BEST_MODEL_PATH)
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1}. Best Val PR-AUC: {best_ap:.4f}")
            break

# Load best model
model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
model.eval()


# ============================================================
# 10) CHOOSE THRESHOLD ON VALIDATION (Change #2)
# ============================================================

val_true, val_prob = get_probs_and_labels(val_loader)
best_thr, best_rec, best_prec = choose_threshold_for_recall(val_true, val_prob, TARGET_RECALL)

print("\n--- Threshold selection on VALIDATION ---")
print(f"Target Recall: {TARGET_RECALL:.2f}")
print(f"Chosen threshold: {best_thr:.2f} | Val Recall: {best_rec:.3f} | Val Precision: {best_prec:.3f}")


# ============================================================
# 11) TEST EVALUATION + PR-AUC + ROC-AUC
# ============================================================

print("\n--- Evaluating on TEST set ---")

test_true, test_prob = get_probs_and_labels(test_loader)
test_pred = [1 if p >= best_thr else 0 for p in test_prob]

print("\nConfusion Matrix [[TN, FP], [FN, TP]]:")
print(confusion_matrix(test_true, test_pred))

print("\nClassification Report (using chosen threshold):")
print(classification_report(test_true, test_pred, target_names=["Normal", "Danger"], zero_division=0))

# PR-AUC (new metric)
test_pr_auc = average_precision_score(test_true, test_prob)
print(f"\nTEST PR-AUC (Average Precision): {test_pr_auc:.4f}")

# ROC-AUC (optional)
if len(set(test_true)) == 2:
    test_roc_auc = roc_auc_score(test_true, test_prob)
    print(f"TEST ROC-AUC: {test_roc_auc:.4f}")
else:
    print("TEST ROC-AUC: not defined (only one class in test)")