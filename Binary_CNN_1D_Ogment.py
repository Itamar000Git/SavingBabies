import os
import glob
import json
from pathlib import Path

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

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

# ============================================================
# CONFIG
# ============================================================

csv_dir = "csv_output"
ph_file_path = "dataset/ph_levels.csv"

FS = 4
SEQ_LEN = 4800                 # 20 minutes @ 4Hz
BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-3

PH_DANGER_THRESHOLD = 7.10     # label: Danger if pH < 7.10
TARGET_RECALL = 0.70           # target recall for Danger on VALIDATION

# pos_weight = (neg/pos) * POS_WEIGHT_MULT
# Bigger value usually increases Danger recall but creates more false alarms.
POS_WEIGHT_MULT = 0.75

PATIENCE = 6
MIN_DELTA_AP = 1e-3

BEST_MODEL_PATH = "best_cnn_binary_by_ap.pt"

# Web app artifacts
WEIGHTS_DIR = Path("fetal_health_web_app") / "backend" / "weights"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

FINAL_MODEL_PATH = WEIGHTS_DIR / "binarycnn_model.pt"
FINAL_STATS_PATH = WEIGHTS_DIR / "binarycnn_stats.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# AUGMENTATION CONFIG
# ============================================================

# Important:
# Augmentation is applied ONLY to the training set after train/val/test split.
# Validation and test remain untouched.

AUGMENT_DANGER_ONLY = True

# Time shifts in seconds.
# Positive shift = move signal to the right.
# Negative shift = move signal to the left.
SHIFT_SECONDS = [-30, 30]

# If you want only duplication without shift, set SHIFT_SECONDS = []
# If you want more aggressive augmentation, you can add [-180, 180],
# but start carefully.

ADD_GAUSSIAN_NOISE_TO_AUGMENTED_DANGER = False
FHR_NOISE_STD = 1.5     # bpm
UC_NOISE_STD = 0.5


# ============================================================
# 1) LOAD pH LABELS
# ============================================================

ph_df = pd.read_csv(ph_file_path)
ph_df["record_id"] = ph_df["record_id"].astype(str)
ph_dict = dict(zip(ph_df["record_id"], ph_df["pH"]))


# ============================================================
# 2) HELPERS
# ============================================================

def make_fixed_length(fhr, uc, mask, seq_len):
    """
    Make sequence length fixed:
    - If longer: keep LAST seq_len samples.
    - If shorter: pad at the END.
      FHR/UC use edge padding.
      FHR_mask uses 0 for padded area.
    """
    n = len(fhr)

    if n >= seq_len:
        return fhr[-seq_len:], uc[-seq_len:], mask[-seq_len:]

    pad = seq_len - n

    fhr_pad = np.pad(fhr, (0, pad), mode="edge")
    uc_pad = np.pad(uc, (0, pad), mode="edge")
    mask_pad = np.concatenate([mask, np.zeros(pad, dtype=np.float32)])

    return fhr_pad, uc_pad, mask_pad


def shift_sequence(seq, shift_samples):
    """
    Time-shift one CTG sequence.

    seq shape: (SEQ_LEN, 3)
    channels:
      0 = FHR
      1 = UC
      2 = FHR_mask

    Positive shift_samples:
      move the signal to the right.
      beginning is padded.

    Negative shift_samples:
      move the signal to the left.
      end is padded.

    Padded areas:
      FHR/UC get edge values.
      FHR_mask = 0, because these areas are artificial.
    """
    if shift_samples == 0:
        return seq.copy().astype(np.float32)

    shifted = np.empty_like(seq)

    if shift_samples > 0:
        k = min(shift_samples, len(seq) - 1)

        shifted[k:] = seq[:-k]

        shifted[:k, 0] = seq[0, 0]   # FHR edge value
        shifted[:k, 1] = seq[0, 1]   # UC edge value
        shifted[:k, 2] = 0.0         # artificial padded area

    else:
        k = min(abs(shift_samples), len(seq) - 1)

        shifted[:-k] = seq[k:]

        shifted[-k:, 0] = seq[-1, 0]  # FHR edge value
        shifted[-k:, 1] = seq[-1, 1]  # UC edge value
        shifted[-k:, 2] = 0.0         # artificial padded area

    return shifted.astype(np.float32)


def add_small_noise(seq, fhr_std=1.5, uc_std=0.5):
    """
    Optional augmentation:
    Add small noise to FHR and UC only where FHR_mask indicates real signal.

    This is disabled by default.
    """
    noisy = seq.copy().astype(np.float32)

    real_mask = noisy[:, 2] > 0.5

    fhr_noise = np.random.normal(0, fhr_std, size=noisy.shape[0]).astype(np.float32)
    uc_noise = np.random.normal(0, uc_std, size=noisy.shape[0]).astype(np.float32)

    noisy[real_mask, 0] += fhr_noise[real_mask]
    noisy[:, 1] += uc_noise

    # Keep values in reasonable ranges.
    noisy[:, 0] = np.clip(noisy[:, 0], 50, 220)
    noisy[:, 1] = np.clip(noisy[:, 1], 0, None)

    return noisy.astype(np.float32)


def augment_train_set_danger_only(X_train, y_train, fs=4):
    """
    Augment ONLY the training set.

    For every sample:
      - keep the original.

    For Danger samples only:
      - add shifted versions using SHIFT_SECONDS.
      - optionally add small noise to shifted versions.

    Validation and test sets are not passed to this function.
    """
    X_aug = []
    y_aug = []

    shift_samples_list = [int(sec * fs) for sec in SHIFT_SECONDS]

    original_normal = int((y_train == 0).sum())
    original_danger = int((y_train == 1).sum())

    for seq, label in zip(X_train, y_train):
        # Always keep original sample.
        X_aug.append(seq)
        y_aug.append(label)

        # Only augment Danger class.
        if AUGMENT_DANGER_ONLY and label == 1:
            for shift_samples in shift_samples_list:
                shifted = shift_sequence(seq, shift_samples)

                if ADD_GAUSSIAN_NOISE_TO_AUGMENTED_DANGER:
                    shifted = add_small_noise(
                        shifted,
                        fhr_std=FHR_NOISE_STD,
                        uc_std=UC_NOISE_STD,
                    )

                X_aug.append(shifted)
                y_aug.append(label)

    X_aug = np.array(X_aug, dtype=np.float32)
    y_aug = np.array(y_aug, dtype=np.int64)

    augmented_normal = int((y_aug == 0).sum())
    augmented_danger = int((y_aug == 1).sum())

    print("\n--- Train augmentation summary ---")
    print(f"Original train Normal: {original_normal}")
    print(f"Original train Danger: {original_danger}")
    print(f"Shift seconds:         {SHIFT_SECONDS}")
    print(f"Augmented train Normal: {augmented_normal}")
    print(f"Augmented train Danger: {augmented_danger}")
    print(f"Total train samples:    {len(y_aug)}")
    print("----------------------------------")

    return X_aug, y_aug


def safe_average_precision(y_true, y_prob):
    if len(set(y_true)) < 2:
        return 0.0
    return average_precision_score(y_true, y_prob)


# ============================================================
# 3) LOAD CTG SEQUENCES
# ============================================================
# X shape = (N, SEQ_LEN, 3)
# Channels:
#   [FHR, UC, FHR_mask]
#
# FHR_mask:
#   1 = real FHR point
#   0 = dropout / missing / invalid

print("Loading real data from CSV files...")

csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))

X_list = []
y_list = []
record_ids = []

for file in csv_files:
    record_id = os.path.basename(file).split(".")[0]

    if record_id not in ph_dict:
        continue

    try:
        df = pd.read_csv(file)

        if "FHR" not in df.columns or "UC" not in df.columns:
            continue

        fhr_raw = df["FHR"].values.astype(np.float32)
        uc_raw = df["UC"].values.astype(np.float32)

        # Mask for real FHR points.
        # Negative or zero FHR is treated as missing/invalid.
        mask = np.ones_like(fhr_raw, dtype=np.float32)
        mask[(fhr_raw <= 0) | np.isnan(fhr_raw)] = 0.0

        # Clean FHR:
        # invalid values -> NaN -> interpolate -> fill baseline.
        fhr_clean = (
            pd.Series(fhr_raw)
            .mask((fhr_raw <= 0) | np.isnan(fhr_raw), np.nan)
            .interpolate(method="linear")
            .fillna(140)
            .values.astype(np.float32)
        )

        # Clean UC:
        # Keep zeros because zeros can be real.
        # Only fill NaN.
        uc_clean = (
            pd.Series(uc_raw)
            .interpolate(method="linear")
            .fillna(0)
            .values.astype(np.float32)
        )

        fhr_fix, uc_fix, mask_fix = make_fixed_length(
            fhr_clean,
            uc_clean,
            mask,
            SEQ_LEN,
        )

        seq = np.stack([fhr_fix, uc_fix, mask_fix], axis=1)

        ph_val = float(ph_dict[record_id])
        y = 1 if ph_val < PH_DANGER_THRESHOLD else 0

        X_list.append(seq)
        y_list.append(y)
        record_ids.append(record_id)

    except Exception as e:
        print(f"Failed to process {file}: {e}")

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.int64)
record_ids = np.array(record_ids)

if len(X) == 0:
    raise RuntimeError("No valid records were loaded. Check csv_output and dataset/ph_levels.csv.")

print(f"Successfully loaded {len(X)} records.")
print(f"Signal shape: {X.shape}")
print(f"Class counts before split (Normal=0, Danger=1): {np.bincount(y)}")


# ============================================================
# 4) SPLIT TRAIN / VAL / TEST WITH STRATIFY
# ============================================================

X_temp, X_test, y_temp, y_test, ids_temp, ids_test = train_test_split(
    X,
    y,
    record_ids,
    test_size=0.15,
    random_state=42,
    stratify=y,
)

X_train, X_val, y_train, y_val, ids_train, ids_val = train_test_split(
    X_temp,
    y_temp,
    ids_temp,
    test_size=0.15,
    random_state=42,
    stratify=y_temp,
)

print("\n--- Split summary before augmentation ---")
print(f"Train class counts: {np.bincount(y_train)}")
print(f"Val class counts:   {np.bincount(y_val)}")
print(f"Test class counts:  {np.bincount(y_test)}")


# ============================================================
# 5) AUGMENT TRAIN ONLY
# ============================================================
# Important:
# Do this after split, so validation/test stay real and untouched.
# Do this before normalization, so train normalization stats include
# the actual training distribution used by the model.

X_train_aug, y_train_aug = augment_train_set_danger_only(
    X_train,
    y_train,
    fs=FS,
)


# ============================================================
# 6) NORMALIZE USING TRAIN STATS ONLY
# ============================================================

train_mean = X_train_aug.reshape(-1, X_train_aug.shape[-1]).mean(axis=0)
train_std = X_train_aug.reshape(-1, X_train_aug.shape[-1]).std(axis=0) + 1e-8

def normalize(X_arr):
    return (X_arr - train_mean) / train_std

X_train_n = normalize(X_train_aug)
X_val_n = normalize(X_val)
X_test_n = normalize(X_test)


# ============================================================
# 7) DATASET / DATALOADER
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


train_loader = DataLoader(
    CTGBinaryDataset(X_train_n, y_train_aug),
    batch_size=BATCH_SIZE,
    shuffle=True,
)

val_loader = DataLoader(
    CTGBinaryDataset(X_val_n, y_val),
    batch_size=BATCH_SIZE,
    shuffle=False,
)

test_loader = DataLoader(
    CTGBinaryDataset(X_test_n, y_test),
    batch_size=1,
    shuffle=False,
)


# ============================================================
# 8) CNN-ONLY MODEL
# ============================================================

class CNNBinaryCTG(nn.Module):
    """
    CNN-only model.

    Output:
      one logit before sigmoid.

    sigmoid(logit):
      probability of Danger.
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
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = x.transpose(1, 2)             # (B, L, C) -> (B, C, L)
        x = self.fe(x)                    # (B, 128, L')
        x = self.gap(x).squeeze(-1)       # (B, 128)
        logit = self.head(x).squeeze(-1)  # (B,)
        return logit


model = CNNBinaryCTG(in_ch=3).to(DEVICE)


# ============================================================
# 9) LOSS WITH POSITIVE CLASS WEIGHT + OPTIMIZER
# ============================================================
# Important:
# pos_weight is computed on the augmented train set.
#
# Because we already increased Danger through augmentation, pos_weight
# will usually be smaller than before. This prevents over-amplifying Danger.

neg = int((y_train_aug == 0).sum())
pos = int((y_train_aug == 1).sum())

base_pos_weight = neg / max(pos, 1)
pos_weight_value = base_pos_weight * POS_WEIGHT_MULT

pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32).to(DEVICE)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=LR)

print("\nClass weighting after augmentation:")
print(f"Train Normal count: {neg}")
print(f"Train Danger count: {pos}")
print(f"Base pos_weight:    {base_pos_weight:.3f}")
print(f"POS_WEIGHT_MULT:    {POS_WEIGHT_MULT:.3f}")
print(f"Final pos_weight:   {pos_weight_value:.3f}")


# ============================================================
# 10) HELPERS: PROBABILITIES + THRESHOLD SELECTION
# ============================================================

def get_probs_and_labels(loader):
    """
    Run model on a loader and return:
      y_true: list of 0/1
      y_prob: list of probabilities.

    y_prob = sigmoid(logit) = probability of Danger.
    """
    model.eval()

    y_true = []
    y_prob = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)

            logits = model(xb)
            probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)

            y_true.extend(yb.numpy().astype(int).tolist())
            y_prob.extend(probs.tolist())

    return y_true, y_prob


def choose_threshold_for_recall(y_true, y_prob, target_recall=0.70):
    """
    Choose threshold automatically using VALIDATION set.

    Scan thresholds and pick one that:
      - achieves recall >= target_recall for Danger
      - among those, has the highest precision

    If none reaches target recall, choose threshold with max recall.
    """
    thresholds = np.linspace(0.05, 0.95, 19)

    best_thr = 0.5
    best_prec = -1.0
    best_rec = -1.0

    best_rec_thr = 0.5
    best_rec_val = -1.0

    for thr in thresholds:
        preds = [1 if p >= thr else 0 for p in y_prob]

        rec = recall_score(y_true, preds, pos_label=1, zero_division=0)
        pre = precision_score(y_true, preds, pos_label=1, zero_division=0)

        if rec > best_rec_val:
            best_rec_val = rec
            best_rec_thr = thr

        if rec >= target_recall:
            if pre > best_prec:
                best_prec = pre
                best_rec = rec
                best_thr = thr

    if best_prec < 0:
        preds = [1 if p >= best_rec_thr else 0 for p in y_prob]

        return (
            best_rec_thr,
            best_rec_val,
            precision_score(y_true, preds, pos_label=1, zero_division=0),
        )

    return best_thr, best_rec, best_prec


# ============================================================
# 11) TRAINING LOOP WITH EARLY STOPPING BY VALIDATION PR-AUC
# ============================================================

print("\n--- Starting Binary CNN Training with Danger Augmentation ---")
print("Early stopping metric: Validation PR-AUC")

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

    # ----- VALIDATION PR-AUC -----
    val_true, val_prob = get_probs_and_labels(val_loader)
    val_ap = safe_average_precision(val_true, val_prob)

    # ----- Validation loss, only for logging -----
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
        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val PR-AUC: {val_ap:.4f}"
        )

    # ----- Early stopping -----
    if val_ap > best_ap + MIN_DELTA_AP:
        best_ap = val_ap
        patience_counter = 0
        torch.save(model.state_dict(), BEST_MODEL_PATH)
    else:
        patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch + 1}. Best Val PR-AUC: {best_ap:.4f}")
            break


# Load best model
model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
model.eval()


# ============================================================
# 12) CHOOSE THRESHOLD ON VALIDATION
# ============================================================

val_true, val_prob = get_probs_and_labels(val_loader)

best_thr, best_rec, best_prec = choose_threshold_for_recall(
    val_true,
    val_prob,
    TARGET_RECALL,
)

print("\n--- Threshold selection on VALIDATION ---")
print(f"Target Recall:    {TARGET_RECALL:.2f}")
print(f"Chosen threshold: {best_thr:.2f}")
print(f"Val Recall:       {best_rec:.3f}")
print(f"Val Precision:    {best_prec:.3f}")


# ============================================================
# 13) TEST EVALUATION - CLEAR SUMMARY
# ============================================================

print("\n" + "=" * 55)
print("TEST RESULTS SUMMARY - BinaryCNN with Train-Only Augmentation")
print("=" * 55)

test_true, test_prob = get_probs_and_labels(test_loader)
test_pred = [1 if p >= best_thr else 0 for p in test_prob]

cm = confusion_matrix(test_true, test_pred, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()

accuracy = accuracy_score(test_true, test_pred)

danger_precision = precision_score(
    test_true,
    test_pred,
    pos_label=1,
    zero_division=0,
)

danger_recall = recall_score(
    test_true,
    test_pred,
    pos_label=1,
    zero_division=0,
)

danger_f1 = f1_score(
    test_true,
    test_pred,
    pos_label=1,
    zero_division=0,
)

test_pr_auc = safe_average_precision(test_true, test_prob)

if len(set(test_true)) == 2:
    test_roc_auc = roc_auc_score(test_true, test_prob)
else:
    test_roc_auc = None

print(f"Decision threshold: {best_thr:.2f}")
print()

print("Confusion Matrix:")
print(f"TN = {tn} | FP = {fp}")
print(f"FN = {fn} | TP = {tp}")
print()

print("Main Metrics:")
print(f"Accuracy:          {accuracy:.3f}")
print(f"Danger Precision:  {danger_precision:.3f}")
print(f"Danger Recall:     {danger_recall:.3f}")
print(f"Danger F1-score:   {danger_f1:.3f}")
print()

print("Medical Risk View:")
print(f"False Negatives - Danger predicted as Normal: {fn}")
print(f"False Positives - Normal predicted as Danger: {fp}")
print()

print(f"PR-AUC:            {test_pr_auc:.3f}")

if test_roc_auc is not None:
    print(f"ROC-AUC:           {test_roc_auc:.3f}")
else:
    print("ROC-AUC:           not defined")

print("=" * 55)


# ============================================================
# 14) SAVE MODEL + STATS FOR WEB APP
# ============================================================

torch.save(model.state_dict(), FINAL_MODEL_PATH)

stats = {
    "model_type": "BinaryCNN",
    "augmentation": {
        "train_only": True,
        "danger_only": bool(AUGMENT_DANGER_ONLY),
        "shift_seconds": SHIFT_SECONDS,
        "add_gaussian_noise": bool(ADD_GAUSSIAN_NOISE_TO_AUGMENTED_DANGER),
        "fhr_noise_std": float(FHR_NOISE_STD),
        "uc_noise_std": float(UC_NOISE_STD),
    },

    # Compatibility with existing BinaryCNNAdapter.
    "train_mean": train_mean.tolist(),
    "train_std": train_std.tolist(),

    # Explicit names.
    "signal_channels": ["FHR", "UC", "FHR_mask"],
    "fs": FS,
    "seq_len": SEQ_LEN,
    "in_ch": 3,

    # Decision threshold.
    "threshold": float(best_thr),
    "target_recall": float(TARGET_RECALL),

    # Training setup.
    "ph_danger_threshold": float(PH_DANGER_THRESHOLD),
    "pos_weight_mult": float(POS_WEIGHT_MULT),
    "pos_weight": float(pos_weight_value),

    # Evaluation results.
    "test_metrics": {
        "accuracy": float(accuracy),
        "danger_precision": float(danger_precision),
        "danger_recall": float(danger_recall),
        "danger_f1": float(danger_f1),
        "false_negatives": int(fn),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "true_positives": int(tp),
        "pr_auc": float(test_pr_auc),
        "roc_auc": None if test_roc_auc is None else float(test_roc_auc),
    },
}

with open(FINAL_STATS_PATH, "w", encoding="utf-8") as f:
    json.dump(stats, f, indent=2)

print("\nSaved artifacts for web app:")
print(f"Model state_dict: {FINAL_MODEL_PATH}")
print(f"Stats JSON:       {FINAL_STATS_PATH}")
print()
print("Important:")
print("Restart the FastAPI backend so the web app reloads the new model weights.")