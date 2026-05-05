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
info_dir = "dataset/info" 
risk_file_path = "dataset/ph_levels_with_risk.csv" 

FS = 4
SEQ_LEN = 4800                 
BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-3

POS_WEIGHT_MULT = 0.75
PATIENCE = 6
MIN_DELTA_AP = 1e-3

# --- שמות קבצים חדשים כדי לא לדרוס את ה-CNN ! ---
WEIGHTS_DIR = Path("fetal_health_web_app") / "backend" / "weights"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

FINAL_MODEL_PATH = WEIGHTS_DIR / "multimodal_rnn_model.pt"
FINAL_STATS_PATH = WEIGHTS_DIR / "multimodal_rnn_stats.json"
BEST_MODEL_PATH = "best_multimodal_rnn_binary.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# AUGMENTATION CONFIG
# ============================================================

AUGMENT_DANGER_ONLY = True
SHIFT_SECONDS = [-30, 30]

TABULAR_FEATURES = [
    'Gest. weeks', 
    'Sex',         
    'Age',         
    'Gravidity',   
    'Parity',      
    'Diabetes',    
    'Pyrexia',     
    'Meconium',    
    'Induced'      
]

# ============================================================
# 1) LOAD RISK LABELS
# ============================================================
risk_df = pd.read_csv(risk_file_path)
risk_df["record_id"] = risk_df["record_id"].astype(str)
risk_dict = dict(zip(risk_df["record_id"], risk_df["RISK"]))

# ============================================================
# 2) HELPERS FOR SIGNAL PROCESSING
# ============================================================

def make_fixed_length(fhr, uc, mask, seq_len):
    n = len(fhr)
    if n >= seq_len:
        return fhr[-seq_len:], uc[-seq_len:], mask[-seq_len:]
    pad = seq_len - n
    fhr_pad = np.pad(fhr, (0, pad), mode="edge")
    uc_pad = np.pad(uc, (0, pad), mode="edge")
    mask_pad = np.concatenate([mask, np.zeros(pad, dtype=np.float32)])
    return fhr_pad, uc_pad, mask_pad

def shift_sequence(seq, shift_samples):
    if shift_samples == 0:
        return seq.copy().astype(np.float32)
    shifted = np.empty_like(seq)
    if shift_samples > 0:
        k = min(shift_samples, len(seq) - 1)
        shifted[k:] = seq[:-k]
        shifted[:k, 0] = seq[0, 0] 
        shifted[:k, 1] = seq[0, 1] 
        shifted[:k, 2] = 0.0        
    else:
        k = min(abs(shift_samples), len(seq) - 1)
        shifted[:-k] = seq[k:]
        shifted[-k:, 0] = seq[-1, 0] 
        shifted[-k:, 1] = seq[-1, 1] 
        shifted[-k:, 2] = 0.0        
    return shifted.astype(np.float32)

def augment_train_set_multimodal(X_seq_train, X_tab_train, y_train, fs=4):
    X_seq_aug, X_tab_aug, y_aug = [], [], []
    shift_samples_list = [int(sec * fs) for sec in SHIFT_SECONDS]

    original_normal = int((y_train == 0).sum())
    original_risk = int((y_train == 1).sum())

    for seq, tab, label in zip(X_seq_train, X_tab_train, y_train):
        X_seq_aug.append(seq)
        X_tab_aug.append(tab)
        y_aug.append(label)

        if AUGMENT_DANGER_ONLY and label == 1:
            for shift_samples in shift_samples_list:
                shifted = shift_sequence(seq, shift_samples)
                X_seq_aug.append(shifted)
                X_tab_aug.append(tab) 
                y_aug.append(label)

    y_aug_arr = np.array(y_aug, dtype=np.int64)
    aug_normal = int((y_aug_arr == 0).sum())
    aug_risk = int((y_aug_arr == 1).sum())

    print("\n--- Train Augmentation Summary ---")
    print(f"Original Train -> Normal: {original_normal} | Risk: {original_risk}")
    print(f"Shift Seconds applied: {SHIFT_SECONDS}")
    print(f"Augmented Train -> Normal: {aug_normal} | Risk: {aug_risk} (Multiplied!)")
    print(f"Total training samples now: {len(y_aug_arr)}")
    print("----------------------------------\n")

    return (np.array(X_seq_aug, dtype=np.float32), 
            np.array(X_tab_aug, dtype=np.float32), 
            y_aug_arr)

def safe_average_precision(y_true, y_prob):
    if len(set(y_true)) < 2: return 0.0
    return average_precision_score(y_true, y_prob)

# ============================================================
# 3) LOAD MULTIMODAL DATA (CTG + JSON)
# ============================================================

print("Loading Real-Time Multimodal data (CTG + Tabular)...")
csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))

X_seq_list, X_tab_list, y_list, record_ids = [], [], [], []

for file in csv_files:
    record_id = os.path.basename(file).split(".")[0]

    if record_id not in risk_dict:
        continue
        
    json_path = os.path.join(info_dir, f"{record_id}.json")
    if not os.path.exists(json_path):
        continue

    try:
        with open(json_path, 'r') as f:
            patient_info = json.load(f)
            
        tab_vector = [float(patient_info.get(feature, 0.0)) for feature in TABULAR_FEATURES]

        df = pd.read_csv(file)
        if "FHR" not in df.columns or "UC" not in df.columns:
            continue

        fhr_raw = df["FHR"].values.astype(np.float32)
        uc_raw = df["UC"].values.astype(np.float32)

        mask = np.ones_like(fhr_raw, dtype=np.float32)
        mask[(fhr_raw <= 0) | np.isnan(fhr_raw)] = 0.0

        fhr_clean = (pd.Series(fhr_raw).mask((fhr_raw <= 0) | np.isnan(fhr_raw), np.nan)
                     .interpolate(method="linear").fillna(140).values.astype(np.float32))

        uc_clean = (pd.Series(uc_raw).interpolate(method="linear")
                    .fillna(0).values.astype(np.float32))

        fhr_fix, uc_fix, mask_fix = make_fixed_length(fhr_clean, uc_clean, mask, SEQ_LEN)
        seq = np.stack([fhr_fix, uc_fix, mask_fix], axis=1)

        y = int(risk_dict[record_id])

        X_seq_list.append(seq)
        X_tab_list.append(tab_vector)
        y_list.append(y)
        record_ids.append(record_id)

    except Exception as e:
        print(f"Failed to process {file}: {e}")

X_seq = np.array(X_seq_list, dtype=np.float32)
X_tab = np.array(X_tab_list, dtype=np.float32)
y = np.array(y_list, dtype=np.int64)

X_seq = np.nan_to_num(X_seq, nan=0.0, posinf=0.0, neginf=0.0)
X_tab = np.nan_to_num(X_tab, nan=0.0, posinf=0.0, neginf=0.0)

print(f"Successfully loaded {len(X_seq)} patients.")
print(f"Class counts (Normal=0, Risk=1): {np.bincount(y)}")

# ============================================================
# 4) SPLIT TRAIN / VAL / TEST 
# ============================================================

idx = np.arange(len(y))

idx_temp, idx_test = train_test_split(idx, test_size=0.15, random_state=42, stratify=y)
idx_train, idx_val = train_test_split(idx_temp, test_size=0.15, random_state=42, stratify=y[idx_temp])

X_seq_train, X_tab_train, y_train = X_seq[idx_train], X_tab[idx_train], y[idx_train]
X_seq_val, X_tab_val, y_val = X_seq[idx_val], X_tab[idx_val], y[idx_val]
X_seq_test, X_tab_test, y_test = X_seq[idx_test], X_tab[idx_test], y[idx_test]

# ============================================================
# 5) AUGMENT TRAIN ONLY
# ============================================================
X_seq_train_aug, X_tab_train_aug, y_train_aug = augment_train_set_multimodal(
    X_seq_train, X_tab_train, y_train, fs=FS
)

# ============================================================
# 6) NORMALIZE BOTH SEQUENCE AND TABULAR DATA
# ============================================================
seq_train_mean = np.nanmean(X_seq_train_aug.reshape(-1, X_seq_train_aug.shape[-1]), axis=0)
seq_train_std = np.nanstd(X_seq_train_aug.reshape(-1, X_seq_train_aug.shape[-1]), axis=0) + 1e-8

def normalize_seq(X_arr):
    norm = (X_arr - seq_train_mean) / seq_train_std
    return np.nan_to_num(norm, nan=0.0, posinf=0.0, neginf=0.0)

X_seq_train_n = normalize_seq(X_seq_train_aug)
X_seq_val_n = normalize_seq(X_seq_val)
X_seq_test_n = normalize_seq(X_seq_test)

tab_train_mean = np.nanmean(X_tab_train_aug, axis=0)
tab_train_std = np.nanstd(X_tab_train_aug, axis=0) + 1e-8

def normalize_tab(X_arr):
    norm = (X_arr - tab_train_mean) / tab_train_std
    return np.nan_to_num(norm, nan=0.0, posinf=0.0, neginf=0.0)

X_tab_train_n = normalize_tab(X_tab_train_aug)
X_tab_val_n = normalize_tab(X_tab_val)
X_tab_test_n = normalize_tab(X_tab_test)

# ============================================================
# 7) DATASET / DATALOADER
# ============================================================

class MultimodalDataset(Dataset):
    def __init__(self, X_seq, X_tab, y_arr):
        self.X_seq = torch.tensor(X_seq, dtype=torch.float32)
        self.X_tab = torch.tensor(X_tab, dtype=torch.float32)
        self.y = torch.tensor(y_arr, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_seq[idx], self.X_tab[idx], self.y[idx]

train_loader = DataLoader(MultimodalDataset(X_seq_train_n, X_tab_train_n, y_train_aug), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(MultimodalDataset(X_seq_val_n, X_tab_val_n, y_val), batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(MultimodalDataset(X_seq_test_n, X_tab_test_n, y_test), batch_size=1, shuffle=False)

# ============================================================
# 8) MULTIMODAL ARCHITECTURE (RNN / Bi-GRU)
# ============================================================

class MultimodalRNNModel(nn.Module):
    def __init__(self, seq_in_ch=3, tab_in_features=len(TABULAR_FEATURES), hidden_dim=64):
        super().__init__()

        # כיווץ הסדרה מ-4800 ל-1200 כדי שה-RNN יעבוד ביעילות ויזכור טוב יותר
        self.pool = nn.AvgPool1d(kernel_size=4, stride=4) 

        # רשת ה-RNN עצמה: Bi-directional GRU
        self.rnn = nn.GRU(
            input_size=seq_in_ch,
            hidden_size=hidden_dim,
            num_layers=2,               # שתי קומות של זיכרון
            batch_first=True,           
            bidirectional=True,         # קריאה קדימה ואחורה
            dropout=0.3
        )

        # עיבוד הנתונים הקליניים (אותו MLP כמו קודם)
        self.mlp_extractor = nn.Sequential(
            nn.Linear(tab_in_features, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )

        # הרשת הסופית שמקבלת החלטה
        # hidden_dim * 2 בגלל ה-Bidirectional (קדימה+אחורה) + 16 נתונים קליניים
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 16, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1) 
        )

    def forward(self, x_seq, x_tab):
        # סידור המימדים לכיווץ
        x_seq = x_seq.transpose(1, 2)            
        x_seq = self.pool(x_seq)
        
        # החזרת המימדים לטובת ה-RNN (Batch, Length, Channels)
        x_seq = x_seq.transpose(1, 2)
        
        # העברת האותות ב-RNN
        rnn_out, _ = self.rnn(x_seq)
        
        # Max Pooling על פני כל ציר הזמן - מושך את הפיצ'רים הכי חזקים שהרשת זיהתה
        seq_features, _ = torch.max(rnn_out, dim=1) 
        
        # העברת הנתונים הקליניים
        tab_features = self.mlp_extractor(x_tab)             
        
        # חיבור והחלטה סופית
        combined_features = torch.cat((seq_features, tab_features), dim=1) 
        logit = self.classifier(combined_features).squeeze(-1)
        return logit

model = MultimodalRNNModel().to(DEVICE)

# ============================================================
# 9) LOSS & OPTIMIZER
# ============================================================

neg = int((y_train_aug == 0).sum())
pos = int((y_train_aug == 1).sum())

base_pos_weight = neg / max(pos, 1)
pos_weight_value = base_pos_weight * POS_WEIGHT_MULT
pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32).to(DEVICE)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=LR)

# ============================================================
# 10) EVALUATION HELPERS 
# ============================================================

def get_probs_and_labels(loader):
    model.eval()
    y_true, y_prob = [], []
    with torch.no_grad():
        for x_seq, x_tab, yb in loader:
            x_seq, x_tab = x_seq.to(DEVICE), x_tab.to(DEVICE)
            logits = model(x_seq, x_tab) 
            probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
            y_true.extend(yb.numpy().astype(int).tolist())
            y_prob.extend(probs.tolist())
    return y_true, y_prob

def choose_threshold_for_custom_metric(y_true, y_prob):
    """ (2 * Recall) + Precision - מקסום המדד שבחרת """
    thresholds = np.linspace(0.05, 0.95, 91)
    
    best_thr = 0.5
    best_custom_score = -1.0
    best_rec = -1.0
    best_prec = -1.0

    for thr in thresholds:
        preds = [1 if p >= thr else 0 for p in y_prob]
        
        rec = recall_score(y_true, preds, pos_label=1, zero_division=0)
        prec = precision_score(y_true, preds, pos_label=1, zero_division=0)

        # המדד המיוחד שלך
        current_score = (2 * rec) + prec

        if current_score > best_custom_score:
            best_custom_score = current_score
            best_thr = thr
            best_rec = rec
            best_prec = prec

    return best_thr, best_custom_score, best_rec, best_prec

# ============================================================
# 11) TRAINING LOOP
# ============================================================

print("\n--- Starting Multimodal RNN (Bi-GRU) Training ---")
best_ap = -1.0
patience_counter = 0

for epoch in range(EPOCHS):
    model.train()
    train_loss_sum = 0.0

    for x_seq, x_tab, yb in train_loader:
        x_seq, x_tab, yb = x_seq.to(DEVICE), x_tab.to(DEVICE), yb.to(DEVICE)

        optimizer.zero_grad()
        logits = model(x_seq, x_tab)
        loss = criterion(logits, yb)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss_sum += loss.item()

    train_loss = train_loss_sum / max(len(train_loader), 1)

    val_true, val_prob = get_probs_and_labels(val_loader)
    val_ap = safe_average_precision(val_true, val_prob)

    if (epoch + 1) % 2 == 0 or epoch == 0:
        print(f"Epoch [{epoch + 1}/{EPOCHS}] | Train Loss: {train_loss:.4f} | Val PR-AUC: {val_ap:.4f}")

    if val_ap > best_ap + MIN_DELTA_AP:
        best_ap = val_ap
        patience_counter = 0
        torch.save(model.state_dict(), BEST_MODEL_PATH)
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch + 1}. Best Val PR-AUC: {best_ap:.4f}")
            break

model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))

# ============================================================
# 12) EVALUATION ON TEST SET
# ============================================================

val_true, val_prob = get_probs_and_labels(val_loader)
best_thr, best_custom_val, _, _ = choose_threshold_for_custom_metric(val_true, val_prob)

print("\n" + "=" * 55)
print("TEST RESULTS SUMMARY - Multimodal RNN (Bi-GRU + MLP)")
print("=" * 55)

test_true, test_prob = get_probs_and_labels(test_loader)
test_pred = [1 if p >= best_thr else 0 for p in test_prob]

cm = confusion_matrix(test_true, test_pred, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()

accuracy = accuracy_score(test_true, test_pred)
danger_precision = precision_score(test_true, test_pred, pos_label=1, zero_division=0)
danger_recall = recall_score(test_true, test_pred, pos_label=1, zero_division=0)
danger_f1 = f1_score(test_true, test_pred, pos_label=1, zero_division=0)
test_pr_auc = safe_average_precision(test_true, test_prob)

if len(set(test_true)) == 2:
    test_roc_auc = roc_auc_score(test_true, test_prob)
else:
    test_roc_auc = None

print(f"Chosen threshold (Maximized 2*R + P): {best_thr:.2f}")
print(f"Accuracy: {accuracy:.3f}")
print(f"Recall (Danger): {danger_recall:.3f}")
print(f"Precision (Danger): {danger_precision:.3f}")
print(f"False Negatives (Missed Danger): {fn}")
print(f"False Positives (False Alarms): {fp}")
print(f"PR-AUC: {test_pr_auc:.3f}")
if test_roc_auc is not None:
    print(f"ROC-AUC: {test_roc_auc:.3f}")
print("=" * 55)

# ============================================================
# 13) SAVE MODEL + STATS FOR WEB APP
# ============================================================

torch.save(model.state_dict(), FINAL_MODEL_PATH)

stats = {
    "model_type": "MultimodalRNN_MLP",
    "tabular_features": TABULAR_FEATURES,
    "augmentation": {
        "train_only": True,
        "danger_only": bool(AUGMENT_DANGER_ONLY),
        "shift_seconds": SHIFT_SECONDS,
    },
    
    "seq_train_mean": seq_train_mean.tolist(),
    "seq_train_std": seq_train_std.tolist(),
    "tab_train_mean": tab_train_mean.tolist(),
    "tab_train_std": tab_train_std.tolist(),

    "signal_channels": ["FHR", "UC", "FHR_mask"],
    "fs": FS,
    "seq_len": SEQ_LEN,
    "seq_in_ch": 3,
    "tab_in_features": len(TABULAR_FEATURES),

    "threshold": float(best_thr),
    "pos_weight_mult": float(POS_WEIGHT_MULT),
    "pos_weight": float(pos_weight_value),

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