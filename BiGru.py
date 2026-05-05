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
risk_file_path = "dataset/ph_levels_with_be_risk_filled.csv" 

FS = 4
SEQ_LEN = 4800                 
BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-3

POS_WEIGHT_MULT = 0.75
PATIENCE = 6
MIN_DELTA_AP = 1e-3

# --- שמות קבצים ---
WEIGHTS_DIR = Path("fetal_health_web_app") / "backend" / "weights"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

FINAL_MODEL_PATH = WEIGHTS_DIR / "multimodal_rnn_multitask_model.pt"
FINAL_STATS_PATH = WEIGHTS_DIR / "multimodal_rnn_multitask_stats.json"
BEST_MODEL_PATH = "best_multimodal_rnn_multitask.pt"

# זיהוי ושימוש ב-GPU אם קיים
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n[SYSTEM] Running computation on device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"[SYSTEM] GPU Name: {torch.cuda.get_device_name(0)}\n")

# ============================================================
# TABULAR FEATURES DEFINITION
# ============================================================
# BE אינו נמצא כאן כדי למנוע זליגת נתונים
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
# 1) LOAD RISK LABELS (MULTI-TASK)
# ============================================================
risk_df = pd.read_csv(risk_file_path)
risk_df["record_id"] = risk_df["record_id"].astype(str)

# חילוץ שתי התוויות במקביל
risk_dict = {str(row['record_id']): [int(row['RISK']), int(row['riskBE'])] for _, row in risk_df.iterrows()}

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

        y = risk_dict[record_id]

        X_seq_list.append(seq)
        X_tab_list.append(tab_vector)
        y_list.append(y)
        record_ids.append(record_id)

    except Exception as e:
        print(f"Failed to process {file}: {e}")

X_seq = np.array(X_seq_list, dtype=np.float32)
X_tab = np.array(X_tab_list, dtype=np.float32)
y = np.array(y_list, dtype=np.float32)

X_seq = np.nan_to_num(X_seq, nan=0.0, posinf=0.0, neginf=0.0)
X_tab = np.nan_to_num(X_tab, nan=0.0, posinf=0.0, neginf=0.0)

print(f"Successfully loaded {len(X_seq)} patients.")

# ספירת המטופלים עם *כל* סוג של סיכון להדפסה בלבד
any_risk_counts = np.bincount(((y[:, 0] == 1) | (y[:, 1] == 1)).astype(int))
print(f"Overall Class counts (Normal=0, Any Risk=1): {any_risk_counts}")

# ============================================================
# 4) SPLIT TRAIN / VAL / TEST 
# ============================================================

idx = np.arange(len(y))

idx_temp, idx_test = train_test_split(idx, test_size=0.15, random_state=42, stratify=y[:, 0])
idx_train, idx_val = train_test_split(idx_temp, test_size=0.15, random_state=42, stratify=y[idx_temp, 0])

X_seq_train, X_tab_train, y_train = X_seq[idx_train], X_tab[idx_train], y[idx_train]
X_seq_val, X_tab_val, y_val = X_seq[idx_val], X_tab[idx_val], y[idx_val]
X_seq_test, X_tab_test, y_test = X_seq[idx_test], X_tab[idx_test], y[idx_test]

# ============================================================
# 5) NORMALIZE BOTH SEQUENCE AND TABULAR DATA
# ============================================================
seq_train_mean = np.nanmean(X_seq_train.reshape(-1, X_seq_train.shape[-1]), axis=0)
seq_train_std = np.nanstd(X_seq_train.reshape(-1, X_seq_train.shape[-1]), axis=0) + 1e-8

def normalize_seq(X_arr):
    norm = (X_arr - seq_train_mean) / seq_train_std
    return np.nan_to_num(norm, nan=0.0, posinf=0.0, neginf=0.0)

X_seq_train_n = normalize_seq(X_seq_train)
X_seq_val_n = normalize_seq(X_seq_val)
X_seq_test_n = normalize_seq(X_seq_test)

tab_train_mean = np.nanmean(X_tab_train, axis=0)
tab_train_std = np.nanstd(X_tab_train, axis=0) + 1e-8

def normalize_tab(X_arr):
    norm = (X_arr - tab_train_mean) / tab_train_std
    return np.nan_to_num(norm, nan=0.0, posinf=0.0, neginf=0.0)

X_tab_train_n = normalize_tab(X_tab_train)
X_tab_val_n = normalize_tab(X_tab_val)
X_tab_test_n = normalize_tab(X_tab_test)

# ============================================================
# 6) DATASET / DATALOADER
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

train_loader = DataLoader(MultimodalDataset(X_seq_train_n, X_tab_train_n, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(MultimodalDataset(X_seq_val_n, X_tab_val_n, y_val), batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(MultimodalDataset(X_seq_test_n, X_tab_test_n, y_test), batch_size=1, shuffle=False)

# ============================================================
# 7) MULTIMODAL ARCHITECTURE (RNN / Bi-GRU / MULTI-TASK)
# ============================================================

class MultimodalRNNModel(nn.Module):
    def __init__(self, seq_in_ch=3, tab_in_features=len(TABULAR_FEATURES), hidden_dim=64):
        super().__init__()

        self.pool = nn.AvgPool1d(kernel_size=4, stride=4) 

        self.rnn = nn.GRU(
            input_size=seq_in_ch,
            hidden_size=hidden_dim,
            num_layers=2,               
            batch_first=True,           
            bidirectional=True,         
            dropout=0.3
        )

        self.mlp_extractor = nn.Sequential(
            nn.Linear(tab_in_features, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 16, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2) # מנבא 2 תוויות
        )

    def forward(self, x_seq, x_tab):
        x_seq = x_seq.transpose(1, 2)            
        x_seq = self.pool(x_seq)
        
        x_seq = x_seq.transpose(1, 2)
        
        rnn_out, _ = self.rnn(x_seq)
        
        seq_features, _ = torch.max(rnn_out, dim=1) 
        
        tab_features = self.mlp_extractor(x_tab)             
        
        combined_features = torch.cat((seq_features, tab_features), dim=1) 
        logits = self.classifier(combined_features)
        return logits

model = MultimodalRNNModel().to(DEVICE)

# ============================================================
# 8) LOSS & OPTIMIZER (MULTI-TASK)
# ============================================================

neg_risk = (y_train[:, 0] == 0).sum()
pos_risk = (y_train[:, 0] == 1).sum()
pw_risk = (neg_risk / max(pos_risk, 1)) * POS_WEIGHT_MULT

neg_be = (y_train[:, 1] == 0).sum()
pos_be = (y_train[:, 1] == 1).sum()
pw_be = (neg_be / max(pos_be, 1)) * POS_WEIGHT_MULT

pos_weight = torch.tensor([pw_risk, pw_be], dtype=torch.float32).to(DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=LR)

# ============================================================
# 9) EVALUATION HELPERS 
# ============================================================

def get_probs_and_labels(loader):
    model.eval()
    y_true, y_prob = [], []
    with torch.no_grad():
        for x_seq, x_tab, yb in loader:
            x_seq, x_tab = x_seq.to(DEVICE), x_tab.to(DEVICE)
            logits = model(x_seq, x_tab) 
            probs = torch.sigmoid(logits).cpu().numpy()
            y_true.extend(yb.cpu().numpy().tolist())
            y_prob.extend(probs.tolist())
    return np.array(y_true), np.array(y_prob)

def choose_threshold_for_custom_metric(y_true, y_prob):
    """ (2 * Recall) + Precision """
    thresholds = np.linspace(0.05, 0.95, 91)
    
    best_thr = 0.5
    best_custom_score = -1.0

    for thr in thresholds:
        preds = [1 if p >= thr else 0 for p in y_prob]
        
        rec = recall_score(y_true, preds, pos_label=1, zero_division=0)
        prec = precision_score(y_true, preds, pos_label=1, zero_division=0)

        current_score = (2 * rec) + prec

        if current_score > best_custom_score:
            best_custom_score = current_score
            best_thr = thr

    return best_thr

# ============================================================
# 10) TRAINING LOOP
# ============================================================

print("\n--- Starting Multi-Task RNN (Bi-GRU) Training (No Augmentation) ---")
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
    
    val_ap_risk = safe_average_precision(val_true[:, 0], val_prob[:, 0])
    val_ap_be = safe_average_precision(val_true[:, 1], val_prob[:, 1])
    val_ap_mean = (val_ap_risk + val_ap_be) / 2.0

    if (epoch + 1) % 2 == 0 or epoch == 0:
        print(f"Epoch [{epoch + 1}/{EPOCHS}] | Train Loss: {train_loss:.4f} | Val Mean PR-AUC: {val_ap_mean:.4f}")

    if val_ap_mean > best_ap + MIN_DELTA_AP:
        best_ap = val_ap_mean
        patience_counter = 0
        torch.save(model.state_dict(), BEST_MODEL_PATH)
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch + 1}. Best Mean Val PR-AUC: {best_ap:.4f}")
            break

model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))

# ============================================================
# 11) EVALUATION ON TEST SET (COMBINED ANY-RISK)
# ============================================================

val_true, val_prob = get_probs_and_labels(val_loader)

# בחירת סף אופטימלי לכל מודל על בסיס ה-Validation
best_thr_risk = choose_threshold_for_custom_metric(val_true[:, 0], val_prob[:, 0])
best_thr_be = choose_threshold_for_custom_metric(val_true[:, 1], val_prob[:, 1])

test_true, test_prob = get_probs_and_labels(test_loader)

# יצירת מטרת האמת המאוחדת (סיכון pH או סיכון BE)
test_true_any_risk = (test_true[:, 0] == 1) | (test_true[:, 1] == 1)

# יצירת תחזית מאוחדת (המערכת צפצפה לאחת מהסיבות)
test_pred_risk = (test_prob[:, 0] >= best_thr_risk)
test_pred_be = (test_prob[:, 1] >= best_thr_be)
test_pred_any_risk = (test_pred_risk | test_pred_be).astype(int)

# חישוב מדדים על המטופל השלם
cm = confusion_matrix(test_true_any_risk, test_pred_any_risk, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()

acc = accuracy_score(test_true_any_risk, test_pred_any_risk)
rec = recall_score(test_true_any_risk, test_pred_any_risk, pos_label=1, zero_division=0)
prec = precision_score(test_true_any_risk, test_pred_any_risk, pos_label=1, zero_division=0)
f1 = f1_score(test_true_any_risk, test_pred_any_risk, pos_label=1, zero_division=0)

print("\n" + "=" * 55)
print("TEST RESULTS SUMMARY - OVERALL PATIENT RISK (Any Risk)")
print("=" * 55)
print(f"Thresholds -> pH Risk: {best_thr_risk:.2f} | BE Risk: {best_thr_be:.2f}")
print(f"Accuracy:  {acc:.3f}")
print(f"Recall:    {rec:.3f} (Identified {tp} out of {tp+fn} babies at risk)")
print(f"Precision: {prec:.3f} (True alarms: {tp} | False alarms: {fp})")
print(f"F1-Score:  {f1:.3f}")
print(f"False Negatives (Missed Danger): {fn}")
print(f"False Positives (False Alarms):  {fp}")
print("=" * 55)

# ============================================================
# 12) SAVE MODEL + STATS FOR WEB APP
# ============================================================

torch.save(model.state_dict(), FINAL_MODEL_PATH)

stats = {
    "model_type": "MultimodalRNN_MLP_MultiTask",
    "tabular_features": TABULAR_FEATURES,
    "augmentation": "None",
    
    "seq_train_mean": seq_train_mean.tolist(),
    "seq_train_std": seq_train_std.tolist(),
    "tab_train_mean": tab_train_mean.tolist(),
    "tab_train_std": tab_train_std.tolist(),

    "signal_channels": ["FHR", "UC", "FHR_mask"],
    "fs": FS,
    "seq_len": SEQ_LEN,
    "seq_in_ch": 3,
    "tab_in_features": len(TABULAR_FEATURES),

    "threshold_risk_pH": float(best_thr_risk),
    "threshold_risk_BE": float(best_thr_be),

    "test_metrics": {
        "OVERALL_ANY_RISK": {
            "accuracy": float(acc),
            "recall": float(rec),
            "precision": float(prec),
            "f1": float(f1),
            "fn": int(fn),
            "fp": int(fp),
            "tn": int(tn),
            "tp": int(tp)
        }
    },
}

with open(FINAL_STATS_PATH, "w", encoding="utf-8") as f:
    json.dump(stats, f, indent=2)

print("\nSaved artifacts for web app:")
print(f"Model state_dict: {FINAL_MODEL_PATH}")
print(f"Stats JSON:       {FINAL_STATS_PATH}")