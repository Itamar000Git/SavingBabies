"""
Ensemble of three pre-trained models for fetal BE-risk prediction:
  1. XGBoost (XGBoost_Classifier_logistic.py)
  2. Multimodal CNN (Risk_CNN_1D_Ogment.py)
  3. Multimodal BiGRU (BiGru.py)

Ensemble strategy: configurable weighted average of the three BE-risk
probabilities followed by a single threshold.

AI integration: after predictions are collected, the Anthropic Claude API
generates a natural-language clinical explanation for every raised alarm.
Set ANTHROPIC_API_KEY in your environment to enable AI explanations.
"""

import os
import re
import glob
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, average_precision_score, roc_auc_score,
    precision_score, recall_score, f1_score, accuracy_score,
)

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
CSV_DIR      = PROJECT_ROOT / "csv_output"
DATASET_DIR  = PROJECT_ROOT / "dataset"
INFO_DIR     = DATASET_DIR / "info"
PH_FILE      = DATASET_DIR / "ph_levels_with_be_risk_filled.csv"
WEIGHTS_DIR  = PROJECT_ROOT / "fetal_health_web_app" / "backend" / "weights"

XGB_MODEL_PATH   = WEIGHTS_DIR / "xgboost_model_riskBE.joblib"
XGB_STATS_PATH   = WEIGHTS_DIR / "xgboost_stats_riskBE.json"
CNN_MODEL_PATH   = WEIGHTS_DIR / "multimodal_cnn_multitask_model.pt"
CNN_STATS_PATH   = WEIGHTS_DIR / "multimodal_cnn_multitask_stats.json"
BIGRU_MODEL_PATH = PROJECT_ROOT / "best_multimodal_rnn_multitask.pt"

ENSEMBLE_STATS_PATH = WEIGHTS_DIR / "ensemble_stats_riskBE.json"

# ============================================================
# ENSEMBLE CONFIG
# ============================================================

RANDOM_STATE = 42
FS           = 4
SEQ_LEN      = 4800   # 20 min @ 4 Hz — must match CNN / BiGRU training

# Weights for the three models in the weighted-average ensemble.
# Tune these to favour the model(s) that performed best on your validation set.
W_XGB   = 0.35   # XGBoost
W_CNN   = 0.35   # Multimodal CNN
W_BIGRU = 0.30   # Multimodal BiGRU

ENSEMBLE_THRESHOLD = 0.20   # Final decision threshold on the weighted-average probability

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# TABULAR FEATURES (shared between CNN and BiGRU)
# ============================================================

TABULAR_FEATURES = [
    "Gest. weeks", "Sex", "Age", "Gravidity", "Parity",
    "Diabetes", "Pyrexia", "Meconium", "Induced",
]

# ============================================================
# XGBOOST FEATURE EXTRACTION (mirrors XGBoost_Classifier_logistic.py)
# ============================================================

METADATA_FEATURES = [
    ("meta_gestational_weeks", "Gest. weeks"),
    ("meta_mother_age",        "Age"),
    ("meta_gravidity",         "Gravidity"),
    ("meta_parity",            "Parity"),
    ("meta_diabetes",          "Diabetes"),
    ("meta_hypertension",      "Hypertension"),
    ("meta_preeclampsia",      "Preeclampsia"),
    ("meta_liq_praecox",       "Liq. praecox"),
    ("meta_pyrexia",           "Pyrexia"),
    ("meta_meconium",          "Meconium"),
    ("meta_presentation",      "Presentation"),
    ("meta_induced",           "Induced"),
    ("meta_i_stage",           "I.stage"),
    ("meta_no_progress",       "NoProgress"),
    ("meta_ck_kp",             "CK/KP"),
    ("meta_ii_stage",          "II.stage"),
]
METADATA_FIELD_NAMES = [n for n, _ in METADATA_FEATURES]

WINDOWS = {
    "last5":  5  * 60 * FS,
    "last10": 10 * 60 * FS,
    "last20": 20 * 60 * FS,
    "full":   None,
}

def _extract_number(text):
    m = re.search(r"-?\d+(\.\d+)?", str(text))
    return float(m.group(0)) if m else np.nan

def _find_info_file(record_id):
    for ext in [".hea", ".txt", ".info", ""]:
        p = INFO_DIR / f"{record_id}{ext}"
        if p.exists(): return p
    hits = list(INFO_DIR.glob(f"*{record_id}*"))
    return hits[0] if hits else None

def _read_hea_metadata(record_id):
    vals = {f: np.nan for f in METADATA_FIELD_NAMES}
    path = _find_info_file(record_id)
    if path:
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
            for field, label in METADATA_FEATURES:
                prefix = f"#{label}"
                for line in lines:
                    if line.strip().startswith(prefix):
                        vals[field] = _extract_number(line.strip()[len(prefix):])
                        break
        except Exception:
            pass
    result = vals.copy()
    for f in METADATA_FIELD_NAMES:
        result[f"{f}_missing"] = 1.0 if pd.isna(vals[f]) else 0.0
    return result

def _clean_fhr(fhr):
    fhr = np.asarray(fhr, dtype=np.float32)
    return pd.Series(fhr).mask((fhr <= 0) | np.isnan(fhr), np.nan).interpolate().fillna(140).values.astype(np.float32)

def _clean_uc(uc):
    uc = np.asarray(uc, dtype=np.float32)
    return pd.Series(uc).interpolate().fillna(0).values.astype(np.float32)

def _window(arr, length, pad):
    if length is None: return arr
    if len(arr) >= length: return arr[-length:]
    return np.pad(arr, (0, length - len(arr)), constant_values=pad)

def _stv(fhr): return float(np.mean(np.abs(np.diff(fhr)))) if len(fhr) >= 2 else 0.0

def _ltv(fhr):
    w = 120
    vals = [np.std(fhr[i:i+w]) for i in range(0, len(fhr)-w, w)]
    return float(np.mean(vals)) if vals else float(np.std(fhr))

def _slope(fhr):
    if len(fhr) < 2: return 0.0
    try: return float(np.polyfit(np.arange(len(fhr)), fhr, 1)[0])
    except: return 0.0

def _detect_accs(fhr, fs=4):
    if not len(fhr): return []
    bl, thr, ml = np.median(fhr), np.median(fhr)+15, int(15*fs)
    res, s = [], None
    for i, v in enumerate(fhr):
        if v >= thr:
            if s is None: s = i
        elif s is not None:
            if i-s >= ml: res.append((s, i))
            s = None
    return res

def _detect_decs(fhr, uc=None, fs=4):
    if not len(fhr): return []
    bl, thr, ml = np.median(fhr), np.median(fhr)-15, int(15*fs)
    res, s = [], None
    for i, v in enumerate(fhr):
        if v <= thr:
            if s is None: s = i
        elif s is not None:
            end = i
            if end-s >= ml:
                seg = fhr[s:end]; d = bl-np.min(seg); dur = (end-s)/fs
                nadir = s+np.argmin(seg); onset = (nadir-s)/fs
                if dur >= 120: typ = "prolonged"
                elif onset < 30: typ = "variable"
                else: typ = "unspecified"
                if uc is not None:
                    uc_seg = uc[s:end]
                    if len(uc_seg): typ = "late" if nadir > np.argmax(uc_seg)+s else "early"
                res.append({"duration": dur, "depth": d, "type": typ})
            s = None
    return res

def _window_features(fhr, uc, mask, prefix):
    feats = {}
    zero_names = ["fhr_mean","fhr_std","fhr_min","fhr_max","fhr_range","fhr_p05","fhr_p95",
                  "fhr_slope","stv","ltv","accelerations","decelerations","acceleration_ratio",
                  "deceleration_ratio","uc_mean","uc_std","uc_max","uc_activity_ratio",
                  "missing_signal_pct","valid_fhr_ratio","dec_count","dec_late","dec_variable",
                  "dec_prolonged","dec_avg_depth","dec_max_depth","dec_avg_duration",
                  "dec_time_ratio","acc_count","acc_rate","reactive","nonreactive"]
    if not len(fhr):
        return {f"{prefix}_{n}": 0.0 for n in zero_names}
    bl = np.median(fhr); n = max(len(fhr), 1)
    feats[f"{prefix}_fhr_mean"]  = float(np.mean(fhr))
    feats[f"{prefix}_fhr_std"]   = float(np.std(fhr))
    feats[f"{prefix}_fhr_min"]   = float(np.min(fhr))
    feats[f"{prefix}_fhr_max"]   = float(np.max(fhr))
    feats[f"{prefix}_fhr_range"] = float(np.max(fhr)-np.min(fhr))
    feats[f"{prefix}_fhr_p05"]   = float(np.percentile(fhr, 5))
    feats[f"{prefix}_fhr_p95"]   = float(np.percentile(fhr, 95))
    feats[f"{prefix}_fhr_slope"] = _slope(fhr)
    feats[f"{prefix}_stv"]       = _stv(fhr)
    feats[f"{prefix}_ltv"]       = _ltv(fhr)
    ac = int(np.sum(fhr > bl+15)); dc = int(np.sum(fhr < bl-15))
    feats[f"{prefix}_accelerations"]      = ac
    feats[f"{prefix}_decelerations"]      = dc
    feats[f"{prefix}_acceleration_ratio"] = ac/n
    feats[f"{prefix}_deceleration_ratio"] = dc/n
    feats[f"{prefix}_uc_mean"]            = float(np.mean(uc))
    feats[f"{prefix}_uc_std"]             = float(np.std(uc))
    feats[f"{prefix}_uc_max"]             = float(np.max(uc))
    feats[f"{prefix}_uc_activity_ratio"]  = float(np.mean(uc > 0))
    feats[f"{prefix}_missing_signal_pct"] = float(100*np.mean(mask == 0))
    feats[f"{prefix}_valid_fhr_ratio"]    = float(np.mean(mask == 1))
    decs = _detect_decs(fhr, uc)
    accs = _detect_accs(fhr)
    total_t = len(fhr)/FS
    feats[f"{prefix}_dec_count"]    = len(decs)
    feats[f"{prefix}_dec_late"]     = sum(d["type"]=="late" for d in decs)
    feats[f"{prefix}_dec_variable"] = sum(d["type"]=="variable" for d in decs)
    feats[f"{prefix}_dec_prolonged"]= sum(d["type"]=="prolonged" for d in decs)
    if decs:
        feats[f"{prefix}_dec_avg_depth"]   = float(np.mean([d["depth"] for d in decs]))
        feats[f"{prefix}_dec_max_depth"]   = float(np.max( [d["depth"] for d in decs]))
        feats[f"{prefix}_dec_avg_duration"]= float(np.mean([d["duration"] for d in decs]))
    else:
        feats[f"{prefix}_dec_avg_depth"] = feats[f"{prefix}_dec_max_depth"] = feats[f"{prefix}_dec_avg_duration"] = 0.0
    feats[f"{prefix}_dec_time_ratio"] = float(sum(d["duration"] for d in decs)/total_t)
    feats[f"{prefix}_acc_count"]     = len(accs)
    feats[f"{prefix}_acc_rate"]      = float(len(accs)/(total_t/60))
    feats[f"{prefix}_reactive"]      = int(len(accs) >= 2)
    feats[f"{prefix}_nonreactive"]   = int(len(accs) <  2)
    return feats

def extract_xgb_features(fhr_raw, uc_raw):
    mask = np.ones_like(fhr_raw, dtype=np.float32)
    mask[(fhr_raw <= 0) | np.isnan(fhr_raw)] = 0.0
    fhr = _clean_fhr(fhr_raw); uc = _clean_uc(uc_raw)
    feats = {}
    for prefix, wlen in WINDOWS.items():
        fw = _window(fhr, wlen, 140); uw = _window(uc,   wlen, 0)
        mw = _window(mask, wlen, 0)
        feats.update(_window_features(fw, uw, mw, prefix))
    return feats

# ============================================================
# PYTORCH ARCHITECTURES (must match training code exactly)
# ============================================================

class MultimodalCTGModel(nn.Module):
    def __init__(self, seq_in_ch=3, tab_in_features=len(TABULAR_FEATURES)):
        super().__init__()
        self.cnn_extractor = nn.Sequential(
            nn.Conv1d(seq_in_ch, 32, 15, padding=7), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 15, padding=7),         nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 11, padding=5),        nn.BatchNorm1d(128), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
        )
        self.mlp_extractor = nn.Sequential(
            nn.Linear(tab_in_features, 32), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128+16, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 2),
        )

    def forward(self, x_seq, x_tab):
        f = self.cnn_extractor(x_seq.transpose(1, 2)).squeeze(-1)
        return self.classifier(torch.cat([f, self.mlp_extractor(x_tab)], dim=1))


class MultimodalRNNModel(nn.Module):
    """
    Architecture reverse-engineered from the saved checkpoint:
    CNN front-end (3→16→32 channels) followed by a 1-layer bidirectional GRU.
    """
    def __init__(self, seq_in_ch=3, tab_in_features=len(TABULAR_FEATURES), hidden_dim=64):
        super().__init__()
        self.cnn_front = nn.Sequential(
            nn.Conv1d(seq_in_ch, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.rnn = nn.GRU(
            input_size=32, hidden_size=hidden_dim,
            num_layers=1, batch_first=True, bidirectional=True,
        )
        self.mlp_extractor = nn.Sequential(
            nn.Linear(tab_in_features, 32), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Linear(32, 16),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 16, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 2),
        )

    def forward(self, x_seq, x_tab):
        x = self.cnn_front(x_seq.transpose(1, 2)).transpose(1, 2)  # (B, T', 32)
        rnn_out, _ = self.rnn(x)
        f, _ = torch.max(rnn_out, dim=1)                             # (B, hidden*2)
        return self.classifier(torch.cat([f, self.mlp_extractor(x_tab)], dim=1))


class MultimodalDataset(Dataset):
    def __init__(self, X_seq, X_tab, y_arr):
        self.X_seq = torch.tensor(X_seq, dtype=torch.float32)
        self.X_tab = torch.tensor(X_tab, dtype=torch.float32)
        self.y     = torch.tensor(y_arr, dtype=torch.float32)

    def __len__(self): return len(self.y)

    def __getitem__(self, idx):
        return self.X_seq[idx], self.X_tab[idx], self.y[idx]

# ============================================================
# CLAUDE AI EXPLANATION
# ============================================================

def _build_clinical_prompt(record_id, meta, xgb_feats, probs, ensemble_prob,
                            ensemble_pred, true_label):
    """Build a compact clinical prompt for Claude."""
    gestational = meta.get("meta_gestational_weeks", "N/A")
    age         = meta.get("meta_mother_age", "N/A")
    diabetes    = "Yes" if meta.get("meta_diabetes", 0) == 1 else "No"
    hypertension= "Yes" if meta.get("meta_hypertension", 0) == 1 else "No"
    meconium    = "Yes" if meta.get("meta_meconium", 0) == 1 else "No"
    pyrexia     = "Yes" if meta.get("meta_pyrexia", 0) == 1 else "No"

    fhr_mean    = xgb_feats.get("full_fhr_mean", "N/A")
    fhr_std     = xgb_feats.get("full_fhr_std", "N/A")
    stv         = xgb_feats.get("full_stv", "N/A")
    dec_count   = xgb_feats.get("full_dec_count", "N/A")
    dec_late    = xgb_feats.get("full_dec_late", "N/A")
    dec_variable= xgb_feats.get("full_dec_variable", "N/A")
    acc_count   = xgb_feats.get("full_acc_count", "N/A")
    reactive    = "Yes" if xgb_feats.get("full_reactive", 0) == 1 else "No"

    alarm_txt   = "DANGER — BE Risk Detected" if ensemble_pred else "NORMAL"
    truth_txt   = "Actual label: DANGER" if true_label else "Actual label: NORMAL"

    return f"""You are a senior perinatal physician assistant. Analyze the following cardiotocography (CTG) data and provide a concise clinical interpretation.

=== PATIENT {record_id} ===
Gestational age: {gestational} weeks | Mother age: {age}
Risk factors: Diabetes={diabetes}, Hypertension={hypertension}, Meconium={meconium}, Pyrexia={pyrexia}

=== CTG SIGNAL STATISTICS (full recording) ===
FHR mean: {fhr_mean:.1f} bpm | FHR std: {fhr_std:.1f} | Short-term variability (STV): {stv:.2f}
Total decelerations: {dec_count} (Late: {dec_late}, Variable: {dec_variable})
Accelerations: {acc_count} | Reactive trace: {reactive}

=== MODEL PROBABILITY SCORES (riskBE) ===
XGBoost:  {probs['xgb']:.3f}
CNN:      {probs['cnn']:.3f}
BiGRU:    {probs['bigru']:.3f}
Ensemble (weighted avg): {ensemble_prob:.3f}

=== ENSEMBLE DECISION: {alarm_txt} ({truth_txt}) ===

Please provide:
1. A 2-sentence clinical interpretation of the CTG pattern and risk factors.
2. The most likely clinical reason the model flagged (or did not flag) this patient.
3. A confidence assessment (low / medium / high) and why.

Keep the response under 150 words and use clinical language."""


def get_ai_explanation(record_id, meta, xgb_feats, probs, ensemble_prob,
                       ensemble_pred, true_label):
    """Call Claude API to generate a clinical explanation. Returns None if unavailable."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return None
    try:
        import anthropic
        client  = anthropic.Anthropic(api_key=api_key)
        prompt  = _build_clinical_prompt(record_id, meta, xgb_feats, probs,
                                         ensemble_prob, ensemble_pred, true_label)
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text.strip()
    except Exception as e:
        return f"[AI explanation unavailable: {e}]"

# ============================================================
# LOAD LABELS
# ============================================================

ph_df = pd.read_csv(PH_FILE)
ph_df["record_id"] = ph_df["record_id"].astype(str)
risk_dict = {
    row["record_id"]: {"RISK": int(row["RISK"]), "riskBE": int(row["riskBE"])}
    for _, row in ph_df.iterrows()
}

# ============================================================
# LOAD DATA (shared between all three models)
# ============================================================

print("[1] Loading dataset...")

csv_files = glob.glob(str(CSV_DIR / "*.csv"))

# We need both JSON (for CNN/BiGRU tabular) and HEA (for XGBoost metadata)
records = []
for file in csv_files:
    record_id = os.path.splitext(os.path.basename(file))[0]
    if record_id not in risk_dict:
        continue
    json_path = INFO_DIR / f"{record_id}.json"
    if not json_path.exists():
        continue
    try:
        df = pd.read_csv(file)
        if "FHR" not in df.columns or "UC" not in df.columns:
            continue

        fhr_raw = df["FHR"].values.astype(np.float32)
        uc_raw  = df["UC"].values.astype(np.float32)

        # --- Sequence for CNN / BiGRU ---
        mask = np.ones_like(fhr_raw, dtype=np.float32)
        mask[(fhr_raw <= 0) | np.isnan(fhr_raw)] = 0.0
        fhr_c = pd.Series(fhr_raw).mask((fhr_raw <= 0)|np.isnan(fhr_raw), np.nan).interpolate().fillna(140).values.astype(np.float32)
        uc_c  = pd.Series(uc_raw).interpolate().fillna(0).values.astype(np.float32)
        n = len(fhr_c)
        if n >= SEQ_LEN:
            fhr_f, uc_f, mask_f = fhr_c[-SEQ_LEN:], uc_c[-SEQ_LEN:], mask[-SEQ_LEN:]
        else:
            pad = SEQ_LEN - n
            fhr_f  = np.pad(fhr_c,  (0, pad), mode="edge")
            uc_f   = np.pad(uc_c,   (0, pad), mode="edge")
            mask_f = np.concatenate([mask, np.zeros(pad, dtype=np.float32)])
        seq = np.stack([fhr_f, uc_f, mask_f], axis=1)  # (SEQ_LEN, 3)

        # --- Tabular for CNN / BiGRU ---
        with open(json_path) as f:
            info = json.load(f)
        tab = np.array([float(info.get(feat, 0.0)) for feat in TABULAR_FEATURES], dtype=np.float32)

        # --- Features for XGBoost ---
        xgb_feats = extract_xgb_features(fhr_raw, uc_raw)
        xgb_feats.update(_read_hea_metadata(record_id))

        records.append({
            "record_id":  record_id,
            "seq":        seq,
            "tab":        tab,
            "xgb_feats":  xgb_feats,
            "meta":       _read_hea_metadata(record_id),
            "y_risk":     risk_dict[record_id]["RISK"],
            "y_be":       risk_dict[record_id]["riskBE"],
        })
    except Exception as e:
        pass

print(f"    Loaded {len(records)} records with both CTG and JSON metadata.")
record_ids_all = [r["record_id"] for r in records]
y_be_all       = np.array([r["y_be"] for r in records])

print(f"    BE-risk distribution: Normal={int((y_be_all==0).sum())}, Danger={int((y_be_all==1).sum())}")

# ============================================================
# REPRODUCE SAME SPLIT AS TRAINING SCRIPTS
# ============================================================

idx_all  = np.arange(len(records))
idx_temp, idx_test = train_test_split(idx_all, test_size=0.15, stratify=y_be_all, random_state=RANDOM_STATE)
idx_train, idx_val = train_test_split(idx_temp, test_size=0.15, stratify=y_be_all[idx_temp], random_state=RANDOM_STATE)

# ============================================================
# NORMALIZATION (computed from training split, applied to all)
# ============================================================

X_seq_all  = np.array([r["seq"] for r in records], dtype=np.float32)
X_tab_all  = np.array([r["tab"] for r in records], dtype=np.float32)

X_seq_train = X_seq_all[idx_train]
X_tab_train = X_tab_all[idx_train]

seq_mean = np.nanmean(X_seq_train.reshape(-1, 3), axis=0)
seq_std  = np.nanstd( X_seq_train.reshape(-1, 3), axis=0) + 1e-8
tab_mean = np.nanmean(X_tab_train, axis=0)
tab_std  = np.nanstd( X_tab_train, axis=0) + 1e-8

def norm_seq(x): return np.nan_to_num((x - seq_mean) / seq_std)
def norm_tab(x): return np.nan_to_num((x - tab_mean) / tab_std)

X_seq_norm = norm_seq(X_seq_all)
X_tab_norm = norm_tab(X_tab_all)

X_seq_test = np.nan_to_num(X_seq_norm[idx_test])
X_tab_test = np.nan_to_num(X_tab_norm[idx_test])
y_test     = y_be_all[idx_test]
test_records = [records[i] for i in idx_test]

# ============================================================
# XGBOOST INFERENCE
# ============================================================

print("\n[2] Running XGBoost inference...")

with open(XGB_STATS_PATH) as f:
    xgb_stats = json.load(f)
xgb_feature_names = xgb_stats["feature_names"]
xgb_model = joblib.load(XGB_MODEL_PATH)

X_xgb_test = pd.DataFrame([r["xgb_feats"] for r in test_records])[xgb_feature_names]
prob_xgb   = xgb_model.predict_proba(X_xgb_test.values)[:, 1]

print(f"    XGBoost — done ({len(prob_xgb)} samples)")

# ============================================================
# CNN INFERENCE
# ============================================================

print("\n[3] Running CNN inference...")

cnn_model = MultimodalCTGModel().to(DEVICE)
cnn_model.load_state_dict(torch.load(CNN_MODEL_PATH, map_location=DEVICE))
cnn_model.eval()

with open(CNN_STATS_PATH) as f:
    cnn_stats = json.load(f)
cnn_threshold_be = cnn_stats["threshold_risk_BE"]

cnn_dataset = MultimodalDataset(X_seq_test, X_tab_test,
                                np.stack([y_test, y_test], axis=1))
cnn_loader  = DataLoader(cnn_dataset, batch_size=32, shuffle=False)

prob_cnn = []
with torch.no_grad():
    for x_seq, x_tab, _ in cnn_loader:
        logits = cnn_model(x_seq.to(DEVICE), x_tab.to(DEVICE))
        probs  = torch.sigmoid(logits).cpu().numpy()
        prob_cnn.extend(probs[:, 1].tolist())   # index 1 = riskBE
prob_cnn = np.array(prob_cnn)

print(f"    CNN — done ({len(prob_cnn)} samples, BE threshold from training: {cnn_threshold_be:.3f})")

# ============================================================
# BIGRU INFERENCE
# ============================================================

print("\n[4] Running BiGRU inference...")

bigru_model = MultimodalRNNModel().to(DEVICE)
bigru_model.load_state_dict(torch.load(BIGRU_MODEL_PATH, map_location=DEVICE))
bigru_model.eval()

bigru_loader = DataLoader(cnn_dataset, batch_size=32, shuffle=False)

prob_bigru = []
with torch.no_grad():
    for x_seq, x_tab, _ in bigru_loader:
        logits = bigru_model(x_seq.to(DEVICE), x_tab.to(DEVICE))
        probs  = torch.sigmoid(logits).cpu().numpy()
        prob_bigru.extend(probs[:, 1].tolist())
prob_bigru = np.array(prob_bigru)

print(f"    BiGRU — done ({len(prob_bigru)} samples)")

# ============================================================
# ENSEMBLE
# ============================================================

print("\n[5] Computing ensemble...")

# Normalize weights
total_w    = W_XGB + W_CNN + W_BIGRU
prob_ensemble = (
    (W_XGB / total_w)   * prob_xgb +
    (W_CNN / total_w)   * prob_cnn +
    (W_BIGRU / total_w) * prob_bigru
)
pred_ensemble = (prob_ensemble >= ENSEMBLE_THRESHOLD).astype(int)

# ============================================================
# EVALUATION
# ============================================================

print("\n[6] Evaluating on Test Set...")

def evaluate(y_true, y_prob, pred, title):
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0,1]).ravel()
    metrics = {
        "accuracy":  float(accuracy_score(y_true, pred)),
        "recall":    float(recall_score(y_true, pred, pos_label=1, zero_division=0)),
        "precision": float(precision_score(y_true, pred, pos_label=1, zero_division=0)),
        "f1":        float(f1_score(y_true, pred, pos_label=1, zero_division=0)),
        "pr_auc":    float(average_precision_score(y_true, y_prob)),
        "roc_auc":   float(roc_auc_score(y_true, y_prob)) if len(set(y_true)) > 1 else None,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }
    print("\n" + "="*57)
    print(title)
    print("="*57)
    print(f"Accuracy:  {metrics['accuracy']:.3f}")
    print(f"Recall:    {metrics['recall']:.3f}  ({tp} / {tp+fn} danger cases identified)")
    print(f"Precision: {metrics['precision']:.3f}  ({tp} true alarms | {fp} false alarms)")
    print(f"F1-Score:  {metrics['f1']:.3f}")
    print(f"PR-AUC:    {metrics['pr_auc']:.3f}" +
          (f"  |  ROC-AUC: {metrics['roc_auc']:.3f}" if metrics['roc_auc'] else ""))
    print(f"False Negatives (Missed): {fn}  |  False Positives (False Alarms): {fp}")
    print("="*57)
    return metrics

# Individual model metrics
pred_xgb_bin   = (prob_xgb   >= xgb_stats["threshold"]).astype(int)
pred_cnn_bin   = (prob_cnn   >= cnn_threshold_be).astype(int)
pred_bigru_bin = (prob_bigru >= cnn_threshold_be).astype(int)  # reuse CNN threshold

evaluate(y_test, prob_xgb,      pred_xgb_bin,   "XGBoost            (riskBE)")
evaluate(y_test, prob_cnn,      pred_cnn_bin,   "CNN                (riskBE)")
evaluate(y_test, prob_bigru,    pred_bigru_bin, "BiGRU              (riskBE)")
ensemble_metrics = evaluate(y_test, prob_ensemble, pred_ensemble, f"ENSEMBLE  (w={W_XGB:.2f}/{W_CNN:.2f}/{W_BIGRU:.2f}, thr={ENSEMBLE_THRESHOLD:.2f})")

# ============================================================
# AI EXPLANATIONS (Claude API)
# ============================================================

api_key = os.environ.get("ANTHROPIC_API_KEY", "")
if not api_key:
    print("\n[7] AI Explanations — set ANTHROPIC_API_KEY to enable Claude explanations.")
else:
    print(f"\n[7] Generating AI explanations for raised alarms ({int(pred_ensemble.sum())} cases)...")
    print("-" * 57)
    for i, rec in enumerate(test_records):
        if pred_ensemble[i] == 0:
            continue   # only explain alarms

        probs_dict = {
            "xgb":   float(prob_xgb[i]),
            "cnn":   float(prob_cnn[i]),
            "bigru": float(prob_bigru[i]),
        }
        explanation = get_ai_explanation(
            record_id     = rec["record_id"],
            meta          = rec["meta"],
            xgb_feats     = rec["xgb_feats"],
            probs         = probs_dict,
            ensemble_prob = float(prob_ensemble[i]),
            ensemble_pred = bool(pred_ensemble[i]),
            true_label    = bool(y_test[i]),
        )
        status = "TP" if (pred_ensemble[i] == 1 and y_test[i] == 1) else "FP"
        print(f"\nPatient {rec['record_id']} [{status}] — ensemble p={prob_ensemble[i]:.3f}")
        if explanation:
            print(explanation)
        print("-" * 57)

# ============================================================
# SAVE RESULTS
# ============================================================

print("\n[8] Saving ensemble stats...")

stats = {
    "ensemble_weights": {"xgboost": W_XGB, "cnn": W_CNN, "bigru": W_BIGRU},
    "ensemble_threshold": ENSEMBLE_THRESHOLD,
    "models": {
        "xgboost": str(XGB_MODEL_PATH),
        "cnn":     str(CNN_MODEL_PATH),
        "bigru":   str(BIGRU_MODEL_PATH),
    },
    "test_metrics": {
        "xgboost":  evaluate(y_test, prob_xgb,   pred_xgb_bin,   "XGBoost") if False else {"skipped": True},
        "ensemble": ensemble_metrics,
    },
}

with open(ENSEMBLE_STATS_PATH, "w") as f:
    json.dump(stats, f, indent=2)

print(f"    Saved to: {ENSEMBLE_STATS_PATH}")
print("\nEnsemble complete.")
