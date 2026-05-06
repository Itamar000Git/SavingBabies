from __future__ import annotations
import json
import os
import re

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from models.base_adapter import BaseModelAdapter

_WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "..", "weights")

FS = 4
SEQ_LEN = 4800  # 20 min @ 4 Hz

W_XGB = 0.25
W_CNN = 0.60
W_BIGRU = 0.15
ENSEMBLE_THRESHOLD = 0.32
MARGIN = 0.10

TABULAR_FEATURES = [
    "Gest. weeks", "Sex", "Age", "Gravidity", "Parity",
    "Diabetes", "Pyrexia", "Meconium", "Induced",
]

WINDOWS = {
    "last5":  5  * 60 * FS,
    "last10": 10 * 60 * FS,
    "last20": 20 * 60 * FS,
    "full":   None,
}


# ──────────────────────────────────────────────────────────────
# PyTorch architectures (must match training checkpoints exactly)
# ──────────────────────────────────────────────────────────────

class _MultimodalCTGModel(nn.Module):
    def __init__(self, seq_in_ch: int = 3, tab_in_features: int = 9):
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
            nn.Linear(128 + 16, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 2),
        )

    def forward(self, x_seq: torch.Tensor, x_tab: torch.Tensor) -> torch.Tensor:
        f = self.cnn_extractor(x_seq.transpose(1, 2)).squeeze(-1)
        return self.classifier(torch.cat([f, self.mlp_extractor(x_tab)], dim=1))


class _MultimodalRNNModel(nn.Module):
    def __init__(self, seq_in_ch: int = 3, tab_in_features: int = 9, hidden_dim: int = 64):
        super().__init__()
        self.cnn_front = nn.Sequential(
            nn.Conv1d(seq_in_ch, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32), nn.ReLU(),
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

    def forward(self, x_seq: torch.Tensor, x_tab: torch.Tensor) -> torch.Tensor:
        x = self.cnn_front(x_seq.transpose(1, 2)).transpose(1, 2)
        rnn_out, _ = self.rnn(x)
        f, _ = torch.max(rnn_out, dim=1)
        return self.classifier(torch.cat([f, self.mlp_extractor(x_tab)], dim=1))


# ──────────────────────────────────────────────────────────────
# Signal helpers
# ──────────────────────────────────────────────────────────────

def _clean_fhr(fhr_raw: np.ndarray) -> np.ndarray:
    return (
        pd.Series(fhr_raw.astype(np.float32))
        .mask((pd.Series(fhr_raw.astype(np.float32)) <= 0) | pd.Series(np.isnan(fhr_raw)), np.nan)
        .interpolate()
        .fillna(140)
        .values.astype(np.float32)
    )


def _clean_uc(uc_raw: np.ndarray) -> np.ndarray:
    return (
        pd.Series(uc_raw.astype(np.float32)).interpolate().fillna(0).values.astype(np.float32)
    )


def _stv(fhr: np.ndarray) -> float:
    return float(np.mean(np.abs(np.diff(fhr)))) if len(fhr) >= 2 else 0.0


def _ltv(fhr: np.ndarray) -> float:
    w = 120
    segs = [np.std(fhr[i: i + w]) for i in range(0, len(fhr) - w, w)]
    return float(np.mean(segs)) if segs else float(np.std(fhr))


def _detect_decs(fhr: np.ndarray) -> list[dict]:
    if not len(fhr):
        return []
    bl = np.median(fhr)
    thr = bl - 15
    ml = int(15 * FS)
    res: list[dict] = []
    s = None
    for i, v in enumerate(fhr):
        if v <= thr:
            if s is None:
                s = i
        elif s is not None:
            if i - s >= ml:
                seg = fhr[s:i]
                d = bl - np.min(seg)
                dur = (i - s) / FS
                nadir = s + np.argmin(seg)
                onset = (nadir - s) / FS
                if dur >= 120:
                    typ = "prolonged"
                elif onset < 30:
                    typ = "variable"
                else:
                    typ = "late"
                res.append({"duration": dur, "depth": d, "type": typ})
            s = None
    return res


def _detect_accs(fhr: np.ndarray) -> list:
    if not len(fhr):
        return []
    bl = np.median(fhr)
    thr = bl + 15
    ml = int(15 * FS)
    res, s = [], None
    for i, v in enumerate(fhr):
        if v >= thr:
            if s is None:
                s = i
        elif s is not None:
            if i - s >= ml:
                res.append((s, i))
            s = None
    return res


def _window_arr(arr: np.ndarray, length: int | None, pad_val: float) -> np.ndarray:
    if length is None:
        return arr
    if len(arr) >= length:
        return arr[-length:]
    return np.pad(arr, (0, length - len(arr)), constant_values=pad_val)


def _window_features(prefix: str, fhr_raw: np.ndarray, uc_raw: np.ndarray, n: int | None) -> dict:
    fw = _window_arr(fhr_raw, n, 140.0)
    uw = _window_arr(uc_raw, n, 0.0)
    missing_mask = (fw <= 0) | np.isnan(fw)
    missing_pct = float(missing_mask.mean() * 100)
    valid_ratio = float(1.0 - missing_mask.mean())
    fhr = _clean_fhr(fw)
    uc = _clean_uc(uw)
    bl = float(np.median(fhr))
    n_pts = len(fhr)
    accs = _detect_accs(fhr)
    decs = _detect_decs(fhr)
    total_t = n_pts / FS
    t = np.arange(n_pts, dtype=np.float64)
    slope = float(np.polyfit(t, fhr, 1)[0]) if n_pts >= 2 else 0.0
    feats = {
        f"{prefix}_fhr_mean":           float(np.mean(fhr)),
        f"{prefix}_fhr_std":            float(np.std(fhr)),
        f"{prefix}_fhr_min":            float(np.min(fhr)),
        f"{prefix}_fhr_max":            float(np.max(fhr)),
        f"{prefix}_fhr_range":          float(np.max(fhr) - np.min(fhr)),
        f"{prefix}_fhr_p05":            float(np.percentile(fhr, 5)),
        f"{prefix}_fhr_p95":            float(np.percentile(fhr, 95)),
        f"{prefix}_fhr_slope":          slope,
        f"{prefix}_stv":                _stv(fhr),
        f"{prefix}_ltv":                _ltv(fhr),
        f"{prefix}_accelerations":      int(np.sum(fhr > bl + 15)),
        f"{prefix}_decelerations":      int(np.sum(fhr < bl - 15)),
        f"{prefix}_acceleration_ratio": int(np.sum(fhr > bl + 15)) / n_pts,
        f"{prefix}_deceleration_ratio": int(np.sum(fhr < bl - 15)) / n_pts,
        f"{prefix}_uc_mean":            float(np.mean(uc)),
        f"{prefix}_uc_std":             float(np.std(uc)),
        f"{prefix}_uc_max":             float(np.max(uc)),
        f"{prefix}_uc_activity_ratio":  float(np.mean(uc > 0)),
        f"{prefix}_missing_signal_pct": missing_pct,
        f"{prefix}_valid_fhr_ratio":    valid_ratio,
        f"{prefix}_dec_count":          len(decs),
        f"{prefix}_dec_late":           sum(d["type"] == "late" for d in decs),
        f"{prefix}_dec_variable":       sum(d["type"] == "variable" for d in decs),
        f"{prefix}_dec_prolonged":      sum(d["type"] == "prolonged" for d in decs),
        f"{prefix}_dec_avg_depth":      float(np.mean([d["depth"] for d in decs])) if decs else 0.0,
        f"{prefix}_dec_max_depth":      float(np.max([d["depth"] for d in decs])) if decs else 0.0,
        f"{prefix}_dec_avg_duration":   float(np.mean([d["duration"] for d in decs])) if decs else 0.0,
        f"{prefix}_dec_time_ratio":     float(sum(d["duration"] for d in decs) / total_t) if total_t else 0.0,
        f"{prefix}_acc_count":          len(accs),
        f"{prefix}_acc_rate":           float(len(accs) / (total_t / 60)) if total_t else 0.0,
        f"{prefix}_reactive":           int(len(accs) >= 2),
        f"{prefix}_nonreactive":        int(len(accs) < 2),
    }
    return feats


def _build_sequence(fhr_raw: np.ndarray, uc_raw: np.ndarray) -> np.ndarray:
    mask = np.ones_like(fhr_raw, dtype=np.float32)
    mask[(fhr_raw <= 0) | np.isnan(fhr_raw)] = 0.0
    fhr_c = _clean_fhr(fhr_raw)
    uc_c = _clean_uc(uc_raw)
    n = len(fhr_c)
    if n >= SEQ_LEN:
        fhr_f, uc_f, mask_f = fhr_c[-SEQ_LEN:], uc_c[-SEQ_LEN:], mask[-SEQ_LEN:]
    else:
        pad = SEQ_LEN - n
        fhr_f = np.pad(fhr_c, (0, pad), mode="edge")
        uc_f = np.pad(uc_c, (0, pad), mode="edge")
        mask_f = np.concatenate([mask, np.zeros(pad, dtype=np.float32)])
    return np.stack([fhr_f, uc_f, mask_f], axis=1)  # (SEQ_LEN, 3)


# ──────────────────────────────────────────────────────────────
# Rule-based explanation
# ──────────────────────────────────────────────────────────────

def _rule_based_explanation(
    feats: dict, baby, mother,
    probs: dict[str, float], ensemble_prob: float,
) -> str:
    concerns, reassuring, findings = [], [], []

    fhr_mean = feats.get("full_fhr_mean", 140.0)
    if fhr_mean < 110:
        concerns.append(f"fetal bradycardia (mean FHR {fhr_mean:.0f} bpm)")
    elif fhr_mean > 160:
        concerns.append(f"fetal tachycardia (mean FHR {fhr_mean:.0f} bpm)")
    else:
        reassuring.append(f"normal baseline FHR ({fhr_mean:.0f} bpm)")

    stv = feats.get("full_stv", 0.0)
    if stv < 1.0:
        concerns.append(f"markedly reduced STV ({stv:.2f})")
    elif stv < 2.0:
        findings.append(f"mildly reduced STV ({stv:.2f})")
    else:
        reassuring.append(f"adequate STV ({stv:.2f})")

    dec_late = int(feats.get("full_dec_late", 0))
    dec_var = int(feats.get("full_dec_variable", 0))
    dec_prol = int(feats.get("full_dec_prolonged", 0))
    dec_total = int(feats.get("full_dec_count", 0))
    if dec_late:
        concerns.append(f"{dec_late} late deceleration(s) — possible uteroplacental insufficiency")
    if dec_prol:
        concerns.append(f"{dec_prol} prolonged deceleration(s)")
    if dec_var:
        findings.append(f"{dec_var} variable deceleration(s)")
    if dec_total == 0:
        reassuring.append("no significant decelerations")

    reactive = int(feats.get("full_reactive", 0))
    acc_count = int(feats.get("full_acc_count", 0))
    if reactive:
        reassuring.append(f"reactive trace ({acc_count} accelerations)")
    else:
        concerns.append(f"non-reactive trace ({acc_count} acceleration(s))")

    risk_factors = []
    if getattr(mother, "diabetes", None):
        risk_factors.append("diabetes")
    if getattr(mother, "hypertension", None):
        risk_factors.append("hypertension")
    if getattr(mother, "preeclampsia", None):
        risk_factors.append("pre-eclampsia")

    model_vals = list(probs.values())
    agreement = sum(1 for p in model_vals if p >= 0.30) / len(model_vals)
    if agreement >= 0.67:
        confidence = "HIGH (majority of models agree)"
    elif agreement >= 0.33:
        confidence = "MEDIUM (partial model agreement)"
    else:
        confidence = "LOW (models disagree — borderline case)"

    lines = ["CTG Assessment:"]
    if concerns:
        lines.append("⚠ Concerning: " + "; ".join(concerns) + ".")
    if findings:
        lines.append("ℹ Additional: " + "; ".join(findings) + ".")
    if reassuring:
        lines.append("✓ Reassuring: " + "; ".join(reassuring) + ".")
    if risk_factors:
        lines.append(f"⚠ Maternal risk factors: {', '.join(risk_factors)}.")

    lines.append(
        f"\nModel scores  →  XGBoost: {probs['xgb']:.1%} | CNN: {probs['cnn']:.1%} | BiGRU: {probs['bigru']:.1%}"
    )
    lines.append(f"Ensemble probability: {ensemble_prob:.1%}   |   Confidence: {confidence}")
    return "\n".join(lines)


def _groq_explanation(feats: dict, baby, mother, probs: dict, ensemble_prob: float, label: str) -> str | None:
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        return None
    try:
        from groq import Groq
        gest = getattr(baby, "gestational_weeks", "N/A")
        age = getattr(mother, "mother_age", "N/A")
        diabetes = "Yes" if getattr(mother, "diabetes", False) else "No"
        hypertension = "Yes" if getattr(mother, "hypertension", False) else "No"
        fhr_mean = feats.get("full_fhr_mean", 0)
        fhr_std = feats.get("full_fhr_std", 0)
        stv = feats.get("full_stv", 0)
        dec_count = int(feats.get("full_dec_count", 0))
        dec_late = int(feats.get("full_dec_late", 0))
        dec_var = int(feats.get("full_dec_variable", 0))
        acc_count = int(feats.get("full_acc_count", 0))
        reactive = "Yes" if feats.get("full_reactive", 0) else "No"

        prompt = f"""You are a senior perinatal physician assistant. Analyze this CTG data and provide a concise clinical interpretation.

Gestational age: {gest} weeks | Mother age: {age} | Diabetes: {diabetes} | Hypertension: {hypertension}

CTG: FHR mean={fhr_mean:.1f} bpm | Std={fhr_std:.1f} | STV={stv:.2f}
Decelerations: {dec_count} total (Late={dec_late}, Variable={dec_var}) | Accelerations: {acc_count} | Reactive: {reactive}

Model scores — XGBoost: {probs['xgb']:.3f} | CNN: {probs['cnn']:.3f} | BiGRU: {probs['bigru']:.3f}
Ensemble: {ensemble_prob:.3f} → Decision: {label}

Provide: 1) A 2-sentence clinical interpretation. 2) Most likely reason for this classification. 3) Confidence assessment (low/medium/high).
Keep under 120 words. Use clinical language."""

        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=250,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[AI explanation unavailable: {e}]"


# ──────────────────────────────────────────────────────────────
# Adapter
# ──────────────────────────────────────────────────────────────

class EnsembleAdapter(BaseModelAdapter):
    name = "Ensemble (XGBoost + CNN + BiGRU)"

    def __init__(self) -> None:
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._xgb = None
        self._xgb_features: list[str] = []
        self._cnn: _MultimodalCTGModel | None = None
        self._bigru: _MultimodalRNNModel | None = None
        self._cnn_seq_mean: np.ndarray | None = None
        self._cnn_seq_std: np.ndarray | None = None
        self._cnn_tab_mean: np.ndarray | None = None
        self._cnn_tab_std: np.ndarray | None = None

    def load_model(self) -> None:
        # XGBoost
        xgb_path = os.path.join(_WEIGHTS_DIR, "xgboost_model_riskBE.joblib")
        xgb_stats = os.path.join(_WEIGHTS_DIR, "xgboost_stats_riskBE.json")
        if os.path.exists(xgb_path):
            self._xgb = joblib.load(xgb_path)
            with open(xgb_stats) as f:
                s = json.load(f)
            self._xgb_features = s["feature_names"]
            print(f"[EnsembleAdapter] XGBoost loaded ({len(self._xgb_features)} features)")
        else:
            print(f"[EnsembleAdapter] WARNING: XGBoost weights not found at {xgb_path}")

        # CNN
        cnn_path = os.path.join(_WEIGHTS_DIR, "multimodal_cnn_multitask_model.pt")
        cnn_stats = os.path.join(_WEIGHTS_DIR, "multimodal_cnn_multitask_stats.json")
        if os.path.exists(cnn_path):
            self._cnn = _MultimodalCTGModel(seq_in_ch=3, tab_in_features=9).to(self._device)
            self._cnn.load_state_dict(torch.load(cnn_path, map_location=self._device))
            self._cnn.eval()
            with open(cnn_stats) as f:
                s = json.load(f)
            self._cnn_seq_mean = np.array(s["seq_train_mean"], dtype=np.float32)
            self._cnn_seq_std  = np.array(s["seq_train_std"],  dtype=np.float32)
            self._cnn_tab_mean = np.array(s["tab_train_mean"], dtype=np.float32)
            self._cnn_tab_std  = np.array(s["tab_train_std"],  dtype=np.float32)
            print("[EnsembleAdapter] CNN loaded")
        else:
            print(f"[EnsembleAdapter] WARNING: CNN weights not found at {cnn_path}")

        # BiGRU
        bigru_path = os.path.join(_WEIGHTS_DIR, "bigru_model.pt")
        if os.path.exists(bigru_path):
            self._bigru = _MultimodalRNNModel(seq_in_ch=3, tab_in_features=9, hidden_dim=64).to(self._device)
            self._bigru.load_state_dict(torch.load(bigru_path, map_location=self._device))
            self._bigru.eval()
            print("[EnsembleAdapter] BiGRU loaded")
        else:
            print(f"[EnsembleAdapter] WARNING: BiGRU weights not found at {bigru_path}")

    def _build_tabular(self, baby, mother) -> np.ndarray:
        def _f(v):
            if v is None:
                return 0.0
            if isinstance(v, bool):
                return 1.0 if v else 0.0
            try:
                return float(v)
            except (TypeError, ValueError):
                return 0.0

        sex_map = {"Male": 1.0, "Female": 0.0, "M": 1.0, "F": 0.0}
        sex_raw = getattr(baby, "sex", None)
        sex = sex_map.get(str(sex_raw), 0.0) if sex_raw is not None else 0.0

        mapping = {
            "Gest. weeks": _f(getattr(baby,   "gestational_weeks", None)),
            "Sex":         sex,
            "Age":         _f(getattr(mother, "mother_age",         None)),
            "Gravidity":   _f(getattr(mother, "gravidity",          None)),
            "Parity":      _f(getattr(mother, "parity",             None)),
            "Diabetes":    _f(getattr(mother, "diabetes",           None)),
            "Pyrexia":     0.0,   # not in web app schema
            "Meconium":    0.0,   # not in web app schema
            "Induced":     0.0,   # not in web app schema
        }
        return np.array([mapping[f] for f in TABULAR_FEATURES], dtype=np.float32)

    def preprocess(self, df: pd.DataFrame, **context) -> dict:
        fhr_raw = df["FHR"].values.astype(np.float32)
        uc_raw  = df["UC"].values.astype(np.float32)
        baby   = context.get("baby")
        mother = context.get("mother")

        # Signal sequence (SEQ_LEN, 3)
        seq = _build_sequence(fhr_raw, uc_raw)

        # Tabular vector (9,)
        tab = self._build_tabular(baby, mother)

        # XGBoost feature vector
        xgb_feats: dict = {}
        for prefix, wlen in WINDOWS.items():
            xgb_feats.update(_window_features(prefix, fhr_raw, uc_raw, wlen))
        xgb_vec = np.array(
            [xgb_feats.get(name, np.nan) for name in self._xgb_features],
            dtype=np.float64,
        ) if self._xgb_features else np.array([], dtype=np.float64)

        return {
            "seq":      seq,
            "tab":      tab,
            "xgb_vec":  xgb_vec,
            "xgb_feats": xgb_feats,
            "baby":     baby,
            "mother":   mother,
        }

    def predict(self, processed_data: dict) -> dict:
        seq      = processed_data["seq"]       # (SEQ_LEN, 3)
        tab      = processed_data["tab"]       # (9,)
        xgb_vec  = processed_data["xgb_vec"]
        probs: dict[str, float] = {}

        # XGBoost
        if self._xgb is not None and len(xgb_vec):
            p = self._xgb.predict_proba(xgb_vec.reshape(1, -1))
            probs["xgb"] = float(p[0, 1])
        else:
            probs["xgb"] = 0.0

        # Normalise sequence and tabular for CNN / BiGRU
        seq_norm = seq.copy()
        tab_norm = tab.copy()
        if self._cnn_seq_mean is not None:
            seq_norm = (seq_norm - self._cnn_seq_mean) / (self._cnn_seq_std + 1e-8)
        if self._cnn_tab_mean is not None:
            tab_norm = (tab_norm - self._cnn_tab_mean) / (self._cnn_tab_std + 1e-8)

        t_seq = torch.tensor(seq_norm, dtype=torch.float32).unsqueeze(0).to(self._device)
        t_tab = torch.tensor(tab_norm, dtype=torch.float32).unsqueeze(0).to(self._device)

        # CNN
        if self._cnn is not None:
            with torch.no_grad():
                logits = self._cnn(t_seq, t_tab)
                probs["cnn"] = float(torch.softmax(logits, dim=1)[0, 1].item())
        else:
            probs["cnn"] = 0.0

        # BiGRU
        if self._bigru is not None:
            with torch.no_grad():
                logits = self._bigru(t_seq, t_tab)
                probs["bigru"] = float(torch.softmax(logits, dim=1)[0, 1].item())
        else:
            probs["bigru"] = 0.0

        # Weighted ensemble
        ensemble_prob = W_XGB * probs["xgb"] + W_CNN * probs["cnn"] + W_BIGRU * probs["bigru"]

        healthy_cutoff = round(ENSEMBLE_THRESHOLD - MARGIN, 4)
        danger_cutoff  = round(ENSEMBLE_THRESHOLD + MARGIN, 4)
        if ensemble_prob >= danger_cutoff:
            label = "Danger"
        elif ensemble_prob <= healthy_cutoff:
            label = "Healthy"
        else:
            label = "Borderline"

        return {
            "label":          label,
            "risk_score":     round(ensemble_prob, 4),
            "threshold":      ENSEMBLE_THRESHOLD,
            "healthy_cutoff": healthy_cutoff,
            "danger_cutoff":  danger_cutoff,
            "xgb_score":      round(probs["xgb"],   4),
            "cnn_score":      round(probs["cnn"],   4),
            "bigru_score":    round(probs["bigru"], 4),
        }

    def explain(self, processed_data: dict, prediction: dict, signal_features: dict) -> dict:
        feats  = processed_data.get("xgb_feats", {})
        baby   = processed_data.get("baby")
        mother = processed_data.get("mother")
        probs  = {
            "xgb":   prediction.get("xgb_score",   0.0),
            "cnn":   prediction.get("cnn_score",   0.0),
            "bigru": prediction.get("bigru_score", 0.0),
        }
        ensemble_prob = prediction.get("risk_score", 0.0)
        label = prediction.get("label", "")

        fhr_mean    = signal_features.get("fhr_mean")    or 0.0
        fhr_std     = signal_features.get("fhr_std")     or 0.0
        missing_pct = signal_features.get("missing_signal_pct") or 0.0
        uc_present  = signal_features.get("uc_available", False)

        params = [
            {
                "name": "Ensemble Risk Score",
                "value": f"{ensemble_prob:.1%}",
                "impact": "critical" if label == "Danger" else "elevated" if label == "Borderline" else "normal",
                "description": (
                    f"Weighted average of XGBoost ({probs['xgb']:.1%}), "
                    f"CNN ({probs['cnn']:.1%}), and BiGRU ({probs['bigru']:.1%}) risk probabilities."
                ),
            },
            {
                "name": "FHR Mean",
                "value": f"{fhr_mean} bpm",
                "impact": "normal" if 110 <= fhr_mean <= 160 else "critical",
                "description": "Average fetal heart rate. Normal range: 110–160 bpm.",
            },
            {
                "name": "FHR Variability (Std Dev)",
                "value": f"{fhr_std} bpm",
                "impact": "elevated" if fhr_std < 5 else "normal",
                "description": "Low variability (< 5 bpm) may indicate fetal hypoxia.",
            },
            {
                "name": "Missing Signal",
                "value": f"{missing_pct}%",
                "impact": "critical" if missing_pct > 20 else "elevated" if missing_pct > 5 else "normal",
                "description": "Percentage of the recording with no valid FHR signal.",
            },
            {
                "name": "UC Activity",
                "value": "Present" if uc_present else "Absent",
                "impact": "normal",
                "description": "Uterine contraction activity detected in the recording.",
            },
        ]

        rule_text = _rule_based_explanation(feats, baby, mother, probs, ensemble_prob)
        ai_text   = _groq_explanation(feats, baby, mother, probs, ensemble_prob, label)

        if ai_text:
            summary = f"{rule_text}\n\n— Groq AI (LLaMA 3.3 70B) —\n{ai_text}"
        else:
            summary = rule_text

        missing_warning = "High missing signal may reduce prediction reliability." if missing_pct > 20 else None

        return {
            "important_parameters": params,
            "summary": summary,
            "table_note": (
                f"Ensemble: XGBoost {probs['xgb']:.1%} × 0.25  +  "
                f"CNN {probs['cnn']:.1%} × 0.60  +  "
                f"BiGRU {probs['bigru']:.1%} × 0.15  =  {ensemble_prob:.1%}"
            ),
            "missing_signal_warning": missing_warning,
        }
