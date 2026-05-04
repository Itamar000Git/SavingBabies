from __future__ import annotations
import json
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from models.base_adapter import BaseModelAdapter

_WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "..", "weights")
SEQ_LEN = 4800  # 20 minutes @ 4 Hz — must match training config


class _CNNBinaryCTG(nn.Module):
    """1D CNN for binary CTG classification. Architecture must match training."""

    def __init__(self, in_ch: int = 3):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)          # (B, L, C) -> (B, C, L)
        x = self.fe(x)                 # (B, 128, L')
        x = self.gap(x).squeeze(-1)    # (B, 128)
        return self.head(x).squeeze(-1)  # (B,)


class BinaryCNNAdapter(BaseModelAdapter):
    name = "BinaryCNN"

    def __init__(self) -> None:
        self._model: _CNNBinaryCTG | None = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._threshold: float = 0.5

    def load_model(self) -> None:
        """Load model weights from weights/binarycnn_model.pt.
        If the file does not exist, the adapter runs in placeholder mode."""
        model_path = os.path.join(_WEIGHTS_DIR, "binarycnn_model.pt")
        stats_path = os.path.join(_WEIGHTS_DIR, "binarycnn_stats.json")

        if not os.path.exists(model_path):
            print(f"[BinaryCNNAdapter] WARNING: weights not found at {model_path}. "
                  "Run training/train_binarycnn.py first.")
            return

        self._model = _CNNBinaryCTG(in_ch=3).to(self._device)
        self._model.load_state_dict(torch.load(model_path, map_location=self._device))
        self._model.eval()

        with open(stats_path) as f:
            stats = json.load(f)
        self._mean = np.array(stats["train_mean"], dtype=np.float32)
        self._std = np.array(stats["train_std"], dtype=np.float32)
        self._threshold = float(stats.get("threshold", 0.5))
        print(f"[BinaryCNNAdapter] Loaded model (threshold={self._threshold:.2f})")

    def _build_raw_sequence(self, df: pd.DataFrame) -> np.ndarray:
        """Build (SEQ_LEN, 3) array [FHR, UC, mask] from raw dataframe."""
        fhr_raw = df["FHR"].values.astype(np.float32)
        uc_raw = df["UC"].values.astype(np.float32)

        mask = np.ones_like(fhr_raw)
        mask[(fhr_raw == 0) | np.isnan(fhr_raw)] = 0.0

        fhr_clean = (
            pd.Series(fhr_raw)
            .replace(0, np.nan)
            .interpolate(method="linear")
            .fillna(140)
            .values.astype(np.float32)
        )
        uc_clean = (
            pd.Series(uc_raw)
            .interpolate(method="linear")
            .fillna(0)
            .values.astype(np.float32)
        )

        n = len(fhr_clean)
        if n >= SEQ_LEN:
            fhr_fix, uc_fix, mask_fix = fhr_clean[-SEQ_LEN:], uc_clean[-SEQ_LEN:], mask[-SEQ_LEN:]
        else:
            pad = SEQ_LEN - n
            fhr_fix = np.pad(fhr_clean, (0, pad), mode="edge")
            uc_fix = np.pad(uc_clean, (0, pad), mode="edge")
            mask_fix = np.concatenate([mask, np.zeros(pad, dtype=np.float32)])

        return np.stack([fhr_fix, uc_fix, mask_fix], axis=1)  # (SEQ_LEN, 3)

    def preprocess(self, df: pd.DataFrame) -> np.ndarray:
        raw = self._build_raw_sequence(df)
        if self._mean is not None:
            return (raw - self._mean) / self._std
        return raw

    def predict(self, processed_data: np.ndarray) -> dict:
        if self._model is None:
            return {"label": "Not available (model not trained)", "risk_score": None, "placeholder": True}

        tensor = (
            torch.tensor(processed_data, dtype=torch.float32)
            .unsqueeze(0)
            .to(self._device)
        )
        with torch.no_grad():
            logit = self._model(tensor)
            prob = float(torch.sigmoid(logit).item())

        margin = 0.10
        healthy_cutoff = round(self._threshold - margin, 4)
        danger_cutoff = round(self._threshold + margin, 4)

        if prob >= danger_cutoff:
            label = "Danger"
        elif prob <= healthy_cutoff:
            label = "Healthy"
        else:
            label = "Borderline"

        return {
            "label": label,
            "risk_score": round(prob, 4),
            "threshold": self._threshold,
            "healthy_cutoff": healthy_cutoff,
            "danger_cutoff": danger_cutoff,
        }

    def explain(self, processed_data: np.ndarray, prediction: dict, signal_features: dict) -> dict:
        if prediction.get("placeholder"):
            return {
                "important_parameters": [],
                "summary": "Placeholder prediction — real model weights are not loaded. Run training/train_binarycnn.py first.",
                "table_note": None,
                "missing_signal_warning": None,
            }

        fhr_mean = signal_features.get("fhr_mean") or 0.0
        fhr_std = signal_features.get("fhr_std") or 0.0
        missing_pct = signal_features.get("missing_signal_pct") or 0.0
        uc_present = signal_features.get("uc_available", False)

        params = [
            {
                "name": "FHR Mean",
                "value": f"{fhr_mean} bpm",
                "impact": "normal" if 110 <= fhr_mean <= 160 else "critical",
                "description": "Average fetal heart rate over the full recording. Normal range: 110–160 bpm. Values outside this range may indicate fetal distress or maternal fever.",
            },
            {
                "name": "FHR Variability (Std Dev)",
                "value": f"{fhr_std} bpm",
                "impact": "elevated" if fhr_std < 5 else "normal",
                "description": "Standard deviation of fetal heart rate. Low variability (< 5 bpm) may indicate fetal hypoxia, sedation, or sleep cycles.",
            },
            {
                "name": "Missing Signal",
                "value": f"{missing_pct}%",
                "impact": "critical" if missing_pct > 20 else "elevated" if missing_pct > 5 else "normal",
                "description": "Percentage of the recording with no valid FHR signal (FHR = 0, negative, or missing). High values reduce prediction reliability.",
            },
            {
                "name": "UC Activity",
                "value": "Present" if uc_present else "Absent",
                "impact": "normal",
                "description": "Uterine contraction activity. Contractions provide additional context for interpreting fetal heart rate decelerations.",
            },
        ]

        label = prediction.get("label", "")

        summary = (
            "The model estimates the probability that this recording belongs to the low-pH risk group. "
            "The CNN decision is based on the full CTG signal pattern, including fetal heart rate, "
            "uterine contractions, and missing-signal mask. "
            "The table shows summary indicators for interpretability, "
            "but these are not the only inputs used by the CNN."
        )
        if label == "Borderline":
            summary += (
                " Because the risk score is close to the decision threshold, "
                "this result is treated as borderline rather than a confident Healthy/Danger classification."
            )

        missing_signal_warning = None
        if missing_pct > 20:
            missing_signal_warning = "High missing signal may reduce prediction reliability."

        return {
            "important_parameters": params,
            "summary": summary,
            "table_note": "Summary indicators — not the only inputs used by the CNN.",
            "missing_signal_warning": missing_signal_warning,
        }
