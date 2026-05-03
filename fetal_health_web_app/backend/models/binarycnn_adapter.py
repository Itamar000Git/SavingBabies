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
        self._last_raw_seq: np.ndarray | None = None  # set during preprocess for explain()

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
        self._last_raw_seq = raw.copy()  # store for explain()
        if self._mean is not None:
            return (raw - self._mean) / self._std
        return raw

    def predict(self, processed_data: np.ndarray) -> dict:
        if self._model is None:
            return {"label": "Not available (model not trained)", "confidence": None}

        tensor = (
            torch.tensor(processed_data, dtype=torch.float32)
            .unsqueeze(0)
            .to(self._device)
        )
        with torch.no_grad():
            logit = self._model(tensor)
            prob = float(torch.sigmoid(logit).item())

        label = "Risk" if prob >= self._threshold else "Healthy"
        return {"label": label, "confidence": round(prob, 4)}

    def explain(self, processed_data: np.ndarray, prediction: dict) -> dict:
        raw = self._last_raw_seq if self._last_raw_seq is not None else processed_data
        fhr_col = raw[:, 0]
        mask_col = raw[:, 2]
        uc_col = raw[:, 1]

        missing_pct = round(float((mask_col < 0.5).mean()) * 100, 1)

        valid_fhr = fhr_col[mask_col > 0.5]
        fhr_mean = round(float(valid_fhr.mean()), 1) if len(valid_fhr) > 0 else 0.0
        fhr_std = round(float(valid_fhr.std()), 1) if len(valid_fhr) > 0 else 0.0
        uc_present = bool(np.any(uc_col > 0))

        confidence = prediction.get("confidence") or 0.5
        is_risk = "risk" in prediction["label"].lower() or "danger" in prediction["label"].lower()

        params = [
            {
                "name": "FHR Mean",
                "value": f"{fhr_mean} bpm",
                "impact": "normal" if 110 <= fhr_mean <= 160 else "critical",
            },
            {
                "name": "FHR Variability",
                "value": f"{fhr_std} bpm",
                "impact": "elevated" if fhr_std < 5 else "normal",
            },
            {
                "name": "Missing Signal",
                "value": f"{missing_pct}%",
                "impact": "critical" if missing_pct > 20 else "elevated" if missing_pct > 5 else "normal",
            },
            {
                "name": "UC Activity",
                "value": "Present" if uc_present else "Absent",
                "impact": "normal",
            },
        ]

        risk_factors = [p["name"] for p in params if p["impact"] in ("elevated", "critical")]
        if risk_factors:
            reason = f"Elevated risk indicators: {', '.join(risk_factors)}."
        else:
            reason = "No significant risk indicators detected."

        summary = (
            f"BinaryCNN classified this recording as {'at risk' if is_risk else 'healthy'} "
            f"with {round(confidence * 100, 1)}% confidence. {reason}"
        )

        return {"important_parameters": params, "summary": summary}
