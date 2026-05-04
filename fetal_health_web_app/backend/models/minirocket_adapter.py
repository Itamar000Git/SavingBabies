from __future__ import annotations
import json
import os
import joblib
import numpy as np
import pandas as pd
from models.base_adapter import BaseModelAdapter

_WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "..", "weights")

# These constants MUST match the training script exactly.
SEQ_LEN = 1200       # 5 minutes @ 4 Hz
N_KERNELS = 4000
KERNEL_MIN = 7
KERNEL_MAX = 51
DILATION_MAX = 64
RANDOM_STATE = 42


class MiniRocketAdapter(BaseModelAdapter):
    name = "MiniROCKET"

    def __init__(self) -> None:
        self._clf = None
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._kernels: list[tuple] = []

    def load_model(self) -> None:
        """Load LogisticRegression weights and regenerate ROCKET kernels (deterministic, seed 42)."""
        model_path = os.path.join(_WEIGHTS_DIR, "minirocket_model.joblib")
        stats_path = os.path.join(_WEIGHTS_DIR, "minirocket_stats.json")

        # Always regenerate kernels (deterministic with fixed seed)
        rng = np.random.default_rng(RANDOM_STATE)
        self._kernels = []
        for _ in range(N_KERNELS):
            length = int(rng.integers(KERNEL_MIN, KERNEL_MAX + 1))
            weights = rng.normal(0, 1, size=length).astype(np.float32)
            bias = float(rng.uniform(-1, 1))
            dilation = int(2 ** rng.integers(0, int(np.log2(DILATION_MAX)) + 1))
            self._kernels.append((weights, bias, dilation))

        if not os.path.exists(model_path):
            print(f"[MiniRocketAdapter] WARNING: weights not found at {model_path}. "
                  "Run training/train_minirocket.py first.")
            return

        self._clf = joblib.load(model_path)

        with open(stats_path) as f:
            stats = json.load(f)
        self._mean = np.array(stats["train_mean"], dtype=np.float32)
        self._std = np.array(stats["train_std"], dtype=np.float32)
        print("[MiniRocketAdapter] Loaded model")

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

    @staticmethod
    def _conv1d_dilated(x: np.ndarray, w: np.ndarray, bias: float, dilation: int) -> np.ndarray:
        L, K = x.shape[0], w.shape[0]
        out_len = L - (K - 1) * dilation
        if out_len <= 0:
            return np.array([], dtype=np.float32)
        out = np.empty(out_len, dtype=np.float32)
        for i in range(out_len):
            out[i] = (x[i: i + dilation * K: dilation] * w).sum() + bias
        return out

    def _rocket_features(self, x_rec: np.ndarray) -> np.ndarray:
        """Extract max + PPV features for all kernels x channels."""
        C = x_rec.shape[1]
        feats: list[float] = []
        for (w, b, d) in self._kernels:
            for c in range(C):
                conv_out = self._conv1d_dilated(x_rec[:, c], w, b, d)
                if conv_out.size == 0:
                    feats.extend([0.0, 0.0])
                else:
                    feats.append(float(conv_out.max()))
                    feats.append(float((conv_out > 0).mean()))
        return np.array(feats, dtype=np.float32)

    def preprocess(self, df: pd.DataFrame, **context) -> np.ndarray:
        raw = self._build_raw_sequence(df)
        if self._mean is not None:
            return (raw - self._mean) / self._std
        return raw

    def predict(self, processed_data: np.ndarray) -> dict:
        if self._clf is None:
            return {"label": "Not available (model not trained)", "confidence": None, "placeholder": True}

        feats = self._rocket_features(processed_data).reshape(1, -1)
        prob = float(self._clf.predict_proba(feats)[0, 1])
        label = "Risk" if prob >= 0.5 else "Healthy"
        return {"label": label, "confidence": round(prob, 4)}

    def explain(self, processed_data: np.ndarray, prediction: dict, signal_features: dict) -> dict:
        if prediction.get("placeholder"):
            return {
                "important_parameters": [],
                "summary": "Placeholder prediction — real model weights are not loaded. Run training/train_minirocket.py first.",
                "missing_signal_warning": None,
            }

        fhr_mean = signal_features.get("fhr_mean") or 0.0
        fhr_std = signal_features.get("fhr_std") or 0.0
        missing_pct = signal_features.get("missing_signal_pct") or 0.0
        uc_present = signal_features.get("uc_available", False)

        confidence = prediction.get("confidence") or 0.5
        is_risk = "risk" in prediction["label"].lower() or "danger" in prediction["label"].lower()

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

        risk_factors = [p["name"] for p in params if p["impact"] in ("elevated", "critical")]
        reason = f"Elevated risk indicators: {', '.join(risk_factors)}." if risk_factors else "No significant risk indicators detected."

        summary = (
            f"MiniROCKET classified this recording as {'at risk' if is_risk else 'healthy'} "
            f"with {round(confidence * 100, 1)}% confidence. {reason}"
        )

        missing_signal_warning = None
        if missing_pct > 20:
            missing_signal_warning = "High missing signal may reduce prediction reliability."

        return {"important_parameters": params, "summary": summary, "missing_signal_warning": missing_signal_warning}
