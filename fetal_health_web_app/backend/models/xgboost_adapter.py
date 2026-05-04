from __future__ import annotations
import json
import os
import joblib
import numpy as np
import pandas as pd
from models.base_adapter import BaseModelAdapter

_WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "..", "weights")

SEQ_LEN = 1200   # 5 minutes @ 4 Hz — must match training config
MARGIN = 0.10


class XGBoostAdapter(BaseModelAdapter):
    name = "XGBoost"

    def __init__(self) -> None:
        self._model = None
        self._threshold: float = 0.5
        self._feature_names: list[str] = []

    def load_model(self) -> None:
        """Load model from weights/xgboost_model.joblib. Placeholder mode if missing."""
        model_path = os.path.join(_WEIGHTS_DIR, "xgboost_model.joblib")
        stats_path = os.path.join(_WEIGHTS_DIR, "xgboost_stats.json")

        if not os.path.exists(model_path):
            print(f"[XGBoostAdapter] WARNING: weights not found at {model_path}. "
                  "Run XGBoost_Classifier.py first.")
            return

        self._model = joblib.load(model_path)

        with open(stats_path) as f:
            stats = json.load(f)

        self._threshold = float(stats.get("threshold", 0.5))
        self._feature_names = stats["feature_names"]
        print(f"[XGBoostAdapter] Loaded model (threshold={self._threshold:.3f}, "
              f"{len(self._feature_names)} features)")

    # ------------------------------------------------------------------
    # Feature extraction — must exactly mirror XGBoost_Classifier.py
    # ------------------------------------------------------------------

    def _build_window(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return (fhr, uc) arrays of length SEQ_LEN, cleaned and windowed."""
        fhr_raw = df["FHR"].values.astype(np.float32)
        uc_raw = df["UC"].values.astype(np.float32)

        fhr_clean = (
            pd.Series(fhr_raw)
            .replace(0, np.nan)
            .mask(pd.Series(fhr_raw) < 0, np.nan)
            .interpolate()
            .fillna(140)
            .values.astype(np.float32)
        )
        uc_clean = (
            pd.Series(uc_raw).interpolate().fillna(0).values.astype(np.float32)
        )

        n = len(fhr_clean)
        if n >= SEQ_LEN:
            return fhr_clean[-SEQ_LEN:], uc_clean[-SEQ_LEN:]

        pad = SEQ_LEN - n
        return (
            np.pad(fhr_clean, (0, pad), constant_values=140.0),
            np.pad(uc_clean, (0, pad), constant_values=0.0),
        )

    @staticmethod
    def _ctg_features(fhr: np.ndarray, uc: np.ndarray) -> dict:
        baseline = float(np.median(fhr))
        t = np.arange(len(fhr))

        window = 120
        ltv_segs = [np.std(fhr[i: i + window]) for i in range(0, len(fhr) - window, window)]
        ltv = float(np.mean(ltv_segs)) if ltv_segs else float(np.std(fhr))

        return {
            "fhr_mean":      float(np.mean(fhr)),
            "fhr_std":       float(np.std(fhr)),
            "fhr_min":       float(np.min(fhr)),
            "fhr_max":       float(np.max(fhr)),
            "fhr_p05":       float(np.percentile(fhr, 5)),
            "fhr_p95":       float(np.percentile(fhr, 95)),
            "fhr_slope":     float(np.polyfit(t, fhr, 1)[0]),
            "stv":           float(np.mean(np.abs(np.diff(fhr)))),
            "ltv":           ltv,
            "accelerations": int(np.sum(fhr > baseline + 15)),
            "decelerations": int(np.sum(fhr < baseline - 15)),
            "uc_mean":       float(np.mean(uc)),
            "uc_std":        float(np.std(uc)),
            "uc_max":        float(np.max(uc)),
        }

    @staticmethod
    def _metadata_features(baby, mother) -> dict:
        def to_f(v):
            if v is None:
                return np.nan
            try:
                return float(v)
            except (TypeError, ValueError):
                return np.nan

        return {
            "gestational_weeks": to_f(getattr(baby,   "gestational_weeks", None)),
            "mother_age":        to_f(getattr(mother, "mother_age",        None)),
            "gravidity":         to_f(getattr(mother, "gravidity",         None)),
            "parity":            to_f(getattr(mother, "parity",            None)),
            "diabetes":          to_f(getattr(mother, "diabetes",          None)),
            "hypertension":      to_f(getattr(mother, "hypertension",      None)),
            "preeclampsia":      to_f(getattr(mother, "preeclampsia",      None)),
            # Fields not yet parsed from .hea — XGBoost handles NaN natively
            "liq_praecox":   np.nan,
            "pyrexia":        np.nan,
            "meconium":       np.nan,
            "presentation":   np.nan,
            "induced":        np.nan,
        }

    def preprocess(self, df: pd.DataFrame, **context) -> np.ndarray:
        """Build the ordered feature vector expected by the trained XGBoost model."""
        fhr, uc = self._build_window(df)
        combined = {
            **self._ctg_features(fhr, uc),
            **self._metadata_features(context.get("baby"), context.get("mother")),
        }

        # Order must match training — fill any unknown name with NaN
        vec = np.array(
            [combined.get(name, np.nan) for name in self._feature_names],
            dtype=np.float64,
        )
        return vec

    def predict(self, processed_data: np.ndarray) -> dict:
        if self._model is None:
            return {"label": "Not available (model not trained)", "risk_score": None, "placeholder": True}

        prob = float(self._model.predict_proba(processed_data.reshape(1, -1))[:, 1])

        healthy_cutoff = round(self._threshold - MARGIN, 4)
        danger_cutoff = round(self._threshold + MARGIN, 4)

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
                "summary": (
                    "Placeholder prediction — XGBoost weights not found. "
                    "Run XGBoost_Classifier.py first."
                ),
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
                "description": (
                    "Average fetal heart rate over the full recording. "
                    "Normal range: 110–160 bpm."
                ),
            },
            {
                "name": "FHR Variability (Std Dev)",
                "value": f"{fhr_std} bpm",
                "impact": "elevated" if fhr_std < 5 else "normal",
                "description": (
                    "Standard deviation of fetal heart rate. "
                    "Low variability (< 5 bpm) may indicate fetal hypoxia."
                ),
            },
            {
                "name": "Missing Signal",
                "value": f"{missing_pct}%",
                "impact": "critical" if missing_pct > 20 else "elevated" if missing_pct > 5 else "normal",
                "description": (
                    "Percentage of the recording with no valid FHR signal. "
                    "High values reduce prediction reliability."
                ),
            },
            {
                "name": "UC Activity",
                "value": "Present" if uc_present else "Absent",
                "impact": "normal",
                "description": "Uterine contraction activity.",
            },
        ]

        label = prediction.get("label", "")
        summary = (
            "XGBoost estimates the probability of low-pH fetal risk using handcrafted CTG features "
            "(heart rate statistics, variability, accelerations, decelerations, UC) "
            "and available clinical metadata (gestational age, maternal conditions). "
            "The table shows key signal summary indicators. "
            "Additional features computed from the signal (percentiles, slope, short- and long-term "
            "variability) and metadata fields are used internally but not listed here."
        )
        if label == "Borderline":
            summary += (
                " Because the risk score is close to the decision threshold, "
                "this result is treated as borderline rather than a confident Healthy/Danger classification."
            )

        missing_signal_warning = (
            "High missing signal may reduce prediction reliability." if missing_pct > 20 else None
        )

        return {
            "important_parameters": params,
            "summary": summary,
            "table_note": "Summary indicators — additional CTG features and clinical metadata are also used.",
            "missing_signal_warning": missing_signal_warning,
        }
