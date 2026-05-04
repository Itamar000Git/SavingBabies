from __future__ import annotations
import json
import os
import joblib
import numpy as np
import pandas as pd
from models.base_adapter import BaseModelAdapter

_WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "..", "weights")

MARGIN = 0.10
FS = 4  # Hz

# Window sizes in samples — must match training
WINDOWS = {
    "last5":  1200,   # 5 min
    "last10": 2400,   # 10 min
    "last20": 4800,   # 20 min
    "full":   None,   # entire recording
}


def _clean_fhr(fhr_raw: np.ndarray) -> np.ndarray:
    return (
        pd.Series(fhr_raw.astype(np.float32))
        .replace(0, np.nan)
        .mask(pd.Series(fhr_raw.astype(np.float32)) < 0, np.nan)
        .interpolate()
        .fillna(140)
        .values.astype(np.float32)
    )


def _clean_uc(uc_raw: np.ndarray) -> np.ndarray:
    return (
        pd.Series(uc_raw.astype(np.float32))
        .interpolate()
        .fillna(0)
        .values.astype(np.float32)
    )


def _window_features(prefix: str, fhr_raw: np.ndarray, uc_raw: np.ndarray, n: int | None) -> dict:
    """
    Compute all 20 per-window features from raw (uncleaned) arrays.
    n=None means use the full array.  n>len → pad with edge values.
    Feature names must match training: {prefix}_{feature}.
    """
    total = len(fhr_raw)

    if n is None:
        raw_fhr_w = fhr_raw
        raw_uc_w = uc_raw
    elif total >= n:
        raw_fhr_w = fhr_raw[-n:]
        raw_uc_w = uc_raw[-n:]
    else:
        pad = n - total
        raw_fhr_w = np.pad(fhr_raw, (0, pad), mode="edge")
        raw_uc_w = np.pad(uc_raw, (0, pad), mode="edge")

    # Missing-signal mask on RAW values (before interpolation)
    missing_mask = (raw_fhr_w <= 0) | np.isnan(raw_fhr_w)
    missing_pct = float(missing_mask.mean() * 100)
    valid_ratio = float(1.0 - missing_mask.mean())

    # Cleaned arrays for feature computation
    fhr = _clean_fhr(raw_fhr_w)
    uc = _clean_uc(raw_uc_w)

    baseline = float(np.median(fhr))
    t = np.arange(len(fhr), dtype=np.float64)

    # Long-term variability: mean std across 30-second segments
    seg = FS * 30  # 120 samples @ 4 Hz
    ltv_segs = [np.std(fhr[i: i + seg]) for i in range(0, len(fhr) - seg, seg)]
    ltv = float(np.mean(ltv_segs)) if ltv_segs else float(np.std(fhr))

    accel = int(np.sum(fhr > baseline + 15))
    decel = int(np.sum(fhr < baseline - 15))
    n_pts = len(fhr)

    return {
        f"{prefix}_fhr_mean":           float(np.mean(fhr)),
        f"{prefix}_fhr_std":            float(np.std(fhr)),
        f"{prefix}_fhr_min":            float(np.min(fhr)),
        f"{prefix}_fhr_max":            float(np.max(fhr)),
        f"{prefix}_fhr_range":          float(np.max(fhr) - np.min(fhr)),
        f"{prefix}_fhr_p05":            float(np.percentile(fhr, 5)),
        f"{prefix}_fhr_p95":            float(np.percentile(fhr, 95)),
        f"{prefix}_fhr_slope":          float(np.polyfit(t, fhr, 1)[0]),
        f"{prefix}_stv":                float(np.mean(np.abs(np.diff(fhr)))),
        f"{prefix}_ltv":                ltv,
        f"{prefix}_accelerations":      accel,
        f"{prefix}_decelerations":      decel,
        f"{prefix}_acceleration_ratio": accel / n_pts,
        f"{prefix}_deceleration_ratio": decel / n_pts,
        f"{prefix}_uc_mean":            float(np.mean(uc)),
        f"{prefix}_uc_std":             float(np.std(uc)),
        f"{prefix}_uc_max":             float(np.max(uc)),
        f"{prefix}_uc_activity_ratio":  float(np.mean(uc > 0)),
        f"{prefix}_missing_signal_pct": missing_pct,
        f"{prefix}_valid_fhr_ratio":    valid_ratio,
    }


class XGBoostAdapter(BaseModelAdapter):
    name = "XGBoost"

    def __init__(self) -> None:
        self._model = None
        self._threshold: float = 0.5
        self._feature_names: list[str] = []

    def load_model(self) -> None:
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
            # Not yet parsed from .hea — XGBoost handles NaN natively
            "liq_praecox":   np.nan,
            "pyrexia":        np.nan,
            "meconium":       np.nan,
            "presentation":   np.nan,
            "induced":        np.nan,
        }

    def preprocess(self, df: pd.DataFrame, **context) -> np.ndarray:
        fhr_raw = df["FHR"].values.astype(np.float32)
        uc_raw = df["UC"].values.astype(np.float32)

        combined: dict = {}
        for prefix, n_samples in WINDOWS.items():
            combined.update(_window_features(prefix, fhr_raw, uc_raw, n_samples))

        combined.update(self._metadata_features(context.get("baby"), context.get("mother")))

        # Order must exactly match training; unknown names → NaN (safe for XGBoost)
        vec = np.array(
            [combined.get(name, np.nan) for name in self._feature_names],
            dtype=np.float64,
        )
        return vec

    def predict(self, processed_data: np.ndarray) -> dict:
        if self._model is None:
            return {"label": "Not available (model not trained)", "risk_score": None, "placeholder": True}

        proba = self._model.predict_proba(processed_data.reshape(1, -1))
        prob = float(proba[0, 1])

        margin = MARGIN
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
            "XGBoost estimates the probability of low-pH fetal risk. "
            "It uses 80 CTG signal features computed across four time windows "
            "(last 5, 10, 20 minutes and the full recording), covering heart rate statistics, "
            "variability, accelerations, decelerations, and uterine contractions, "
            "plus available clinical metadata. "
            "The table shows key signal summary indicators from the full recording."
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
            "table_note": "Summary indicators — XGBoost also uses 80 multi-window CTG features internally.",
            "missing_signal_warning": missing_signal_warning,
        }
