from __future__ import annotations
import io
import numpy as np
import pandas as pd
from core.schemas import MedicalMetadata

MAX_DURATION_MIN = 90.0


def parse_csv(content: bytes) -> pd.DataFrame:
    """Parse raw CSV bytes. Raises ValueError if FHR or UC columns are missing."""
    df = pd.read_csv(io.BytesIO(content))
    missing = {"FHR", "UC"} - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")
    return df


def extract_signal_features(df: pd.DataFrame) -> dict:
    """Compute signal statistics from the full recording. Single source of truth used by both metadata and explanation."""
    fhr = df["FHR"].values.astype(np.float32)
    uc = df["UC"].values.astype(np.float32)
    duration_min = None
    if "t_sec" in df.columns:
        t = df["t_sec"].values.astype(np.float64)
        duration_min = round(float(t[-1] - t[0]) / 60.0, 2)
    missing_mask = (fhr == 0) | np.isnan(fhr)
    missing_pct = round(float(missing_mask.mean()) * 100, 2)
    valid_fhr = fhr[~missing_mask]
    fhr_mean = round(float(valid_fhr.mean()), 2) if len(valid_fhr) > 0 else None
    fhr_std = round(float(valid_fhr.std()), 2) if len(valid_fhr) > 0 else None
    uc_available = bool(np.any(uc > 0))
    return {
        "recording_duration_min": duration_min,
        "missing_signal_pct": missing_pct,
        "fhr_mean": fhr_mean,
        "fhr_std": fhr_std,
        "uc_available": uc_available,
    }


def extract_medical_metadata(df: pd.DataFrame) -> MedicalMetadata:
    return MedicalMetadata(**extract_signal_features(df))


def validate_duration(medical: MedicalMetadata) -> None:
    """Raises ValueError if recording exceeds MAX_DURATION_MIN."""
    if medical.recording_duration_min is not None and medical.recording_duration_min > MAX_DURATION_MIN:
        raise ValueError(
            f"Recording too long: {medical.recording_duration_min:.1f} min (max {MAX_DURATION_MIN:.0f} min)"
        )
