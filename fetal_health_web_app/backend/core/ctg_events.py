from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional

FS = 4  # Hz — same as training


def _rolling_median_baseline(fhr: np.ndarray, window_sec: int = 600) -> np.ndarray:
    """10-minute rolling median baseline, forward-filled at edges."""
    window = window_sec * FS
    series = pd.Series(fhr.astype(np.float64))
    baseline = series.rolling(window, min_periods=1, center=True).median().values
    return baseline.astype(np.float64)


def _find_segments(mask: np.ndarray, min_samples: int, max_samples: Optional[int] = None) -> list[tuple[int, int]]:
    """Return list of (start, end) inclusive index pairs where mask is True for ≥min_samples."""
    segments: list[tuple[int, int]] = []
    n = len(mask)
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            length = j - i
            if length >= min_samples:
                if max_samples is None or length <= max_samples:
                    segments.append((i, j - 1))
            i = j
        else:
            i += 1
    return segments


def detect_fhr_events(
    fhr_raw: np.ndarray,
    t_sec: Optional[np.ndarray] = None,
    gestational_weeks: Optional[int] = None,
    fs: int = FS,
) -> dict:
    """
    Detect FHR accelerations and decelerations using approximate ACOG/NICHD definitions.

    Returns a dict with keys "accelerations" and "decelerations", each a list of event dicts.
    Each event dict keys: event_type, start_index, end_index, peak_or_nadir_index,
    start_min, end_min, peak_or_nadir_min, duration_sec, max_height_bpm,
    max_depth_bpm, subtype.
    """
    fhr = fhr_raw.astype(np.float64).copy()

    # Replace missing signal with NaN for baseline, then interpolate for detection
    missing = (fhr <= 0) | np.isnan(fhr)
    fhr[missing] = np.nan
    fhr_series = pd.Series(fhr).interpolate().fillna(140.0).values

    if t_sec is None:
        t_sec = np.arange(len(fhr_series), dtype=np.float64) / fs

    baseline = _rolling_median_baseline(fhr_series)

    # Acceleration amplitude threshold: 15 bpm (10 bpm before 32 weeks)
    accel_thresh = 10.0 if (gestational_weeks is not None and gestational_weeks < 32) else 15.0
    decel_thresh = 15.0

    min_dur_samples = int(15 * fs)       # 15 seconds
    max_accel_samples = int(120 * fs)    # 2 minutes (prolonged if > 2 min → not acceleration)
    prolonged_samples = int(120 * fs)    # 2 minutes threshold for prolonged decel

    accel_mask = (fhr_series - baseline) >= accel_thresh
    decel_mask = (baseline - fhr_series) >= decel_thresh

    accelerations = []
    for start, end in _find_segments(accel_mask, min_dur_samples, max_accel_samples):
        segment = fhr_series[start: end + 1]
        peak_idx = int(start + np.argmax(segment))
        height = float(fhr_series[peak_idx] - baseline[peak_idx])
        accelerations.append({
            "event_type": "acceleration",
            "start_index": int(start),
            "end_index": int(end),
            "peak_or_nadir_index": peak_idx,
            "start_min": round(float(t_sec[start]) / 60.0, 3),
            "end_min": round(float(t_sec[end]) / 60.0, 3),
            "peak_or_nadir_min": round(float(t_sec[peak_idx]) / 60.0, 3),
            "duration_sec": round(float(t_sec[end] - t_sec[start]), 1),
            "max_height_bpm": round(height, 1),
            "max_depth_bpm": None,
            "subtype": "acceleration",
        })

    decelerations = []
    for start, end in _find_segments(decel_mask, min_dur_samples):
        segment = fhr_series[start: end + 1]
        nadir_idx = int(start + np.argmin(segment))
        depth = float(baseline[nadir_idx] - fhr_series[nadir_idx])
        duration_sec = float(t_sec[end] - t_sec[start])

        # Classify subtype: prolonged (>= 2 min), variable (onset-to-nadir < 30s), gradual (>=30s)
        onset_to_nadir_sec = float(t_sec[nadir_idx] - t_sec[start])
        if (end - start + 1) >= prolonged_samples:
            subtype = "prolonged"
        elif onset_to_nadir_sec < 30.0:
            subtype = "variable"
        else:
            subtype = "gradual"

        decelerations.append({
            "event_type": "deceleration",
            "start_index": int(start),
            "end_index": int(end),
            "peak_or_nadir_index": nadir_idx,
            "start_min": round(float(t_sec[start]) / 60.0, 3),
            "end_min": round(float(t_sec[end]) / 60.0, 3),
            "peak_or_nadir_min": round(float(t_sec[nadir_idx]) / 60.0, 3),
            "duration_sec": round(duration_sec, 1),
            "max_height_bpm": None,
            "max_depth_bpm": round(depth, 1),
            "subtype": subtype,
        })

    return {"accelerations": accelerations, "decelerations": decelerations}


def prepare_signal_data(df: pd.DataFrame, max_points: int = 4800) -> dict:
    """
    Downsample signal arrays to at most max_points for chart rendering.

    Returns dict with keys: time_min, fhr (None where missing), uc.
    """
    fhr_raw = df["FHR"].values.astype(np.float32)
    uc_raw = df["UC"].values.astype(np.float32)

    n = len(fhr_raw)
    if "t_sec" in df.columns:
        t_sec = df["t_sec"].values.astype(np.float64)
    else:
        t_sec = np.arange(n, dtype=np.float64) / FS

    if n > max_points:
        step = n / max_points
        indices = np.round(np.arange(max_points) * step).astype(int)
        indices = np.clip(indices, 0, n - 1)
        fhr_raw = fhr_raw[indices]
        uc_raw = uc_raw[indices]
        t_sec = t_sec[indices]

    missing = (fhr_raw <= 0) | np.isnan(fhr_raw)
    fhr_out = [None if missing[i] else round(float(fhr_raw[i]), 1) for i in range(len(fhr_raw))]
    uc_out = [round(float(v), 1) for v in uc_raw]
    time_out = [round(float(v) / 60.0, 4) for v in t_sec]

    return {"time_min": time_out, "fhr": fhr_out, "uc": uc_out}
