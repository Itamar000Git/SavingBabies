from __future__ import annotations
import os
import pandas as pd

PH_DANGER_THRESHOLD = 7.10


def lookup_ground_truth(record_id: str, ph_file_path: str) -> dict:
    """
    Look up ground truth label from ph_levels.csv.
    Returns a dict compatible with GroundTruth schema.
    """
    if not record_id:
        return {"actual_label": None, "ph_value": None, "correctness": "Unknown", "available": False}

    try:
        ph_df = pd.read_csv(ph_file_path)
        ph_df["record_id"] = ph_df["record_id"].astype(str)
        row = ph_df[ph_df["record_id"] == record_id]
        if len(row) == 0:
            return {"actual_label": None, "ph_value": None, "correctness": "Unknown", "available": False}
        ph_value = float(row.iloc[0]["pH"])
        actual_label = "Risk" if ph_value < PH_DANGER_THRESHOLD else "Healthy"
        return {
            "actual_label": actual_label,
            "ph_value": round(ph_value, 3),
            "correctness": None,  # filled in by caller after comparing to prediction
            "available": True,
        }
    except Exception:
        return {"actual_label": None, "ph_value": None, "correctness": "Unknown", "available": False}
