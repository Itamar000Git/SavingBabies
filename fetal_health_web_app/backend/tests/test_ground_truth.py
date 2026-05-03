import os
import tempfile
import pandas as pd
import pytest
from core.ground_truth import lookup_ground_truth


def _make_ph_file(records: dict) -> str:
    df = pd.DataFrame([{"record_id": k, "pH": v} for k, v in records.items()])
    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w")
    df.to_csv(tmp.name, index=False)
    tmp.close()
    return tmp.name


def test_lookup_healthy_record():
    path = _make_ph_file({"1001": 7.25})
    result = lookup_ground_truth("1001", path)
    assert result["available"] is True
    assert result["actual_label"] == "Healthy"
    assert result["ph_value"] == 7.25
    os.unlink(path)


def test_lookup_risk_record():
    path = _make_ph_file({"1002": 7.05})
    result = lookup_ground_truth("1002", path)
    assert result["available"] is True
    assert result["actual_label"] == "Risk"
    os.unlink(path)


def test_lookup_unknown_record():
    path = _make_ph_file({"1001": 7.25})
    result = lookup_ground_truth("9999", path)
    assert result["available"] is False
    assert result["actual_label"] is None
    assert result["correctness"] == "Unknown"
    os.unlink(path)


def test_lookup_missing_file():
    result = lookup_ground_truth("1001", "/nonexistent/path.csv")
    assert result["available"] is False


def test_lookup_empty_record_id():
    path = _make_ph_file({"1001": 7.25})
    result = lookup_ground_truth("", path)
    assert result["available"] is False
    os.unlink(path)
