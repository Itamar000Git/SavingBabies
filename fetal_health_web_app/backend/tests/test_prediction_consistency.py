import io
import re

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient


def _csv_bytes(fhr_values, uc_values=None, step_sec=0.25):
    if uc_values is None:
        uc_values = [5.0] * len(fhr_values)

    t = np.arange(len(fhr_values), dtype=np.float32) * step_sec

    return pd.DataFrame({
        "t_sec": t,
        "FHR": fhr_values,
        "UC": uc_values,
    }).to_csv(index=False).encode()


def _first_number(value):
    """
    Extracts the first numeric value from either:
    - 118.68
    - "118.68 bpm"
    - "22.02%"
    """
    if value is None:
        return None

    if isinstance(value, (int, float)):
        return float(value)

    match = re.search(r"-?\d+(\.\d+)?", str(value))
    if not match:
        return None

    return float(match.group(0))


def _find_param(parameters, name_contains):
    for p in parameters:
        if name_contains.lower() in p["name"].lower():
            return p
    raise AssertionError(f"Could not find explanation parameter containing: {name_contains}")


@pytest.fixture(scope="module")
def client():
    from main import app
    return TestClient(app)


@pytest.mark.parametrize("model_name", ["binarycnn", "minirocket"])
def test_medical_metadata_and_signal_features_are_consistent(client, model_name):
    csv_data = _csv_bytes(
        fhr_values=[120.0, 121.0, 122.0, 0.0, 124.0],
        uc_values=[5.0, 5.5, 6.0, 5.0, 4.5],
    )

    r = client.post(
        "/predict",
        data={"model_name": model_name},
        files={"file": ("1001.csv", io.BytesIO(csv_data), "text/csv")},
    )

    assert r.status_code == 200
    body = r.json()

    assert "signal_features" in body

    medical = body["metadata"]["medical"]
    features = body["signal_features"]

    assert medical["fhr_mean"] == features["fhr_mean"]
    assert medical["fhr_std"] == features["fhr_std"]
    assert medical["missing_signal_pct"] == features["missing_signal_pct"]
    assert medical["uc_available"] == features["uc_available"]
    assert medical["recording_duration_min"] == features["recording_duration_min"]


@pytest.mark.parametrize("model_name", ["binarycnn", "minirocket"])
def test_explanation_uses_same_values_as_medical_metadata(client, model_name):
    csv_data = _csv_bytes(
        fhr_values=[118.0, 119.0, 120.0, 0.0, 122.0],
        uc_values=[2.0, 2.0, 2.0, 2.0, 2.0],
    )

    r = client.post(
        "/predict",
        data={"model_name": model_name},
        files={"file": ("1002.csv", io.BytesIO(csv_data), "text/csv")},
    )

    assert r.status_code == 200
    body = r.json()

    if body["prediction"].get("placeholder"):
        pytest.skip("Skipped in placeholder mode — explanation params are empty until weights are trained")

    medical = body["metadata"]["medical"]
    params = body["explanation"]["important_parameters"]

    fhr_mean_param = _find_param(params, "FHR Mean")
    missing_signal_param = _find_param(params, "Missing Signal")

    assert abs(_first_number(fhr_mean_param["value"]) - medical["fhr_mean"]) < 0.01
    assert abs(_first_number(missing_signal_param["value"]) - medical["missing_signal_pct"]) < 0.01


@pytest.mark.parametrize("model_name", ["binarycnn", "minirocket"])
def test_prediction_confidence_is_valid_when_present(client, model_name):
    csv_data = _csv_bytes([150.0] * 1200)

    r = client.post(
        "/predict",
        data={"model_name": model_name},
        files={"file": ("test.csv", io.BytesIO(csv_data), "text/csv")},
    )

    assert r.status_code == 200
    body = r.json()

    confidence = body["prediction"].get("confidence")

    if confidence is not None:
        assert 0.0 <= confidence <= 1.0


@pytest.mark.parametrize("model_name", ["binarycnn", "minirocket"])
def test_prediction_label_is_one_of_allowed_labels(client, model_name):
    csv_data = _csv_bytes([150.0] * 1200)

    r = client.post(
        "/predict",
        data={"model_name": model_name},
        files={"file": ("test.csv", io.BytesIO(csv_data), "text/csv")},
    )

    assert r.status_code == 200
    body = r.json()

    if body["prediction"].get("placeholder"):
        pytest.skip("Skipped in placeholder mode — placeholder label is expected until weights are trained")

    allowed_labels = {"Normal", "Danger", "Healthy", "Risk", "Suspicious", "Pathological"}
    assert body["prediction"]["label"] in allowed_labels


def test_models_endpoint_contains_required_models(client):
    r = client.get("/models")

    assert r.status_code == 200
    body = r.json()

    assert "models" in body
    assert "binarycnn" in body["models"]
    assert "minirocket" in body["models"]


def test_predict_missing_model_name_returns_validation_error(client):
    csv_data = _csv_bytes([150.0] * 100)

    r = client.post(
        "/predict",
        files={"file": ("test.csv", io.BytesIO(csv_data), "text/csv")},
    )

    assert r.status_code in (400, 422)


def test_predict_missing_file_returns_validation_error(client):
    r = client.post(
        "/predict",
        data={"model_name": "binarycnn"},
    )

    assert r.status_code in (400, 422)