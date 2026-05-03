import io
import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

# Build a minimal 5-minute CSV at 4 Hz
N = 1200
t = np.arange(N, dtype=np.float32) * 0.25
fhr = np.full(N, 150.0, dtype=np.float32)
uc = np.full(N, 5.0, dtype=np.float32)
_SAMPLE_CSV = pd.DataFrame({"t_sec": t, "FHR": fhr, "UC": uc}).to_csv(index=False).encode()

# Build a >90-minute CSV at 4 Hz (21,840 rows) — used to test the too-long rejection path
_N_LONG = int(91 * 60 * 4)
_LONG_CSV = pd.DataFrame({
    "t_sec": np.arange(_N_LONG, dtype=np.float32) * 0.25,
    "FHR": np.full(_N_LONG, 150.0),
    "UC": np.full(_N_LONG, 5.0),
}).to_csv(index=False).encode()


@pytest.fixture(scope="module")
def client():
    from main import app
    return TestClient(app)


def test_get_models_returns_list(client):
    r = client.get("/models")
    assert r.status_code == 200
    data = r.json()
    assert "models" in data
    assert isinstance(data["models"], list)
    assert len(data["models"]) >= 1


def test_predict_binarycnn_returns_structure(client):
    r = client.post(
        "/predict",
        data={"model_name": "binarycnn"},
        files={"file": ("test.csv", io.BytesIO(_SAMPLE_CSV), "text/csv")},
    )
    assert r.status_code == 200
    body = r.json()
    assert "prediction" in body
    assert "label" in body["prediction"]
    assert "metadata" in body
    assert "baby" in body["metadata"]
    assert "mother" in body["metadata"]
    assert "medical" in body["metadata"]
    assert "explanation" in body
    assert "important_parameters" in body["explanation"]
    assert "summary" in body["explanation"]


def test_predict_minirocket_returns_structure(client):
    r = client.post(
        "/predict",
        data={"model_name": "minirocket"},
        files={"file": ("test.csv", io.BytesIO(_SAMPLE_CSV), "text/csv")},
    )
    assert r.status_code == 200
    body = r.json()
    assert "prediction" in body
    assert "label" in body["prediction"]
    assert "metadata" in body
    assert "baby" in body["metadata"]
    assert "mother" in body["metadata"]
    assert "medical" in body["metadata"]
    assert "explanation" in body
    assert "important_parameters" in body["explanation"]
    assert "summary" in body["explanation"]
    assert body["model_name"] == "MiniROCKET"


def test_predict_unknown_model_returns_404(client):
    r = client.post(
        "/predict",
        data={"model_name": "nonexistent"},
        files={"file": ("test.csv", io.BytesIO(_SAMPLE_CSV), "text/csv")},
    )
    assert r.status_code == 404


def test_predict_bad_csv_returns_400(client):
    bad = b"col1,col2\n1,2\n"
    r = client.post(
        "/predict",
        data={"model_name": "binarycnn"},
        files={"file": ("bad.csv", io.BytesIO(bad), "text/csv")},
    )
    assert r.status_code == 400


def test_predict_too_long_returns_400(client):
    r = client.post(
        "/predict",
        data={"model_name": "binarycnn"},
        files={"file": ("long.csv", io.BytesIO(_LONG_CSV), "text/csv")},
    )
    assert r.status_code == 400


def test_signal_features_in_response(client):
    """Response must include signal_features field with all 5 stats."""
    r = client.post(
        "/predict",
        data={"model_name": "binarycnn"},
        files={"file": ("test.csv", io.BytesIO(_SAMPLE_CSV), "text/csv")},
    )
    assert r.status_code == 200
    body = r.json()
    assert "signal_features" in body
    sf = body["signal_features"]
    assert "fhr_mean" in sf
    assert "fhr_std" in sf
    assert "missing_signal_pct" in sf
    assert "uc_available" in sf
    assert "recording_duration_min" in sf


def test_medical_and_signal_features_consistent(client):
    """metadata.medical and signal_features must be identical — single source of truth."""
    r = client.post(
        "/predict",
        data={"model_name": "binarycnn"},
        files={"file": ("test.csv", io.BytesIO(_SAMPLE_CSV), "text/csv")},
    )
    assert r.status_code == 200
    body = r.json()
    medical = body["metadata"]["medical"]
    sf = body["signal_features"]
    assert sf["fhr_mean"] == medical["fhr_mean"]
    assert sf["fhr_std"] == medical["fhr_std"]
    assert sf["missing_signal_pct"] == medical["missing_signal_pct"]
    assert sf["uc_available"] == medical["uc_available"]
    assert sf["recording_duration_min"] == medical["recording_duration_min"]
