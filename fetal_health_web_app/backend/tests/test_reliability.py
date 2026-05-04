import pytest
from core.reliability import compute_reliability, get_display_label


def test_high_confidence():
    r = compute_reliability({"label": "Healthy", "confidence": 0.85})
    assert r["level"] == "High confidence"
    assert r["recommend_review"] is False


def test_low_confidence():
    r = compute_reliability({"label": "Healthy", "confidence": 0.62})
    assert r["level"] == "Low confidence"
    assert r["recommend_review"] is True


def test_borderline_uncertain():
    r = compute_reliability({"label": "Risk", "confidence": 0.54})
    assert r["level"] == "Borderline / Uncertain"
    assert r["recommend_review"] is True


def test_boundary_055_is_low_confidence():
    r = compute_reliability({"label": "Healthy", "confidence": 0.55})
    assert r["level"] == "Low confidence"


def test_boundary_070_is_high_confidence():
    r = compute_reliability({"label": "Healthy", "confidence": 0.70})
    assert r["level"] == "High confidence"


def test_placeholder_reliability():
    r = compute_reliability({"label": "Not available", "confidence": None, "placeholder": True})
    assert r["recommend_review"] is True
    assert "not loaded" in r["message"].lower()


def test_display_label_unchanged_at_high_confidence():
    assert get_display_label({"label": "Healthy", "confidence": 0.85}) == "Healthy"
    assert get_display_label({"label": "Risk", "confidence": 0.75}) == "Risk"


def test_display_label_borderline_healthy():
    result = get_display_label({"label": "Healthy", "confidence": 0.54})
    assert "Borderline" in result
    assert "Healthy" in result


def test_display_label_borderline_danger():
    result = get_display_label({"label": "Risk", "confidence": 0.51})
    assert "Borderline" in result
    assert "Danger" in result


def test_display_label_at_boundary_060_unchanged():
    assert get_display_label({"label": "Healthy", "confidence": 0.60}) == "Healthy"


def test_display_label_placeholder_unchanged():
    pred = {"label": "Not available", "confidence": None, "placeholder": True}
    assert get_display_label(pred) == "Not available"


# --- BinaryCNN risk_score path ---

def _binarycnn_pred(risk_score, label):
    return {"label": label, "risk_score": risk_score, "threshold": 0.55, "healthy_cutoff": 0.45, "danger_cutoff": 0.65}


def test_risk_score_danger_label():
    r = compute_reliability(_binarycnn_pred(0.72, "Danger"))
    assert r["level"] == "Danger"
    assert r["recommend_review"] is True
    assert "72.0%" in r["message"]


def test_risk_score_borderline_label():
    r = compute_reliability(_binarycnn_pred(0.52, "Borderline"))
    assert r["level"] == "Borderline / Uncertain"
    assert r["recommend_review"] is True


def test_risk_score_healthy_label():
    r = compute_reliability(_binarycnn_pred(0.30, "Healthy"))
    assert r["level"] == "Healthy"
    assert r["recommend_review"] is False


def test_risk_score_display_label_unchanged():
    for label in ("Healthy", "Borderline", "Danger"):
        pred = _binarycnn_pred(0.5, label)
        assert get_display_label(pred) == label
