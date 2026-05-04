"""Unit tests for BinaryCNN margin-based label logic (no model weights needed)."""
import pytest


MARGIN = 0.10


def _classify(risk_score: float, threshold: float) -> str:
    healthy_cutoff = threshold - MARGIN
    danger_cutoff = threshold + MARGIN
    if risk_score >= danger_cutoff:
        return "Danger"
    elif risk_score <= healthy_cutoff:
        return "Healthy"
    return "Borderline"


@pytest.mark.parametrize("risk_score,threshold,expected", [
    (0.40, 0.55, "Healthy"),     # below healthy_cutoff (0.45)
    (0.45, 0.55, "Healthy"),     # exactly at healthy_cutoff → Healthy
    (0.46, 0.55, "Borderline"),  # just above healthy_cutoff
    (0.50, 0.55, "Borderline"),  # squarely in the middle
    (0.64, 0.55, "Borderline"),  # just below danger_cutoff
    (0.65, 0.55, "Danger"),      # exactly at danger_cutoff → Danger
    (0.70, 0.55, "Danger"),      # clearly above danger_cutoff
    (0.00, 0.55, "Healthy"),     # zero score → Healthy
    (1.00, 0.55, "Danger"),      # maximum score → Danger
    (0.38, 0.48, "Healthy"),     # threshold=0.48, healthy_cutoff=0.38; risk_score=0.38 → Healthy
    (0.39, 0.48, "Borderline"),  # just above 0.38
    (0.57, 0.48, "Borderline"),  # just below danger_cutoff (0.58)
    (0.58, 0.48, "Danger"),      # exactly at danger_cutoff
])
def test_margin_classification(risk_score, threshold, expected):
    assert _classify(risk_score, threshold) == expected
