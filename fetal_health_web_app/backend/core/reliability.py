from __future__ import annotations


def compute_reliability(prediction: dict) -> dict:
    """Compute reliability level and message from a prediction dict."""
    label = prediction.get("label", "")

    if prediction.get("placeholder"):
        return {
            "level": "Not applicable",
            "message": (
                "Placeholder prediction — real model weights are not loaded. "
                "Do not use this as a real prediction."
            ),
            "recommend_review": True,
        }

    # BinaryCNN: risk_score + margin-based label (Healthy / Borderline / Danger)
    if prediction.get("risk_score") is not None:
        risk_pct = round(prediction["risk_score"] * 100, 1)
        if label == "Danger":
            return {
                "level": "Danger",
                "message": f"Risk score is {risk_pct}% — above the danger cutoff. Immediate clinical review recommended.",
                "recommend_review": True,
            }
        elif label == "Borderline":
            return {
                "level": "Borderline / Uncertain",
                "message": f"Risk score is {risk_pct}% — within the borderline range. Further review recommended.",
                "recommend_review": True,
            }
        else:
            return {
                "level": "Healthy",
                "message": f"Risk score is {risk_pct}% — below the healthy cutoff. Continue routine monitoring.",
                "recommend_review": False,
            }

    # Confidence-based models (MiniROCKET)
    confidence = prediction.get("confidence")
    if confidence is None:
        return {
            "level": "Unknown",
            "message": "Confidence is not available for this prediction.",
            "recommend_review": True,
        }

    pct = round(confidence * 100, 1)
    if confidence >= 0.70:
        return {
            "level": "High confidence",
            "message": (
                f"The model predicts {label} with high confidence ({pct}%). "
                "This is a strong prediction."
            ),
            "recommend_review": False,
        }
    elif confidence >= 0.55:
        return {
            "level": "Low confidence",
            "message": (
                f"The model leans toward {label}, but confidence is moderate ({pct}%). "
                "This result should be reviewed before clinical use."
            ),
            "recommend_review": True,
        }
    else:
        return {
            "level": "Borderline / Uncertain",
            "message": (
                f"The model slightly leans toward {label}, but confidence is low ({pct}%). "
                "This result should be reviewed and should not be treated as a strong prediction."
            ),
            "recommend_review": True,
        }


def get_display_label(prediction: dict) -> str:
    """Return a display label — may be modified if confidence is below 0.60."""
    label = prediction.get("label", "")
    confidence = prediction.get("confidence")

    if prediction.get("placeholder"):
        return label

    if confidence is None or confidence >= 0.60:
        return label

    is_risk = any(k in label.lower() for k in ("risk", "danger"))
    return f"Borderline {'Danger' if is_risk else 'Healthy'}"
