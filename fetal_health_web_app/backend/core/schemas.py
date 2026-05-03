from __future__ import annotations
from typing import Optional
from pydantic import BaseModel


class PredictionLabel(BaseModel):
    label: str                          # raw model label ("Healthy", "Risk")
    display_label: str                  # possibly modified ("Borderline Healthy", etc.)
    confidence: Optional[float] = None
    placeholder: Optional[bool] = None


class PredictionReliability(BaseModel):
    level: str           # "High confidence" | "Low confidence" | "Borderline / Uncertain" | "Not applicable" | "Unknown"
    message: str
    recommend_review: bool


class GroundTruth(BaseModel):
    actual_label: Optional[str] = None   # "Healthy" | "Risk" | None
    ph_value: Optional[float] = None
    correctness: Optional[str] = None    # "Correct" | "Incorrect" | "Unknown"
    available: bool = False


class BabyMetadata(BaseModel):
    baby_id: Optional[str] = None
    gestational_weeks: Optional[int] = None
    weight_g: Optional[int] = None
    sex: Optional[str] = None
    apgar1: Optional[int] = None
    apgar5: Optional[int] = None


class MotherMetadata(BaseModel):
    mother_age: Optional[int] = None
    gravidity: Optional[int] = None
    parity: Optional[int] = None
    diabetes: Optional[bool] = None
    hypertension: Optional[bool] = None
    preeclampsia: Optional[bool] = None


class MedicalMetadata(BaseModel):
    recording_duration_min: Optional[float] = None
    missing_signal_pct: Optional[float] = None
    fhr_mean: Optional[float] = None
    fhr_std: Optional[float] = None
    uc_available: Optional[bool] = None


class RecordingMetadata(BaseModel):
    baby: BabyMetadata
    mother: MotherMetadata
    medical: MedicalMetadata


class ImportantParameter(BaseModel):
    name: str
    value: float | str
    impact: str                          # "normal" | "elevated" | "critical"
    description: Optional[str] = None   # medical explanation of this parameter


class Explanation(BaseModel):
    important_parameters: list[ImportantParameter]
    summary: str
    missing_signal_warning: Optional[str] = None  # set if missing_signal_pct > 20%


class PredictionResponse(BaseModel):
    model_name: str
    prediction: PredictionLabel
    reliability: PredictionReliability
    ground_truth: GroundTruth
    metadata: RecordingMetadata
    explanation: Explanation
    signal_features: MedicalMetadata


class ModelsResponse(BaseModel):
    models: list[str]
