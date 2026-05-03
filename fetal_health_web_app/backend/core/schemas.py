from __future__ import annotations
from typing import Optional
from pydantic import BaseModel


class PredictionLabel(BaseModel):
    label: str
    confidence: Optional[float] = None


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
    impact: str  # "normal" | "elevated" | "critical"


class Explanation(BaseModel):
    important_parameters: list[ImportantParameter]
    summary: str


class PredictionResponse(BaseModel):
    model_name: str
    prediction: PredictionLabel
    metadata: RecordingMetadata
    explanation: Explanation


class ModelsResponse(BaseModel):
    models: list[str]
