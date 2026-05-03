from __future__ import annotations
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from core.schemas import (
    Explanation,
    ImportantParameter,
    MedicalMetadata,
    ModelsResponse,
    PredictionLabel,
    PredictionResponse,
    RecordingMetadata,
)
from core.file_parser import extract_medical_metadata, extract_signal_features, parse_csv, validate_duration
from core.hea_parser import parse_hea_file
from models.registry import MODEL_REGISTRY

DATASET_DIR = os.getenv(
    "DATASET_DIR",
    os.path.join(os.path.dirname(__file__), "..", "..", "dataset"),
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    for adapter in MODEL_REGISTRY.values():
        adapter.load_model()
    yield


app = FastAPI(title="Fetal Health Prediction API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/models", response_model=ModelsResponse)
def get_models() -> ModelsResponse:
    return ModelsResponse(models=list(MODEL_REGISTRY.keys()))


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    model_name: str = Form(...),
) -> PredictionResponse:
    if model_name not in MODEL_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found.")

    content = await file.read()
    try:
        df = parse_csv(content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    signal_features = extract_signal_features(df)
    medical = MedicalMetadata(**signal_features)
    try:
        validate_duration(medical)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    record_id = os.path.splitext(file.filename or "")[0]
    hea_path = os.path.join(DATASET_DIR, f"{record_id}.hea")
    baby, mother = parse_hea_file(hea_path, record_id=record_id)

    metadata = RecordingMetadata(baby=baby, mother=mother, medical=medical)

    adapter = MODEL_REGISTRY[model_name]
    processed = adapter.preprocess(df)
    prediction = adapter.predict(processed)
    explanation_data = adapter.explain(processed, prediction, signal_features)

    return PredictionResponse(
        model_name=adapter.name,
        prediction=PredictionLabel(**prediction),
        metadata=metadata,
        explanation=Explanation(
            important_parameters=[
                ImportantParameter(**p) for p in explanation_data["important_parameters"]
            ],
            summary=explanation_data["summary"],
        ),
        signal_features=medical,
    )
