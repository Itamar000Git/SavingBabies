from __future__ import annotations
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from core.schemas import (
    Explanation,
    FHREvents,
    FHREvent,
    GroundTruth,
    ImportantParameter,
    MedicalMetadata,
    ModelsResponse,
    PredictionLabel,
    PredictionReliability,
    PredictionResponse,
    RecordingMetadata,
    SignalData,
)
from core.file_parser import extract_signal_features, parse_csv, validate_duration
from core.hea_parser import parse_hea_file, parse_outcome_fields
from core.reliability import compute_reliability, get_display_label
from core.ground_truth import lookup_ground_truth
from core.ctg_events import detect_fhr_events, prepare_signal_data
from models.registry import MODEL_REGISTRY

DATASET_DIR = os.getenv(
    "DATASET_DIR",
    os.path.join(os.path.dirname(__file__), "..", "..", "dataset"),
)
PH_FILE = os.getenv("PH_FILE", os.path.join(DATASET_DIR, "ph_levels.csv"))


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
    processed = adapter.preprocess(df, baby=baby, mother=mother)
    prediction = adapter.predict(processed)
    explanation_data = adapter.explain(processed, prediction, signal_features)

    reliability_data = compute_reliability(prediction)
    display_label = get_display_label(prediction)

    # Signal arrays and event detection
    sig = prepare_signal_data(df)
    signal_data = SignalData(**sig)
    fhr_raw = df["FHR"].values.astype(float)
    t_sec = df["t_sec"].values.astype(float) if "t_sec" in df.columns else None
    events_raw = detect_fhr_events(fhr_raw, t_sec=t_sec,
                                   gestational_weeks=baby.gestational_weeks)
    fhr_events = FHREvents(
        accelerations=[FHREvent(**e) for e in events_raw["accelerations"]],
        decelerations=[FHREvent(**e) for e in events_raw["decelerations"]],
    )

    gt_data = lookup_ground_truth(record_id, PH_FILE)
    gt_data["bdecf"] = parse_outcome_fields(hea_path).get("bdecf")
    if gt_data["available"] and gt_data["actual_label"] and not prediction.get("placeholder"):
        pred_is_risk = any(k in prediction["label"].lower() for k in ("risk", "danger"))
        actual_is_risk = gt_data["actual_label"] == "Risk"
        gt_data["correctness"] = "Correct" if pred_is_risk == actual_is_risk else "Incorrect"
    else:
        gt_data["correctness"] = "Unknown"

    return PredictionResponse(
        model_name=adapter.name,
        prediction=PredictionLabel(
            label=prediction["label"],
            display_label=display_label,
            confidence=prediction.get("confidence"),
            risk_score=prediction.get("risk_score"),
            threshold=prediction.get("threshold"),
            healthy_cutoff=prediction.get("healthy_cutoff"),
            danger_cutoff=prediction.get("danger_cutoff"),
            placeholder=prediction.get("placeholder"),
        ),
        reliability=PredictionReliability(**reliability_data),
        ground_truth=GroundTruth(**gt_data),
        metadata=metadata,
        explanation=Explanation(
            important_parameters=[
                ImportantParameter(**p) for p in explanation_data["important_parameters"]
            ],
            summary=explanation_data["summary"],
            table_note=explanation_data.get("table_note"),
            missing_signal_warning=explanation_data.get("missing_signal_warning"),
        ),
        signal_features=medical,
        signal_data=signal_data,
        fhr_events=fhr_events,
    )
