# Fetal Health Web App — Design Spec
**Date:** 2026-05-03
**Status:** Approved

---

## Overview

A generic medical dashboard web application that accepts a fetal CTG recording (CSV), lets the user select a prediction model, runs inference, and displays the result alongside extracted metadata. Built to support adding new models without touching existing code.

**Existing models in scope:** MiniROCKET, BinaryCNN (both binary: Normal / Danger).
**Future models** can be added by registering a new adapter — no other changes needed.

---

## Constraints & Decisions

| Decision | Choice | Reason |
|---|---|---|
| Upload format | CSV only (`t_sec, FHR, UC`) | Matches `csv_output/` format already used by both models |
| Metadata source | Match filename to `dataset/{id}.hea` on server | Richest metadata; graceful fallback to "Not available" |
| Max recording length | 90 minutes | Validated server-side |
| Backend | Python / FastAPI | Models are Python/PyTorch/sklearn |
| Frontend | React + Vite | Clean component model, best DX for a multi-panel dashboard |
| Dev servers | Separate (FastAPI :8000, Vite :5173) | Independent dev; production = Vite build served by FastAPI |
| Weight reload | Manual server restart | Simplest; matches research workflow of retrain → restart |
| MiniROCKET kernels | Regenerated at load time (seed 42) | Deterministic — no need to save ~MB of kernel arrays |

---

## Folder Structure

```
fetal_health_web_app/
├── backend/
│   ├── main.py                      # FastAPI app, lifespan startup, routes
│   ├── core/
│   │   ├── schemas.py               # Pydantic request/response models
│   │   ├── file_parser.py           # CSV parsing + signal feature extraction
│   │   └── hea_parser.py            # .hea metadata parser
│   ├── models/
│   │   ├── base_adapter.py          # Abstract BaseModelAdapter
│   │   ├── registry.py              # MODEL_REGISTRY dict
│   │   ├── minirocket_adapter.py    # MiniROCKET inference adapter
│   │   └── binarycnn_adapter.py     # BinaryCNN inference adapter
│   ├── training/                    # Standalone training scripts (save weights)
│   │   ├── train_minirocket.py      # Trains + saves minirocket_model.joblib + stats.json
│   │   └── train_binarycnn.py       # Trains + saves binarycnn_model.pt + stats.json
│   └── weights/                     # Git-ignored; populated by training scripts
│       ├── minirocket_model.joblib
│       ├── minirocket_stats.json
│       ├── binarycnn_model.pt
│       └── binarycnn_stats.json
├── frontend/
│   ├── src/
│   │   ├── App.jsx                  # Root component, state machine
│   │   ├── api/
│   │   │   └── client.js            # fetch wrappers for /models and /predict
│   │   └── components/
│   │       ├── UploadPanel.jsx      # File input + model dropdown + Run button
│   │       ├── BabyVisual.jsx       # SVG outline + prediction label + confidence
│   │       ├── ExplanationPanel.jsx # Feature importance list with impact icons
│   │       └── MetadataPanel.jsx    # Right-side framed panel (Baby/Mother/Medical)
│   ├── index.html
│   └── vite.config.js               # Proxy /api → :8000 in dev
└── README.md
```

---

## Backend

### API Endpoints

| Method | Path | Input | Output |
|---|---|---|---|
| `GET` | `/models` | — | `{ "models": ["minirocket", "binarycnn"] }` |
| `POST` | `/predict` | multipart: `file` (CSV), `model_name` (str) | `PredictionResponse` JSON |

### Pydantic Schemas (`schemas.py`)

```python
class PredictionLabel(BaseModel):
    label: str           # e.g. "Healthy", "Risk"
    confidence: float | None

class MetadataSection(BaseModel):
    # all fields Optional[str | float | int], default None → serialized as null

class BabyMetadata(MetadataSection):
    baby_id: str | None
    gestational_weeks: int | None
    weight_g: int | None
    sex: str | None
    apgar1: int | None
    apgar5: int | None

class MotherMetadata(MetadataSection):
    mother_age: int | None
    gravidity: int | None
    parity: int | None
    diabetes: bool | None
    hypertension: bool | None
    preeclampsia: bool | None

class MedicalMetadata(MetadataSection):
    recording_duration_min: float | None
    missing_signal_pct: float | None
    fhr_mean: float | None
    fhr_std: float | None
    uc_available: bool | None

class ImportantParameter(BaseModel):
    name: str
    value: float | str
    impact: str
    # impact values and their frontend icon mapping:
    # "normal"   → ✓ (green)
    # "elevated" → ⚠ (amber)
    # "critical" → ✗ (red)

class Explanation(BaseModel):
    important_parameters: list[ImportantParameter]
    summary: str

class RecordingMetadata(BaseModel):
    baby: BabyMetadata
    mother: MotherMetadata
    medical: MedicalMetadata

class PredictionResponse(BaseModel):
    model_name: str
    prediction: PredictionLabel
    metadata: RecordingMetadata
    explanation: Explanation
```

### BaseModelAdapter (`models/base_adapter.py`)

```python
class BaseModelAdapter(ABC):
    name: str

    @abstractmethod
    def load_model(self) -> None:
        """Called once at server startup. Load weights from disk."""

    @abstractmethod
    def preprocess(self, df: pd.DataFrame) -> np.ndarray:
        """Clean + normalize signal. Returns array ready for inference."""

    @abstractmethod
    def predict(self, processed_data: np.ndarray) -> dict:
        """Returns {"label": str, "confidence": float | None}"""

    @abstractmethod
    def explain(self, processed_data: np.ndarray, prediction: dict) -> dict:
        """Returns {"important_parameters": [...], "summary": str}"""
```

### MODEL_REGISTRY (`models/registry.py`)

```python
MODEL_REGISTRY: dict[str, BaseModelAdapter] = {
    "minirocket": MiniRocketAdapter(),
    "binarycnn":  BinaryCNNAdapter(),
}
```

All adapters in the registry have `load_model()` called during FastAPI's lifespan startup event.

### Request Flow (`POST /predict`)

1. Receive multipart: `file` (CSV bytes) + `model_name` (str)
2. Look up adapter in `MODEL_REGISTRY`; 404 if not found
3. Parse CSV → `pd.DataFrame`; 400 if columns missing
4. Extract recording duration; 400 if > 90 minutes
5. Try `hea_parser.parse(filename)` → baby/mother/medical metadata; silently fall back to `None` fields
6. `adapter.preprocess(df)` → `adapter.predict(data)` → `adapter.explain(data, pred)`
7. Assemble `PredictionResponse` and return

### Training Scripts (`training/`)

Both scripts are standalone — run from the project root with the existing `csv_output/` and `dataset/` directories present.

- `train_minirocket.py`: trains LogisticRegression on ROCKET features, saves `weights/minirocket_model.joblib` + `weights/minirocket_stats.json` (train_mean, train_std per channel)
- `train_binarycnn.py`: trains `CNNBinaryCTG`, saves `weights/binarycnn_model.pt` (state dict) + `weights/binarycnn_stats.json`

To pick up new weights: run the training script, then restart the FastAPI server.

---

## Frontend

### State Machine (`App.jsx`)

```
idle
  → (file selected + model selected) → ready
  → (click Run) → loading
    → (response ok) → result
    → (response error) → error
  → (new file selected from result/error) → ready
```

### Component Responsibilities

| Component | Responsibility |
|---|---|
| `UploadPanel` | File input, model dropdown (from `GET /models`), Run button (disabled unless ready) |
| `BabyVisual` | Inline SVG baby outline; label + confidence badge rendered inside; green/amber/red by label |
| `ExplanationPanel` | Ordered list of `important_parameters`; impact rendered as icon (✓ / ⚠ / ✗) |
| `MetadataPanel` | Three framed sections (Baby / Mother / Medical); "Not available" for null fields |

### Visual Design

- Clean medical dashboard — white background, subtle card shadows, muted blues/greens
- Left column (60%): `BabyVisual` + `ExplanationPanel`
- Right column (40%): `MetadataPanel`
- Full-screen loading overlay with spinner while `POST /predict` is in flight
- Error banner (dismissible) if request fails
- Medical disclaimer footer: *"This tool is for research and decision-support only and is not a medical diagnosis."*

### API Client (`api/client.js`)

```js
export async function getModels() { ... }           // GET /api/models
export async function runPrediction(file, model) {  // POST /api/predict
    const form = new FormData();
    form.append("file", file);
    form.append("model_name", model);
    ...
}
```

Vite proxies `/api/*` → `http://localhost:8000` in dev so no CORS issues.

---

## Adding a New Model (Future)

1. Create `backend/models/my_new_adapter.py` implementing `BaseModelAdapter`
2. Create `backend/training/train_mynewmodel.py`, save weights to `weights/`
3. Add one line to `registry.py`: `"mynewmodel": MyNewAdapter()`
4. The dropdown updates automatically — frontend needs no changes

---

## Out of Scope

- User authentication
- Database / result persistence
- Multi-file batch prediction
- Training triggered from the web UI
- Deployment infrastructure (Docker, cloud)
