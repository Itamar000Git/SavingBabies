# Fetal Health Web App Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a FastAPI + React web application in `fetal_health_web_app/` that accepts a CTG CSV recording, runs MiniROCKET or BinaryCNN inference, and displays a structured medical dashboard with prediction, explanation, and metadata.

**Architecture:** FastAPI backend (port 8000) with a generic model adapter registry; React + Vite frontend (port 5173) that proxies `/api/*` to the backend. Models load weights once at startup; retraining is done by running a standalone training script then restarting the server.

**Tech Stack:** Python 3.10+, FastAPI, Pydantic v2, PyTorch, scikit-learn, joblib, pandas, numpy, React 18, Vite, plain CSS (no UI framework).

---

## File Map

### Backend (`fetal_health_web_app/backend/`)
| File | Responsibility |
|------|---------------|
| `main.py` | FastAPI app, lifespan startup, `/models` and `/predict` routes |
| `core/__init__.py` | Empty package marker |
| `core/schemas.py` | All Pydantic request/response models |
| `core/hea_parser.py` | Parse `.hea` metadata files → `BabyMetadata`, `MotherMetadata` |
| `core/file_parser.py` | Parse uploaded CSV → `pd.DataFrame` + `MedicalMetadata`, validate 90-min limit |
| `models/__init__.py` | Empty package marker |
| `models/base_adapter.py` | `BaseModelAdapter` abstract class |
| `models/registry.py` | `MODEL_REGISTRY` dict wiring adapters |
| `models/binarycnn_adapter.py` | `BinaryCNNAdapter`: load `.pt` weights, preprocess, predict, explain |
| `models/minirocket_adapter.py` | `MiniRocketAdapter`: regenerate kernels (seed 42), load `.joblib`, predict, explain |
| `training/__init__.py` | Empty package marker |
| `training/train_binarycnn.py` | Full CNN training → saves `weights/binarycnn_model.pt` + `binarycnn_stats.json` |
| `training/train_minirocket.py` | Full ROCKET training → saves `weights/minirocket_model.joblib` + `minirocket_stats.json` |
| `weights/.gitkeep` | Keep empty weights dir in git |
| `requirements.txt` | Python dependencies |
| `pytest.ini` | `pythonpath = .` so tests import `core.*`, `models.*` directly |
| `tests/__init__.py` | Empty |
| `tests/test_hea_parser.py` | Unit tests for `.hea` parsing |
| `tests/test_file_parser.py` | Unit tests for CSV parsing and validation |
| `tests/test_predict_endpoint.py` | Integration tests for POST `/predict` |

### Frontend (`fetal_health_web_app/frontend/`)
| File | Responsibility |
|------|---------------|
| `package.json` | Deps: react, react-dom, vite, @vitejs/plugin-react |
| `vite.config.js` | Proxy `/api` → `http://localhost:8000` |
| `index.html` | Root HTML with `<div id="root">` |
| `src/main.jsx` | React DOM render entry point |
| `src/App.jsx` | Root component, state machine (idle→loading→result/error) |
| `src/App.css` | All styles (medical dashboard theme) |
| `src/api/client.js` | `getModels()` and `runPrediction(file, modelName)` fetch wrappers |
| `src/components/UploadPanel.jsx` | File input + model dropdown (from `/api/models`) + Run button |
| `src/components/BabyVisual.jsx` | SVG baby outline + prediction label + confidence badge |
| `src/components/ExplanationPanel.jsx` | Feature table with impact icons (✓ ⚠ ✗) + summary text |
| `src/components/MetadataPanel.jsx` | Three framed sections: Baby / Mother / Medical |
| `src/components/LoadingOverlay.jsx` | Full-screen spinner overlay |
| `src/components/ErrorBanner.jsx` | Dismissible red error bar |

### Root
| File | Responsibility |
|------|---------------|
| `fetal_health_web_app/.gitignore` | Ignore `weights/*.pt`, `weights/*.joblib`, `weights/*.json`, `node_modules/`, `__pycache__/` |
| `fetal_health_web_app/README.md` | Setup and usage instructions |

---

## Task 1: Project Scaffold

**Files:**
- Create: `fetal_health_web_app/backend/requirements.txt`
- Create: `fetal_health_web_app/backend/pytest.ini`
- Create: `fetal_health_web_app/backend/weights/.gitkeep`
- Create: `fetal_health_web_app/backend/core/__init__.py`
- Create: `fetal_health_web_app/backend/models/__init__.py`
- Create: `fetal_health_web_app/backend/training/__init__.py`
- Create: `fetal_health_web_app/backend/tests/__init__.py`
- Create: `fetal_health_web_app/.gitignore`

- [ ] **Step 1: Create directory structure**

Run from the `SavingBabies/` project root:
```bash
mkdir -p fetal_health_web_app/backend/core
mkdir -p fetal_health_web_app/backend/models
mkdir -p fetal_health_web_app/backend/training
mkdir -p fetal_health_web_app/backend/tests
mkdir -p fetal_health_web_app/backend/weights
```

- [ ] **Step 2: Create package markers and config files**

`fetal_health_web_app/backend/core/__init__.py` — empty file

`fetal_health_web_app/backend/models/__init__.py` — empty file

`fetal_health_web_app/backend/training/__init__.py` — empty file

`fetal_health_web_app/backend/tests/__init__.py` — empty file

`fetal_health_web_app/backend/weights/.gitkeep` — empty file

`fetal_health_web_app/backend/pytest.ini`:
```ini
[pytest]
pythonpath = .
testpaths = tests
```

`fetal_health_web_app/backend/requirements.txt`:
```
fastapi>=0.111.0
uvicorn[standard]>=0.30.0
python-multipart>=0.0.9
pydantic>=2.7.0
pandas>=2.2.0
numpy>=1.26.0
torch>=2.3.0
scikit-learn>=1.5.0
joblib>=1.4.0
pytest>=8.2.0
httpx>=0.27.0
```

`fetal_health_web_app/.gitignore`:
```
# Python
__pycache__/
*.pyc
*.pyo
.venv/
venv/

# Model weights
backend/weights/*.pt
backend/weights/*.joblib
backend/weights/*.json

# Frontend
frontend/node_modules/
frontend/dist/

# IDE
.idea/
.vscode/
```

- [ ] **Step 3: Install backend dependencies**

```bash
cd fetal_health_web_app/backend
pip install -r requirements.txt
```

Expected: all packages install without errors.

- [ ] **Step 4: Commit scaffold**

```bash
git add fetal_health_web_app/
git commit -m "feat: scaffold fetal_health_web_app folder structure"
```

---

## Task 2: Pydantic Schemas

**Files:**
- Create: `fetal_health_web_app/backend/core/schemas.py`

- [ ] **Step 1: Write schemas**

`fetal_health_web_app/backend/core/schemas.py`:
```python
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
```

- [ ] **Step 2: Verify schemas import cleanly**

Run from `fetal_health_web_app/backend/`:
```bash
python -c "from core.schemas import PredictionResponse, ModelsResponse; print('OK')"
```
Expected output: `OK`

- [ ] **Step 3: Commit**

```bash
git add fetal_health_web_app/backend/core/schemas.py
git commit -m "feat: add Pydantic schemas for prediction API"
```

---

## Task 3: `.hea` Metadata Parser

**Files:**
- Create: `fetal_health_web_app/backend/core/hea_parser.py`
- Create: `fetal_health_web_app/backend/tests/test_hea_parser.py`

- [ ] **Step 1: Write the failing tests**

`fetal_health_web_app/backend/tests/test_hea_parser.py`:
```python
import os
import tempfile
import pytest
from core.hea_parser import parse_hea_file

SAMPLE_HEA = """\
1001 2 4 19200
1001.dat 16 100(0)/bpm 12 0 15050 20101 0 FHR
1001.dat 16 100/nd 12 0 700 378 0 UC

#----- Additional parameters for record 1001

#-- Outcome measures
#pH           7.14

#-- Fetus/Neonate descriptors
#Gest. weeks  37
#Weight(g)    2660
#Sex          2
#Apgar1       6
#Apgar5       8

#-- Maternal (risk-)factors
#Age          32
#Gravidity    1
#Parity       0
#Diabetes     1
#Hypertension 0
#Preeclampsia 0
"""


@pytest.fixture
def hea_file(tmp_path):
    p = tmp_path / "1001.hea"
    p.write_text(SAMPLE_HEA)
    return str(p)


def test_parse_baby_gestational_weeks(hea_file):
    baby, _ = parse_hea_file(hea_file, "1001")
    assert baby.gestational_weeks == 37


def test_parse_baby_weight(hea_file):
    baby, _ = parse_hea_file(hea_file, "1001")
    assert baby.weight_g == 2660


def test_parse_baby_sex_female(hea_file):
    baby, _ = parse_hea_file(hea_file, "1001")
    assert baby.sex == "Female"


def test_parse_baby_apgar(hea_file):
    baby, _ = parse_hea_file(hea_file, "1001")
    assert baby.apgar1 == 6
    assert baby.apgar5 == 8


def test_parse_baby_id(hea_file):
    baby, _ = parse_hea_file(hea_file, "1001")
    assert baby.baby_id == "1001"


def test_parse_mother_age(hea_file):
    _, mother = parse_hea_file(hea_file, "1001")
    assert mother.mother_age == 32


def test_parse_mother_gravidity(hea_file):
    _, mother = parse_hea_file(hea_file, "1001")
    assert mother.gravidity == 1
    assert mother.parity == 0


def test_parse_mother_diabetes_true(hea_file):
    _, mother = parse_hea_file(hea_file, "1001")
    assert mother.diabetes is True


def test_parse_mother_hypertension_false(hea_file):
    _, mother = parse_hea_file(hea_file, "1001")
    assert mother.hypertension is False


def test_missing_file_returns_defaults():
    baby, mother = parse_hea_file("/nonexistent/path.hea", "9999")
    assert baby.baby_id == "9999"
    assert baby.gestational_weeks is None
    assert mother.mother_age is None
```

- [ ] **Step 2: Run tests to confirm they fail**

Run from `fetal_health_web_app/backend/`:
```bash
pytest tests/test_hea_parser.py -v
```
Expected: `ImportError` or `ModuleNotFoundError` — `core.hea_parser` does not exist yet.

- [ ] **Step 3: Implement the parser**

`fetal_health_web_app/backend/core/hea_parser.py`:
```python
from __future__ import annotations
import re
from typing import Optional
from core.schemas import BabyMetadata, MotherMetadata


def _parse_fields(hea_path: str) -> dict[str, str]:
    """Read all #Key   Value lines from a .hea file into a dict."""
    fields: dict[str, str] = {}
    try:
        with open(hea_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line.startswith("#"):
                    continue
                content = line[1:]  # strip leading #
                # Split on 2+ spaces to separate key from value
                parts = re.split(r"\s{2,}", content, maxsplit=1)
                if len(parts) == 2:
                    fields[parts[0].strip()] = parts[1].strip()
    except FileNotFoundError:
        pass
    return fields


def _int(fields: dict[str, str], key: str) -> Optional[int]:
    v = fields.get(key)
    if v is None:
        return None
    try:
        return int(float(v))
    except (ValueError, TypeError):
        return None


def _bool(fields: dict[str, str], key: str) -> Optional[bool]:
    v = _int(fields, key)
    return bool(v) if v is not None else None


_SEX_MAP = {"1": "Male", "2": "Female"}


def parse_hea_file(hea_path: str, record_id: str) -> tuple[BabyMetadata, MotherMetadata]:
    """
    Parse a WFDB .hea file and return structured baby and mother metadata.
    Returns objects with all-None fields if the file is missing or unreadable.
    """
    fields = _parse_fields(hea_path)

    sex_raw = fields.get("Sex")
    baby = BabyMetadata(
        baby_id=record_id,
        gestational_weeks=_int(fields, "Gest. weeks"),
        weight_g=_int(fields, "Weight(g)"),
        sex=_SEX_MAP.get(sex_raw) if sex_raw else None,
        apgar1=_int(fields, "Apgar1"),
        apgar5=_int(fields, "Apgar5"),
    )

    mother = MotherMetadata(
        mother_age=_int(fields, "Age"),
        gravidity=_int(fields, "Gravidity"),
        parity=_int(fields, "Parity"),
        diabetes=_bool(fields, "Diabetes"),
        hypertension=_bool(fields, "Hypertension"),
        preeclampsia=_bool(fields, "Preeclampsia"),
    )

    return baby, mother
```

- [ ] **Step 4: Run tests and confirm they pass**

```bash
pytest tests/test_hea_parser.py -v
```
Expected: all 10 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add fetal_health_web_app/backend/core/hea_parser.py fetal_health_web_app/backend/tests/test_hea_parser.py
git commit -m "feat: add .hea metadata parser with tests"
```

---

## Task 4: CSV File Parser

**Files:**
- Create: `fetal_health_web_app/backend/core/file_parser.py`
- Create: `fetal_health_web_app/backend/tests/test_file_parser.py`

- [ ] **Step 1: Write the failing tests**

`fetal_health_web_app/backend/tests/test_file_parser.py`:
```python
import io
import pytest
import numpy as np
import pandas as pd
from core.file_parser import parse_csv, extract_medical_metadata, validate_duration

# 4 rows, 1 missing FHR (=0), t_sec spans 0.0–0.75
SAMPLE_CSV = b"t_sec,FHR,UC\n0.0,150.5,7.0\n0.25,151.0,8.5\n0.5,0.0,9.0\n0.75,152.0,7.5\n"


def test_parse_csv_returns_dataframe():
    df = parse_csv(SAMPLE_CSV)
    assert list(df.columns) >= ["FHR", "UC"]
    assert len(df) == 4


def test_parse_csv_missing_uc_raises():
    bad = b"t_sec,FHR\n0.0,150.0\n"
    with pytest.raises(ValueError, match="UC"):
        parse_csv(bad)


def test_parse_csv_missing_fhr_raises():
    bad = b"t_sec,UC\n0.0,5.0\n"
    with pytest.raises(ValueError, match="FHR"):
        parse_csv(bad)


def test_extract_missing_signal_pct():
    df = parse_csv(SAMPLE_CSV)
    medical = extract_medical_metadata(df)
    assert medical.missing_signal_pct == 25.0  # 1 of 4 rows has FHR=0


def test_extract_fhr_mean():
    df = parse_csv(SAMPLE_CSV)
    medical = extract_medical_metadata(df)
    # valid FHR rows: 150.5, 151.0, 152.0 → mean = 151.1667
    assert medical.fhr_mean is not None
    assert abs(medical.fhr_mean - 151.17) < 0.1


def test_extract_uc_available():
    df = parse_csv(SAMPLE_CSV)
    medical = extract_medical_metadata(df)
    assert medical.uc_available is True


def test_extract_duration_minutes():
    df = parse_csv(SAMPLE_CSV)
    medical = extract_medical_metadata(df)
    # t_sec goes from 0.0 to 0.75 → duration = 0.75 sec = 0.0125 min
    assert medical.recording_duration_min is not None
    assert abs(medical.recording_duration_min - round(0.75 / 60, 2)) < 0.001


def test_validate_duration_ok():
    df = parse_csv(SAMPLE_CSV)
    medical = extract_medical_metadata(df)
    validate_duration(medical)  # should not raise


def test_validate_duration_too_long():
    n = int(91 * 60 * 4)  # 91 minutes at 4 Hz
    t = np.arange(n, dtype=np.float32) * 0.25
    fhr = np.full(n, 150.0, dtype=np.float32)
    uc = np.full(n, 5.0, dtype=np.float32)
    csv_bytes = pd.DataFrame({"t_sec": t, "FHR": fhr, "UC": uc}).to_csv(index=False).encode()
    df = parse_csv(csv_bytes)
    medical = extract_medical_metadata(df)
    with pytest.raises(ValueError, match="too long"):
        validate_duration(medical)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_file_parser.py -v
```
Expected: `ImportError` — `core.file_parser` does not exist yet.

- [ ] **Step 3: Implement the parser**

`fetal_health_web_app/backend/core/file_parser.py`:
```python
from __future__ import annotations
import io
import numpy as np
import pandas as pd
from core.schemas import MedicalMetadata

MAX_DURATION_MIN = 90.0


def parse_csv(content: bytes) -> pd.DataFrame:
    """Parse raw CSV bytes. Raises ValueError if FHR or UC columns are missing."""
    df = pd.read_csv(io.BytesIO(content))
    missing = {"FHR", "UC"} - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")
    return df


def extract_medical_metadata(df: pd.DataFrame) -> MedicalMetadata:
    """Compute signal-level features from the CTG dataframe."""
    fhr = df["FHR"].values.astype(np.float32)
    uc = df["UC"].values.astype(np.float32)

    duration_min: float | None = None
    if "t_sec" in df.columns:
        t = df["t_sec"].values.astype(np.float64)
        duration_min = round(float(t[-1] - t[0]) / 60.0, 2)

    missing_mask = (fhr == 0) | np.isnan(fhr)
    missing_pct = round(float(missing_mask.mean()) * 100, 2)

    valid_fhr = fhr[~missing_mask]
    fhr_mean = round(float(valid_fhr.mean()), 2) if len(valid_fhr) > 0 else None
    fhr_std = round(float(valid_fhr.std()), 2) if len(valid_fhr) > 0 else None

    uc_available = bool(np.any(uc > 0))

    return MedicalMetadata(
        recording_duration_min=duration_min,
        missing_signal_pct=missing_pct,
        fhr_mean=fhr_mean,
        fhr_std=fhr_std,
        uc_available=uc_available,
    )


def validate_duration(medical: MedicalMetadata) -> None:
    """Raises ValueError if recording exceeds MAX_DURATION_MIN."""
    if medical.recording_duration_min is not None and medical.recording_duration_min > MAX_DURATION_MIN:
        raise ValueError(
            f"Recording too long: {medical.recording_duration_min:.1f} min (max {MAX_DURATION_MIN:.0f} min)"
        )
```

- [ ] **Step 4: Run tests and confirm they pass**

```bash
pytest tests/test_file_parser.py -v
```
Expected: all 9 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add fetal_health_web_app/backend/core/file_parser.py fetal_health_web_app/backend/tests/test_file_parser.py
git commit -m "feat: add CSV file parser with duration validation and tests"
```

---

## Task 5: Base Adapter + Registry

**Files:**
- Create: `fetal_health_web_app/backend/models/base_adapter.py`
- Create: `fetal_health_web_app/backend/models/registry.py`

- [ ] **Step 1: Write BaseModelAdapter**

`fetal_health_web_app/backend/models/base_adapter.py`:
```python
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class BaseModelAdapter(ABC):
    name: str  # human-readable display name, set in each subclass

    @abstractmethod
    def load_model(self) -> None:
        """Load weights from disk. Called once at server startup."""

    @abstractmethod
    def preprocess(self, df: pd.DataFrame) -> np.ndarray:
        """
        Clean and normalize the CTG signal.
        Stores raw signal stats on self for use by explain().
        Returns normalized array ready for inference.
        """

    @abstractmethod
    def predict(self, processed_data: np.ndarray) -> dict:
        """
        Run inference on preprocessed data.
        Returns {"label": str, "confidence": float | None}
        """

    @abstractmethod
    def explain(self, processed_data: np.ndarray, prediction: dict) -> dict:
        """
        Produce a human-readable explanation.
        Returns {"important_parameters": list[dict], "summary": str}
        Each parameter dict: {"name": str, "value": float|str, "impact": "normal"|"elevated"|"critical"}
        """
```

- [ ] **Step 2: Write placeholder registry**

`fetal_health_web_app/backend/models/registry.py`:
```python
from __future__ import annotations
from models.base_adapter import BaseModelAdapter

# Adapters are imported lazily here; populated in task 6 and 7.
# To add a new model: import its adapter and add one line below.
MODEL_REGISTRY: dict[str, BaseModelAdapter] = {}
```

- [ ] **Step 3: Verify imports**

```bash
python -c "from models.base_adapter import BaseModelAdapter; from models.registry import MODEL_REGISTRY; print('OK')"
```
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add fetal_health_web_app/backend/models/base_adapter.py fetal_health_web_app/backend/models/registry.py
git commit -m "feat: add BaseModelAdapter ABC and empty MODEL_REGISTRY"
```

---

## Task 6: BinaryCNN Adapter

**Files:**
- Create: `fetal_health_web_app/backend/models/binarycnn_adapter.py`

- [ ] **Step 1: Write the adapter**

`fetal_health_web_app/backend/models/binarycnn_adapter.py`:
```python
from __future__ import annotations
import json
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from models.base_adapter import BaseModelAdapter

_WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "..", "weights")
SEQ_LEN = 4800  # 20 minutes @ 4 Hz — must match training config


class _CNNBinaryCTG(nn.Module):
    """1D CNN for binary CTG classification. Architecture must match training."""

    def __init__(self, in_ch: int = 3):
        super().__init__()
        self.fe = nn.Sequential(
            nn.Conv1d(in_ch, 32, kernel_size=15, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=15, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=11, padding=5),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)          # (B, L, C) → (B, C, L)
        x = self.fe(x)                 # (B, 128, L')
        x = self.gap(x).squeeze(-1)    # (B, 128)
        return self.head(x).squeeze(-1)  # (B,)


class BinaryCNNAdapter(BaseModelAdapter):
    name = "BinaryCNN"

    def __init__(self) -> None:
        self._model: _CNNBinaryCTG | None = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._threshold: float = 0.5
        self._last_raw_seq: np.ndarray | None = None  # set during preprocess for explain()

    def load_model(self) -> None:
        """Load model weights from weights/binarycnn_model.pt.
        If the file does not exist, the adapter runs in placeholder mode."""
        model_path = os.path.join(_WEIGHTS_DIR, "binarycnn_model.pt")
        stats_path = os.path.join(_WEIGHTS_DIR, "binarycnn_stats.json")

        if not os.path.exists(model_path):
            print(f"[BinaryCNNAdapter] WARNING: weights not found at {model_path}. "
                  "Run training/train_binarycnn.py first.")
            return

        self._model = _CNNBinaryCTG(in_ch=3).to(self._device)
        self._model.load_state_dict(torch.load(model_path, map_location=self._device))
        self._model.eval()

        with open(stats_path) as f:
            stats = json.load(f)
        self._mean = np.array(stats["train_mean"], dtype=np.float32)
        self._std = np.array(stats["train_std"], dtype=np.float32)
        self._threshold = float(stats.get("threshold", 0.5))
        print(f"[BinaryCNNAdapter] Loaded model (threshold={self._threshold:.2f})")

    def _build_raw_sequence(self, df: pd.DataFrame) -> np.ndarray:
        """Build (SEQ_LEN, 3) array [FHR, UC, mask] from raw dataframe."""
        fhr_raw = df["FHR"].values.astype(np.float32)
        uc_raw = df["UC"].values.astype(np.float32)

        mask = np.ones_like(fhr_raw)
        mask[(fhr_raw == 0) | np.isnan(fhr_raw)] = 0.0

        fhr_clean = (
            pd.Series(fhr_raw)
            .replace(0, np.nan)
            .interpolate(method="linear")
            .fillna(140)
            .values.astype(np.float32)
        )
        uc_clean = (
            pd.Series(uc_raw)
            .interpolate(method="linear")
            .fillna(0)
            .values.astype(np.float32)
        )

        n = len(fhr_clean)
        if n >= SEQ_LEN:
            fhr_fix, uc_fix, mask_fix = fhr_clean[-SEQ_LEN:], uc_clean[-SEQ_LEN:], mask[-SEQ_LEN:]
        else:
            pad = SEQ_LEN - n
            fhr_fix = np.pad(fhr_clean, (0, pad), mode="edge")
            uc_fix = np.pad(uc_clean, (0, pad), mode="edge")
            mask_fix = np.concatenate([mask, np.zeros(pad, dtype=np.float32)])

        return np.stack([fhr_fix, uc_fix, mask_fix], axis=1)  # (SEQ_LEN, 3)

    def preprocess(self, df: pd.DataFrame) -> np.ndarray:
        raw = self._build_raw_sequence(df)
        self._last_raw_seq = raw.copy()  # store for explain()
        if self._mean is not None:
            return (raw - self._mean) / self._std
        return raw

    def predict(self, processed_data: np.ndarray) -> dict:
        if self._model is None:
            return {"label": "Not available (model not trained)", "confidence": None}

        tensor = (
            torch.tensor(processed_data, dtype=torch.float32)
            .unsqueeze(0)
            .to(self._device)
        )
        with torch.no_grad():
            logit = self._model(tensor)
            prob = float(torch.sigmoid(logit).item())

        label = "Risk" if prob >= self._threshold else "Healthy"
        return {"label": label, "confidence": round(prob, 4)}

    def explain(self, processed_data: np.ndarray, prediction: dict) -> dict:
        raw = self._last_raw_seq if self._last_raw_seq is not None else processed_data
        fhr_col = raw[:, 0]
        mask_col = raw[:, 2]
        uc_col = raw[:, 1]

        missing_pct = round(float((mask_col < 0.5).mean()) * 100, 1)

        valid_fhr = fhr_col[mask_col > 0.5]
        fhr_mean = round(float(valid_fhr.mean()), 1) if len(valid_fhr) > 0 else 0.0
        fhr_std = round(float(valid_fhr.std()), 1) if len(valid_fhr) > 0 else 0.0
        uc_present = bool(np.any(uc_col > 0))

        confidence = prediction.get("confidence") or 0.5
        is_risk = "risk" in prediction["label"].lower() or "danger" in prediction["label"].lower()

        params = [
            {
                "name": "FHR Mean",
                "value": f"{fhr_mean} bpm",
                "impact": "normal" if 110 <= fhr_mean <= 160 else "critical",
            },
            {
                "name": "FHR Variability",
                "value": f"{fhr_std} bpm",
                "impact": "elevated" if fhr_std < 5 else "normal",
            },
            {
                "name": "Missing Signal",
                "value": f"{missing_pct}%",
                "impact": "critical" if missing_pct > 20 else "elevated" if missing_pct > 5 else "normal",
            },
            {
                "name": "UC Activity",
                "value": "Present" if uc_present else "Absent",
                "impact": "normal",
            },
        ]

        risk_factors = [p["name"] for p in params if p["impact"] in ("elevated", "critical")]
        if risk_factors:
            reason = f"Elevated risk indicators: {', '.join(risk_factors)}."
        else:
            reason = "No significant risk indicators detected."

        summary = (
            f"BinaryCNN classified this recording as {'at risk' if is_risk else 'healthy'} "
            f"with {round(confidence * 100, 1)}% confidence. {reason}"
        )

        return {"important_parameters": params, "summary": summary}
```

- [ ] **Step 2: Register the adapter**

Update `fetal_health_web_app/backend/models/registry.py`:
```python
from __future__ import annotations
from models.base_adapter import BaseModelAdapter
from models.binarycnn_adapter import BinaryCNNAdapter

# To add a new model: import its adapter and add one line below.
MODEL_REGISTRY: dict[str, BaseModelAdapter] = {
    "binarycnn": BinaryCNNAdapter(),
}
```

- [ ] **Step 3: Verify adapter loads without error (placeholder mode)**

```bash
python -c "
from models.registry import MODEL_REGISTRY
for name, adapter in MODEL_REGISTRY.items():
    adapter.load_model()
    print(f'{name}: loaded (placeholder={adapter._model is None})')
"
```
Expected output: `binarycnn: loaded (placeholder=True)`

- [ ] **Step 4: Commit**

```bash
git add fetal_health_web_app/backend/models/binarycnn_adapter.py fetal_health_web_app/backend/models/registry.py
git commit -m "feat: add BinaryCNNAdapter with placeholder mode"
```

---

## Task 7: MiniROCKET Adapter

**Files:**
- Create: `fetal_health_web_app/backend/models/minirocket_adapter.py`

- [ ] **Step 1: Write the adapter**

`fetal_health_web_app/backend/models/minirocket_adapter.py`:
```python
from __future__ import annotations
import json
import os
import joblib
import numpy as np
import pandas as pd
from models.base_adapter import BaseModelAdapter

_WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "..", "weights")

# These constants MUST match the training script exactly.
SEQ_LEN = 1200       # 5 minutes @ 4 Hz
N_KERNELS = 4000
KERNEL_MIN = 7
KERNEL_MAX = 51
DILATION_MAX = 64
RANDOM_STATE = 42


class MiniRocketAdapter(BaseModelAdapter):
    name = "MiniROCKET"

    def __init__(self) -> None:
        self._clf = None
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._kernels: list[tuple] = []
        self._last_raw_seq: np.ndarray | None = None

    def load_model(self) -> None:
        """Load LogisticRegression weights and regenerate ROCKET kernels (deterministic, seed 42)."""
        model_path = os.path.join(_WEIGHTS_DIR, "minirocket_model.joblib")
        stats_path = os.path.join(_WEIGHTS_DIR, "minirocket_stats.json")

        # Always regenerate kernels (deterministic with fixed seed)
        rng = np.random.default_rng(RANDOM_STATE)
        self._kernels = []
        for _ in range(N_KERNELS):
            length = int(rng.integers(KERNEL_MIN, KERNEL_MAX + 1))
            weights = rng.normal(0, 1, size=length).astype(np.float32)
            bias = float(rng.uniform(-1, 1))
            dilation = int(2 ** rng.integers(0, int(np.log2(DILATION_MAX)) + 1))
            self._kernels.append((weights, bias, dilation))

        if not os.path.exists(model_path):
            print(f"[MiniRocketAdapter] WARNING: weights not found at {model_path}. "
                  "Run training/train_minirocket.py first.")
            return

        self._clf = joblib.load(model_path)

        with open(stats_path) as f:
            stats = json.load(f)
        self._mean = np.array(stats["train_mean"], dtype=np.float32)
        self._std = np.array(stats["train_std"], dtype=np.float32)
        print("[MiniRocketAdapter] Loaded model")

    def _build_raw_sequence(self, df: pd.DataFrame) -> np.ndarray:
        """Build (SEQ_LEN, 3) array [FHR, UC, mask] from raw dataframe."""
        fhr_raw = df["FHR"].values.astype(np.float32)
        uc_raw = df["UC"].values.astype(np.float32)

        mask = np.ones_like(fhr_raw)
        mask[(fhr_raw == 0) | np.isnan(fhr_raw)] = 0.0

        fhr_clean = (
            pd.Series(fhr_raw)
            .replace(0, np.nan)
            .interpolate(method="linear")
            .fillna(140)
            .values.astype(np.float32)
        )
        uc_clean = (
            pd.Series(uc_raw)
            .interpolate(method="linear")
            .fillna(0)
            .values.astype(np.float32)
        )

        n = len(fhr_clean)
        if n >= SEQ_LEN:
            fhr_fix, uc_fix, mask_fix = fhr_clean[-SEQ_LEN:], uc_clean[-SEQ_LEN:], mask[-SEQ_LEN:]
        else:
            pad = SEQ_LEN - n
            fhr_fix = np.pad(fhr_clean, (0, pad), mode="edge")
            uc_fix = np.pad(uc_clean, (0, pad), mode="edge")
            mask_fix = np.concatenate([mask, np.zeros(pad, dtype=np.float32)])

        return np.stack([fhr_fix, uc_fix, mask_fix], axis=1)  # (SEQ_LEN, 3)

    @staticmethod
    def _conv1d_dilated(x: np.ndarray, w: np.ndarray, bias: float, dilation: int) -> np.ndarray:
        L, K = x.shape[0], w.shape[0]
        out_len = L - (K - 1) * dilation
        if out_len <= 0:
            return np.array([], dtype=np.float32)
        out = np.empty(out_len, dtype=np.float32)
        for i in range(out_len):
            out[i] = (x[i: i + dilation * K: dilation] * w).sum() + bias
        return out

    def _rocket_features(self, x_rec: np.ndarray) -> np.ndarray:
        """Extract max + PPV features for all kernels × channels."""
        C = x_rec.shape[1]
        feats: list[float] = []
        for (w, b, d) in self._kernels:
            for c in range(C):
                conv_out = self._conv1d_dilated(x_rec[:, c], w, b, d)
                if conv_out.size == 0:
                    feats.extend([0.0, 0.0])
                else:
                    feats.append(float(conv_out.max()))
                    feats.append(float((conv_out > 0).mean()))
        return np.array(feats, dtype=np.float32)

    def preprocess(self, df: pd.DataFrame) -> np.ndarray:
        raw = self._build_raw_sequence(df)
        self._last_raw_seq = raw.copy()
        if self._mean is not None:
            return (raw - self._mean) / self._std
        return raw

    def predict(self, processed_data: np.ndarray) -> dict:
        if self._clf is None:
            return {"label": "Not available (model not trained)", "confidence": None}

        feats = self._rocket_features(processed_data).reshape(1, -1)
        prob = float(self._clf.predict_proba(feats)[0, 1])
        label = "Risk" if prob >= 0.5 else "Healthy"
        return {"label": label, "confidence": round(prob, 4)}

    def explain(self, processed_data: np.ndarray, prediction: dict) -> dict:
        raw = self._last_raw_seq if self._last_raw_seq is not None else processed_data
        fhr_col = raw[:, 0]
        mask_col = raw[:, 2]
        uc_col = raw[:, 1]

        missing_pct = round(float((mask_col < 0.5).mean()) * 100, 1)
        valid_fhr = fhr_col[mask_col > 0.5]
        fhr_mean = round(float(valid_fhr.mean()), 1) if len(valid_fhr) > 0 else 0.0
        fhr_std = round(float(valid_fhr.std()), 1) if len(valid_fhr) > 0 else 0.0
        uc_present = bool(np.any(uc_col > 0))

        confidence = prediction.get("confidence") or 0.5
        is_risk = "risk" in prediction["label"].lower() or "danger" in prediction["label"].lower()

        params = [
            {
                "name": "FHR Mean",
                "value": f"{fhr_mean} bpm",
                "impact": "normal" if 110 <= fhr_mean <= 160 else "critical",
            },
            {
                "name": "FHR Variability",
                "value": f"{fhr_std} bpm",
                "impact": "elevated" if fhr_std < 5 else "normal",
            },
            {
                "name": "Missing Signal",
                "value": f"{missing_pct}%",
                "impact": "critical" if missing_pct > 20 else "elevated" if missing_pct > 5 else "normal",
            },
            {
                "name": "UC Activity",
                "value": "Present" if uc_present else "Absent",
                "impact": "normal",
            },
        ]

        risk_factors = [p["name"] for p in params if p["impact"] in ("elevated", "critical")]
        reason = f"Elevated indicators: {', '.join(risk_factors)}." if risk_factors else "No significant risk indicators."

        summary = (
            f"MiniROCKET classified this recording as {'at risk' if is_risk else 'healthy'} "
            f"with {round(confidence * 100, 1)}% confidence. {reason}"
        )

        return {"important_parameters": params, "summary": summary}
```

- [ ] **Step 2: Register the adapter**

Update `fetal_health_web_app/backend/models/registry.py`:
```python
from __future__ import annotations
from models.base_adapter import BaseModelAdapter
from models.binarycnn_adapter import BinaryCNNAdapter
from models.minirocket_adapter import MiniRocketAdapter

# To add a new model: import its adapter and add one line below.
MODEL_REGISTRY: dict[str, BaseModelAdapter] = {
    "binarycnn": BinaryCNNAdapter(),
    "minirocket": MiniRocketAdapter(),
}
```

- [ ] **Step 3: Verify both adapters load**

```bash
python -c "
from models.registry import MODEL_REGISTRY
for name, adapter in MODEL_REGISTRY.items():
    adapter.load_model()
    print(name, 'ready')
"
```
Expected: both print `ready` (may show WARNING about missing weights).

- [ ] **Step 4: Commit**

```bash
git add fetal_health_web_app/backend/models/minirocket_adapter.py fetal_health_web_app/backend/models/registry.py
git commit -m "feat: add MiniRocketAdapter with deterministic kernel regeneration"
```

---

## Task 8: FastAPI Main + Endpoint Tests

**Files:**
- Create: `fetal_health_web_app/backend/main.py`
- Create: `fetal_health_web_app/backend/tests/test_predict_endpoint.py`

- [ ] **Step 1: Write the failing endpoint tests**

`fetal_health_web_app/backend/tests/test_predict_endpoint.py`:
```python
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
    n = int(91 * 60 * 4)
    t_long = np.arange(n, dtype=np.float32) * 0.25
    long_csv = pd.DataFrame({
        "t_sec": t_long,
        "FHR": np.full(n, 150.0),
        "UC": np.full(n, 5.0),
    }).to_csv(index=False).encode()
    r = client.post(
        "/predict",
        data={"model_name": "binarycnn"},
        files={"file": ("long.csv", io.BytesIO(long_csv), "text/csv")},
    )
    assert r.status_code == 400
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_predict_endpoint.py -v
```
Expected: `ImportError` — `main` module not found.

- [ ] **Step 3: Implement main.py**

`fetal_health_web_app/backend/main.py`:
```python
from __future__ import annotations
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from core.schemas import (
    Explanation,
    ImportantParameter,
    ModelsResponse,
    PredictionLabel,
    PredictionResponse,
    RecordingMetadata,
)
from core.file_parser import extract_medical_metadata, parse_csv, validate_duration
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

    medical = extract_medical_metadata(df)
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
    explanation_data = adapter.explain(processed, prediction)

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
    )
```

- [ ] **Step 4: Run tests and confirm they pass**

```bash
pytest tests/test_predict_endpoint.py -v
```
Expected: all 6 tests PASS.

- [ ] **Step 5: Smoke-test the live server**

In one terminal:
```bash
uvicorn main:app --reload --port 8000
```
In another:
```bash
curl http://localhost:8000/models
```
Expected: `{"models":["binarycnn","minirocket"]}`

- [ ] **Step 6: Commit**

```bash
git add fetal_health_web_app/backend/main.py fetal_health_web_app/backend/tests/test_predict_endpoint.py
git commit -m "feat: add FastAPI app with /models and /predict endpoints"
```

---

## Task 9: Training Script — BinaryCNN

**Files:**
- Create: `fetal_health_web_app/backend/training/train_binarycnn.py`

> Run this script from the **`SavingBabies/` project root** after completing this task to produce `backend/weights/binarycnn_model.pt`.

- [ ] **Step 1: Write the training script**

`fetal_health_web_app/backend/training/train_binarycnn.py`:
```python
"""
BinaryCNN training script.

Run from SavingBabies/ project root:
    python fetal_health_web_app/backend/training/train_binarycnn.py

Outputs to fetal_health_web_app/backend/weights/:
    binarycnn_model.pt       — model state dict
    binarycnn_stats.json     — train_mean, train_std, threshold
"""
from __future__ import annotations
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(_HERE, "..", "weights")
CSV_DIR = os.getenv("CSV_DIR", "csv_output")
PH_FILE = os.getenv("PH_FILE", "dataset/ph_levels.csv")

# ── Hyperparameters (keep in sync with binarycnn_adapter.py SEQ_LEN) ──────────
FS = 4
SEQ_LEN = 4800
BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-3
PH_DANGER_THRESHOLD = 7.10
TARGET_RECALL = 0.70
POS_WEIGHT_MULT = 1.0
PATIENCE = 6
MIN_DELTA_AP = 1e-3
RANDOM_STATE = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Model definition (must match binarycnn_adapter.py _CNNBinaryCTG) ──────────
class CNNBinaryCTG(nn.Module):
    def __init__(self, in_ch: int = 3):
        super().__init__()
        self.fe = nn.Sequential(
            nn.Conv1d(in_ch, 32, kernel_size=15, padding=7),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=15, padding=7),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=11, padding=5),
            nn.BatchNorm1d(128), nn.ReLU(),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.fe(x)
        x = self.gap(x).squeeze(-1)
        return self.head(x).squeeze(-1)


class CTGDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def make_fixed_length(fhr, uc, mask, seq_len):
    n = len(fhr)
    if n >= seq_len:
        return fhr[-seq_len:], uc[-seq_len:], mask[-seq_len:]
    pad = seq_len - n
    return (
        np.pad(fhr, (0, pad), mode="edge"),
        np.pad(uc, (0, pad), mode="edge"),
        np.concatenate([mask, np.zeros(pad, dtype=np.float32)]),
    )


def load_dataset():
    ph_df = pd.read_csv(PH_FILE)
    ph_df["record_id"] = ph_df["record_id"].astype(str)
    ph_dict = dict(zip(ph_df["record_id"], ph_df["pH"]))

    X_list, y_list = [], []
    for file in glob.glob(os.path.join(CSV_DIR, "*.csv")):
        record_id = os.path.basename(file).split(".")[0]
        if record_id not in ph_dict:
            continue
        try:
            df = pd.read_csv(file)
            if "FHR" not in df.columns or "UC" not in df.columns:
                continue
            fhr_raw = df["FHR"].values.astype(np.float32)
            uc_raw = df["UC"].values.astype(np.float32)
            mask = np.ones_like(fhr_raw)
            mask[(fhr_raw == 0) | np.isnan(fhr_raw)] = 0.0
            fhr_clean = (
                pd.Series(fhr_raw).replace(0, np.nan)
                .interpolate(method="linear").fillna(140).values.astype(np.float32)
            )
            uc_clean = (
                pd.Series(uc_raw).interpolate(method="linear")
                .fillna(0).values.astype(np.float32)
            )
            fhr_fix, uc_fix, mask_fix = make_fixed_length(fhr_clean, uc_clean, mask, SEQ_LEN)
            X_list.append(np.stack([fhr_fix, uc_fix, mask_fix], axis=1))
            y_list.append(1 if float(ph_dict[record_id]) < PH_DANGER_THRESHOLD else 0)
        except Exception as e:
            print(f"  skip {file}: {e}")

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int64)


def get_probs(model, loader):
    model.eval()
    yt, yp = [], []
    with torch.no_grad():
        for xb, yb in loader:
            logit = model(xb.to(DEVICE))
            prob = torch.sigmoid(logit).cpu().numpy().reshape(-1)
            yt.extend(yb.numpy().astype(int).tolist())
            yp.extend(prob.tolist())
    return yt, yp


def choose_threshold(y_true, y_prob, target_recall=0.70):
    best_thr, best_prec, best_rec = 0.5, -1.0, -1.0
    best_rec_thr, best_rec_val = 0.5, -1.0
    for thr in np.linspace(0.05, 0.95, 19):
        preds = [1 if p >= thr else 0 for p in y_prob]
        rec = recall_score(y_true, preds, zero_division=0)
        pre = precision_score(y_true, preds, zero_division=0)
        if rec > best_rec_val:
            best_rec_val, best_rec_thr = rec, thr
        if rec >= target_recall and pre > best_prec:
            best_prec, best_rec, best_thr = pre, rec, thr
    if best_prec < 0:
        return best_rec_thr
    return best_thr


if __name__ == "__main__":
    print(f"Loading dataset from {CSV_DIR}/ ...")
    X, y = load_dataset()
    print(f"Loaded {len(X)} records. Class counts: {np.bincount(y)}")

    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=RANDOM_STATE, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.15, random_state=RANDOM_STATE, stratify=y_temp)

    train_mean = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0)
    train_std = X_train.reshape(-1, X_train.shape[-1]).std(axis=0) + 1e-8
    X_train_n = (X_train - train_mean) / train_std
    X_val_n = (X_val - train_mean) / train_std
    X_test_n = (X_test - train_mean) / train_std

    train_loader = DataLoader(CTGDataset(X_train_n, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(CTGDataset(X_val_n, y_val), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(CTGDataset(X_test_n, y_test), batch_size=1, shuffle=False)

    model = CNNBinaryCTG(in_ch=3).to(DEVICE)
    neg, pos = int((y_train == 0).sum()), int((y_train == 1).sum())
    pos_weight = torch.tensor([neg / max(pos, 1) * POS_WEIGHT_MULT]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_ap, patience_counter = -1.0, 0
    _tmp_path = os.path.join(WEIGHTS_DIR, "_tmp_best.pt")
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    print("\nTraining ...")
    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()

        val_true, val_prob = get_probs(model, val_loader)
        val_ap = average_precision_score(val_true, val_prob)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS} | Val PR-AUC: {val_ap:.4f}")

        if val_ap > best_ap + MIN_DELTA_AP:
            best_ap, patience_counter = val_ap, 0
            torch.save(model.state_dict(), _tmp_path)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stop at epoch {epoch+1}. Best Val PR-AUC: {best_ap:.4f}")
                break

    model.load_state_dict(torch.load(_tmp_path, map_location=DEVICE))
    model.eval()

    val_true, val_prob = get_probs(model, val_loader)
    best_thr = choose_threshold(val_true, val_prob, TARGET_RECALL)
    print(f"\nChosen threshold: {best_thr:.2f}")

    test_true, test_prob = get_probs(model, test_loader)
    test_pred = [1 if p >= best_thr else 0 for p in test_prob]
    print("\n--- TEST RESULTS ---")
    print(confusion_matrix(test_true, test_pred))
    print(classification_report(test_true, test_pred, target_names=["Normal", "Danger"], zero_division=0))
    print(f"PR-AUC: {average_precision_score(test_true, test_prob):.4f}")
    if len(set(test_true)) == 2:
        print(f"ROC-AUC: {roc_auc_score(test_true, test_prob):.4f}")

    # ── Save weights ────────────────────────────────────────────────────────────
    out_model = os.path.join(WEIGHTS_DIR, "binarycnn_model.pt")
    out_stats = os.path.join(WEIGHTS_DIR, "binarycnn_stats.json")
    torch.save(model.state_dict(), out_model)
    with open(out_stats, "w") as f:
        json.dump(
            {
                "train_mean": train_mean.tolist(),
                "train_std": train_std.tolist(),
                "threshold": float(best_thr),
            },
            f,
            indent=2,
        )
    os.remove(_tmp_path)
    print(f"\nSaved: {out_model}")
    print(f"Saved: {out_stats}")
    print("Done. Restart the FastAPI server to load the new weights.")
```

- [ ] **Step 2: Run the training script (optional at this point)**

From `SavingBabies/` root:
```bash
python fetal_health_web_app/backend/training/train_binarycnn.py
```
Expected: trains, prints test results, saves two files to `fetal_health_web_app/backend/weights/`.
This step can be deferred until model development is ready.

- [ ] **Step 3: Commit**

```bash
git add fetal_health_web_app/backend/training/train_binarycnn.py
git commit -m "feat: add BinaryCNN training script with weight saving"
```

---

## Task 10: Training Script — MiniROCKET

**Files:**
- Create: `fetal_health_web_app/backend/training/train_minirocket.py`

- [ ] **Step 1: Write the training script**

`fetal_health_web_app/backend/training/train_minirocket.py`:
```python
"""
MiniROCKET training script.

Run from SavingBabies/ project root:
    python fetal_health_web_app/backend/training/train_minirocket.py

Outputs to fetal_health_web_app/backend/weights/:
    minirocket_model.joblib  — fitted LogisticRegression
    minirocket_stats.json    — train_mean, train_std
"""
from __future__ import annotations
import glob
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(_HERE, "..", "weights")
CSV_DIR = os.getenv("CSV_DIR", "csv_output")
PH_FILE = os.getenv("PH_FILE", "dataset/ph_levels.csv")

# ── Hyperparameters (keep in sync with minirocket_adapter.py) ─────────────────
FS = 4
SEQ_LEN = 1200       # 5 min @ 4 Hz
PH_DANGER_THRESHOLD = 7.10
N_KERNELS = 4000
KERNEL_MIN = 7
KERNEL_MAX = 51
DILATION_MAX = 64
RANDOM_STATE = 42


def load_dataset():
    ph_df = pd.read_csv(PH_FILE)
    ph_df["record_id"] = ph_df["record_id"].astype(str)
    ph_dict = dict(zip(ph_df["record_id"], ph_df["pH"]))

    X_list, y_list = [], []
    for file in glob.glob(os.path.join(CSV_DIR, "*.csv")):
        record_id = os.path.basename(file).split(".")[0]
        if record_id not in ph_dict:
            continue
        try:
            df = pd.read_csv(file)
            if "FHR" not in df.columns or "UC" not in df.columns:
                continue
            fhr_raw = df["FHR"].values.astype(np.float32)
            uc_raw = df["UC"].values.astype(np.float32)
            mask = np.ones_like(fhr_raw)
            mask[(fhr_raw == 0) | np.isnan(fhr_raw)] = 0.0
            fhr_clean = (
                pd.Series(fhr_raw).replace(0, np.nan)
                .interpolate(method="linear").fillna(140).values.astype(np.float32)
            )
            uc_clean = (
                pd.Series(uc_raw).interpolate(method="linear")
                .fillna(0).values.astype(np.float32)
            )
            n = len(fhr_clean)
            if n >= SEQ_LEN:
                fhr_fix, uc_fix, mask_fix = fhr_clean[-SEQ_LEN:], uc_clean[-SEQ_LEN:], mask[-SEQ_LEN:]
            else:
                pad = SEQ_LEN - n
                fhr_fix = np.pad(fhr_clean, (0, pad), mode="edge")
                uc_fix = np.pad(uc_clean, (0, pad), mode="edge")
                mask_fix = np.concatenate([mask, np.zeros(pad, dtype=np.float32)])
            X_list.append(np.stack([fhr_fix, uc_fix, mask_fix], axis=1))
            y_list.append(1 if float(ph_dict[record_id]) < PH_DANGER_THRESHOLD else 0)
        except Exception as e:
            print(f"  skip {file}: {e}")

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int64)


def generate_kernels(n_kernels, rng):
    kernels = []
    for _ in range(n_kernels):
        length = int(rng.integers(KERNEL_MIN, KERNEL_MAX + 1))
        weights = rng.normal(0, 1, size=length).astype(np.float32)
        bias = float(rng.uniform(-1, 1))
        dilation = int(2 ** rng.integers(0, int(np.log2(DILATION_MAX)) + 1))
        kernels.append((weights, bias, dilation))
    return kernels


def conv1d_dilated(x, w, bias, dilation):
    L, K = x.shape[0], w.shape[0]
    out_len = L - (K - 1) * dilation
    if out_len <= 0:
        return np.array([], dtype=np.float32)
    out = np.empty(out_len, dtype=np.float32)
    for i in range(out_len):
        out[i] = (x[i: i + dilation * K: dilation] * w).sum() + bias
    return out


def rocket_features_one(x_rec, kernels):
    C = x_rec.shape[1]
    feats: list[float] = []
    for (w, b, d) in kernels:
        for c in range(C):
            conv_out = conv1d_dilated(x_rec[:, c], w, b, d)
            if conv_out.size == 0:
                feats.extend([0.0, 0.0])
            else:
                feats.append(float(conv_out.max()))
                feats.append(float((conv_out > 0).mean()))
    return np.array(feats, dtype=np.float32)


def rocket_transform(X_arr, kernels, name):
    N = X_arr.shape[0]
    out = []
    t0 = time.time()
    for i in range(N):
        out.append(rocket_features_one(X_arr[i], kernels))
        if (i + 1) % 20 == 0 or (i + 1) == N:
            elapsed = time.time() - t0
            print(f"  [{name}] {i+1}/{N} | {elapsed:.0f}s elapsed")
    return np.vstack(out)


if __name__ == "__main__":
    print(f"Loading dataset from {CSV_DIR}/ ...")
    X, y = load_dataset()
    print(f"Loaded {len(X)} records. Class counts: {np.bincount(y)}")

    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=RANDOM_STATE, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.15, random_state=RANDOM_STATE, stratify=y_temp)

    train_mean = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0)
    train_std = X_train.reshape(-1, X_train.shape[-1]).std(axis=0) + 1e-8
    X_train_n = (X_train - train_mean) / train_std
    X_val_n = (X_val - train_mean) / train_std
    X_test_n = (X_test - train_mean) / train_std

    print("\nGenerating ROCKET kernels (seed=42) ...")
    rng = np.random.default_rng(RANDOM_STATE)
    kernels = generate_kernels(N_KERNELS, rng)

    print("\nExtracting features ...")
    F_train = rocket_transform(X_train_n, kernels, "train")
    F_val = rocket_transform(X_val_n, kernels, "val")
    F_test = rocket_transform(X_test_n, kernels, "test")
    print(f"Feature shape: {F_train.shape}")

    print("\nTraining LogisticRegression ...")
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=-1)
    clf.fit(F_train, y_train)

    test_prob = clf.predict_proba(F_test)[:, 1]
    test_pred = (test_prob >= 0.5).astype(int)

    print("\n--- TEST RESULTS ---")
    print(confusion_matrix(y_test, test_pred))
    print(classification_report(y_test, test_pred, target_names=["Normal", "Danger"], zero_division=0))
    print(f"PR-AUC:  {average_precision_score(y_test, test_prob):.4f}")
    if len(set(y_test)) == 2:
        print(f"ROC-AUC: {roc_auc_score(y_test, test_prob):.4f}")

    # ── Save weights ────────────────────────────────────────────────────────────
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    out_model = os.path.join(WEIGHTS_DIR, "minirocket_model.joblib")
    out_stats = os.path.join(WEIGHTS_DIR, "minirocket_stats.json")
    joblib.dump(clf, out_model)
    with open(out_stats, "w") as f:
        json.dump(
            {"train_mean": train_mean.tolist(), "train_std": train_std.tolist()},
            f,
            indent=2,
        )
    print(f"\nSaved: {out_model}")
    print(f"Saved: {out_stats}")
    print("Done. Restart the FastAPI server to load the new weights.")
```

- [ ] **Step 2: Commit**

```bash
git add fetal_health_web_app/backend/training/train_minirocket.py
git commit -m "feat: add MiniROCKET training script with weight saving"
```

---

## Task 11: Frontend Scaffold

**Files:**
- Create: `fetal_health_web_app/frontend/package.json`
- Create: `fetal_health_web_app/frontend/vite.config.js`
- Create: `fetal_health_web_app/frontend/index.html`
- Create: `fetal_health_web_app/frontend/src/main.jsx`

- [ ] **Step 1: Create frontend directory and install Vite + React**

```bash
cd fetal_health_web_app/frontend
npm create vite@latest . -- --template react
npm install
```

This scaffolds `package.json`, `vite.config.js`, `index.html`, and `src/`.

- [ ] **Step 2: Replace vite.config.js with proxy config**

`fetal_health_web_app/frontend/vite.config.js`:
```javascript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
})
```

- [ ] **Step 3: Replace index.html title**

Edit `fetal_health_web_app/frontend/index.html` — change the `<title>` tag:
```html
<title>Fetal Health Prediction</title>
```

- [ ] **Step 4: Replace src/main.jsx**

`fetal_health_web_app/frontend/src/main.jsx`:
```jsx
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './App.css'
import App from './App.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
```

- [ ] **Step 5: Delete Vite boilerplate files**

Delete these default files (they will be replaced):
```bash
rm fetal_health_web_app/frontend/src/App.css
rm fetal_health_web_app/frontend/src/App.jsx
rm fetal_health_web_app/frontend/src/index.css
rm fetal_health_web_app/frontend/src/assets/react.svg
rm fetal_health_web_app/frontend/public/vite.svg
```

- [ ] **Step 6: Create component and api folders**

```bash
mkdir -p fetal_health_web_app/frontend/src/components
mkdir -p fetal_health_web_app/frontend/src/api
```

- [ ] **Step 7: Verify dev server starts**

In one terminal start the FastAPI backend:
```bash
cd fetal_health_web_app/backend && uvicorn main:app --reload --port 8000
```

In another terminal:
```bash
cd fetal_health_web_app/frontend && npm run dev
```
Expected: Vite reports `http://localhost:5173` — page loads (may show errors since App.jsx is missing).

- [ ] **Step 8: Commit**

```bash
git add fetal_health_web_app/frontend/
git commit -m "feat: scaffold React + Vite frontend with /api proxy"
```

---

## Task 12: API Client + UploadPanel

**Files:**
- Create: `fetal_health_web_app/frontend/src/api/client.js`
- Create: `fetal_health_web_app/frontend/src/components/UploadPanel.jsx`

- [ ] **Step 1: Write API client**

`fetal_health_web_app/frontend/src/api/client.js`:
```javascript
const BASE = '/api'

export async function getModels() {
  const res = await fetch(`${BASE}/models`)
  if (!res.ok) throw new Error('Failed to fetch model list')
  return res.json()
}

export async function runPrediction(file, modelName) {
  const form = new FormData()
  form.append('file', file)
  form.append('model_name', modelName)
  const res = await fetch(`${BASE}/predict`, { method: 'POST', body: form })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Unknown server error' }))
    throw new Error(err.detail || 'Prediction failed')
  }
  return res.json()
}
```

- [ ] **Step 2: Write UploadPanel component**

`fetal_health_web_app/frontend/src/components/UploadPanel.jsx`:
```jsx
import { useState, useEffect } from 'react'
import { getModels } from '../api/client'

export default function UploadPanel({ onSubmit, isLoading }) {
  const [models, setModels] = useState([])
  const [selectedModel, setSelectedModel] = useState('')
  const [file, setFile] = useState(null)

  useEffect(() => {
    getModels()
      .then(data => {
        setModels(data.models)
        if (data.models.length > 0) setSelectedModel(data.models[0])
      })
      .catch(err => console.error('Could not load models:', err))
  }, [])

  const canRun = file !== null && selectedModel !== '' && !isLoading

  return (
    <div className="upload-panel">
      <h1 className="app-title">Fetal Health Prediction</h1>
      <div className="controls">
        <label className="file-label">
          <input
            type="file"
            accept=".csv"
            onChange={e => setFile(e.target.files[0] ?? null)}
          />
          <span>{file ? file.name : 'Choose CSV file…'}</span>
        </label>

        <select
          value={selectedModel}
          onChange={e => setSelectedModel(e.target.value)}
          disabled={models.length === 0}
        >
          {models.map(m => (
            <option key={m} value={m}>{m}</option>
          ))}
        </select>

        <button
          className="run-btn"
          onClick={() => canRun && onSubmit(file, selectedModel)}
          disabled={!canRun}
        >
          {isLoading ? 'Running…' : 'Run Prediction'}
        </button>
      </div>
      {file && (
        <p className="file-info">
          {file.name} — {(file.size / 1024).toFixed(1)} KB
        </p>
      )}
    </div>
  )
}
```

- [ ] **Step 3: Commit**

```bash
git add fetal_health_web_app/frontend/src/api/client.js fetal_health_web_app/frontend/src/components/UploadPanel.jsx
git commit -m "feat: add API client and UploadPanel component"
```

---

## Task 13: MetadataPanel

**Files:**
- Create: `fetal_health_web_app/frontend/src/components/MetadataPanel.jsx`

- [ ] **Step 1: Write MetadataPanel**

`fetal_health_web_app/frontend/src/components/MetadataPanel.jsx`:
```jsx
function MetaField({ label, value }) {
  const display = value == null ? 'Not available' : String(value)
  return (
    <div className="meta-field">
      <span className="meta-label">{label}</span>
      <span className={`meta-value${value == null ? ' not-available' : ''}`}>{display}</span>
    </div>
  )
}

function MetaSection({ title, children }) {
  return (
    <div className="meta-section">
      <h3 className="meta-section-title">{title}</h3>
      {children}
    </div>
  )
}

export default function MetadataPanel({ metadata }) {
  if (!metadata) {
    return (
      <div className="metadata-panel">
        <p className="meta-placeholder">Upload a recording to see patient details</p>
      </div>
    )
  }

  const { baby, mother, medical } = metadata

  return (
    <div className="metadata-panel">
      <MetaSection title="Baby Details">
        <MetaField label="Baby ID" value={baby.baby_id} />
        <MetaField label="Gestational Age" value={baby.gestational_weeks != null ? `${baby.gestational_weeks} weeks` : null} />
        <MetaField label="Weight" value={baby.weight_g != null ? `${baby.weight_g} g` : null} />
        <MetaField label="Sex" value={baby.sex} />
        <MetaField label="Apgar 1 min" value={baby.apgar1} />
        <MetaField label="Apgar 5 min" value={baby.apgar5} />
      </MetaSection>

      <MetaSection title="Mother Details">
        <MetaField label="Age" value={mother.mother_age != null ? `${mother.mother_age} years` : null} />
        <MetaField label="Gravidity" value={mother.gravidity} />
        <MetaField label="Parity" value={mother.parity} />
        <MetaField label="Diabetes" value={mother.diabetes != null ? (mother.diabetes ? 'Yes' : 'No') : null} />
        <MetaField label="Hypertension" value={mother.hypertension != null ? (mother.hypertension ? 'Yes' : 'No') : null} />
        <MetaField label="Preeclampsia" value={mother.preeclampsia != null ? (mother.preeclampsia ? 'Yes' : 'No') : null} />
      </MetaSection>

      <MetaSection title="Medical / Recording Status">
        <MetaField label="Duration" value={medical.recording_duration_min != null ? `${medical.recording_duration_min} min` : null} />
        <MetaField label="Missing Signal" value={medical.missing_signal_pct != null ? `${medical.missing_signal_pct}%` : null} />
        <MetaField label="FHR Mean" value={medical.fhr_mean != null ? `${medical.fhr_mean} bpm` : null} />
        <MetaField label="FHR Std Dev" value={medical.fhr_std != null ? `${medical.fhr_std} bpm` : null} />
        <MetaField label="UC Activity" value={medical.uc_available != null ? (medical.uc_available ? 'Present' : 'Absent') : null} />
      </MetaSection>
    </div>
  )
}
```

- [ ] **Step 2: Commit**

```bash
git add fetal_health_web_app/frontend/src/components/MetadataPanel.jsx
git commit -m "feat: add MetadataPanel with baby/mother/medical sections"
```

---

## Task 14: BabyVisual

**Files:**
- Create: `fetal_health_web_app/frontend/src/components/BabyVisual.jsx`

- [ ] **Step 1: Write BabyVisual**

`fetal_health_web_app/frontend/src/components/BabyVisual.jsx`:
```jsx
const LABEL_COLOR = {
  healthy: '#22c55e',
  normal: '#22c55e',
  risk: '#ef4444',
  danger: '#ef4444',
  suspicious: '#f59e0b',
}

function getColor(label) {
  if (!label) return '#94a3b8'
  const key = label.toLowerCase()
  for (const [k, v] of Object.entries(LABEL_COLOR)) {
    if (key.includes(k)) return v
  }
  return '#94a3b8'
}

export default function BabyVisual({ prediction }) {
  const color = getColor(prediction?.label)

  return (
    <div className="baby-visual">
      <svg viewBox="0 0 200 320" width="180" height="288" aria-label="Baby silhouette">
        {/* Head */}
        <ellipse cx="100" cy="52" rx="36" ry="40" fill="none" stroke={color} strokeWidth="3" />
        {/* Body */}
        <ellipse cx="100" cy="168" rx="52" ry="72" fill="none" stroke={color} strokeWidth="3" />
        {/* Left arm */}
        <path d="M50 135 Q22 165 18 195" fill="none" stroke={color} strokeWidth="3" strokeLinecap="round" />
        {/* Right arm */}
        <path d="M150 135 Q178 165 182 195" fill="none" stroke={color} strokeWidth="3" strokeLinecap="round" />
        {/* Left leg */}
        <path d="M80 235 Q70 270 65 300" fill="none" stroke={color} strokeWidth="3" strokeLinecap="round" />
        {/* Right leg */}
        <path d="M120 235 Q130 270 135 300" fill="none" stroke={color} strokeWidth="3" strokeLinecap="round" />
      </svg>

      {prediction ? (
        <div className="prediction-badge" style={{ borderColor: color, color }}>
          <span className="prediction-label">{prediction.label}</span>
          {prediction.confidence != null && (
            <span className="prediction-confidence">
              {(prediction.confidence * 100).toFixed(1)}% confidence
            </span>
          )}
        </div>
      ) : (
        <div className="prediction-badge prediction-badge--empty">
          <span className="prediction-label">Awaiting prediction</span>
        </div>
      )}
    </div>
  )
}
```

- [ ] **Step 2: Commit**

```bash
git add fetal_health_web_app/frontend/src/components/BabyVisual.jsx
git commit -m "feat: add BabyVisual SVG component with color-coded prediction"
```

---

## Task 15: ExplanationPanel, LoadingOverlay, ErrorBanner

**Files:**
- Create: `fetal_health_web_app/frontend/src/components/ExplanationPanel.jsx`
- Create: `fetal_health_web_app/frontend/src/components/LoadingOverlay.jsx`
- Create: `fetal_health_web_app/frontend/src/components/ErrorBanner.jsx`

- [ ] **Step 1: Write ExplanationPanel**

`fetal_health_web_app/frontend/src/components/ExplanationPanel.jsx`:
```jsx
const IMPACT = {
  normal:   { icon: '✓', cls: 'impact-normal' },
  elevated: { icon: '⚠', cls: 'impact-elevated' },
  critical: { icon: '✗', cls: 'impact-critical' },
}

export default function ExplanationPanel({ explanation }) {
  if (!explanation) return null

  return (
    <div className="explanation-panel">
      <h3 className="explanation-title">Why this result?</h3>
      <table className="params-table">
        <thead>
          <tr>
            <th>Parameter</th>
            <th>Value</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          {explanation.important_parameters.map((p, i) => {
            const { icon, cls } = IMPACT[p.impact] ?? IMPACT.normal
            return (
              <tr key={i}>
                <td>{p.name}</td>
                <td>{p.value}</td>
                <td className={cls}>{icon}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
      <p className="explanation-summary">{explanation.summary}</p>
    </div>
  )
}
```

- [ ] **Step 2: Write LoadingOverlay**

`fetal_health_web_app/frontend/src/components/LoadingOverlay.jsx`:
```jsx
export default function LoadingOverlay({ visible }) {
  if (!visible) return null
  return (
    <div className="loading-overlay" role="status" aria-label="Loading">
      <div className="spinner" />
      <p>Running prediction…</p>
    </div>
  )
}
```

- [ ] **Step 3: Write ErrorBanner**

`fetal_health_web_app/frontend/src/components/ErrorBanner.jsx`:
```jsx
export default function ErrorBanner({ error, onDismiss }) {
  if (!error) return null
  return (
    <div className="error-banner" role="alert">
      <span>{error}</span>
      <button className="dismiss-btn" onClick={onDismiss} aria-label="Dismiss">×</button>
    </div>
  )
}
```

- [ ] **Step 4: Commit**

```bash
git add fetal_health_web_app/frontend/src/components/ExplanationPanel.jsx fetal_health_web_app/frontend/src/components/LoadingOverlay.jsx fetal_health_web_app/frontend/src/components/ErrorBanner.jsx
git commit -m "feat: add ExplanationPanel, LoadingOverlay, and ErrorBanner components"
```

---

## Task 16: App.jsx — State Machine and Layout Wiring

**Files:**
- Create: `fetal_health_web_app/frontend/src/App.jsx`

- [ ] **Step 1: Write App.jsx**

`fetal_health_web_app/frontend/src/App.jsx`:
```jsx
import { useState } from 'react'
import UploadPanel from './components/UploadPanel'
import BabyVisual from './components/BabyVisual'
import ExplanationPanel from './components/ExplanationPanel'
import MetadataPanel from './components/MetadataPanel'
import LoadingOverlay from './components/LoadingOverlay'
import ErrorBanner from './components/ErrorBanner'
import { runPrediction } from './api/client'

// States: 'idle' | 'loading' | 'result' | 'error'

export default function App() {
  const [appState, setAppState] = useState('idle')
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  async function handleSubmit(file, modelName) {
    setAppState('loading')
    setError(null)
    setResult(null)
    try {
      const data = await runPrediction(file, modelName)
      setResult(data)
      setAppState('result')
    } catch (err) {
      setError(err.message)
      setAppState('error')
    }
  }

  function handleDismissError() {
    setError(null)
    setAppState('idle')
  }

  const prediction = result?.prediction ?? null
  const metadata = result?.metadata ?? null
  const explanation = result?.explanation ?? null

  return (
    <div className="app">
      <LoadingOverlay visible={appState === 'loading'} />

      <UploadPanel onSubmit={handleSubmit} isLoading={appState === 'loading'} />

      <ErrorBanner error={error} onDismiss={handleDismissError} />

      <div className="dashboard">
        <div className="left-panel">
          <BabyVisual prediction={prediction} />
          {explanation && <ExplanationPanel explanation={explanation} />}
        </div>
        <div className="right-panel">
          <MetadataPanel metadata={metadata} />
        </div>
      </div>

      <footer className="disclaimer">
        ⚠ This tool is for research and decision-support only and is not a medical diagnosis.
      </footer>
    </div>
  )
}
```

- [ ] **Step 2: Verify the app renders**

With both servers running (`uvicorn` on :8000, `npm run dev` on :5173), open `http://localhost:5173`.
Expected:
- Header with title, file picker, model dropdown, disabled Run button
- Empty baby silhouette on the left
- "Upload a recording..." placeholder on the right
- Disclaimer footer

- [ ] **Step 3: Commit**

```bash
git add fetal_health_web_app/frontend/src/App.jsx
git commit -m "feat: wire App state machine connecting all components"
```

---

## Task 17: CSS Styling

**Files:**
- Create: `fetal_health_web_app/frontend/src/App.css`

- [ ] **Step 1: Write App.css**

`fetal_health_web_app/frontend/src/App.css`:
```css
/* ── Reset & Base ─────────────────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
  background: #f0f4f8;
  color: #1e293b;
  min-height: 100vh;
}

/* ── App shell ────────────────────────────────────────────────────────────── */
.app {
  display: flex;
  flex-direction: column;
  min-height: 100vh;
}

/* ── Upload panel ─────────────────────────────────────────────────────────── */
.upload-panel {
  background: #ffffff;
  padding: 1.25rem 2rem;
  border-bottom: 1px solid #e2e8f0;
  box-shadow: 0 1px 4px rgba(0, 0, 0, 0.07);
}
.app-title {
  font-size: 1.4rem;
  font-weight: 700;
  color: #0f172a;
  margin-bottom: 0.875rem;
}
.controls {
  display: flex;
  gap: 0.75rem;
  align-items: center;
  flex-wrap: wrap;
}
.file-label {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  padding: 0.45rem 1rem;
  border: 2px dashed #94a3b8;
  border-radius: 8px;
  cursor: pointer;
  color: #64748b;
  font-size: 0.9rem;
  transition: border-color 0.15s, color 0.15s;
}
.file-label:hover { border-color: #3b82f6; color: #3b82f6; }
.file-label input[type="file"] { display: none; }
select {
  padding: 0.45rem 0.75rem;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  font-size: 0.9rem;
  color: #1e293b;
  background: #fff;
  cursor: pointer;
}
.run-btn {
  padding: 0.45rem 1.75rem;
  background: #3b82f6;
  color: #fff;
  border: none;
  border-radius: 8px;
  font-size: 0.9rem;
  font-weight: 600;
  cursor: pointer;
  transition: background 0.15s;
}
.run-btn:hover:not(:disabled) { background: #2563eb; }
.run-btn:disabled { background: #94a3b8; cursor: not-allowed; }
.file-info { margin-top: 0.5rem; font-size: 0.78rem; color: #64748b; }

/* ── Dashboard layout ─────────────────────────────────────────────────────── */
.dashboard {
  display: flex;
  flex: 1;
  gap: 1.25rem;
  padding: 1.25rem 2rem;
  max-width: 1400px;
  margin: 0 auto;
  width: 100%;
}
.left-panel  { flex: 6; display: flex; flex-direction: column; gap: 1rem; }
.right-panel { flex: 4; }

/* ── Baby visual ──────────────────────────────────────────────────────────── */
.baby-visual {
  background: #fff;
  border-radius: 14px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.07);
  padding: 2rem;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 1.25rem;
}
.prediction-badge {
  border: 3px solid #94a3b8;
  border-radius: 12px;
  padding: 0.875rem 2.5rem;
  text-align: center;
  transition: border-color 0.3s, color 0.3s;
}
.prediction-badge--empty { border-color: #e2e8f0; color: #94a3b8; }
.prediction-label       { display: block; font-size: 1.9rem; font-weight: 700; }
.prediction-confidence  { display: block; font-size: 0.95rem; margin-top: 0.2rem; opacity: 0.8; }

/* ── Explanation panel ────────────────────────────────────────────────────── */
.explanation-panel {
  background: #fff;
  border-radius: 14px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.07);
  padding: 1.5rem;
}
.explanation-title {
  font-size: 0.95rem;
  font-weight: 600;
  color: #475569;
  margin-bottom: 1rem;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}
.params-table { width: 100%; border-collapse: collapse; }
.params-table th {
  text-align: left;
  padding: 0.4rem 0.5rem;
  font-size: 0.78rem;
  color: #94a3b8;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  border-bottom: 1px solid #e2e8f0;
}
.params-table td {
  padding: 0.45rem 0.5rem;
  font-size: 0.88rem;
  border-bottom: 1px solid #f8fafc;
}
.impact-normal   { color: #22c55e; font-weight: 700; font-size: 1rem; }
.impact-elevated { color: #f59e0b; font-weight: 700; font-size: 1rem; }
.impact-critical { color: #ef4444; font-weight: 700; font-size: 1rem; }
.explanation-summary {
  margin-top: 1rem;
  font-size: 0.85rem;
  color: #64748b;
  line-height: 1.6;
}

/* ── Metadata panel ───────────────────────────────────────────────────────── */
.metadata-panel {
  background: #fff;
  border-radius: 14px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.07);
  padding: 1.5rem;
  height: 100%;
}
.meta-placeholder { color: #94a3b8; padding: 2rem; text-align: center; }
.meta-section     { margin-bottom: 1.5rem; }
.meta-section:last-child { margin-bottom: 0; }
.meta-section-title {
  font-size: 0.75rem;
  font-weight: 700;
  color: #3b82f6;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  padding-bottom: 0.4rem;
  border-bottom: 2px solid #eff6ff;
  margin-bottom: 0.6rem;
}
.meta-field {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  padding: 0.22rem 0;
  font-size: 0.88rem;
  gap: 0.5rem;
}
.meta-label       { color: #64748b; flex-shrink: 0; }
.meta-value       { font-weight: 500; text-align: right; }
.meta-value.not-available { color: #94a3b8; font-weight: 400; font-style: italic; }

/* ── Loading overlay ──────────────────────────────────────────────────────── */
.loading-overlay {
  position: fixed;
  inset: 0;
  background: rgba(255, 255, 255, 0.82);
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  z-index: 200;
  gap: 1rem;
  font-size: 1rem;
  color: #475569;
}
.spinner {
  width: 52px;
  height: 52px;
  border: 5px solid #e2e8f0;
  border-top-color: #3b82f6;
  border-radius: 50%;
  animation: spin 0.75s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }

/* ── Error banner ─────────────────────────────────────────────────────────── */
.error-banner {
  background: #fef2f2;
  border-top: 1px solid #fecaca;
  border-bottom: 1px solid #fecaca;
  color: #dc2626;
  padding: 0.75rem 2rem;
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 0.9rem;
}
.dismiss-btn {
  background: none;
  border: none;
  font-size: 1.3rem;
  cursor: pointer;
  color: #dc2626;
  line-height: 1;
  padding: 0 0.25rem;
}

/* ── Disclaimer footer ────────────────────────────────────────────────────── */
.disclaimer {
  text-align: center;
  padding: 0.875rem;
  font-size: 0.78rem;
  color: #94a3b8;
  border-top: 1px solid #e2e8f0;
  background: #fff;
  margin-top: auto;
}

/* ── Responsive ───────────────────────────────────────────────────────────── */
@media (max-width: 768px) {
  .dashboard { flex-direction: column; padding: 1rem; }
  .left-panel, .right-panel { flex: none; width: 100%; }
}
```

- [ ] **Step 2: Test full golden path in browser**

With both servers running:
1. Open `http://localhost:5173`
2. Upload `csv_output/1001.csv`
3. Select `binarycnn` from the dropdown
4. Click "Run Prediction"
5. Verify:
   - Loading spinner appears
   - Baby silhouette turns green (Healthy) or red (Risk) with confidence %
   - Right panel shows baby/mother metadata from `1001.hea` (gestational age, weight, etc.)
   - Explanation table appears with ✓/⚠/✗ icons
   - Disclaimer footer visible

- [ ] **Step 3: Test edge cases**

- Upload a CSV with no FHR column → red error banner appears, can be dismissed
- Select `minirocket` → prediction runs and returns structure
- If `1001.csv` is uploaded (filename matches `1001.hea`), metadata fields should be populated, not "Not available"
- If a CSV without a matching `.hea` file is uploaded, all metadata fields show "Not available" gracefully

- [ ] **Step 4: Commit**

```bash
git add fetal_health_web_app/frontend/src/App.css
git commit -m "feat: add full dashboard CSS styling"
```

---

## Task 18: README

**Files:**
- Create: `fetal_health_web_app/README.md`

- [ ] **Step 1: Write README**

`fetal_health_web_app/README.md`:
```markdown
# Fetal Health Prediction Web App

A research-grade web dashboard for fetal CTG analysis. Upload a recording CSV, select a model, and view the prediction with extracted metadata and explanations.

> **Disclaimer:** This tool is for research and decision-support only and is not a medical diagnosis.

---

## Prerequisites

- Python 3.10+
- Node.js 18+
- pip

---

## Setup

### 1. Install backend dependencies

```bash
cd fetal_health_web_app/backend
pip install -r requirements.txt
```

### 2. Train models (first time only)

Run from the **project root** (`SavingBabies/`):

```bash
# BinaryCNN (~10–30 min depending on hardware)
python fetal_health_web_app/backend/training/train_binarycnn.py

# MiniROCKET (~5–20 min — CPU-bound)
python fetal_health_web_app/backend/training/train_minirocket.py
```

Both scripts save weights to `fetal_health_web_app/backend/weights/`.

### 3. Start the backend

```bash
cd fetal_health_web_app/backend
uvicorn main:app --reload --port 8000
```

### 4. Install and start the frontend

```bash
cd fetal_health_web_app/frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

---

## Usage

1. Upload a CTG recording CSV (`t_sec, FHR, UC` columns). Max duration: 90 minutes.
2. Select a model from the dropdown.
3. Click **Run Prediction**.
4. The dashboard shows:
   - **Left:** Baby silhouette with prediction label + confidence, and a feature explanation table.
   - **Right:** Patient metadata (baby, mother, medical). If the CSV filename matches a record in `dataset/` (e.g., `1001.csv` → `1001.hea`), full metadata is shown; otherwise fields show "Not available".

---

## Retraining a Model

1. Run the training script (see step 2 above).
2. Restart the FastAPI server — it reloads weights on startup.

---

## Adding a New Model

1. Create `backend/models/my_model_adapter.py` implementing `BaseModelAdapter`.
2. Create `backend/training/train_mymodel.py` saving weights to `backend/weights/`.
3. In `backend/models/registry.py`, add:
   ```python
   from models.my_model_adapter import MyModelAdapter
   MODEL_REGISTRY["mymodel"] = MyModelAdapter()
   ```
4. The dropdown updates automatically — no frontend changes needed.

---

## Running Tests

```bash
cd fetal_health_web_app/backend
pytest -v
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `DATASET_DIR` | `../../dataset` | Path to directory containing `.hea` files |
| `CSV_DIR` | `csv_output` | Path to CSV files (training scripts only) |
| `PH_FILE` | `dataset/ph_levels.csv` | pH labels file (training scripts only) |
```

- [ ] **Step 2: Commit**

```bash
git add fetal_health_web_app/README.md
git commit -m "docs: add README with setup, usage, and model-adding instructions"
```

---

## Self-Review Checklist

### Spec Coverage
- [x] CSV-only upload with 90-min validation → `file_parser.py` Task 4
- [x] `.hea` metadata matching by filename, graceful fallback → `hea_parser.py` Task 3, `main.py` Task 8
- [x] Generic `BaseModelAdapter` + `MODEL_REGISTRY` → Tasks 5–7
- [x] MiniROCKET adapter (deterministic kernels, `joblib`) → Task 7
- [x] BinaryCNN adapter (PyTorch, threshold from stats.json) → Task 6
- [x] Training scripts that save weights → Tasks 9–10
- [x] `GET /models` → dropdown populated from backend → Task 8, Task 12
- [x] `POST /predict` → structured `PredictionResponse` → Tasks 2, 8
- [x] Baby silhouette with color-coded label → Task 14
- [x] Right-side metadata panel (Baby / Mother / Medical) → Task 13
- [x] Explanation table with impact icons → Task 15
- [x] Loading overlay, error banner, disclaimer → Tasks 15, 16
- [x] Medical disclaimer footer → Task 16
- [x] React + Vite frontend → Task 11
- [x] Proxy `/api/*` → `:8000` → Task 11
- [x] README with setup instructions → Task 18

### Type Consistency
- `BinaryCNNAdapter.name = "BinaryCNN"` → used as `model_name` in response ✓
- `MiniRocketAdapter.name = "MiniROCKET"` → used as `model_name` in response ✓
- `impact` values: `"normal"` / `"elevated"` / `"critical"` used consistently in adapters and CSS classes ✓
- `_last_raw_seq` pattern used identically in both adapters ✓
- `SEQ_LEN = 4800` in `binarycnn_adapter.py` and `train_binarycnn.py` ✓
- `SEQ_LEN = 1200` in `minirocket_adapter.py` and `train_minirocket.py` ✓
- `RANDOM_STATE = 42`, `N_KERNELS = 4000`, `KERNEL_MIN/MAX`, `DILATION_MAX` identical in adapter and training script ✓
