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
