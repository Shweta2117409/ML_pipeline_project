# Heart Disease ML Pipeline

A production-style, end-to-end machine learning system that predicts heart disease from clinical features, served via a FastAPI REST API with a polished HTML frontend.

---

## Project Overview

This project demonstrates a complete ML engineering workflow:

```
Raw CSV → Validate → Preprocess → Train → Evaluate → Save → Serve
```

The system trains two models (Decision Tree baseline and Random Forest), selects the best by F1 score, persists it with `joblib`, and exposes it through a REST API.

---

## Folder Structure

```
ml_pipeline_project/
├── data/
│   └── raw.csv                  # Raw heart disease dataset (CSV)
│
├── pipeline/
│   ├── __init__.py
│   ├── data_ingestion.py        # Load CSV → DataFrame
│   ├── data_validation.py       # Schema, nulls, duplicate checks
│   ├── data_preprocessing.py    # Imputation, IQR outlier capping, feature split
│   └── model_trainer.py         # Train, evaluate, and select best model
│
├── api/
│   ├── __init__.py
│   └── api.py                   # FastAPI application (GET / + POST /predict)
│
├── frontend/
│   └── index.html               # HTML + JS interface for predictions
│
├── notebooks/
│   └── eda.ipynb                # Exploratory data analysis notebook
│
├── reports/
│   └── before_vs_after.md       # Baseline vs improved model analysis
│
├── train_model.py               # End-to-end training entrypoint
├── model.pkl                    # Saved best model (generated after training)
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## Pipeline Architecture

```
┌─────────────────┐
│  data/raw.csv   │
└────────┬────────┘
         │
         ▼
┌─────────────────────┐
│  data_ingestion.py  │  ← Loads CSV, checks existence & emptiness
└────────┬────────────┘
         │
         ▼
┌──────────────────────┐
│  data_validation.py  │  ← Schema, missing values, duplicates, target integrity
└────────┬─────────────┘
         │
         ▼
┌────────────────────────┐
│ data_preprocessing.py  │  ← Impute nulls, coerce types, IQR outlier capping
└────────┬───────────────┘
         │
         ▼
┌─────────────────────┐
│  model_trainer.py   │  ← Train DecisionTree + RandomForest, evaluate, select best
└────────┬────────────┘
         │
         ▼
┌──────────────────┐
│   model.pkl      │  ← Serialized best model
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   api/api.py     │  ← FastAPI loads model.pkl, exposes POST /predict
└────────┬─────────┘
         │
         ▼
┌──────────────────────┐
│  frontend/index.html │  ← HTML form → fetch /predict → display result
└──────────────────────┘
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
python train_model.py
```

This runs the full pipeline and saves `model.pkl`.

### 3. Start the API server (and optional frontend from same server)

From the **project root**:

```bash
uvicorn api.api:app --reload
```

- **Frontend (one terminal):** Open **http://127.0.0.1:8000/** in your browser. The same server serves both the API and the UI.
- **API only:** The API is at `http://127.0.0.1:8000`. Interactive docs: **http://127.0.0.1:8000/docs**
- **Frontend as file:** You can still open `frontend/index.html` directly in your browser; it will call the API at `http://127.0.0.1:8000` if the server is running.

---

## API Reference

### `GET /`
Serves the frontend UI (single-page app). Use **http://127.0.0.1:8000/** to run frontend and backend together.

### `GET /welcome`
Returns a welcome message.

```json
{ "message": "Heart Disease Prediction API" }
```

### `GET /health`
Returns model status.

### `POST /predict`
**Request body:**
```json
{
  "age": 63,
  "sex": 1,
  "cp": 3,
  "trestbps": 145,
  "chol": 233,
  "fbs": 1,
  "restecg": 0,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 2.3,
  "slope": 0,
  "ca": 0,
  "thal": 1
}
```

**Response:**
```json
{
  "prediction": 1,
  "confidence": 0.87
}
```

---

## Feature Descriptions

| Feature | Description |
|---|---|
| age | Age in years |
| sex | Sex (1 = male, 0 = female) |
| cp | Chest pain type (0–3) |
| trestbps | Resting blood pressure (mm Hg) |
| chol | Serum cholesterol (mg/dl) |
| fbs | Fasting blood sugar > 120 mg/dl (1 = true) |
| restecg | Resting ECG results (0–2) |
| thalach | Maximum heart rate achieved |
| exang | Exercise-induced angina (1 = yes) |
| oldpeak | ST depression induced by exercise |
| slope | Slope of peak exercise ST segment (0–2) |
| ca | Number of major vessels coloured by fluoroscopy (0–4) |
| thal | Thalassemia type (0–3) |
| target | Heart disease present (1) or absent (0) |
