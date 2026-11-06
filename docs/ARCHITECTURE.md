# Maternalyze — Architecture

<!--
Companion to PRD.md.
PRD says WHAT the system does. This says HOW.
Audience: an engineer who needs to understand the system well
enough to build it, debug it, or extend it.
No marketing language here — be precise.
-->

---

## 1. Stack

| Layer | Tech |
|---|---|
| Framework | FastAPI (Python 3.11, uvicorn) |
| ML Models | LightGBM — pre-trained binary classifiers, loaded via joblib at startup |
| Data Processing | pandas · NumPy · scikit-learn (preprocessing, encoding, oversampling) |
| Frontend | Vanilla HTML · CSS · JavaScript |
| Charts | Chart.js (loaded via CDN) |
| Hosting | Render — single web service, auto-deploy on push to `main` |

---

## 2. Components

```
Maternalyze/
  backend/          FastAPI server, ML inference endpoints, categorical encoders
  frontend/         Static HTML pages, form inputs, Chart.js visualizations
  models/           Pre-trained LightGBM .joblib model files (loaded at startup)
  data/raw/         Training dataset (gdm_dataset.csv — offline use only)
  notebooks/        Model training scripts (offline, not part of production serving)
```

### backend/app.py

The single FastAPI application. Mounts the `frontend/` directory as a static file route, loads both `.joblib` models at startup, and exposes two POST prediction endpoints. Static HTML pages are served via a `GET /{file_name}` catch-all route. Does not connect to any database — every request is stateless.

### backend/model_utils.py

Categorical encoding dictionaries and two preprocessing functions (`preprocess_gdm_input`, `preprocess_child_input`) that convert Pydantic model instances into NumPy arrays ready for LightGBM inference. Also contains `get_gdm_precautions` and `get_child_precautions` — deterministic lookup functions that return fixed clinical recommendation lists based on the binary prediction label.

### frontend/

Three HTML pages:
- `index.html` — landing page with navigation buttons and a sample Chart.js bar chart.
- `gdm_prediction.html` — 7-field form; on submit POSTs to `/predict_gdm` and renders a doughnut chart.
- `child_prediction.html` — 45-field form; on submit POSTs to `/predict_child` and renders a feature-importance bar chart.

`config.js` resolves the API base URL: `window.location.origin` when running on Render, `http://127.0.0.1:8000` locally. `script.js` handles both form submissions via `fetch`, destroys and recreates Chart.js instances on each prediction, and renders results as an HTML table.

### models/

Two pre-trained LightGBM binary classifiers serialized with joblib:
- `gdm_classifier_small.joblib` — 7-feature GDM model, trained with RandomOverSampler for class balance.
- `child_outcome_model_full.joblib` — 45-feature child outcome (SCBU admission) model.

Both are loaded once at server startup via `joblib.load()`.

### notebooks/

Offline training scripts not shipped to production. `preprocess_and_train_gdm.py` trains the GDM model with early stopping (2000 rounds, patience 100). `preprocess_and_train_child.py` trains the child outcome model with early stopping (patience 20). Both save the final model to `../models/`.

---

## 3. Data Flow

```
[Browser] ──GET /──────────────────────────────────────────────► [FastAPI]
                                                                    │
                                                        serves index.html

[User selects prediction type]
    │
    ▼
[Browser loads gdm_prediction.html or child_prediction.html]
    │
[User fills form and submits]
    │
    ▼
POST /predict_gdm  or  POST /predict_child
    │
    ▼
[FastAPI] ── Pydantic validation ──► preprocess_*_input()
                                          │ categorical dict lookups
                                          │ numpy array reshape(1, -1)
                                          ▼
                                     LightGBM model.predict()
                                          │ probability float
                                          ▼
                                     get_*_precautions(label)
                                          │
                                          ▼
                              JSON response: { Prediction, Probability,
                                              Precautions, [TopFeatureImportances] }
    │
    ▼
[Browser] renders result table + Chart.js doughnut / bar chart
```

1. Browser requests the root URL; FastAPI serves `index.html`.
2. User navigates to a prediction page and fills in the clinical form.
3. `script.js` intercepts form submission, serializes to JSON, POSTs to the API.
4. FastAPI validates the request body against the Pydantic model (`GDMInput` or `ChildInput`).
5. `preprocess_*_input()` applies categorical mappings and returns a `(1, N)` NumPy array.
6. The pre-loaded LightGBM model calls `.predict()` and returns a probability float.
7. A binary label is derived (`> 0.5`), and `get_*_precautions()` returns a fixed recommendation list.
8. For child outcome, feature importances are extracted via `.feature_importance(importance_type='gain')` and the top 5 are returned by name.
9. The JSON response is rendered as an HTML table and Chart.js chart in the browser.

---

## 4. Database Schema

There is no database. Both models are pre-trained and loaded from `.joblib` files at startup. Predictions are stateless — no request is persisted.

---

## 5. ML Model Design

### GDM Model

**Input:** 7 features — `AgeAtStartOfSpell`, `WeightMeasured`, `Height`, `BodyMassIndexAtBooking`, `Obese` (binary), `Ethnicity` (integer-encoded), `Glucoselevelblood`.

**Training:**
- LightGBM GBDT, binary classification, AUC + binary_logloss metrics.
- `RandomOverSampler` applied to the training set to address GDM class imbalance.
- Early stopping: 2000 rounds max, patience 100.
- Hyperparameters: `num_leaves=31`, `learning_rate=0.05`, `feature_fraction=0.9`, `bagging_fraction=0.8`.

**Output:** Probability float. Label derived as `int(prob > 0.5)`.

### Child Outcome Model

**Input:** 45 features spanning the full maternity care pathway: maternal demographics, antenatal indicators (blood pressure, vitamin D, thyroid), delivery details (onset of labour, Caesarean category, blood loss), and neonatal outcomes (APGAR scores, birth weight centile, sex, gestational age).

**Target:** `SCBU_adm` — whether the newborn required Special Care Baby Unit admission.

**Training:**
- LightGBM GBDT, binary classification, AUC metric.
- Early stopping: patience 20.
- Categorical fields encoded via `pd.factorize()` during training; production uses hand-coded dictionary maps for consistency.

**Output:** Probability float, binary label, and top-5 feature importances by gain.

### Failure Handling

- Both model files are loaded at server startup; if a file is missing, the server fails to start.
- A `try/except` in each endpoint returns `HTTP 400` with the exception message if preprocessing or inference fails.
- No retry logic — errors surface directly to the client.

---

## 6. API Routes

| Method | Route | Description |
|---|---|---|
| `GET` | `/` | Serves `index.html` |
| `GET` | `/{file_name}` | Serves any `.html`, `.css`, `.js`, or `.ico` file from the frontend directory |
| `POST` | `/predict_gdm` | GDM risk prediction — accepts `GDMInput`, returns prediction, probability, precautions |
| `POST` | `/predict_child` | Child outcome prediction — accepts `ChildInput`, returns prediction, probability, precautions, top-5 feature importances |

Interactive API docs available at `/docs` (Swagger UI) and `/redoc`.

---

## 7. Security

- **API keys:** None required — no third-party AI provider or cloud database. No secrets in the codebase.
- **CORS:** Currently set to `allow_origins=["*"]` — sufficient for a demo; should be restricted to the Render origin in production.
- **User data:** No user input is stored. Each prediction is entirely stateless.
- **Input validation:** Pydantic models enforce field types and reject malformed requests before they reach model inference with `422 Unprocessable Entity`.

---

## 8. Error Handling & Reliability

| Failure | Behaviour |
|---|---|
| Missing model file at startup | Server fails to start — `joblib.load()` raises `FileNotFoundError` |
| Invalid or missing form field | Pydantic returns `422 Unprocessable Entity` before inference |
| Unexpected inference error | `HTTP 400` returned with exception detail string |
| Unknown categorical value | Dictionary `.get(value, default)` falls back to `"Unknown"` encoding gracefully |
| Render cold start (free tier) | First request after inactivity may take 30–60s; subsequent requests are fast |

---

## 9. Deployment

1. Render web service linked to the GitHub repository — auto-deploy on push to `main`.
2. Build command: `pip install -r backend/requirements.txt`.
3. Start command: `cd backend && uvicorn app:app --host 0.0.0.0 --port $PORT`.
4. Python version pinned to `3.11.0` via `render.yaml` environment variable (`PYTHON_VERSION`).
5. No database migrations — no database.
6. Model `.joblib` files are committed to the repository and loaded at startup.

---

## 10. Explicit Scope Cuts

- **User authentication** — predictions are stateless; no user accounts or session management in v1.
- **Patient record storage** — no database; no history or audit log in v1.
- **EHR/FHIR integration** — standalone form-based input only; institutional API access not available.
- **Model retraining in production** — offline training workflow only; raw data is not served.
- **Batch prediction** — single-patient, single-request only in v1.
