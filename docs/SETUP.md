# Local Setup — Maternalyze

> **Just want to try it?** Use the live demo at [maternalyze.tanisheesh.in](https://maternalyze.tanisheesh.in) — no setup needed.
> This guide is for running Maternalyze locally or self-hosting it.

---

## Prerequisites

- Python 3.11+
- pip (comes with Python)
- Git

No Node.js, no Docker, no database required.

---

## 1. Clone and install

```bash
git clone https://github.com/tanisheesh/maternalyze
cd maternalyze
pip install -r backend/requirements.txt
```

Key dependencies installed: `fastapi`, `uvicorn[standard]`, `lightgbm>=4.1.0`, `scikit-learn>=1.3.0`, `pandas>=2.1.0`, `numpy>=1.26.0`, `joblib>=1.3.0`.

---

## 2. Environment variables

None required. Maternalyze has no external API keys, no database, and no third-party service credentials. The only runtime configuration is the API base URL, which is resolved automatically in `frontend/config.js`:

- On Render (`*.onrender.com`): uses `window.location.origin`
- Locally: falls back to `http://127.0.0.1:8000`

---

## 3. Models

The pre-trained LightGBM model files are committed to the repository and loaded at startup:

```
models/
  gdm_classifier_small.joblib       7-feature GDM classifier
  child_outcome_model_full.joblib   45-feature SCBU admission classifier
```

If you want to retrain the models from scratch, place `gdm_dataset.csv` in `data/raw/` and run:

```bash
python notebooks/preprocess_and_train_gdm.py
python notebooks/preprocess_and_train_child.py
```

Both scripts require `imblearn` (`pip install imbalanced-learn`) in addition to the standard requirements.

---

## 4. Run locally

```bash
cd backend
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

Maternalyze will be running at `http://127.0.0.1:8000`.

- Landing page: `http://127.0.0.1:8000/`
- GDM form: `http://127.0.0.1:8000/gdm_prediction.html`
- Child outcome form: `http://127.0.0.1:8000/child_prediction.html`
- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

---

## 5. Deploy to production

Render is the reference deployment target. The `render.yaml` and `Procfile` at the project root contain the full configuration.

**Manual Render setup:**

1. Create a new **Web Service** on [render.com](https://render.com) linked to this repository.
2. Set **Build Command:** `pip install -r backend/requirements.txt`
3. Set **Start Command:** `cd backend && uvicorn app:app --host 0.0.0.0 --port $PORT`
4. Set environment variable `PYTHON_VERSION=3.11.0`.
5. Push to `main` — Render auto-deploys on every push.

No database provisioning, no secret injection, no migration step required.

---

## Known local-only limitations

- **Cold start on Render free tier** — the service sleeps after 15 minutes of inactivity. The first request after sleep takes 30–60 seconds. This does not affect local runs.
- **No HTTPS locally** — `http://127.0.0.1:8000` only. Render handles TLS termination automatically in production.
- **Chart.js loaded via CDN** — `index.html` and the prediction pages load Chart.js from `cdn.jsdelivr.net`. If you are running offline, the charts will not render; the prediction results table still displays correctly.
