<p align="center">
  <img src="frontend/favicon.ico" width="64" height="64" alt="Maternalyze">
</p>

<h1 align="center">Maternalyze</h1>

<p align="center">
  <strong>ML-powered pregnancy risk assessment — GDM screening and child outcome prediction for clinical support.</strong>
</p>

<p align="center">
  <a href="https://maternalyze.tanisheesh.in">
    <img src="https://img.shields.io/badge/live_demo-maternalyze.tanisheesh.in-36a2eb?style=flat-square" alt="Live Demo">
  </a>
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/LightGBM-00C853?style=flat-square" alt="LightGBM">
  <img src="https://img.shields.io/badge/JavaScript-F7DF1E?style=flat-square&logo=javascript&logoColor=black" alt="JavaScript">
  <img src="https://img.shields.io/badge/Render-46E3B7?style=flat-square&logo=render&logoColor=white" alt="Render">
  <img src="https://img.shields.io/badge/license-GPL--3.0-36a2eb?style=flat-square" alt="License">
</p>

---

## What is Maternalyze?

Maternalyze is a web-based clinical decision-support tool that gives healthcare professionals instant, ML-powered risk scores for two of the most consequential pregnancy outcomes: Gestational Diabetes Mellitus (GDM) and newborn SCBU (Special Care Baby Unit) admission. A clinician fills in a structured form, submits it, and within milliseconds receives a binary risk prediction, the probability score, and evidence-based precautionary steps — no specialist ML knowledge required. Where existing screening tools rely on fixed threshold rules, Maternalyze uses LightGBM gradient-boosted models trained on real maternity data, and for child outcome predictions it surfaces the top-5 driving features so the result is auditable, not a black box.

> **Live demo →** [maternalyze.tanisheesh.in](https://maternalyze.tanisheesh.in)

---

## What you get

- **GDM Risk Screening** — enter 7 clinical inputs (age, weight, BMI, blood glucose, ethnicity) and receive a binary risk label, probability score, and dietary/monitoring recommendations in under a second.
- **Child Outcome Prediction** — enter 45 maternal and delivery indicators to predict SCBU admission risk, with top-5 feature importances shown as a bar chart so the model's reasoning is transparent.
- **Interactive Probability Charts** — doughnut chart for GDM risk and bar chart for feature importance, rendered client-side with Chart.js — no page reload required.
- **Clinical Precautions** — every prediction returns a list of actionable clinical steps tailored to the predicted risk level.

---

## Stack

| Layer | Tech |
|---|---|
| Frontend | Vanilla HTML · CSS · JavaScript · Chart.js |
| Backend | Python 3.11 · FastAPI · uvicorn |
| ML | LightGBM · scikit-learn · pandas · NumPy · joblib |
| Infra | Render (single web service, auto-deploy from GitHub) |

---

## Engineering Decisions

**Why LightGBM over logistic regression or a neural network?**
LightGBM achieves strong AUC on tabular clinical data without feature scaling, handles missing categorical values natively, and serializes to a small `.joblib` file — making it a practical fit for a single-server deployment without a GPU or heavy infrastructure.

**Why pre-trained model files rather than training on demand?**
Training requires the raw dataset which is not served in production. Loading a pre-trained `.joblib` at startup keeps the server stateless and cold-start fast. The tradeoff is that the model becomes stale unless manually retrained and redeployed.

**Why vanilla HTML/JS instead of React or Next.js?**
Three form pages with Chart.js visualizations do not justify a SPA framework and a build pipeline. FastAPI serves the static files directly from the same process — zero npm dependencies, one Render service, simpler deployment.

**What would you do differently in v2?**
Restrict CORS from `"*"` to the known Render origin, add structured per-request logging (model version, input fingerprint, prediction) for drift monitoring, and containerize with Docker to pin the exact LightGBM version across environments.

---

## Docs

| Document | Description |
|---|---|
| [PRD](docs/PRD.md) | Product requirements — goals, user stories, non-goals |
| [Architecture](docs/ARCHITECTURE.md) | System design, data flow, component breakdown |
| [Decisions](docs/DECISIONS.md) | Every major technical decision and why |
| [Setup](docs/SETUP.md) | Local dev setup, env vars, deployment |

---

## Author

**Tanish Poddar** — [tanisheesh.in](https://tanisheesh.in) · [LinkedIn](https://linkedin.com/in/tanisheesh) · [GitHub](https://github.com/tanisheesh)
