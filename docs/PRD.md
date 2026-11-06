# Maternalyze — Product Requirements Document

**Status:** Final
**Owner:** Tanish Poddar
**One-liner:** ML-powered pregnancy risk assessment — GDM screening and child outcome prediction for clinical support.

---

## 1. Problem

Clinicians assessing pregnancy risk typically rely on fixed threshold rules (e.g. "glucose > X = high risk") that ignore the interaction between multiple patient indicators. A clinician running a mid-appointment assessment has no tool that quickly combines demographics, vitals, and lab results into a calibrated probability — they either refer to a specialist (slow, expensive) or apply a heuristic (imprecise). Maternalyze addresses this gap with two pre-trained LightGBM classifiers that score GDM risk from 7 inputs and SCBU admission risk from 45 inputs, returning a result in under a second with actionable clinical precautions attached.

---

## 2. Goals (v1 / MVP)

1. GDM prediction endpoint: accept 7 clinical inputs, return binary risk label, probability score, and dietary/monitoring recommendations.
2. Child outcome prediction endpoint: accept 45 maternal and delivery indicators, return binary SCBU admission risk, probability score, top-5 feature importances, and clinical precautions.
3. Transparent ML output: feature importance bar chart displayed alongside every child outcome prediction so the model's reasoning is auditable.
4. No-install clinical tool: deploy as a single web service with a public URL — no signup, no app install, browser only.
5. Working live demo deployed to Render with auto-deploy on push to `main`.

---

## 3. Non-Goals (explicit scope cuts)

- **User authentication** — no accounts, no sessions; predictions are public and stateless in v1. Would be required before any real patient data is processed.
- **Patient record storage** — no database in v1; no history, audit log, or longitudinal tracking. A hard prerequisite for any clinical audit trail.
- **Batch prediction** — single-patient requests only; CSV upload or bulk scoring deferred to v2.
- **EHR / FHIR integration** — standalone form-based input only; institutional API access is an infrastructure problem outside the scope of a portfolio project.
- **Model retraining pipeline** — offline training scripts exist but are not wired into production; retraining requires a manual run and redeploy.
- **Mobile-native app** — the web UI is responsive but not a PWA or native app.

---

## 4. Users

**Primary:** Healthcare professionals (midwives, obstetricians, GPs) who need a fast second opinion on pregnancy risk classification during or between appointments.

**Secondary:** Developers and recruiters evaluating the project as a portfolio piece — the tool needs to be demonstrably functional on its live URL with realistic inputs.

---

## 5. User Stories

1. *As a clinician,* I enter a patient's age, weight, BMI, blood glucose, and ethnicity into the GDM form so that I receive an immediate risk label and probability score without manual threshold checks.
2. *As a clinician,* I receive actionable dietary and monitoring recommendations alongside the GDM risk label so that I can act on the result without leaving the tool.
3. *As a clinician,* I enter 45 antenatal and delivery indicators into the child outcome form so that I receive a SCBU admission risk probability and know which factors are driving the prediction.
4. *As a clinician,* I see a bar chart of the top-5 feature importances on each child outcome prediction so that I can verify the model is weighting clinically plausible factors.
5. *As a developer evaluating the project,* I open the live URL and submit a test prediction within 30 seconds so that I can confirm the system is working end-to-end without reading documentation.
6. *As a developer,* I open `/docs` on the Render URL so that I can inspect the API schema and test endpoints without writing any code.

---

## 6. Functional Requirements

### 6.1 GDM Prediction

- Accept `POST /predict_gdm` with a JSON body matching `GDMInput` (7 fields: `AgeAtStartOfSpell`, `WeightMeasured`, `Height`, `BodyMassIndexAtBooking`, `Obese`, `Ethnicity`, `Glucoselevelblood`).
- Return `GDM_Prediction` (0 or 1), `Probability` (float 0–1), and `Precautions` (list of strings).
- Reject malformed requests with `422 Unprocessable Entity` before model inference.
- Return `HTTP 400` with exception detail on unexpected inference error.
- Render doughnut chart client-side from the returned probability.

### 6.2 Child Outcome Prediction

- Accept `POST /predict_child` with a JSON body matching `ChildInput` (45 fields covering demographics, antenatal indicators, delivery details, neonatal outcomes).
- Return `Child_OutcomePrediction` (0 or 1), `Probability` (float 0–1), `Precautions` (list of strings), and `TopFeatureImportances` (dict of top-5 feature names to gain scores).
- Render feature importance bar chart client-side from the returned dict.

### 6.3 Static File Serving

- `GET /` serves `index.html`.
- `GET /{file_name}` serves any `.html`, `.css`, `.js`, or `.ico` file from the `frontend/` directory.
- Interactive API docs served at `/docs` (Swagger UI) and `/redoc` by FastAPI default.

### 6.4 Clinical Precautions

- GDM risk = 1: return four dietary, monitoring, and check-up recommendations.
- GDM risk = 0: return single "continue regular prenatal care" message.
- Child outcome risk = 1: return four fetal monitoring and neonatal care recommendations.
- Child outcome risk = 0: return single "standard neonatal care recommended" message.

---

## 7. Non-Functional Requirements

- **Latency:** Model inference must complete in under 1 second for a single-patient request on standard Render hardware. Cold-start (free tier) may take up to 60s on first request after inactivity — this is acceptable for a demo.
- **Security:** No API keys or secrets required for the core prediction flow. CORS currently open to all origins — acceptable for demo; must be restricted before handling real patient data.
- **Reliability:** Both `.joblib` model files are committed to the repository; the server fails fast at startup if they are missing, rather than silently serving wrong predictions.
- **Cost:** Zero external API spend — no LLM, no managed database. Render free tier is sufficient.
- **Accessibility:** Forms use labelled inputs; results are rendered as HTML tables readable by screen readers.

---

## 8. Success Metrics

| Metric | Target |
|---|---|
| Live demo availability | Returns a valid prediction for a test input within 60s of first cold-start load |
| End-to-end prediction latency | < 1s for both endpoints after warm start |
| API documentation | `/docs` returns the full OpenAPI schema with no manual configuration |
| Feature importance transparency | Top-5 features returned by name and displayed as a labeled bar chart on every child prediction |

---

## 9. Risks & Open Questions

- **Render free-tier cold starts** — the service sleeps after inactivity; first request after sleep takes 30–60s. Mitigated by adding a note to the live demo page, and by upgrading to a paid tier for production use.
- **Model staleness** — pre-trained models become stale as population statistics shift. No retraining trigger exists in v1; mitigated by documenting the offline retraining workflow.
- **CORS open to `"*"`** — currently safe for a demo with no patient data stored. Must be restricted to the Render origin before any real clinical use.
- **Open question** — should categorical encoding maps in `model_utils.py` be auto-generated from training data rather than hand-coded? Hand-coded maps ensure consistency between training and inference but drift silently if the training encoder changes.

---

## 10. v2 Candidates

- **Docker containerization** — pin LightGBM and Python versions exactly; eliminate environment drift between local training and Render deployment.
- **CORS restriction** — restrict `allow_origins` to the known Render origin; required before any real patient data flows through the system.
- **Structured per-request logging** — log model version, input fingerprint (hashed), and prediction label for drift monitoring and audit.
- **Batch prediction endpoint** — accept CSV upload and return predictions for multiple patients in one request.
- **Patient record storage** — optional Postgres/Supabase backend behind auth for clinics that need an audit trail.
- **EHR / FHIR integration** — receive patient data via FHIR R4 `Patient` + `Observation` resources instead of manual form entry.

---

**Tanish Poddar** — [tanisheesh.in](https://tanisheesh.in) · [LinkedIn](https://linkedin.com/in/tanisheesh) · [GitHub](https://github.com/tanisheesh)
