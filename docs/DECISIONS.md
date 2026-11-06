# Engineering Decisions — Maternalyze

<!-- PURPOSE OF THIS FILE
     This is not user documentation. This is for technical interviewers
     and senior engineers who want to understand WHY the system is built
     the way it is. Every entry should answer a question an interviewer
     might ask. If you can't explain the tradeoff, the decision doesn't
     belong here yet.
-->

---

## Decision 1 — LightGBM over logistic regression or a neural network

**Context:** The GDM and SCBU admission targets are binary classification problems on tabular clinical data with a mix of numeric and categorical features and a meaningful class imbalance (GDM is relatively rare in the population).

**Decision:** LightGBM GBDT with `RandomOverSampler` for the GDM model, LightGBM GBDT with early stopping for the child outcome model.

**Reason:** LightGBM achieves strong AUC on tabular data without feature scaling, handles integer-encoded categoricals natively, trains in seconds on a laptop, and serializes to a compact `.joblib` file (< 10 MB). Logistic regression lacks the capacity to model non-linear interactions between clinical indicators (e.g., BMI × glucose). A neural network would require more data, GPU training, and a larger serving footprint — none of which are justified by a dataset of this size.

**Tradeoff:** LightGBM predictions are less interpretable than a logistic regression coefficient table. Mitigated by returning top-5 feature importances by gain on every child outcome prediction so the model's reasoning is visible. Full SHAP values were not added in v1 — that would be the richer v2 solution.

---

## Decision 2 — Pre-trained .joblib files committed to the repository

**Context:** The raw training dataset (`gdm_dataset.csv`) is sensitive maternity data that cannot be served in production. Two options: (a) train on demand at server startup, (b) commit pre-trained model files to the repo.

**Decision:** Pre-trained `.joblib` files committed to the repository and loaded at startup.

**Reason:** Training requires the raw CSV which is not present on the Render instance. Loading a pre-trained file is instantaneous compared to a full training run (which takes 10–60 seconds). The server becomes stateless: no training data, no training dependencies at runtime, no startup latency beyond `joblib.load()`.

**Tradeoff:** The model becomes stale as new maternity data arrives; there is no automated retraining trigger. A manual retrain-and-redeploy workflow is documented in `notebooks/` but is not wired into CI. For a clinical production system this would be unacceptable; for a portfolio demo it is a deliberate, scoped tradeoff.

---

## Decision 3 — Vanilla HTML/JS over React or Next.js

**Context:** The frontend consists of three pages: a landing page and two prediction forms, each with one Chart.js visualization. No shared state between pages. No auth. No SSR requirements.

**Decision:** Vanilla HTML, CSS, and JavaScript served as static files directly by FastAPI.

**Reason:** Three mostly-static pages with Chart.js do not warrant a SPA framework, a build pipeline, or npm. FastAPI's `StaticFiles` mount serves them from the same process — one Render service, zero npm dependencies, zero build step. Simpler to deploy, simpler to debug, no cold-start cost from a Node.js layer.

**Tradeoff:** Scaling to more pages, shared components, or client-side routing would require a framework. If the tool grows to include a dashboard, patient history, or auth flow, this decision should be revisited. The explicit trigger for migrating to React would be adding a second page that shares non-trivial state with the first.

---

## Decision 4 — No database — fully stateless architecture

**Context:** The tool predicts risk from clinical inputs. Two design options: (a) persist each prediction for audit and history, (b) treat every request as stateless and discard the result.

**Decision:** No database. Every prediction is computed and returned; nothing is stored.

**Reason:** A database adds operational complexity (provisioning, migrations, connection pooling, backups) that is not justified for an MVP demo. More importantly, storing real patient inputs without a HIPAA/GDPR-compliant data governance framework would be irresponsible. Making the system explicitly stateless is the safer default: it forces the clinic to own their data rather than having it silently accumulate in a third-party database.

**Tradeoff:** No audit trail, no longitudinal tracking, no ability to monitor model performance over time on real inputs. These are all valid clinical requirements that would be required before any real-world deployment — they are deliberately deferred to v2, contingent on proper data governance being in place.

---

## What I'd do differently in v2

- **Restrict CORS to the known Render origin** — `allow_origins=["*"]` was acceptable for a demo with no persistent data. Any version that touches real patient data must restrict this before the first request reaches the server.
- **Containerize with Docker** — pin Python 3.11 and LightGBM 4.x exactly. The current `render.yaml` pins Python via an env var but does not pin the LightGBM wheel version, which can drift between deploys.
- **Add structured per-request logging** — log model version, a hashed input fingerprint, and the binary prediction to a time-series store. Without this, detecting model drift is impossible.
- **Auto-generate categorical encoding maps from training artifacts** — the hand-coded dicts in `model_utils.py` mirror the `LabelEncoder` fit during training but are not derived from it. A mismatch (e.g., a new ethnicity code added to training data) would silently produce the wrong encoding in production. The fix is to serialize the encoder alongside the model and load both at startup.
- **Add SHAP values to child outcome predictions** — `feature_importance(importance_type='gain')` returns global feature importances, not per-prediction SHAP values. SHAP would make each individual prediction more clinically interpretable.

---

## Explicit non-decisions (deferred to v2)

| Feature | Why deferred |
|---|---|
| User authentication | No persistent data in v1 makes auth unnecessary; would be required the moment patient records are stored |
| Patient record storage | Requires data governance (HIPAA/GDPR) before implementation; cannot be added ad-hoc |
| EHR / FHIR integration | Requires institutional API access and FHIR R4 client implementation — outside scope of a portfolio project |
| Batch prediction | Single-patient flow is sufficient for the demo; CSV upload adds form complexity without demonstrating new ML capability |
| Model retraining pipeline | Offline workflow is documented; automating it requires a data pipeline and scheduled jobs not justified at this scale |
