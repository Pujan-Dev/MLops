# Week 17 MLOps — Track A (Data Science MLOps): Telco Customer Churn

End-to-end, production-style MLOps project on the **IBM Telco Customer Churn** dataset (7,043 customers, target `Churn` ≈ 26.5% positive).
It covers reproducible environments with **uv**, experiment tracking + registry with **MLflow**, model serving with **FastAPI**, and drift monitoring with **Evidently AI** — all in clean, commented, runnable Python.

**Stack (only):** Python 3.11 · scikit-learn · MLflow 3.x · Evidently 0.7.x · FastAPI (+ Uvicorn) · uv · pandas/numpy/matplotlib (supporting libs)

---

## 1. Project overview

| Step | What happens | Code |
|---|---|---|
| Ingest & clean | Download CSV from IBM's public mirror, strip whitespace, coerce `TotalCharges` to numeric, map `Churn` Yes/No → 1/0, drop `customerID` | `src/data_loader.py` |
| Preprocess | Stratified 80/20 split; numeric → median-impute + scale, categorical → most-frequent + one-hot (single `ColumnTransformer`, reused everywhere so train/serve can't skew) | `src/preprocess.py` |
| Train (3 configs) | Logistic Regression · Random Forest · Gradient Boosting — each run logs params, accuracy, precision, recall, F1, ROC-AUC, confusion-matrix PNG, ROC-curve PNG, and the full Pipeline artifact | `src/train.py` |
| Register | Best run by F1 → registered model `TelcoChurnClassifier`, version promoted **Staging → Production** | `src/register_model.py` |
| Serve | FastAPI loads `models:/TelcoChurnClassifier/Production`; `GET /`, `GET /health`, `POST /predict` | `src/app.py` |
| Monitor | 70% reference / 30% current split, synthetic drift injected, Evidently Data-Drift report → `reports/drift_report.html` + custom metric, both logged to MLflow | `src/monitor.py` |

```
track-1/
├── pyproject.toml          # dependencies + Python pin (>=3.11,<3.13)
├── uv.lock                 # fully resolved, reproducible lockfile (uv sync)
├── README.md
├── src/
│   ├── data_loader.py      # download + clean Telco CSV
│   ├── preprocess.py       # split + ColumnTransformer
│   ├── train.py            # train 3 configs, log to MLflow
│   ├── register_model.py   # best run -> registry, Staging -> Production
│   ├── app.py              # FastAPI serving app
│   └── monitor.py          # Evidently drift monitoring
├── data/                   # Telco-Customer-Churn.csv (auto-downloaded)
├── models/                 # optional local .pkl fallback dir
├── artifacts/              # confusion-matrix + ROC PNGs (also in MLflow)
├── reports/                # drift_report.html
└── mlruns/                 # local MLflow tracking store (file backend)
```

---

## 2. Installation (uv)

### Why uv improves dependency reproducibility

1. **One declarative file + one lockfile.** `pyproject.toml` states *intent* (`scikit-learn>=1.3`, `requires-python = ">=3.11,<3.13"`); `uv.lock` records the *exact* resolved versions, hashes, and Python constraints for every transitive dependency. `uv sync` installs byte-for-byte what the lockfile says — no "works on my machine" drift.
2. **Deterministic resolution.** uv's resolver pins the full dependency graph (139 packages here) and verifies hashes at install time, unlike an unpinned `pip install -r requirements.txt` which floats to newest versions on each fresh install.
3. **Pinned interpreter.** `uv sync --python 3.11` provisions the exact Python version, so native wheels (numpy/scipy/sklearn) resolve identically across machines.
4. **Speed + isolation.** uv creates `.venv` automatically and installs in seconds, making clean-room reproduction (`rm -rf .venv && uv sync`) trivial — the standard reproducibility check.

```bash
# 0) Install uv once:  curl -LsSf https://astral.sh/uv/install.sh | sh

# 1) Reproducible install (creates .venv from uv.lock, Python 3.11)
uv sync --python 3.11

# 2) Run anything inside the locked environment
uv run python -m src.train
```

> MLflow 3.x blocks the local file backend by default; every entrypoint in `src/` sets `MLFLOW_ALLOW_FILE_STORE=true` programmatically, so `./mlruns` works with zero manual exports.

---

## 3. Dataset

- **Source:** IBM Telco Customer Churn — [`Telco-Customer-Churn.csv`](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv) (Kaggle mirror: `blastchar/telco-customer-churn`).
- **Shape:** 7,043 rows × 21 columns → 20 after cleaning (identifier `customerID` dropped).
- **Target:** `Churn` (Yes/No → 1/0; 26.5% churn).
- **Features:** demographics (`gender`, `SeniorCitizen`, `Partner`, `Dependents`), tenure/billing (`tenure`, `MonthlyCharges`, `TotalCharges`, `Contract`, `PaperlessBilling`, `PaymentMethod`), services (`PhoneService`, `InternetService`, `OnlineSecurity`, …).
- **Cleaning** (`src/data_loader.py`): auto-download if `data/` is empty → strip whitespace → `TotalCharges` blanks→median → target mapping → drop ID → drop residual NaNs.

```bash
uv run python -m src.data_loader   # downloads + prints class balance
```

---

## 4. Training — 3 model configurations

```bash
uv run python -m src.train
```

Each of the three runs (experiment `telco-churn`) logs:

- **Parameters** — full classifier hyperparameters + `model_family`, `test_size`, `random_state`
- **Metrics** — `accuracy`, `precision`, `recall`, `f1`, `roc_auc`
- **Artifacts** — `plots/<config>_confusion_matrix.png`, `plots/<config>_roc_curve.png`, and the fitted `Pipeline` under `model/` (preprocessor + classifier, so inference needs no separate encoding step)

Local PNG copies are also saved to `artifacts/` for the report.

---

## 5. MLflow usage — compare runs & register the best

```bash
uv run mlflow ui --port 5000   # open http://127.0.0.1:5000
```

1. Open experiment **`telco-churn`** → 3 runs (`logistic_regression`, `random_forest`, `gradient_boosting`) + 1 `drift-monitoring` run.
2. **Compare:** select the 3 training runs → *Compare* → sort by `metrics.f1`; open a run to see params, metric charts, the confusion-matrix/ROC images, and the `model` artifact.
3. **Register + promote** (or run the script that does exactly this):

```bash
uv run python -m src.register_model
```

`src/register_model.py` searches runs ordered by `metrics.f1 DESC`, registers that run's logged model (`models:/m-…`, the MLflow 3.x model entity) as **`TelcoChurnClassifier`**, then transitions the new version **Staging → Production** (with alias fallback `staging`/`production` on MLflow 3.x where stages are deprecated).

> ![MLflow experiment runs table](docs/screenshots/mlflow-runs.png)
> *Placeholder: screenshot of the `telco-churn` experiment — 3 training runs + `drift-monitoring`, sorted by F1.*

> ![MLflow run detail with plots](docs/screenshots/mlflow-run-detail.png)
> *Placeholder: run detail showing params, metric values, confusion-matrix and ROC-curve artifacts.*

> ![MLflow model registry Production](docs/screenshots/mlflow-registry.png)
> *Placeholder: registered model `TelcoChurnClassifier`, version in `Production` stage.*

---

## 6. Model serving (FastAPI)

The app resolves the model as `models:/TelcoChurnClassifier/Production` (via `mlflow.sklearn`, so `predict_proba` yields calibrated probabilities), with fallbacks to Staging → best run's logged-model ID → local `models/*.pkl`.

```bash
uv run uvicorn src.app:app --host 0.0.0.0 --port 8000
# docs: http://127.0.0.1:8000/docs
```

### `GET /` — service info

```bash
curl http://127.0.0.1:8000/
```

```json
{
  "message": "Telco Customer Churn API — Track A MLOps",
  "model": "TelcoChurnClassifier",
  "model_source": "models:/TelcoChurnClassifier/Production [sklearn]",
  "model_status": "loaded",
  "endpoints": {"docs": "/docs", "predict": "POST /predict", "health": "GET /health"}
}
```

### `POST /predict` — churn prediction

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
    "Dependents": "No", "tenure": 12, "PhoneService": "Yes",
    "MultipleLines": "No", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "Yes",
    "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "Yes", "StreamingMovies": "No",
    "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 89.1, "TotalCharges": 1070.5
  }'
```

Response (high-risk customer):

```json
{"churn_prediction": 1, "churn_label": "Yes", "churn_probability": 0.6249}
```

Response (loyal customer: 60-month tenure, two-year contract, DSL, low charges):

```json
{"churn_prediction": 0, "churn_label": "No", "churn_probability": 0.022}
```

---

## 7. Drift monitoring (Evidently AI)

```bash
uv run python -m src.monitor
```

Protocol, exactly per spec:

1. **Split** cleaned data into **70% reference** (4,930 rows) / **30% current** (2,113 rows).
2. **Inject synthetic drift into current only:**
   - Numerical: `MonthlyCharges += 30` (mean 64.55 → 94.55)
   - Categorical: force 45% of `InternetService` → `"Fiber optic"` (44.0% → 68.7% share)
3. **Reports:** Evidently `Report([DataDriftPreset(), ClassificationQuality()])` — the drift preset covers feature drift *and* target (`Churn`) drift via the classification mapping — exported to **`reports/drift_report.html`** (open it in a browser).
4. **Custom metric:** `|mean(MonthlyCharges_current) − mean(MonthlyCharges_reference)|` = **29.6676**, printed and logged.
5. **MLflow logging:** HTML report (under `monitoring/`) + custom metric + drift parameters → run `drift-monitoring` in experiment `telco-churn`.

> ![Evidently drift report](docs/screenshots/evidently-drift.png)
> *Placeholder: screenshot of `reports/drift_report.html` — dataset drift summary with drifted columns highlighted.*

---

## 8. Results and model selection

Test set: 1,409 rows (stratified 80/20 split, `random_state=42`).

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Verdict |
|---|---|---|---|---|---|---|
| **Logistic Regression** (C=1.0, max_iter=1000) | **0.8055** | 0.6572 | **0.5588** | **0.6040** | **0.8419** | ✅ **selected (best F1)** |
| Random Forest (200 trees, depth 10) | 0.8027 | **0.6633** | 0.5214 | 0.5838 | 0.8409 | runner-up (best precision) |
| Gradient Boosting (150 est., lr 0.1, depth 3) | 0.8006 | 0.6598 | 0.5134 | 0.5774 | 0.8405 | competitive, slowest to train |

**Selection rule:** highest test **F1** (the right single number for 26.5%-positive churn data, balancing missed churners against false alarms). Logistic Regression wins on F1, recall, accuracy, and ROC-AUC while being the simplest, fastest, and most interpretable — registered as `TelcoChurnClassifier` v2, stage **Production**.

---

## 9. Reproduce everything (cheat sheet)

```bash
uv sync --python 3.11              # 1. reproducible env (writes .venv from uv.lock)
uv run python -m src.train         # 2. train 3 configs -> MLflow experiment telco-churn
uv run python -m src.register_model# 3. best run -> TelcoChurnClassifier, Staging -> Production
uv run uvicorn src.app:app --port 8000        # 4. serve
uv run python -m src.monitor       # 5. drift report -> reports/drift_report.html + MLflow
uv run mlflow ui --port 5000       # 6. inspect runs, compare, registry
```

To re-verify from scratch: `rm -rf .venv mlruns && uv sync --python 3.11 && uv run python -m src.train`.
