# Week 17: MLOps — Track A (Data Science) + Track B (Agentic AI)

Submission repository for the **Week 17 MLOps problem set**: applying MLOps
discipline — environment reproducibility, experiment tracking, and monitoring —
to (A) a classical ML pipeline on the Telco Customer Churn dataset and (B) the
W15/W16 RAG assistant. Single repository, one folder per track, each track with
its own `pyproject.toml` + committed `uv.lock`.

| Tool | Track A — Data Science (`track-1/`) | Track B — Agentic AI (`track-2/`) |
|---|---|---|
| **uv** | Reproducible env for the ML pipeline | Reproducible env for the assistant |
| **MLflow** | Training runs, best-model registry | Prompt/config runs, traces, regression metrics |
| **Evidently AI** | Data + target drift on tabular data | Regression testing / LLM evaluation |
| Airflow (bonus) | Not implemented | Not implemented |

## Repository layout

```text
mlops/
├── README.md                  # this file — submission documentation
├── .gitignore                 # .venv, mlruns, data, reports, secrets
├── track-1/                   # TRACK A — Telco Churn MLOps
│   ├── pyproject.toml / uv.lock
│   ├── src/
│   │   ├── data_loader.py     # download + clean Telco CSV
│   │   ├── preprocess.py      # stratified split + shared ColumnTransformer
│   │   ├── train.py           # 3 model configs → MLflow
│   │   ├── register_model.py  # best run → registry, Staging → Production
│   │   ├── app.py             # FastAPI: GET /, GET /health, POST /predict
│   │   └── monitor.py         # Evidently drift: 70/30 + synthetic drift
│   ├── data/                  # Telco-Customer-Churn.csv (auto-downloaded)
│   ├── artifacts/             # confusion-matrix + ROC PNGs (all 3 models)
│   ├── reports/               # drift_report.html
│   ├── models/                # optional local .pkl fallback
│   └── mlruns/                # local MLflow file store
└── track-2/
    ├── w16/                   # W16 baseline: stdlib TF-IDF RAG + single-agent loop
    │                          # (rag_tool.py, agent.py, eval.py, results.md, …)
    └── agentic-mlops/         # TRACK B — Agentic MLOps
        ├── pyproject.toml / uv.lock
        ├── prompts/           # prompt_v1/v2/v3.txt (versioned policies)
        ├── src/               # assistant, agent (+traces), tracker, evaluate, app
        ├── traces/            # 36 structured traces (12 queries × 3 versions)
        ├── evaluations/       # eval_v*.json/.md + evidently_input_v*.csv
        └── mlflow.db          # SQLite MLflow backend
```

Per-project deep dives: [`track-1/README.md`](track-1/README.md),
[`track-2/agentic-mlops/README.md`](track-2/agentic-mlops/README.md),
[`track-2/w16/README.md`](track-2/w16/README.md).

---

## Track A — Data Science MLOps (`track-1/`)

**Objective:** bring a churn-prediction pipeline to production MLOps standards.
Dataset: IBM Telco Customer Churn (~7,043 customers, mixed categorical/numeric
features, binary `Churn` ≈ 26.5% positive). Standard workflow implemented
end-to-end: **data → training → tracking → registry → serving → monitoring**.

### 1. Environment management (uv)

- `pyproject.toml` pins `requires-python = ">=3.11,<3.13"` (MLflow/Evidently
  compatibility) and declares sklearn / mlflow / evidently / fastapi / uvicorn.
- `uv.lock` (139 resolved packages, hashes) is committed.
- One-command reproduction: `cd track-1 && uv sync --python 3.11`, then
  `uv run python -m src.train`. Deleting `.venv` and re-syncing restores the
  exact environment — no version drift between machines.

### 2. Experiment tracking (MLflow) — 3 genuinely different models

`uv run python -m src.train` trains three configurations (different families
and hyperparameters, not just seeds) on a stratified 80/20 split
(`random_state=42`), logging per run: all hyperparameters, accuracy,
precision, recall, F1, ROC-AUC, confusion-matrix PNG, ROC-curve PNG, and the
full fitted `Pipeline` (preprocessor + classifier) as the model artifact.

| Run (experiment `telco-churn`) | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression (C=1.0, max_iter=1000) | **0.8055** | 0.6572 | **0.5588** | **0.6040** | **0.8419** |
| Random Forest (n_estimators=200, max_depth=10) | 0.8027 | **0.6633** | 0.5214 | 0.5838 | 0.8409 |
| Gradient Boosting (n_estimators=150, lr=0.1, depth=3) | 0.8006 | 0.6598 | 0.5134 | 0.5774 | 0.8405 |

**Justification for registration:** Logistic Regression wins on F1 (+2 points
over the forest) and ROC-AUC despite the forest's marginally higher precision
(0.6633 vs 0.6572) — accuracy alone is misleading on this imbalanced target,
so F1 is the selection metric. It is also the simplest, fastest, and most
interpretable model. `uv run python -m src.register_model` registers its
logged model as **`TelcoChurnClassifier`** and transitions the version
**Staging → Production** (verified: `models:/TelcoChurnClassifier/Production`
loads and serves). Compare runs side-by-side: `uv run mlflow ui --port 5000`.

### 3. Model serving (FastAPI)

`uv run uvicorn src.app:app --port 8000` — loads the Production model from the
registry via `mlflow.sklearn` (calibrated `predict_proba`, with Staging →
logged-model-ID → local-pickle fallbacks).

- `GET /` — service info + backing model source; `GET /health` — liveness.
- `POST /predict` — one customer record → churn label + probability.
  Verified: high-risk profile → `{"churn_prediction": 1, "churn_label": "Yes",
  "churn_probability": 0.6249}`; loyal profile (60-month tenure, two-year
  contract) → `{"churn_prediction": 0, "churn_label": "No",
  "churn_probability": 0.022}`.

### 4. Monitoring (Evidently AI)

`uv run python -m src.monitor`:

- **Reference vs current:** random 70% reference (4,930 rows, "training-time")
  / 30% current (2,113 rows, "incoming production").
- **Synthetic drift injected into current only:** numerical —
  `MonthlyCharges += 30` (mean 64.55 → 94.55); categorical — 45% of
  `InternetService` forced to `"Fiber optic"` (44.0% → 68.7% share).
- **Data Drift report** (`DataDriftPreset`) flags the perturbed columns;
  **target drift** is covered through the classification mapping on `Churn`;
  **custom metric** `|mean(MonthlyCharges_current) − mean(MonthlyCharges_ref)|`
  = **29.6676** goes beyond Evidently's defaults.
- **Interpretation:** the engineered drifts are detected on exactly the
  columns perturbed, confirming the pipeline works; in production this pattern
  (rising charges + plan-mix shift) would signal upstream data/featurization
  changes that invalidate the training distribution and warrant investigation
  before retraining — blind retraining on drifted data would bake the shift in.
- Report exported to `reports/drift_report.html` **and** logged to MLflow
  (run `drift-monitoring`, artifact path `monitoring/`).

---

## Track B — Agentic AI MLOps (`track-2/agentic-mlops`, built on `track-2/w16`)

**Objective:** extend the W15 RAG assistant + W16 agentic loop with the same
three disciplines, applied to configurations and behavior (prompts, retrieval
settings, loop parameters, harness metrics) instead of a trained model.

### 1. Environment management (uv)

`pyproject.toml` + committed `uv.lock` (fastapi, uvicorn, mlflow, evidently,
pydantic, python-dotenv; `requires-python >=3.11,<3.13`). One command:
`cd track-2/agentic-mlops && uv sync --python 3.11`. LLM keys are optional
(`.env`, git-ignored; see `.env.example`) — deterministic offline policies run
without them.

### 2. Experiment tracking (MLflow) — 3 prompt versions driven by traces

Prompts are versioned explicitly (`prompts/prompt_v1|v2|v3.txt`) alongside the
config each version varies (temperature, top_k, chunk_size, max_iterations):

- **V1** naive (temp 0.7, top_k 3, max_iter 2) → **V2** abstention + keyword
  retry (temp 0.3, top_k 5, max_iter 3) → **V3** robust + synonym expansion
  (temp 0.0, top_k 5, chunk 500, max_iter 5).

Every `run_agent()` call emits a **structured trace** (each tool call with
args + raw result, the intermediate decision/reasoning at each step, total
iterations + termination reason: answered / clarified / abstained /
max_iterations) — 36 saved to `traces/` (12 queries × 3 versions) and logged
as MLflow artifacts, including failure cases. Each revision responds to a
specific traced failure: V1 fabricated from "general knowledge" on
zero-overlap paraphrases → V2 abstains but still fails paraphrases (no
synonyms) → V3 adds `SYNONYM_MAP` + multi-iteration compositional coverage.

Per-version MLflow params (prompt_version, model, temperature, top_k,
chunk_size, max_iterations), metrics (token_usage, latency,
task_completion_score, evidently_pass_rate, correctness_avg,
preservation_avg), artifacts (prompt text, all 12 traces, eval JSON/MD).
Backend is SQLite (`mlflow.db`):
`uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000`.

| Version | Pass rate | Correctness | Preservation | Avg tokens |
|---|---|---|---|---|
| V1 | 8/12 (67%) | 0.69 | 0.59 | 948 |
| V2 | 9/12 (75%) | 0.78 | 0.60 | 1620 |
| V3 | **12/12 (100%)** | 0.99 | 0.73 | 2334 |

**Trade-off:** V3 is the only non-fabricating version at ~2.5× V1's token
cost — justified for grounded answers; V2 is the budget fallback.

### 3. Monitoring & regression testing (Evidently AI)

- **Golden set:** 12 representative queries in `src/evaluate.py`, each with an
  approved reference response (8 direct, 2 zero-overlap paraphrases, 1 vague,
  1 compositional).
- **Two judge checks per pair** — reference-based **correctness** (golden
  keyword coverage + doc-citation check; wrong-doc citation capped) and
  **information preservation** (token-F1); pass gate `correctness ≥ 0.5 and
  preservation ≥ 0.2` (clarify items judged on correctness alone). The
  deterministic judge mirrors Evidently's correct/incorrect LLM-judge
  semantics so `evaluate` runs fully offline; per-row audit tables are always
  written (`evaluations/evidently_input_v*.csv`), and an Evidently HTML suite
  is generated and logged whenever the installed Evidently exposes the suite
  API (see `eval_v*.json → evidently.detail` for which path ran).
- **Pass/fail logged as `evidently_pass_rate`** per MLflow run, so version
  comparison includes the regression signal next to harness numbers; a
  failing suite blocks promotion of that prompt version.
- **Judge sanity check:** verdicts were read against the responses — V1's
  failures are genuine fabrications ("Based on general knowledge…"), V2's are
  safe clarifications on answerable queries (soft failures), V3's passes cite
  the correct docs; the judge's labels match human reading on all 36 pairs.

Serving for Track B: `uv run uvicorn src.app:app --port 8000` with
`POST /query` (answer + full trace), `POST /evaluate`, `GET /prompts`,
`GET /evaluations/{version}`, `GET /health`.

---

## Documentation cross-reference (grading criteria)

- **(a) Environment & reproducibility:** Track A §1 / Track B §1 above —
  what uv solves per project (locked transitive deps, pinned interpreter,
  hash-verified installs) and the one-command paths, verified from clean
  `.venv` recreation.
- **(b) Experiment tracking strategy:** Track A §2 (varied families +
  hyperparameters, measured 5 metrics + artifacts, F1-backed registration
  with the full side-by-side table); Track B §2 (varied prompt/config,
  trace-driven revisions, pass-rate vs token-cost trade-off with table).
- **(c) Monitoring & drift strategy:** Track A §4 (reference/current meaning,
  injected + detected drift, custom metric, production action); Track B §3
  (golden set, two judge checks, pass-rate gate, judge sanity check).
- **(d) Orchestration:** Airflow bonus not implemented in either track.

## Submission checklist

1. **GitHub repo, both tracks** — this repo: `track-1/` (Track A) +
   `track-2/` (Track B: `w16/` baseline + `agentic-mlops/`).
2. **`pyproject.toml` + `uv.lock` per track** — `track-1/`, `track-2/agentic-mlops/`.
3. **MLflow comparisons + registry (A)** — tables above; registry:
   `TelcoChurnClassifier` v2, Production (`track-1/mlruns`);
   Track B runs in `track-2/agentic-mlops/mlflow.db`, experiment `agentic-mlops`.
4. **Evidently HTML** — Track A: `track-1/reports/drift_report.html` (+ MLflow
   `monitoring/` artifact); Track B: per-row `evidently_input_v*.csv` audit
   tables always, HTML suite when the installed Evidently exposes it.
5. **Airflow DAGs** — not attempted.

## Reproduce everything

```bash
# Track A
cd track-1 && uv sync --python 3.11 \
  && uv run python -m src.train \
  && uv run python -m src.register_model \
  && uv run python -m src.monitor \
  && uv run uvicorn src.app:app --port 8000

# Track B
cd ../track-2/agentic-mlops && uv sync --python 3.11 \
  && uv run python -m src.evaluate --all \
  && uv run uvicorn src.app:app --port 8000

# W16 baseline (stdlib only)
cd ../w16 && python3 eval.py
```
