"""FastAPI model-serving app.

Loads the registered MLflow model ``TelcoChurnClassifier`` (stage Production,
falling back to Staging / latest version / local runs:/ URI / models/*.pkl)
and exposes:

  GET  /        -> health check + model info
  POST /predict -> churn prediction for one customer
  GET  /health  -> liveness probe (alias of /)

Run:
    uv run uvicorn src.app:app --host 0.0.0.0 --port 8000

Request example:
    POST /predict
    {
      "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
      "Dependents": "No", "tenure": 12, "PhoneService": "Yes",
      "MultipleLines": "No", "InternetService": "Fiber optic",
      "OnlineSecurity": "No", "OnlineBackup": "Yes",
      "DeviceProtection": "No", "TechSupport": "No",
      "StreamingTV": "Yes", "StreamingMovies": "No",
      "Contract": "Month-to-month", "PaperlessBilling": "Yes",
      "PaymentMethod": "Electronic check",
      "MonthlyCharges": 89.1, "TotalCharges": 1070.5
    }

Response example:
    {"churn_prediction": 1, "churn_label": "Yes", "churn_probability": 0.83}
"""

from pathlib import Path
from typing import Optional

import os
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MLFLOW_TRACKING_URI = f"file://{PROJECT_ROOT / 'mlruns'}"
REGISTERED_MODEL_NAME = "TelcoChurnClassifier"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class ChurnRequest(BaseModel):
    """One customer's features. Field names match the raw Telco CSV columns."""

    gender: str = Field(default="Female", examples=["Female"])
    SeniorCitizen: int = Field(default=0, ge=0, le=1, examples=[0])
    Partner: str = Field(default="Yes", examples=["Yes"])
    Dependents: str = Field(default="No", examples=["No"])
    tenure: int = Field(default=12, ge=0, examples=[12])
    PhoneService: str = Field(default="Yes", examples=["Yes"])
    MultipleLines: str = Field(default="No", examples=["No"])
    InternetService: str = Field(default="Fiber optic", examples=["Fiber optic"])
    OnlineSecurity: str = Field(default="No", examples=["No"])
    OnlineBackup: str = Field(default="Yes", examples=["Yes"])
    DeviceProtection: str = Field(default="No", examples=["No"])
    TechSupport: str = Field(default="No", examples=["No"])
    StreamingTV: str = Field(default="Yes", examples=["Yes"])
    StreamingMovies: str = Field(default="No", examples=["No"])
    Contract: str = Field(default="Month-to-month", examples=["Month-to-month"])
    PaperlessBilling: str = Field(default="Yes", examples=["Yes"])
    PaymentMethod: str = Field(default="Electronic check", examples=["Electronic check"])
    MonthlyCharges: float = Field(default=89.1, ge=0, examples=[89.1])
    TotalCharges: float = Field(default=1070.5, ge=0, examples=[1070.5])

    model_config = {"extra": "ignore"}  # tolerate extra client fields


class ChurnResponse(BaseModel):
    churn_prediction: int
    churn_label: str
    churn_probability: float


# ---------------------------------------------------------------------------
# Model loading with graceful fallbacks
# ---------------------------------------------------------------------------

_model = None
_model_source: Optional[str] = None


def _try_load(uri: str):
    """Attempt to load an MLflow model URI; return model or None.

    Prefer ``mlflow.sklearn.load_model`` (returns the native Pipeline with
    ``predict`` AND ``predict_proba``) over ``mlflow.pyfunc`` (which only
    exposes class labels, so probabilities would collapse to 0.0/1.0).
    """
    global _model, _model_source
    for loader in ("sklearn", "pyfunc"):
        try:
            if loader == "sklearn":
                _model = mlflow.sklearn.load_model(uri)
            else:
                _model = mlflow.pyfunc.load_model(uri)
            _model_source = f"{uri} [{loader}]"
            print(f"[app] Loaded model from {uri} via {loader}")
            return _model
        except Exception as e:
            print(f"[app] Could not load {uri} via {loader}: {e}")
    return None


def get_model():
    """Lazy-load the model, trying registry stages first, then logged-model ID.

    Resolution order:
      1. models:/TelcoChurnClassifier/Production  (the promoted model)
      2. models:/TelcoChurnClassifier/Staging
      3. models:/<logged-model-id of best run by F1>  (MLflow 3.x dynamic lookup)
      4. newest models/*.pkl sklearn pickle (offline fallback)
    """
    global _model
    if _model is not None:
        return _model
    candidates = [
        f"models:/{REGISTERED_MODEL_NAME}/Production",
        f"models:/{REGISTERED_MODEL_NAME}/Staging",
    ]
    # Dynamic fallback: logged-model ID of the best run (highest test F1).
    # Needed because MLflow 3.x stores log_model outputs as logged models
    # (m-...) rather than run artifacts (runs:/<id>/model).
    try:
        from mlflow.tracking import MlflowClient
        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        exp = client.get_experiment_by_name("telco-churn")
        if exp is not None:
            runs = client.search_runs(
                experiment_ids=[exp.experiment_id],
                order_by=["metrics.f1 DESC"],
                max_results=1,
            )
            if runs:
                logged = client.search_logged_models(
                    experiment_ids=[exp.experiment_id],
                    filter_string=f"source_run_id = '{runs[0].info.run_id}'",
                )
                if logged:
                    candidates.append(f"models:/{logged[0].model_id}")
    except Exception as e:
        print(f"[app] Dynamic logged-model lookup failed: {e}")
    for uri in candidates:
        if _try_load(uri):
            return _model
    # Fallback: newest pickle under models/ (e.g. saved manually).
    pickles = sorted(PROJECT_ROOT.glob("models/*.pkl"))
    if pickles:
        import joblib
        _model = joblib.load(pickles[-1])
        _model_source = str(pickles[-1])
        # Wrap sklearn estimator so the rest of the code can use .predict/_proba.
        print(f"[app] Loaded sklearn pickle from {_model_source}")

        class _SklearnAdapter:
            """Wrap a raw sklearn Pipeline so it quacks like the MLflow one."""

            def __init__(self, pipe):
                self.pipe = pipe

            def predict(self, df: pd.DataFrame):
                return self.pipe.predict(df)

            def predict_proba(self, df: pd.DataFrame):
                return self.pipe.predict_proba(df)

        _model = _SklearnAdapter(_model)
        return _model
    raise RuntimeError(
        "No model found. Train + register first: "
        "'uv run python -m src.train && uv run python -m src.register_model'."
    )


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Telco Churn MLOps API (Track A)",
    version="0.1.0",
    description="Serves the MLflow-registered Telco churn classifier. "
                "GET / for info, POST /predict for inference.",
)


@app.get("/")
def root():
    """Service info + which model artifact is backing predictions."""
    try:
        get_model()
        status, source = "loaded", _model_source
    except Exception as e:
        status, source = f"error: {e}", None
    return {
        "message": "Telco Customer Churn API — Track A MLOps",
        "model": REGISTERED_MODEL_NAME,
        "model_source": source,
        "model_status": status,
        "endpoints": {"docs": "/docs", "predict": "POST /predict", "health": "GET /health"},
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=ChurnResponse)
def predict(req: ChurnRequest):
    """Predict churn (0/1) + probability for a single customer record."""
    try:
        model = get_model()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    row = pd.DataFrame([req.model_dump()])
    try:
        if hasattr(model, "predict_proba"):
            # Native sklearn Pipeline (preferred): real calibrated probabilities.
            prediction = int(model.predict(row)[0])
            probability = float(model.predict_proba(row)[0, 1])
        else:
            # Generic pyfunc model: only class labels available.
            out = model.predict(row)
            import numpy as np
            arr = out.values if hasattr(out, "values") else out
            prediction = int(arr[0])
            probability = float(arr[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")
    return ChurnResponse(
        churn_prediction=prediction,
        churn_label="Yes" if prediction == 1 else "No",
        churn_probability=round(probability, 4),
    )


if __name__ == "__main__":  # pragma: no cover
    import uvicorn
    uvicorn.run("src.app:app", host="0.0.0.0", port=8000, reload=True)
