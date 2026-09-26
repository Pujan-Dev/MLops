"""Find the best run (highest test F1) and register it in the MLflow Model Registry.

Steps:
  1. Search experiment 'telco-churn' ordered by metrics.f1 DESC.
  2. Register that run's 'model' artifact as registered model 'TelcoChurnClassifier'.
  3. Transition the new version Staging -> Production
     (MLflow 3.x prefers aliases; we try stages first, then aliases).

Run:
    uv run python -m src.register_model
"""

from pathlib import Path

import os
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import mlflow
from mlflow.tracking import MlflowClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MLFLOW_TRACKING_URI = f"file://{PROJECT_ROOT / 'mlruns'}"
EXPERIMENT_NAME = "telco-churn"
REGISTERED_MODEL_NAME = "TelcoChurnClassifier"


def main() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(
            f"Experiment '{EXPERIMENT_NAME}' not found. Run 'uv run python -m src.train' first."
        )

    # Best run by F1 score, descending.
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.f1 DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError("No runs found — train models first.")
    best = runs[0]
    run_id = best.info.run_id
    f1 = best.data.metrics.get("f1", float("nan"))
    print(f"[register] Best run: {run_id} (f1={f1:.4f}, run_name={best.info.run_name})")

    # Register the 'model' artifact of that run.
    # NOTE (MLflow 3.x): mlflow.sklearn.log_model(..., name="model") creates a
    # *logged model* entity (id like "m-abc123"), NOT a run artifact, so the
    # classic "runs:/<run_id>/model" URI is empty. We must register via the
    # logged-model URI "models:/<model_id>".
    logged = client.search_logged_models(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"source_run_id = '{run_id}'",
    )
    if not logged:
        raise RuntimeError(f"No logged models found for run {run_id}.")
    logged_model_id = logged[0].model_id
    model_uri = f"models:/{logged_model_id}"
    print(f"[register] Logged model id for best run: {logged_model_id}")
    print(f"[register] Registering {model_uri} as '{REGISTERED_MODEL_NAME}' ...")
    try:
        version = client.create_registered_model(REGISTERED_MODEL_NAME)
        print(f"[register] Created registered model: {version.name}")
    except mlflow.exceptions.MlflowException as e:
        # Already exists -> continue (RESOURCE_ALREADY_EXISTS).
        print(f"[register] Registered model already exists. Continuing... ({e})")

    mv = client.create_model_version(
        name=REGISTERED_MODEL_NAME, source=model_uri, run_id=run_id
    )
    print(f"[register] Created version {mv.version} (stage={mv.current_stage})")

    # --- Staging -> Production (classic stages API, works on MLflow 2.x) ---
    transitioned = False
    for stage in ("Staging", "Production"):
        try:
            client.transition_model_version_stage(
                name=REGISTERED_MODEL_NAME,
                version=mv.version,
                stage=stage,
                archive_existing_versions=(stage == "Production"),
            )
            print(f"[register] Transitioned version {mv.version} -> {stage}")
            transitioned = True
        except Exception as e:  # MLflow 3.x may disable stages
            print(f"[register] Stage transition to {stage} failed: {e}")
            break

    # --- Fallback for MLflow 3.x: aliases (production / staging) ---
    if not transitioned:
        for alias in ("staging", "production"):
            try:
                client.set_registered_model_alias(REGISTERED_MODEL_NAME, alias, mv.version)
                print(f"[register] Set alias '{alias}' -> version {mv.version}")
            except Exception as e:
                print(f"[register] Alias '{alias}' failed: {e}")

    print(f"\n[register] Done. Load with: models:/{REGISTERED_MODEL_NAME}/Production "
          f"or models:/{logged_model_id} (or runs:/{run_id}/model on MLflow 2.x)")


if __name__ == "__main__":
    main()
