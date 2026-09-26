"""Train 3 model configurations and track everything in MLflow.

Configurations:
  1. Logistic Regression      (C=1.0, max_iter=1000)
  2. Random Forest            (n_estimators=200, max_depth=10)
  3. Gradient Boosting        (n_estimators=150, learning_rate=0.1, max_depth=3)

For every run we log:
  - params, accuracy, precision, recall, f1, roc_auc
  - confusion-matrix PNG, ROC-curve PNG
  - the full sklearn Pipeline (preprocessor + classifier) via mlflow.sklearn

Run:
    uv run python -m src.train
Then open the UI:
    uv run mlflow ui --port 5000
Best model = highest test F1. Registration happens in src/register_model.py.
"""

from pathlib import Path

import os
# MLflow 3.x blocks the local file store by default; the assignment
# requires ./mlruns, so opt into the file backend explicitly.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import matplotlib
matplotlib.use("Agg")  # headless: no display needed
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

from src.data_loader import load_clean
from src.preprocess import build_preprocessor, train_test_split_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MLFLOW_TRACKING_URI = f"file://{PROJECT_ROOT / 'mlruns'}"
EXPERIMENT_NAME = "telco-churn"

# The three configurations required by the assignment.
MODEL_CONFIGS = {
    "logistic_regression": LogisticRegression(C=1.0, max_iter=1000, random_state=42),
    "random_forest": RandomForestClassifier(
        n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
    ),
    "gradient_boosting": GradientBoostingClassifier(
        n_estimators=150, learning_rate=0.1, max_depth=3, random_state=42
    ),
}


def plot_confusion_matrix(y_true, y_pred, title: str, save_path: Path) -> None:
    """Render a labelled confusion-matrix heatmap to ``save_path``."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_roc_curve(y_true, y_proba, title: str, save_path: Path) -> None:
    """Render the ROC curve (with AUC in the legend) to ``save_path``."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"ROC (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def evaluate_and_log(name: str, pipe: Pipeline, X_test, y_test, run_dir: Path) -> dict:
    """Compute metrics, save figures, log everything to the active MLflow run."""
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
    print(f"[{name}] " + " | ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    # Log scalar metrics.
    mlflow.log_metrics(metrics)

    # Figures -> MLflow artifacts.
    run_dir.mkdir(parents=True, exist_ok=True)
    cm_path = run_dir / f"{name}_confusion_matrix.png"
    roc_path = run_dir / f"{name}_roc_curve.png"
    plot_confusion_matrix(y_test, y_pred, f"Confusion Matrix — {name}", cm_path)
    plot_roc_curve(y_test, y_proba, f"ROC Curve — {name}", roc_path)
    mlflow.log_artifact(str(cm_path), artifact_path="plots")
    mlflow.log_artifact(str(roc_path), artifact_path="plots")

    # Full pipeline (preprocessor + model) -> reproducible inference artifact.
    # cloudpickle avoids skops 'untrusted types' errors (e.g. numpy.dtype)
    # and keeps pyfunc serving working out of the box.
    mlflow.sklearn.log_model(pipe, name="model", serialization_format="cloudpickle")
    return metrics


def main() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_clean()
    X_train, X_test, y_train, y_test, num_feats, cat_feats = train_test_split_data(df)

    results: dict = {}
    artifacts_dir = PROJECT_ROOT / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)

    for name, classifier in MODEL_CONFIGS.items():
        pre = build_preprocessor(num_feats, cat_feats)
        pipe = Pipeline(steps=[("preprocess", pre), ("model", classifier)])
        with mlflow.start_run(run_name=name) as run:
            print(f"\n=== Training {name} (run_id={run.info.run_id}) ===")
            # Log model-family hyperparameters.
            mlflow.log_params(classifier.get_params())
            mlflow.log_param("model_family", type(classifier).__name__)
            mlflow.log_param("test_size", 0.2)
            mlflow.log_param("random_state", 42)

            pipe.fit(X_train, y_train)
            metrics = evaluate_and_log(name, pipe, X_test, y_test, artifacts_dir)
            results[name] = {"run_id": run.info.run_id, **metrics}

    print("\n========== SUMMARY (sorted by F1) ==========")
    for name, r in sorted(results.items(), key=lambda kv: kv[1]["f1"], reverse=True):
        print(f"{name:22s} f1={r['f1']:.4f} acc={r['accuracy']:.4f} "
              f"auc={r['roc_auc']:.4f} run_id={r['run_id']}")
    best = max(results, key=lambda k: results[k]["f1"])
    print(f"\nBest model by F1: {best} -> run_id={results[best]['run_id']}")
    print("Next: uv run python -m src.register_model  (registers best run)")


if __name__ == "__main__":
    main()
