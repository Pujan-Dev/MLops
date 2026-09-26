"""MLflow experiment tracking wrapper.

Every prompt version logs:
  params:  prompt_version, model_name, temperature, top_k, chunk_size, max_iterations
  metrics: token_usage (avg per query), latency_s (avg), task_completion_score,
           evidently_pass_rate, correctness_avg, preservation_avg
  artifacts: prompt text, agent traces (JSON), evaluation report (JSON/MD)

Tracking store defaults to ./mlruns (local file store) so the assignment runs
offline. View with `mlflow ui`.

If MLflow is not installed, all calls degrade to local JSON logging so the
agent/eval pipeline still runs (a warning is printed).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

try:
    import mlflow  # type: ignore
    _MLFLOW_OK = True
except Exception:  # pragma: no cover
    mlflow = None  # type: ignore
    _MLFLOW_OK = False

EXPERIMENT = "agentic-mlops"


def _ensure_experiment(tracking_uri: str | None = None) -> None:
    if not _MLFLOW_OK:
        return
    # Default to SQLite backend (mlflow.db) — the plain ./mlruns file store is
    # in maintenance mode in MLflow >= 3 and raises unless opted in.
    default_uri = f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
    uri = tracking_uri or default_uri
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(EXPERIMENT)


def start_prompt_run(prompt_version: str, config: dict,
                     tracking_uri: str | None = None):
    """Start (or reuse) an MLflow run for a prompt version.

    Returns a context manager (mlflow.start_run) or a no-op fallback.
    Also logs all config params immediately.
    """
    _ensure_experiment(tracking_uri)
    if not _MLFLOW_OK:
        print("[tracker] WARNING: mlflow not installed; logging to local JSON only.")

        class _Noop:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            @property
            def info(self):
                class _I: run_id = f"local-{prompt_version}-{int(time.time())}"
                return _I()
        return _Noop()
    run = mlflow.start_run(run_name=f"prompt_{prompt_version}", nested=False)
    # Required params per assignment spec.
    mlflow.log_param("prompt_version", config.get("prompt_version", prompt_version))
    mlflow.log_param("model_name", config.get("model_name", ""))
    mlflow.log_param("temperature", config.get("temperature", 0.0))
    mlflow.log_param("top_k", config.get("top_k", 5))
    mlflow.log_param("chunk_size", config.get("chunk_size", 500))
    mlflow.log_param("max_iterations", config.get("max_iterations", 5))
    return run


def log_metrics(metrics: dict) -> None:
    """Log a dict of metrics to the active run (or print when offline)."""
    if not _MLFLOW_OK:
        print("[tracker] metrics (local-only):", json.dumps(metrics, indent=2))
        return
    for k, v in metrics.items():
        try:
            mlflow.log_metric(k, float(v))
        except Exception as e:
            print(f"[tracker] could not log metric {k}: {e}")


def log_artifact(path: str | Path) -> None:
    """Log a single file artifact to the active run."""
    p = Path(path)
    if not p.exists():
        print(f"[tracker] artifact missing, skipped: {p}")
        return
    if not _MLFLOW_OK:
        print(f"[tracker] artifact (local-only, not uploaded): {p}")
        return
    try:
        mlflow.log_artifact(str(p))
    except Exception as e:
        print(f"[tracker] could not log artifact {p}: {e}")


def log_text_artifact(text: str, artifact_filename: str) -> Path:
    """Write text to evaluations/ and log it. Returns the file path."""
    out = PROJECT_ROOT / "evaluations" / artifact_filename
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    log_artifact(out)
    return out
