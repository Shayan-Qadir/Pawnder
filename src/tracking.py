"""Shared MLflow experiment tracking and model registry helpers.

Runs are logged to a local file store by default so the whole pipeline is
reproducible with no network access. Set ``MLFLOW_TRACKING_URI`` to a DagsHub
(or any remote) URI to push the same runs to a shared server.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import mlflow

ROOT = Path(__file__).resolve().parents[1]
# MLflow 3.x deprecates the file store; SQLite is self-contained and offline.
LOCAL_TRACKING_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"


def tracking_uri() -> str:
    return os.getenv("MLFLOW_TRACKING_URI", LOCAL_TRACKING_URI)


def setup() -> None:
    mlflow.set_tracking_uri(tracking_uri())
    mlflow.set_experiment("pawnder-scenario-ranker")


@contextmanager
def start_run(run_name: str, tags: dict[str, str] | None = None) -> Iterator[None]:
    """Start an MLflow run with standardised project tags."""
    setup()
    base_tags = {"project": "pawnder", "task": "scenario-ranking"}
    if tags:
        base_tags.update(tags)
    with mlflow.start_run(run_name=run_name, tags=base_tags):
        yield


def log_params(params: dict[str, Any]) -> None:
    mlflow.log_params({k: v for k, v in params.items() if v is not None})


def log_metrics(metrics: dict[str, Any], step: int | None = None) -> None:
    mlflow.log_metrics(
        {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
        step=step,
    )


def log_artifact(path: Path) -> None:
    if path.exists():
        mlflow.log_artifact(str(path))


def register_model(model_path: Path, name: str = "pawnder-scenario-ranker") -> str | None:
    """Register a versioned checkpoint in the MLflow model registry.

    Returns the registered model version, or None if registration is not
    supported by the configured store.
    """
    if not model_path.exists():
        return None
    try:
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        try:
            client.create_registered_model(name)
        except Exception:
            pass  # already registered
        version = client.create_model_version(
            name=name,
            source=f"file://{model_path.resolve().as_posix()}",
            run_id=mlflow.active_run().info.run_id if mlflow.active_run() else None,
        )
        return version.version
    except Exception as exc:  # remote stores may disallow registry writes
        print(f"Model registry unavailable: {exc}")
        return None
