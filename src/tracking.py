"""Shared MLflow experiment tracking and model registry helpers.

Runs are logged to a local file store by default so the whole pipeline is
reproducible with no network access. Set ``MLFLOW_TRACKING_URI`` to a DagsHub
(or any remote) URI to push the same runs to a shared server.
"""
from __future__ import annotations

import configparser
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import mlflow

ROOT = Path(__file__).resolve().parents[1]
# MLflow 3.x deprecates the file store; SQLite is self-contained and offline.
LOCAL_TRACKING_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
HOSTED_TRACKING_URI = "https://dagshub.com/Shayan-Qadir/Pawnder.mlflow"


def tracking_uri() -> str:
    return os.getenv("MLFLOW_TRACKING_URI", LOCAL_TRACKING_URI)


def _configure_dagshub_auth(uri: str) -> None:
    """Reuse the DVC remote's DagsHub token for hosted MLflow.

    Credentials stay in environment variables only; nothing is printed or
    written to disk. Explicit ``MLFLOW_TRACKING_*`` env vars take precedence.
    """
    if "dagshub.com" not in uri:
        return
    if os.getenv("MLFLOW_TRACKING_USERNAME") and os.getenv("MLFLOW_TRACKING_PASSWORD"):
        return
    cfg = ROOT / ".dvc" / "config.local"
    if not cfg.exists():
        return
    parser = configparser.ConfigParser()
    parser.read(cfg)
    # DVC writes the section as ['remote "origin"'] in this config file.
    section = next(
        (s for s in parser.sections() if "remote" in s and "origin" in s), None
    )
    if section is None:
        return
    token = parser.get(section, "password", fallback="") or parser.get(
        section, "auth", fallback=""
    )
    if not token:
        return
    # DagsHub's MLflow username is the account that owns the token, which is
    # not necessarily the `user` field DVC stores for HTTP auth.
    try:
        from dagshub.common.api import UserAPI

        username = UserAPI.get_user_from_token(token, host="https://dagshub.com").username
    except Exception:
        username = parser.get(section, "user", fallback="")
    if username:
        os.environ.setdefault("MLFLOW_TRACKING_USERNAME", username)
    os.environ.setdefault("MLFLOW_TRACKING_PASSWORD", token)


def setup() -> None:
    # MLflow prints a run URL containing an emoji when a hosted run ends;
    # Windows consoles default to cp1252 and would raise UnicodeEncodeError.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    uri = tracking_uri()
    _configure_dagshub_auth(uri)
    mlflow.set_tracking_uri(uri)
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

        # A local file:// source is meaningless on a hosted server: register
        # the checkpoint that was uploaded to this run's artifact store instead.
        if "dagshub.com" in tracking_uri():
            log_artifact(model_path)
            source = f"{mlflow.get_artifact_uri()}/{model_path.name}"
        else:
            source = f"file://{model_path.resolve().as_posix()}"

        version = client.create_model_version(
            name=name,
            source=source,
            run_id=mlflow.active_run().info.run_id if mlflow.active_run() else None,
        )
        return version.version
    except Exception as exc:  # remote stores may disallow registry writes
        print(f"Model registry unavailable: {exc}")
        return None
