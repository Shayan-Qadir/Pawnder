"""Export MLflow run history and registered model info to committed files.

The export is plain CSV/JSON so reviewers can inspect experiment tracking
evidence without running the MLflow UI.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mlflow
from mlflow.tracking import MlflowClient

from src.tracking import setup

OUT = ROOT / "docs" / "mlflow_export"
EXPERIMENT = "pawnder-scenario-ranker"


def main() -> None:
    setup()
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT)
    OUT.mkdir(parents=True, exist_ok=True)
    if experiment is None:
        print("No MLflow experiment found yet; run the pipeline first.")
        return

    runs = client.search_runs([experiment.experiment_id], max_results=100)
    rows = []
    for run in runs:
        rows.append({
            "run_id": run.info.run_id,
            "run_name": run.data.tags.get("mlflow.runName", ""),
            "status": run.info.status,
            "stage_tag": run.data.tags.get("stage", ""),
            "dataset_tag": run.data.tags.get("dataset", ""),
            **{f"param:{k}": v for k, v in run.data.params.items()},
            **{f"metric:{k}": v for k, v in run.data.metrics.items()},
        })
    columns = sorted({k for row in rows for k in row})
    with open(OUT / "runs.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    registry: dict = {"registered_models": []}
    try:
        for model in client.search_registered_models():
            entry = {"name": model.name, "versions": []}
            for version in model.latest_versions:
                entry["versions"].append({
                    "version": version.version,
                    "source": version.source,
                    "run_id": version.run_id,
                    "status": version.status,
                })
            registry["registered_models"].append(entry)
    except Exception as exc:
        registry["registry_error"] = str(exc)
    (OUT / "registry.json").write_text(json.dumps(registry, indent=2), encoding="utf-8")
    print(f"Exported {len(rows)} MLflow runs to {OUT}")


if __name__ == "__main__":
    main()
