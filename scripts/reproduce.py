"""One-command reproduction of the registered model and all metrics.

Usage:
    python scripts/reproduce.py

This validates every dataset, pulls versioned data from DVC remote if available,
runs the full DVC DAG (ingest -> extract -> train -> evaluate -> usefulness),
regenerates the documentation figures, and exports the MLflow run log.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*args: str, check: bool = True) -> None:
    print(f"\n>>> {' '.join(args)}")
    result = subprocess.run(args, cwd=ROOT, check=False)
    if check and result.returncode != 0:
        raise SystemExit(f"Command failed ({result.returncode}): {' '.join(args)}")


def main() -> None:
    py = sys.executable
    print("=" * 70)
    print("PAWNDER — one-command reproducible ML pipeline")
    print("=" * 70)

    run(py, "-m", "src.data.validation")
    run("dvc", "pull", check=False)  # optional when no network/remote configured
    run("dvc", "repro")
    run(py, "scripts/export_mlflow_runs.py")
    run(py, "scripts/generate_model_figures.py")
    print("\nReproduction complete. Metrics: metrics.json, metrics_usefulness.json")
    print("MLflow export: docs/mlflow_export/runs.csv")


if __name__ == "__main__":
    main()
