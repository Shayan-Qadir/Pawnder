"""Log the real pipeline runs to hosted MLflow on DagsHub (one command).

Usage:
    python scripts/log_to_dagshub.py

Credentials are reused from the existing DVC remote (``.dvc/config.local``),
so no token has to be typed or committed. Override the destination with
``--uri`` or the ``MLFLOW_TRACKING_URI`` environment variable.

This re-executes the three ML stages against the hosted store — the runs are
genuine training/evaluation runs, not copies of the local export. Afterwards
``docs/mlflow_export/`` is refreshed from the hosted server.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URI = "https://dagshub.com/Shayan-Qadir/Pawnder.mlflow"
STAGES = (
    ("train_base", ("-m", "src.training.train_base")),
    ("evaluate", ("-m", "src.training.evaluate")),
    ("evaluate_usefulness", ("-m", "src.training.evaluate_usefulness")),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uri", default=os.getenv("MLFLOW_TRACKING_URI", DEFAULT_URI))
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Only refresh docs/mlflow_export from the hosted server",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        choices=[name for name, _ in STAGES],
        help="Re-run only the listed stages (useful after a network hiccup)",
    )
    args = parser.parse_args()

    env = os.environ.copy()
    env["MLFLOW_TRACKING_URI"] = args.uri
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    # Let src.tracking.read the DVC token; never pass it on the command line.
    env.pop("MLFLOW_TRACKING_USERNAME", None)
    env.pop("MLFLOW_TRACKING_PASSWORD", None)

    if not args.export_only:
        selected = args.only if args.only is not None else [n for n, _ in STAGES]
        for name, module_args in STAGES:
            if name not in selected:
                continue
            for attempt in range(1, 4):
                print(f"\n>>> {name} -> {args.uri} (attempt {attempt}/3)")
                result = subprocess.run([sys.executable, *module_args], cwd=ROOT, env=env)
                if result.returncode == 0:
                    break
                if attempt == 3:
                    raise SystemExit(f"Stage {name} failed after 3 attempts")
                time.sleep(5 * attempt)

    print(f"\n>>> exporting run history from {args.uri}")
    subprocess.run(
        [sys.executable, "scripts/export_mlflow_runs.py"],
        cwd=ROOT,
        env=env,
        check=True,
    )
    print("\nDone. Open", args.uri.replace(".mlflow", ""), "to see the runs and registry.")


if __name__ == "__main__":
    main()
