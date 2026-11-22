# GCR Submission — Pawnder

## GitHub repository
https://github.com/Shayan-Qadir/Pawnder

## DagsHub project / data / MLflow
- Project: https://dagshub.com/Shayan-Qadir/Pawnder
- DVC remote: `https://dagshub.com/Shayan-Qadir/Pawnder.dvc`
- Hosted MLflow UI: https://dagshub.com/Shayan-Qadir/Pawnder.mlflow
- Registered model: `pawnder-scenario-ranker` version **2** (`READY`)

---

## 1. Versioned data and validation

**DVC configuration files** (all committed in the repository):

| File | Purpose |
|---|---|
| `.dvc/config` | DVC remote definition — DagsHub `Pawnder.dvc` |
| `.dvcignore` | DVC ignore rules |
| `dvc.yaml` | Six-stage pipeline: ingest → extract → train → evaluate → usefulness |
| `dvc.lock` | Pinned dependency hashes, params, outputs, and metrics |
| `params.yaml` | Data, blunder, extraction, base-data, and training hyperparameters |
| `data/**/*.dvc` | Pointers for raw/processed datasets versioned in DVC |

The DagsHub access token lives only in the git-ignored `.dvc/config.local`, so no credential is part of this submission.

**Automated pre-training checks:** `src/data/validation.py`

```bash
python -m src.data.validation
```

Checks include required columns/keys, FEN legality via `python-chess`, non-empty candidate pools, bounded clock/move values, duplicate detection, and ownership/platform tags that keep the Lichess base data separate from user-facing Chess.com scenarios. Both `train_base` and `evaluate*` depend on this validation stage input in `dvc.yaml`, so a corrupted or missing dataset fails before training.

---

## 2. Leakage-free splits and baselines

- `src/training/evaluate_usefulness.py`: strict **chronological 80/20** split per real user; the evaluated model is retrained using only the earlier 80%, and the later 20% is never used for fitting or candidate selection.
- Negative controls in the same run: random candidate selection, most-recent candidate, hand-rule/majority, and a history/target shuffle control.
- `src/training/evaluate.py`: classical **k-nearest-neighbour baseline** is computed and logged to MLflow **before** the neural model is evaluated.

Key offline evidence (hosted MLflow run `000624649d4a4b31a6c8d450e70412cd`):

| Metric | Value |
|---|---:|
| Same-category drill selection | 0.9737 |
| Most-recent baseline | 0.7894 |
| Random baseline | 0.7131 |
| Rule/majority baseline | 0.7106 |
| Board similarity to later real blunder | 0.9819 |
| Random similarity | 0.7891 |
| Top-5 target recall | 0.6895 |
| Shuffle-control exact top-1 | 0.0790 |

**Honest claim:** this offline evidence supports useful drill selection for this dataset. It does **not** prove rating improvement, and no A/B test was run.

---

## 3. Experiment tracking and registry

Hosted MLflow experiment: `pawnder-scenario-ranker`  
UI: https://dagshub.com/Shayan-Qadir/Pawnder.mlflow

**Exported logs committed in the repository** (these satisfy the “screenshots or exported logs” requirement):

| Artifact | Contents |
|---|---|
| `docs/mlflow_export/runs.csv` | 4 hosted runs: 3 `FINISHED` + 1 retained `FAILED` attempt; hyperparameters, dataset MD5s, git/DVC-linked dataset tags, split, and metrics |
| `docs/mlflow_export/registry.json` | Registered model `pawnder-scenario-ranker`, version 2, `READY`, with run ID and artifact-store source |

Top registered artifact:

```json
{
  "name": "pawnder-scenario-ranker",
  "version": "2",
  "source": "mlflow-artifacts:/3ff0c0bba82b48e2b265af6924da380a/5650d6fd9b3540959e3e4a4a54db0df7/artifacts/scenario_model.pt",
  "run_id": "5650d6fd9b3540959e3e4a4a54db0df7",
  "status": "READY"
}
```

The model source is an `mlflow-artifacts:/...` URI rather than a local `file://` path, so the registered artifact is loadable from the hosted store.

---

## 4. Reproducible execution

One command reproduces the full local pipeline and exported MLflow logs:

```bash
python scripts/reproduce.py
```

It performs: DVC pull → schema/quality validation → `dvc repro` → MLflow export → metric summary.

To re-run the same stages directly against hosted DagsHub MLflow:

```bash
python scripts/log_to_dagshub.py
```

To refresh only the committed export files from the hosted server:

```bash
python scripts/log_to_dagshub.py --export-only
```

An alternative container path is provided by `Dockerfile.api`, `Dockerfile.frontend`, and `docker-compose.yml`; `docker compose config` passes. The verified reproduction path on this machine is the one-command script above.

---

## Exact text to paste into GCR

```text
GitHub repository: https://github.com/Shayan-Qadir/Pawnder

DVC configuration files:
- .dvc/config
- .dvcignore
- dvc.yaml
- dvc.lock
- params.yaml
- data/**/*.dvc pointers
DVC remote / versioned datasets: https://dagshub.com/Shayan-Qadir/Pawnder (remote: https://dagshub.com/Shayan-Qadir/Pawnder.dvc)

Schema and quality checks: src/data/validation.py (python -m src.data.validation), wired into the DVC pipeline before training.

Leakage-free evaluation and baselines: src/training/evaluate_usefulness.py (chronological 80/20 with random, recent, rule/majority, and shuffle controls) and src/training/evaluate.py (classical kNN baseline logged before the neural model).

MLflow runs (exported logs): docs/mlflow_export/runs.csv
MLflow UI: https://dagshub.com/Shayan-Qadir/Pawnder.mlflow

Registered model (exported log): docs/mlflow_export/registry.json
Registered model: pawnder-scenario-ranker version 2, READY, artifact source mlflow-artifacts:/3ff0c0bba82b48e2b265af6924da380a/5650d6fd9b3540959e3e4a4a54db0df7/artifacts/scenario_model.pt

One-command reproduction: python scripts/reproduce.py
Hosted MLflow re-log: python scripts/log_to_dagshub.py

Claim: offline evidence supports useful personalized drill selection; it does not prove rating improvement, and no A/B test was performed.
```

---

## Final local verification snapshot

- `pytest`: **31 passed**
- `npm run build`: passed
- `dvc status`: **Data and pipelines are up to date.**
- `dvc status -c`: **Cache and remote 'origin' are in sync.**
- Hosted MLflow export refreshed from https://dagshub.com/Shayan-Qadir/Pawnder.mlflow
