# Hosted MLflow export

These files are exported from the hosted DagsHub MLflow server:

- UI: https://dagshub.com/Shayan-Qadir/Pawnder.mlflow
- Registry: `pawnder-scenario-ranker` version **2** (`READY`)

Refresh with:

```bash
python scripts/log_to_dagshub.py --export-only
```

## `runs.csv`

Exported run history with hyperparameters, dataset MD5s, split tags, and evaluation metrics.

| Run name | Run ID | Status | Purpose |
|---|---|---|---|
| `train_base_lichess` | `5650d6fd9b3540959e3e4a4a54db0df7` | FINISHED | Lichess-pretrained base model; source of registered v2 |
| `evaluate_holdout` | `c7f72972ce6b4d3a846aff7b10b16785` | FINISHED | Chronological Chess.com holdout + kNN baseline |
| `evaluate_usefulness` | `000624649d4a4b31a6c8d450e70412cd` | FINISHED | 80/20 chronological usefulness evidence + controls |
| `train_base_lichess` | `7e8b3dc7d23040deaeaccc60dc761e0d` | FAILED | Retained failed attempt from an earlier auth hiccup; not the registered model |

## `registry.json`

Versioned registered artifact:

```json
{
  "name": "pawnder-scenario-ranker",
  "version": "2",
  "source": "mlflow-artifacts:/3ff0c0bba82b48e2b265af6924da380a/5650d6fd9b3540959e3e4a4a54db0df7/artifacts/scenario_model.pt",
  "run_id": "5650d6fd9b3540959e3e4a4a54db0df7",
  "status": "READY"
}
```

The `mlflow-artifacts:/...` source means the model is stored in the hosted artifact store and is not dependent on a local `file://` path.
