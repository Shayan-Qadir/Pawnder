# Pawnder (PressureLab) - Chess Practice with MLOps

> **Predict difficult practice scenarios from player game history and train under high-pressure conditions against Stockfish.**

**Chess.com is the main user platform.** Pawnder imports a player's completed Chess.com games (focusing on losses and evaluation deterioration), extracts *that player's* blunders into a private scenario bank, and fine-tunes a history-conditioned scenario prediction neural network. **Lichess is the base dataset, not a user import source:** public Lichess games pretrain the global network in the DVC pipeline, and every personalized model starts from that Lichess-pretrained checkpoint. Its output is a complete scenario:
- **A legal board position (FEN)**
- **The player's color**
- **Remaining clock times**
- **Time control (10+0 Rapid)**
- **An opponent-rating band**
- **A Stockfish Elo derived from that scenario's opponent, band and clock pressure**

The player then plays out the scenario against a strength-adjusted Stockfish engine, receives post-practice feedback, and logs performance.

### Personalization

Importing a username creates private artifacts under `data/candidate_bank/users/`,
`data/processed/user_profiles/` and `models_saved/users/`. Retrieval is restricted to
that player's own blunders, and a private model is fine-tuned on the player's own
chronological pairs when enough data exists — starting from the global checkpoint
that was pretrained on Lichess games. One user's positions are never served to
another user. Re-importing the same username is a real retrain: newly available
Chess.com games are merged with the cache, but **only games that arrived since the
previous import** are scanned with Stockfish (losses first, up to 12 per import).
Already-scanned games are never re-analyzed, so repeated imports stay fast and the
private bank does not inflate. If no games arrived, the player's cached private
model is reused (a no-op import takes ~20–50 ms); new games trigger a fresh
fine-tune from the Lichess base checkpoint. The audit panel reports
games used, Stockfish-scanned games, and training pairs, and the UI shows explicit
loading indicators during import, scenario generation, and move submission.
A username is required — scenario generation without one returns `422`. The
Lichess base bank and global model are used only by the DVC
pretraining/evaluation pipeline, never to serve scenarios.

### Is the ML actually useful?

Yes for **offline drill selection**: on a chronological holdout over two real
Chess.com users, the ranker picked a drill from the same failure category as
later real blunders **97.4%** of the time (vs 71–79% for random/recent/hand-rule
baselines), with board similarity **0.982 vs 0.789 random**. No A/B test was
run, so we do not claim rating improvement. Full method, numbers, and limits:
[`MODEL_JUSTIFICATION.md`](MODEL_JUSTIFICATION.md). For the picture version with
graphs, see [`MODEL_EXPLAINED.md`](MODEL_EXPLAINED.md).

---

## Architecture Overview

```
   BASE DATASET (pretraining)              MAIN PLATFORM (per-user)
   --------------------------              ------------------------
   +-----------------------------+         +-----------------------------+
   | Lichess public API          |         | Chess.com Published API     |
   | (base accounts)             |         | (the player's own games)    |
   +--------------+--------------+         +--------------+--------------+
                  |                                       |
                  v                                       v
   +-----------------------------+         +-----------------------------+
   | Game Parser & Clock         |         | Game Parser & Clock         |
   | Reconstruction              |         | Reconstruction              |
   +--------------+--------------+         +--------------+--------------+
                  |                                       |
                  v                                       v
   +-----------------------------+         +-----------------------------+
   | Stockfish Blunder Extractor |         | Stockfish Blunder Extractor |
   +--------------+--------------+         +--------------+--------------+
                  |                                       |
                  v                                       v
   +-----------------------------+         +-----------------------------+
   | Lichess base bank           |         | Private user bank + profile |
   | (lichess_base.json)         |         | (candidate_bank/users/)     |
   +--------------+--------------+         +--------------+--------------+
                  |                                       |
                  v                                       |
   +-----------------------------+                         |
   | PRETRAIN global checkpoint  |                         |
   | scenario_model.pt           |                         |
   +--------------+--------------+                         |
                  |                                       |
                  +------------------+--------------------+
                                     | init_from (fine-tune)
                                     v
                      +-----------------------------+
                      | CompactBoardCNN (8x8x14)    |
                      | + HistoryGRU (2-layer)      |
                      | + Scenario Retrieval Head   |
                      +--------------+--------------+
                                     |
                       +-------------+-------------+
                       |                           |
                       v                           v
              +-------------------+       +-------------------+
              | MLflow Tracking & |       | Model Quality Gate|
              | Model Registry    |       | (100% Legal, <150m|
              +-------------------+       +---------+---------+
                                                    |
                                                    v
                                         +---------------------+
                                         | FastAPI + React UI  |
                                         | Stockfish Engine    |
                                         +---------------------+
```

Lichess pretraining is a one-time DVC stage; only Chess.com data is ever served
to a player, and retrieval is restricted to that player's own private bank.

The React UI accepts **Enter** in the username field as an alias for
**Import & Practice**. It shows loading spinners for import, scenario generation,
and move submission; pieces slide in 160 ms for both live moves and ←/→ replay;
and a vertical chess.com-style evaluation bar sits beside the board. Pieces can be
moved either by clicking a square or by dragging the piece onto a highlighted
destination.

---

## MLOps Stack

- **Data Versioning:** DVC (Data Version Control) managing raw PGNs, processed game representations, and legal candidate position banks.
- **Experiment Tracking & Registry:** MLflow logging hyperparameters, loss curves, top-k scenario retrieval accuracy, and clock MAE.
- **Model Quality Gate:** Automated CI/CD regression testing ensuring 100% legal FEN generation, clock boundary compliance, and inference latency under 150ms.
- **Backend Serving:** FastAPI REST API with UCI Stockfish subprocess manager and skill-limiting.
- **Frontend UI:** React + Vite + TailwindCSS interactive chessboard, real-time pressure clocks, weakness diagnosis dashboard, scenario-owned Elo slider, chess.com-style material advantage badges, animated moves, and ←/→ move replay.
- **Containerization & CI/CD:** Docker, Docker Compose, GitHub Actions.

---

## Quick Start (Local Setup)

### 1. Backend & Machine Learning
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

# Validate every dataset, then reproduce the registered model + all metrics
python scripts/reproduce.py
```

`python scripts/reproduce.py` is the **single command** that validates raw data,
pulls DVC-versioned data, runs the full DAG (`dvc repro`), exports the MLflow run
log, and regenerates the documentation figures. Equivalent manual steps:

```bash
python -m src.data.validation   # schema + FEN legality checks, fails loudly
dvc pull                        # fetch versioned datasets from DagsHub
dvc repro                       # ingest -> extract -> train -> evaluate -> usefulness
python scripts/export_mlflow_runs.py
python -m pytest -v             # 31 tests including the model quality gate
```

### 2. Start API Server
```bash
uvicorn src.api.main:app --reload --port 8000
```
API Documentation will be accessible at: `http://localhost:8000/docs`

### 3. Frontend UI
```bash
cd frontend
npm install
npm run dev
```
Open `http://localhost:3000` in your browser.

---

## Docker Compose Full-Stack Deployment
```bash
docker compose up --build
```
- **Web UI:** `http://localhost:3000`
- **FastAPI Documentation:** `http://localhost:8000/docs`
- **MLflow Tracking Server:** `http://localhost:5000`

---

## Submission / Evidence Pack

Everything required to verify the ML pipeline is in this repository:

| Requirement | Where to find it |
|---|---|
| GitHub repository | https://github.com/Shayan-Qadir/Pawnder |
| DVC config files | `.dvc/config`, `dvc.yaml`, `dvc.lock`, `params.yaml`, plus `*.dvc` pointers under `data/` |
| Versioned datasets on DagsHub | https://dagshub.com/Shayan-Qadir/Pawnder (DVC remote: `.../Pawnder.dvc`) |
| Schema / quality checks | `src/data/validation.py` (run `python -m src.data.validation`) |
| Leakage-free split + baselines | `src/training/evaluate.py` (kNN baseline logged before model), `src/training/evaluate_usefulness.py` (chronological 80/20 + random/recent/rule/majority) |
| MLflow runs | `docs/mlflow_export/runs.csv` — 3 FINISHED runs with params, metrics, dataset MD5s and run IDs |
| Registered model | `docs/mlflow_export/registry.json` — `pawnder-scenario-ranker` version 1 (READY), linked to its training run |
| One-command reproduction | `python scripts/reproduce.py` |
| Metrics | `metrics.json`, `metrics_usefulness.json` |
| Honest interpretation | `MODEL_JUSTIFICATION.md`, `MODEL_EXPLAINED.md` (with `docs/images/*.png`) |

MLflow uses a local SQLite backend (`.venv`-incompatible file store was deprecated in MLflow 3).
Set `MLFLOW_TRACKING_URI` to a DagsHub URI to mirror runs to the shared server.
