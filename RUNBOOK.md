# Pawnder — Runbook (setup-complete edition)

> This workspace has been fixed so the project runs end to end. See `SETUP_REPORT.md` for the exact code changes.
> Python environment: `.venv` (Python 3.12). Stockfish 19: `bin/stockfish.exe`.

---

## 1. Quick start

**Terminal 1 — backend:**

```powershell
cd C:\Users\abdul\OneDrive\Desktop\dev\Pawnder
.\.venv\Scripts\Activate.ps1
uvicorn src.api.main:app --reload --port 8000
```

**Terminal 2 — frontend:**

```powershell
cd C:\Users\abdul\OneDrive\Desktop\dev\Pawnder\frontend
npm run dev
```

Open **http://localhost:3000**. API docs: **http://localhost:8000/docs**.

Enter your **Chess.com** username and click **Import & Practice** (or press **Enter** in the field). Your Chess.com blunders are extracted into a private bank; only your own positions will ever be served. Re-importing an existing player merges newly available games but scans **only games that arrived since the previous import** with Stockfish (losses first, up to 12). When new data exists it retrains a **fresh** private model from the Lichess base checkpoint; when nothing changed it reuses the cached private model, so a no-op import finishes in ~20–50 ms. The audit panel reports games used, Stockfish-scanned games, training pairs, and whether the model was reused. Import, scenario generation, and move submission all show visible loading statuses (spinner + text), including an overlay on the board while a new scenario loads.

Each **Generate New Practice Scenario** click avoids recently served FENs, so repeated clicks give different positions. The scenario itself carries its Stockfish Elo, derived from that game's opponent rating/band and clock pressure (clamped to 1000–2400); the slider starts from that value and can be overridden for the current game. Click a piece followed by a highlighted square, or drag the piece directly onto
a highlighted square, to move. Both your move and Stockfish's reply slide across the board in 160 ms, with a ~0.5 s pause before the reply, and you can replay the history with **← / →** or the replay buttons — those steps animate too, including captures. Material advantage appears above/below the board, and a vertical evaluation bar sits to its left like chess.com. Lichess is not a user import source — it is the base dataset that pretrains the model offline. The board uses compact chess.com-style cburnett PNG pieces stored in `frontend/public/pieces/` (regenerate with `npm run pieces`). You must enter a username before either action button is enabled.

---

## 2. Rebuild the environment from scratch

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt     # now includes CPU torch, pyyaml, dvc, mlflow, dagshub
cd frontend && npm install && npm run build
```

Stockfish (already present here): download from https://github.com/official-stockfish/Stockfish/releases/latest, extract `stockfish-windows-x86-64-universal.exe` into `bin/stockfish.exe`. The manager checks `STOCKFISH_PATH`, `PATH`, `bin/stockfish.exe`, and `./stockfish.exe`.

---

## 3. Data & model pipeline

DVC stages now have real entrypoints and use `params.yaml`:

```powershell
.\.venv\Scripts\Activate.ps1
dvc repro          # Chess.com: ingest -> extract_targets
                   # Lichess base: ingest_base -> extract_base -> train_base
                   # evaluate -> metrics.json
```

> Activate the venv first: the DVC stage commands call plain `python`, which must resolve to `.venv\Scripts\python.exe` (the system Python 3.14 has no project dependencies). `dvc status` should report "Data and pipelines are up to date." after a successful run.

This will:
1. Fetch the main platform's games for `data.username` from **Chess.com** (last 3 months, with a labeled offline sample if the API is unreachable).
2. Fetch the **Lichess base dataset** for `base_data.usernames` (`base_data.max_games_per_user` Rapid games each) and tag each game with its loser.
3. Extract blunders from both tracks with **real Stockfish evaluation** (material-only evaluation cannot detect a blunder on a player's own move). Chess.com scenarios go to a user-facing bank; Lichess scenarios go to `data/candidate_bank/lichess_base.json`.
4. Pretrain the global network on the Lichess base bank (`train_base`) and save `models_saved/scenario_model.pt`. Every private user model fine-tunes from this checkpoint.
5. Evaluate on a chronological 20% Chess.com holdout and write `metrics.json`.

To use the pre-built artifacts from DagsHub instead:

```powershell
dvc pull
```

If DagsHub requires credentials: `dagshub login`, then retry.

Manual stage commands (equivalent to `dvc repro`, without hash tracking):

```powershell
python -m src.data.game_ingest          # Chess.com (main platform)
python -m src.data.blunder_detector     # Chess.com blunders
python -m src.data.lichess_base_ingest  # Lichess base dataset
python -m src.data.lichess_base_extract
python -m src.training.train_base      # pretrain global checkpoint on Lichess
python -m src.training.train           # fine-tune on Chess.com data
python -m src.training.evaluate
python -m src.training.evaluate_usefulness  # chronological holdout vs baselines
```

⚠️ `train_base`/`train` overwrite `models_saved/scenario_model.pt`, `evaluate` overwrites `metrics.json`, and `evaluate_usefulness` overwrites `metrics_usefulness.json`. See `MODEL_JUSTIFICATION.md` for how to read the usefulness numbers.

---

## 4. Tests

```powershell
python -m pytest -v
```

31 tests: parser, blunder math, model shapes, API endpoints, per-user isolation, fresh re-import with new-game scanning, scenario variation, scenario-derived Elo, Lichess base-dataset pipeline, Lichess normalization, model quality gate (100% legal FEN, <150 ms inference).

---

## 5. Useful API calls

```powershell
Invoke-RestMethod http://localhost:8000/health
Invoke-RestMethod -Uri http://localhost:8000/api/user/import -Method Post -ContentType application/json -Body '{"username":"bonc_1","max_months":1}'
# Omit target_elo: the response includes the scenario's own target_elo.
Invoke-RestMethod -Uri http://localhost:8000/api/scenario/predict -Method Post -ContentType application/json -Body '{"username":"bonc_1","player_rating":1400,"use_baseline":false}'
Invoke-RestMethod -Uri http://localhost:8000/api/scenario/predict -Method Post -ContentType application/json -Body '{"username":"bonc_1","player_rating":1400,"use_baseline":true}'
# Pass the target_elo returned by /api/scenario/predict to the practice endpoint.
Invoke-RestMethod -Uri http://localhost:8000/api/practice/move -Method Post -ContentType application/json -Body '{"fen":"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1","move_uci":"e2e4","target_elo":1200}'
Invoke-RestMethod http://localhost:8000/api/practice/sessions
```

A username is **required** for `/api/scenario/predict` (missing → `422`) and that user must be imported first (otherwise `422`); retrieval only searches that user's private bank. `target_elo` is returned **by the scenario** (derived from its opponent and clock metadata) and must be passed to `/api/practice/move`; a missing `target_elo` there is now a `422` rather than a silent 1400 default.

---

## 6. Docker (fully fixed)

```powershell
docker compose up --build
```

- UI: http://localhost:3000 (nginx now proxies `/api` to the backend)
- API: http://localhost:8000/docs
- MLflow: http://localhost:5000

Tear down: `docker compose down`.

---

## 7. Links

| What | Link |
|---|---|
| GitHub | https://github.com/Shayan-Qadir/Pawnder |
| DagsHub | https://dagshub.com/Shayan-Qadir/Pawnder |
| DVC remote | https://dagshub.com/Shayan-Qadir/Pawnder.dvc |
| MLflow (configured) | https://dagshub.com/Shayan-Qadir/Pawnder.mlflow |
| Local Swagger | http://localhost:8000/docs |
| Local UI | http://localhost:3000 |
| Chess.com API | https://api.chess.com/pub/player/bonc_1/games/archives |
| Lichess game export API (base dataset only) | https://lichess.org/api/games/user/{username} |
| Stockfish releases | https://github.com/official-stockfish/Stockfish/releases/latest |
| python-chess | https://python-chess.readthedocs.io/en/latest/ |
| DVC docs | https://dvc.org/doc/start |

---

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError` for `chess`/`torch` | Activate `.venv`; `pip install -r requirements.txt` |
| Engine plays trivial moves | Ensure `bin/stockfish.exe` exists or set `STOCKFISH_PATH` |
| `dvc pull` asks for auth | `dagshub login`, then retry |
| API calls fail in Docker | Rebuild after nginx config change: `docker compose up --build` |
| Board deadlocks for Black | Scenario color/FEN alignment is now enforced server-side |
| Latency test flaky | Re-run; it measures a single CPU forward pass |
