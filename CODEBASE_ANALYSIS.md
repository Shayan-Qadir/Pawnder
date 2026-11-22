# Pawnder (PressureLab) — Comprehensive Codebase Analysis

> **Document generated:** 2026-11-22
> **Repository:** `https://github.com/Shayan-Qadir/Pawnder.git`
> **Analyzed commit:** `dba796b` — "Configure DagsHub DVC remote and MLflow tracking for Shayan-Qadir" (`main`, in sync with `origin/main` and `origin/shayan-dev`; clean working tree)
> **Method:** Static analysis of every tracked source, configuration, test, and infrastructure file (~1,750 lines of Python, ~775 lines of TypeScript/TSX, plus YAML/Docker configs).
> **Caveat:** The local environment has Python 3.14.7 but **none of the project dependencies installed** (`import chess` fails). Therefore no test run, training run, or live API probe was performed. All behavioral claims below are derived from reading the code, not from execution. Where execution would be needed to confirm something, I say so explicitly.

---

## 1. Executive Summary

Pawnder is an **MLOps-themed chess training demo**: it downloads a Chess.com user's Rapid games, reconstructs positions and clocks from PGN, finds "blunder" moments, stores those positions in a candidate bank, trains a small PyTorch model to retrieve one of them from a player-history vector, and serves a React + FastAPI practice loop where the user plays the resulting position against Stockfish.

The **concept is coherent and the repository is well organized** for a portfolio/demo project: there is a real DVC pipeline with hashes, real Pydantic schemas, real UCI engine handling with a fallback, a multi-task neural network, a baseline, tests, CI, and Docker files.

However, there is a large and important gap between **the advertised MLOps system** (README, DVC params, config strings) and **the code that actually executes**. The core issues, in one sentence each:

1. **The "history-conditioned" loop is not actually history-conditioned at runtime.** The API feeds a tensor of all zeros into the model, and the user profile cache is imported but never used.
2. **Training and evaluation use hard-coded synthetic samples of size 1 and 2.** No code path converts real parsed games/blunders into `(history → target)` training pairs. The committed `metrics.json` reports `samples_evaluated: 1`, i.e. `100%` accuracy on a single synthetic example.
3. **The board CNN receives no gradients.** Every use of it is wrapped in `torch.no_grad()`, so its randomly-initialized weights are never learned; candidate embeddings are also frozen snapshots.
4. **"Stockfish-based blunder detection" is not wired up.** `BlunderDetector` is constructed with no engine; material-only heuristic evaluation drives all extraction, both in the API and in the DVC stage.
5. **`torch` is missing from `requirements.txt`.** CI installs that file and then imports torch in four test modules — the pipeline cannot pass as written.
6. **The Docker frontend cannot call the backend.** Vite's `/api` proxy exists only in dev; nginx has no proxy config, so a containerized deployment returns 404 for every API call. `Dockerfile.api` also `COPY`s a `models_saved/` directory that does not exist in the repository.
7. **Player color and FEN side-to-move disagree** in several default/seeded candidates, which can deadlock the chessboard UI.
8. **MLflow is configured but unused** — no `mlflow` import exists anywhere; `SESSION_LOGS` and `USER_CACHE` are process-local Python dicts.

Everything else (parsing, clock regex, centipawn math, bucket boundaries, UCI command sequence, React state machine) is internally consistent and mostly correct.

**Bottom line:** treat this as a **well-structured prototype / scaffold with an unfinished ML core**. The surrounding engineering (DVC, CI, Docker, schemas, tests, UI) is more mature than the modeling and data-pairing logic it is meant to serve.

> ### Working-tree update (2026-11-22)
>
> This document is a frozen analysis of commit `dba796b`; the findings above are
> intentionally left as they were. The uncommitted working tree has since fixed
> the major gaps: real chronological pairs, Stockfish-backed extraction, real
> user history at inference, color alignment, nginx `/api` proxy, dependencies,
> and Docker artifacts. It also adds **true per-user personalization** (private
> banks/profiles/models under `data/candidate_bank/users/`,
> `data/processed/user_profiles/`, `models_saved/users/`) and a split data design:
> **Chess.com is the main player-facing platform**, while **Lichess is the base
> dataset** used to pretrain the global checkpoint (`src/data/lichess_base_*`,
> `src/training/train_base.py`, `base_data.*` in `params.yaml`). Private models
> fine-tune from that Lichess-pretrained checkpoint. The latest working tree also
> retrains on re-import only when genuinely new games arrive (otherwise it reuses
> the cached private model), varies repeated scenarios, derives Stockfish Elo
> from the selected scenario, adds material-advantage badges plus animated ←/→
> move replay, a vertical evaluation bar, drag-and-drop or click-to-move,
> Enter-to-import, and loading indicators for import, generation, and move submission.
> See `SETUP_REPORT.md` and `RUNBOOK.md` for the
> current state. Exact next-FEN prediction remains weak (0.05–0 top-1), so the
> final evaluation reframes the task honestly as **useful drill selection**: on a
> chronological 80/20 holdout over two users, the learned ranker matches the
> failure category of later real blunders 97.4% of the time vs 71–79% for
> random / recent / hand-rule baselines. See `MODEL_JUSTIFICATION.md` and the
> final section below for the full numbers and limitations.

---

## 2. Intended Behavior vs. Actual Behavior

| README / design claim | What the code does | Verdict |
|---|---|---|
| "Predict difficult practice scenarios from player game history" | `/api/scenario/predict` builds `torch.zeros((1,5,131))`; `USER_CACHE` is imported but never read | ❌ Not implemented |
| "Stockfish Deterioration & Blunder Scenario Extractor" | `BlunderDetector()` with `engine_evaluator=None`; `heuristic_eval()` = material sum only. `StockfishManager` is never passed in | ❌ Heuristic only |
| "Game Parser & Clock Reconst" | Real `python-chess` PGN walk + `[%clk ...]` regex | ✅ Implemented |
| "Candidate Bank (DVC Tracked)" | Real DVC stage, `dvc.lock` hashes, data ignored from Git by DVC-generated `.gitignore`s | ✅ Implemented (data not present locally; needs `dvc pull`) |
| "CompactBoardCNN + 2-layer GRU + Retrieval Head" | Classes, shapes and forward pass all exist | ⚠️ Implemented but CNN untrained |
| "MLflow Tracking & Model Registry" | Only a URL constant in `config.py` and an MLflow container in Compose; zero `import mlflow` | ❌ Config only |
| "Model Quality Gate (100% legal, <150ms)" | Real pytest gate asserting both | ✅ Implemented, but the latency test excludes per-request embedding recomputation |
| "15 automated tests" | 4 + 4 + 2 + 1 + 4 = 15 | ✅ Count matches |
| "FastAPI + React UI + Stockfish" | Real endpoints, real UI, real UCI manager (with deterministic fallback) | ✅ Implemented |
| "Training Pairs (History → Target)" | No module constructs them; smoke-test literals only | ❌ Missing |
| "Top-k retrieval accuracy, clock MAE" | `__main__` computes top-1 and clock *bucket* accuracy; baseline object is fitted but its predictions are never compared | ⚠️ Partial |
| "Docker Compose Full-Stack Deployment" | Compose starts 3 services, but nginx has no `/api` proxy and API image COPYs a missing directory | ❌ Broken as shipped |

---

## 3. Repository Inventory

### 3.1 Source tree

```
Pawnder/
├── .dvc/config                  # DagsHub DVC remote URL
├── .dvcignore                   # boilerplate only
├── .github/workflows/ci.yml     # test + gate + frontend build
├── data/
│   ├── raw/.gitignore           # ignores bonc_1_games.json (DVC output)
│   └── candidate_bank/.gitignore# ignores legal_positions.json (DVC output)
├── frontend/
│   ├── index.html               # Tailwind via CDN script
│   ├── package.json             # React 18, Vite 5, chess.js beta, lucide
│   ├── vite.config.ts           # dev proxy /api -> :8000
│   ├── tsconfig.json            # strict: false
│   └── src/
│       ├── main.tsx
│       ├── App.tsx              # 335 lines; application state machine
│       ├── components/
│       │   ├── ChessBoard.tsx   # chess.js-driven click-to-move board
│       │   ├── ClockWidget.tsx  # client-side countdown
│       │   └── WeaknessSummary.tsx
│       └── services/api.ts      # 4 fetch wrappers
├── src/
│   ├── config.py                # paths, pydantic Settings, constants
│   ├── api/
│   │   ├── main.py              # FastAPI app, CORS, 3 routers, /health
│   │   ├── schemas.py           # 7 Pydantic models
│   │   └── routes/{user,scenario,practice}.py
│   ├── data/
│   │   ├── chess_com_client.py  # HTTP client + raw cache
│   │   ├── game_parser.py       # PGN -> per-ply states + clocks
│   │   └── blunder_detector.py  # deterioration rules + bucketing
│   ├── engine/
│   │   └── stockfish_manager.py # UCI subprocess + fallback
│   ├── models/
│   │   ├── board_encoder.py     # fen_to_tensor + CompactBoardCNN
│   │   ├── history_gru.py       # HistoryGRU
│   │   ├── scenario_network.py  # multi-task retrieval network
│   │   └── baseline_knn.py      # sklearn cosine kNN benchmark
│   └── training/
│       ├── dataset.py           # CandidateBankManager + HistoryScenarioDataset
│       ├── train.py             # training loop + smoke __main__
│       └── evaluate.py          # metrics + smoke __main__
├── tests/                       # 5 files, 15 tests
├── dvc.yaml / dvc.lock / params.yaml
├── metrics.json                 # committed evaluation output
├── requirements.txt
├── Dockerfile.api / Dockerfile.frontend / docker-compose.yml
└── README.md
```

Note: `src/api/` and `src/api/routes/` have **no `__init__.py`**. This works on Python 3.3+ via implicit namespace packages (PEP 420), but it is inconsistent with the other packages (`src/data`, `src/models`, etc.) and can confuse tooling.

### 3.2 Local data state

Only the two DVC-generated `.gitignore` files exist under `data/`. The actual artifacts (`bonc_1_games.json`, `legal_positions.json`) and `models_saved/scenario_model.pt` are **not present locally** — they live in the DagsHub DVC remote and must be fetched with `dvc pull`. `dvc.lock` records their last-known sizes: raw games ~136 KB, candidate bank 450 B, model ~3.6 MiB.

The 450-byte candidate bank implies roughly **3–4 candidate positions** — which matches the hardcoded defaults. That is a critical scale observation: retrieval over 3 candidates with 100% top-1 accuracy on 1 sample is not a meaningful ML result.

---

## 4. Architecture and End-to-End Data Flow

### 4.1 Intended pipeline (DVC DAG)

```
Chess.com Published Data API
        │  ingest            (chess_com_client.py, params: username, max_months)
        ▼
data/raw/bonc_1_games.json                 [DVC output, cached]
        │  extract_targets   (blunder_detector.py + game_parser.py)
        ▼
data/candidate_bank/legal_positions.json   [DVC output, cached]
        │  train             (scenario_network.py + dataset.py + CNN + GRU)
        ▼
models_saved/scenario_model.pt             [DVC output, cached]
        │  evaluate          (evaluate.py)
        ▼
metrics.json                               [DVC metric, cache: false]
```

The DAG is declared correctly in `dvc.yaml`: dependencies, parameters, and outputs are listed, and `dvc.lock` contains MD5 hashes for each. Missing from the DAG:

- No stage builds **history→target training pairs**. `train` depends directly on the candidate bank and then constructs its own synthetic samples in `__main__`.
- `extract_targets` does not depend on `stockfish_manager.py` (expected if Stockfish were used; consistent with the fact that it isn't).
- `train.params` in `dvc.yaml` lists only epochs/batch/lr, while `params.yaml` also defines `train.embedding_dim` and `train.hidden_dim` and `config.py` defines `BOARD_CHANNELS`, `CLOCK_BUCKETS`, `RATING_BANDS`. Param changes to those won't invalidate the DVC stage.
- `ingest` does not track `data.time_control` from `params.yaml` even though that key exists; the client hardcodes `time_class="rapid"`.

### 4.2 Runtime request flow

```
Browser (React :3000)
   │ POST /api/user/import
   ▼
FastAPI user.py ──► ChessComClient ──► GameParser ──► BlunderDetector
   │                                                       │
   │                                       appends new FENs to candidate bank
   │ USER_CACHE[username] = profile (in-memory only)
   │
   │ POST /api/scenario/predict {username, player_rating, use_baseline}
   ▼
scenario.py ──► CandidateBankManager.load_candidates()
        ├─ use_baseline? ──► BaselineKNNMatcher.predict()
        └─ else ──► load scenario_model.pt → compute CNN embeddings for all candidates
                    → forward(torch.zeros(1,5,131), embeddings) → argmax → candidate
   │
   │ POST /api/practice/move {fen, move_uci, target_elo}
   ▼
practice.py ──► chess.Board validate ──► StockfishManager.get_best_move(elo=target_elo)
             ──► StockfishManager.evaluate_position() ──► fen_after + eval
   │
   │ POST /api/practice/feedback
   ▼
SESSION_LOGS.append(...)  (in-memory only)
```

The important observation: the `username` parameter travels from the UI to `scenario.py` but **dies there** — it is never used to look up `USER_CACHE`, and `player_rating` is used only to set `target_elo` (the engine strength), not to condition retrieval.

---

## 5. Module-by-Module Deep Dive

### 5.1 `src/config.py` — configuration surface

- Computes `BASE_DIR` from `__file__`, then eagerly `mkdir`s five directories at import time: `data/`, `data/raw/`, `data/processed/`, `data/candidate_bank/`, `models_saved/`.
  - **Side effect at import:** merely importing `src.config` mutates the filesystem. Convenient for demo robustness, but surprising for tests and read-only deployments (e.g. a read-only container filesystem would raise).
- Pydantic `Settings` holds constants: API prefix `/api`, default time control `"600"` (10+0), user agent, Stockfish path from env, engine depth 12, blunder thresholds, DagsHub owner/repo, MLflow tracking URI, board channels 14, embedding dim 128, GRU hidden 256, four clock buckets, six rating bands.
- **Only one env var is actually read:** `STOCKFISH_PATH`. `MLFLOW_TRACKING_URI` is defined but never consumed (no MLflow client).
- `DEFAULT_TIME_CONTROL` is a string and is not used by the ingest client; `time_class="rapid"` is used instead. Two different notions of "time control" coexist.

### 5.2 `src/data/chess_com_client.py` — ingestion

- Minimal `requests` wrapper with `BASE_URL = https://api.chess.com/pub`, explicit `User-Agent`, timeouts 10/15 s, `raise_for_status()`.
- `fetch_player_games()` takes the last `max_months` archive URLs, filters games by `time_class == "rapid"`, catches per-month exceptions and logs a warning, then writes the whole list to `data/raw/<username>_games.json`.
- `load_cached_games()` is the offline path used by the API.

**Reasoning / issues:**
- Good practice: timeouts, user agent, per-archive fault isolation, local caching.
- `params.yaml` has `data.time_control: "600"`, but DVC only tracks `username` and `max_months`; changing the time control param would not re-run ingestion and would do nothing anyway.
- `get_player_profile()` exists but is never called by any route.
- No retry/backoff, no pagination concerns (Chess.com archives are monthly), no validation of the JSON shape.

### 5.3 `src/data/game_parser.py` — PGN and clock reconstruction

`parse_clock_string(comment)` uses regex `\[%clk\s+(\d+):(\d+):(\d+(?:\.\d+)?)\]` and returns hours·3600 + minutes·60 + seconds. Handles `0:09:58` → 598.0 and fractional seconds.

`GameParser.parse_pgn()`:
1. `chess.pgn.read_game` from `StringIO`.
2. Reads headers: players, result, `TimeControl` (default `"600"`), Elo fields (digit-checked).
3. Determines `player_color` and `player_won` via case-insensitive username match; draws → 0.5, unknown → `None`.
4. Walks `game.mainline()`, storing `fen_before`, SAN, UCI, `fen_after`, mover color, and both clocks for **every ply**.

**Clock semantics — worth understanding:** Chess.com attaches `[%clk ...]` to the move node, and that value is the clock *after* the mover made their move. The code reads the comment on the node whose `board.turn` is the mover and updates that mover's last clock — so the stored clock is correct for "time remaining after this move". If a comment is missing for a given move, the last known value silently carries forward; there is no flag for missing data. Initial clocks are hardcoded to 600.0 rather than parsed from `TimeControl`, which is fine for the 10+0 focus but wrong for other time controls.

The per-ply snapshot design is the right foundation: it gives both the pre-move position (the practice scenario) and the post-move position (needed to measure the evaluation drop).

### 5.4 `src/data/blunder_detector.py` — target extraction

Pure-Python, no `python-chess` dependency:

- `cp_to_win_prob(cp)` = logistic `1/(1+10^(-cp/400))` with cp clipped to ±2000. Standard Elo-to-win-probability; tests verify 0 → 0.5, ±1000 saturation.
- `clock_to_bucket()` → `0-30s`, `30-60s`, `60-120s`, `120s+`. Boundaries align with `settings.CLOCK_BUCKETS`.
- `rating_to_band()` → six bands, aligning with `settings.RATING_BANDS`.
- `heuristic_eval(fen)` = material sum from a FEN's board string: P100 N320 B330 R500 Q900 K20000 (king value irrelevant for raw differences).
- `evaluate_fen(fen)` prefers `engine.evaluate_position()` if an object with that method is injected, swallows all exceptions, then falls back to material.
- `extract_scenarios_from_game()`:
  - Only games with `player_won == 0` (losses).
  - Opponent rating chosen from the opposite color; converted to a band.
  - For each ply where `turn_color == player_color`: converts eval to the player's perspective (negate for Black), computes `cp_drop = before - after` and `prob_drop = p_before - p_after`.
  - Blunder if **any** of: `cp_drop ≥ 150`, `prob_drop ≥ 0.20`, or `clock < 45s and cp_drop ≥ 100`.
  - Reason label priority: `"Time scramble blunder (<45s)"` if clock < 45, else `"Endgame miscalculation"` if ply > 60, else `"Tactical deterioration"`.
  - Emits a 12-field dict including the **pre-move FEN**, which is exactly what a practice scenario needs.

**Reasoning / issues:**
- The OR-combination with a 0.20 win-probability drop is redundant-ish with the 150 cp threshold around equal positions (150 cp ≈ 0.30 prob drop at 0), but becomes more sensitive near extremes — reasonable.
- The clock-pressure rule is a nice domain touch and produces the weakness strings in the UI.
- `prev_eval = 0.0` at line 110 is dead code.
- **Critical:** no caller injects a `StockfishManager`. `BlunderDetector()` is instantiated bare in `user.py`, and the DVC `extract_targets` stage just runs the module (whose `__main__` section does not even exist — more on that below). So all "deterioration" is pure material change. A quiet positional blunder is invisible; losing a pawn because of a tactic is detected only via the pawn's material.
- **Important DVC detail:** `src/data/blunder_detector.py` has **no `if __name__ == "__main__"` block** (confirmed by reading the full 168-line file). Similarly `src/data/chess_com_client.py` ends at line 78 with no `__main__`. The DVC stages `python -m src.data.chess_com_client` and `python -m src.data.blunder_detector` therefore **execute nothing** — no ingestion, no extraction, no outputs written. This is a major pipeline defect: `dvc.lock` hashes exist, but re-running `dvc repro` from a clean state would produce empty/missing outputs.

### 5.5 `src/engine/stockfish_manager.py` — UCI engine management

- Locates the binary via `shutil.which("stockfish")`/`"stockfish.exe"`, then `./bin/stockfish.exe`, `./stockfish.exe`.
- Starts a `subprocess.Popen` with line-buffered text pipes, performs the UCI handshake (`uci` → wait `uciok`, `isready` → wait `readyok`), sets Elo.
- `set_elo(elo)` sets `UCI_LimitStrength true`, `UCI_Elo` clamped to [1000, 2800], and derives `Skill Level = clamp((elo-1000)/70, 0, 20)`.
- `evaluate_position(fen, depth)` sends `position fen …` + `go depth …`, scans lines in reverse for `score cp` or `score mate`, negates if `board.turn == BLACK` so the result is always **from White's perspective**, and falls back to material eval.
- `get_best_move(fen, movetime_ms, elo)` sends `go movetime …`, parses `bestmove`, falls back to `_fallback_best_move()` = first capture else first legal move.
- `close()` sends `quit` and terminates.

**Reasoning / issues:**
- The handshake and command/response parsing are textbook UCI and correct in the happy path.
- `text=True, bufsize=1` with `readline()` is appropriate; `_wait_for` has a 3-second wall-clock timeout.
- `evaluate_position` asks for `depth 12` by default but waits only 3 seconds — on slower machines `bestmove` may not arrive and the code silently returns the material fallback. There is no distinction in the return value between a real engine eval and a fallback, so downstream consumers cannot know.
- The fallback "engine" is deterministic and very weak (first capture / first legal move), yet the UI displays a chosen Elo. Without Stockfish installed, the Elo slider is cosmetic.
- **No process locking.** The manager is a module-level singleton used by a synchronous FastAPI endpoint; with multiple workers or threads, interleaved `position`/`go` commands on one stdin would corrupt state. Fine for single-worker uvicorn, not for `--workers > 1`.
- `set_elo` is called on **every move** (when `elo` is passed), which sends 4 commands + `isready` each time. Cheap but unnecessary.
- Mate scores: `mate_in > 0 → ±10000`, sign handled by the side-to-move negation. Reasonable approximation.
- `STOCKFISH_PATH` from config is never used by `StockfishManager` — it auto-discovers instead. The Compose env var is therefore ineffective.

### 5.6 Neural models

#### `board_encoder.py`
`fen_to_tensor(fen)` → `torch.Tensor(14, 8, 8)` float32:
- planes 0–5: white P/N/B/R/Q/K; 6–11: black; 12: side-to-move (all ones if White); 13: castling rights encoded spatially (white K/Q at row 0 cols 0–1 / 2–3; black at row 7).
- Square mapping `row = 7 - square//8`, `col = square % 8` — rank 8 at top, correct for conventional board display.
- En-passant target and halfmove/fullmove counters are intentionally not encoded (documented in the docstring).

`CompactBoardCNN`: three `Conv2d(3×3, pad=1)` + BatchNorm + ReLU blocks, channels 14→32→64→128, `AdaptiveAvgPool2d(1,1)`, flatten, `Linear(128 → embedding_dim)`. Parameter count on the order of 10⁵; output `(batch, 128)`.

#### `history_gru.py`
`HistoryGRU(input_dim=131, hidden_dim=256, num_layers=2, dropout=0.1, batch_first=True)` → last timestep → `Linear(256→256)` + ReLU → `(batch, 256)`. 131 = 128 board embedding + 3 context features. Taking only the last hidden state is a simple user-summary strategy; zero-padding is added at the **front** of sequences in the dataset, so the last position always corresponds to the most recent real step — a good choice.

#### `scenario_network.py`
Multi-task head:
```
history (B,5,131) ──GRU──► user_rep (B,256) ──┬─► clock_head   ─► (B,4)
                                             ├─► rating_head  ─► (B,6)
                                             └─► retrieval_proj (B,128)
                                                     │ L2-normalize
candidate_embeddings (N,128) ──L2-normalize──────────┘ cosine matmul
                                                  ▼  retrieval_logits (B,N)
```
Returns `retrieval_logits`, `clock_logits`, `rating_logits`, `query_embedding`. Shape contracts are asserted by `tests/test_models.py`.

**Reasoning / issues:**
- Cosine similarity retrieval is a sensible retrieval-first design: it guarantees the output is always one of the bank's pre-validated legal FENs (hence the "100% legality" claim is structural, not learned).
- No temperature parameter on the cosine logits; cross-entropy over values in [-1, 1] produces small gradients and slow calibration. A learnable scale (e.g. `logit_scale`) is standard practice.
- **The CNN is frozen in practice.** In `train_scenario_model`, candidate embeddings are computed inside `torch.no_grad()`, and in `HistoryScenarioDataset._encode_step` each step's board embedding is also computed inside `torch.no_grad()`. Although CNN parameters are in `model.parameters()` and passed to Adam, **no gradient path ever reaches them**. The network that "encodes boards" is therefore a random feature extractor. This also means the precomputed candidate embeddings are stable across epochs (good for retrieval stability) but semantically arbitrary (bad for quality). A proper design would either (a) train with graphs through both branches, or (b) explicitly freeze the CNN and say so, with a pretrained/engineered encoder.
- `CandidateBankManager.get_embeddings()` can load `candidate_embeddings.pt` produced by a *different* randomly initialized encoder instance; embeddings and encoder would then be inconsistent. The API avoids this by recomputing, but the helper is a trap.
- `model_state_dict` alone is saved; candidate bank order/content is not versioned with the checkpoint. If `/api/user/import` appends new candidates (it does), target indices learned during training no longer correspond to the bank. This silently corrupts retrieval semantics after any user import.

#### `baseline_knn.py`
`BaselineKNNMatcher` uses `NearestNeighbors(metric="cosine")` over a handcrafted 3-vector `[rating/2000, clock/600, loss_rate]`. For all candidates, `loss_rate` is fixed at 0.5 (candidates don't carry it), and the scenario route queries with `avg_clock=60.0` constant. So the baseline is effectively a 2-D nearest-neighbor lookup. It has a valid hardcoded fallback scenario. It is fitted in `evaluate_model` but its predictions are never compared to the neural model — the "benchmark vs baseline" claim in the module docstring is not realized.

### 5.7 `src/training/dataset.py`

- `CandidateBankManager`: load/save JSON bank; precompute CNN embeddings for every FEN with `torch.no_grad()`, save to `candidate_embeddings.pt` (`weights_only=True` on load — requires torch ≥ 2.0). Has a seed-starting-position fallback if the bank is empty.
- `HistoryScenarioDataset`:
  - `_encode_step`: FEN → 128-d embedding; `clock_ratio = clamp(clock/600)`, `rating_delta = (player_rating - opponent_rating)/400`, `ply/100` → 3-d context; concatenated to 131.
  - `__getitem__`: takes the last `max_seq_len=5` history steps, left-pads with zero vectors, and maps target clock bucket / rating band strings to class indices via `settings` lists (unknown → 0).
  - Defaults `target_candidate_idx=0` when missing — convenient for smoke tests, dangerous for real data (silently biases everything to class 0).

The feature engineering is thoughtful (normalizations by 600 s, 400 rating points, 100 ply are all sensible scales), but no code in the repository populates `samples` with real data.

### 5.8 `src/training/train.py`

`train_scenario_model()`:
- Device auto-select CUDA if available.
- Saves candidate bank, builds model, precomputes embeddings, builds dataset/DataLoader (`shuffle=True`, batch 8).
- Loss = `CE(retrieval) + 0.5·CE(clock) + 0.5·CE(rating)`; Adam lr 1e-3; 5 epochs.
- Logs average epoch loss via `logging`, saves `{"model_state_dict", "epochs", "final_loss"}` to `models_saved/scenario_model.pt`.

`__main__` is a **smoke test**: 2 candidates and **1 sample** whose history is a single starting-position FEN and whose target is candidate 0. It trains for 1 epoch with batch 1. This is what DVC's `train` stage executes. The `params.yaml` values (5 epochs, batch 8, lr 1e-3) are declared to DVC but **never read by the Python code** — `train.py` hardcodes its own defaults and `__main__` overrides them with 1/1. So DVC parameter tracking is decorative here.

### 5.9 `src/training/evaluate.py`

`evaluate_model()` computes top-1 retrieval accuracy, clock-bucket accuracy, and mean per-sample latency with `time.perf_counter()`, returns `legality_guarantee_pct: 100.0` as a constant (justified structurally by retrieval-from-bank). It fits a baseline but never compares it. `__main__` uses **1 synthetic test sample**, writes `metrics.json`. That is why the committed metrics read:

```json
{ "samples_evaluated": 1,
  "scenario_top1_accuracy": 1.0,
  "clock_bucket_accuracy": 1.0,
  "avg_inference_latency_ms": 3.76,
  "legality_guarantee_pct": 100.0 }
```

These numbers should be interpreted as "smoke test passed", not as model performance. Also note the module docstring promises **clock MAE**; the code reports bucket accuracy instead. The latency measurement covers only `model(...)`, not the O(N) CNN embedding recomputation that the API performs per request.

### 5.10 FastAPI layer

`src/api/main.py`: app metadata from settings, permissive CORS (`allow_origins=["*"]` **and** `allow_credentials=True` — this combination is invalid per the Fetch spec; browsers reject wildcard origin when credentials are included; no credentials are used by the frontend, so it happens to work). Three routers under `/api`, `/health` at root.

`src/api/schemas.py`: seven Pydantic v2 models — `UserImportRequest`, `UserProfileResponse`, `ScenarioPredictRequest`, `ScenarioResponse`, `PracticeMoveRequest`, `PracticeMoveResponse`, `SessionFeedbackRequest` (with `Field(ge=1, le=5)` validation). Clean contract layer.

Routes:

**`user.py`**
- Module globals: `ChessComClient()`, `BlunderDetector()` (no engine), `CandidateBankManager()`, `USER_CACHE` dict.
- `/user/import`: cached games or live fetch (swallows errors → empty list), parses each PGN, extracts blunders. If nothing parsed, injects a **synthetic seeded profile** ("bonc_1", 12 games, 6 losses, 1 blunder at FEN `r1bqkb1r/... w KQkq - 4 4` with `player_color: "black"` — note the mismatch). Appends new blunder FENs to the candidate bank, deduped by FEN string. Synthesizes weakness strings by counting reason substrings. Caches the profile in `USER_CACHE`.
- `/user/{username}/profile`: cache hit or on-demand import with `max_months=2`.

**`scenario.py`**
- `get_default_candidates()` returns 3 hand-written positions (Italian Game, Sicilian, pawn endgame). Caveat: the second has `player_color: "black"` but FEN starts `w` — side-to-move mismatch.
- `/scenario/predict`: loads bank (or seeds defaults); baseline branch fits kNN and calls `predict(player_rating, avg_clock=60.0)`; neural branch loads the checkpoint, recomputes all candidate embeddings, runs the model on `torch.zeros(1,5,131)`, clamps indices, and returns a `ScenarioResponse` with a fresh 8-char uuid; on any exception, silently falls back to candidate 0.
- `username` is unused; `target_elo = req.player_rating`; clock bucket and rating band from the model's *auxiliary heads* are returned even though they may contradict the selected candidate's own `clock_bucket`/`opponent_rating_band`.

**`practice.py`**
- Module globals: `StockfishManager()` singleton, `SESSION_LOGS` list.
- `/practice/move`: parses FEN (400 on failure), validates player UCI move against `board.legal_moves`, pushes it, requests engine move at `movetime_ms=350` with Elo, pushes the engine move if legal, evaluates the resulting FEN, returns SAN/UCI/FEN/eval/flags.
  - Subtle bug: if the engine returns an illegal UCI (e.g. from the fallback or a race), the move is not pushed but `engine_move_uci` is still echoed to the client → reported engine move and `fen_after` can disagree.
- `/practice/feedback` appends to in-memory list; `/practice/sessions` lists it. No DB, no MLflow.

### 5.11 Frontend

`main.tsx` mounts `<App/>` in `React.StrictMode`. `api.ts` defines TS mirrors of the Pydantic schemas and four `fetch` wrappers against `/api` (relative path; dev proxy handles it).

`App.tsx` is a 335-line single-page state machine with 13 state variables:
- On mount, hardcodes `handleLoadUser('bonc_1')` — the username input is decorative until the user clicks "Import & Practice". In dev StrictMode this effect runs twice.
- `handleLoadUser` → import profile → generate scenario.
- `handleGenerateScenario` calls `predictScenario(uname, targetElo, useBaseline)` — note the **Elo slider value is sent as `player_rating`**, conflating "my rating" with "engine strength".
- `handleMove` disables the board while "Stockfish thinking", calls `/practice/move`, updates FEN/eval/history, and on game over sends feedback with `result = evaluation_cp > 0 ? 'win' : 'loss'`. Since eval is **White-perspective** and not a game result, this feedback label is unreliable (and meaningless for a Black player).
- UI: dark Chess.com-like theme, navbar, username import, weakness panel, model/engine configuration card with baseline toggle and Elo slider (1000–2400), scenario banner with reset button, two clocks, chessboard, eval bar, move history.

`ChessBoard.tsx`:
- Owns a local `chess.js` `Chess` instance rebuilt from `fen` on prop change.
- Click-to-select with legal-target dots; auto-promotion to queen on 8th/1st rank; board flips by `playerColor` (ranks/files arrays reversed for Black).
- Uses Unicode piece glyphs, no image assets.
- **Deadlock risk:** if a scenario's `player_color` doesn't match the FEN's side to move (see the seeded examples), `game.moves({square})` returns nothing for the player's pieces and the game cannot progress.

`ClockWidget.tsx`: local 1-second interval, red pulsing < 30 s, calls `onTimeOut`. The effect depends on `seconds` and `onTimeOut`, recreating the interval every tick and whenever App re-renders (inline callbacks) — functional but inefficient, and clocks are purely cosmetic: no server-side enforcement, no time added per move (correct for 10+0, but the clocks are not synced to the engine call duration).

`WeaknessSummary.tsx`: presentational card: 3 stats, weakness list, scrollable "recent deterioration points". All data comes from the synthesized strings in `user.py`.

Styling: Tailwind is loaded via a **CDN `<script>` in `index.html`**, while `tailwindcss`/`postcss`/`autoprefixer` are also devDependencies with no `tailwind.config.*`, no PostCSS config, and no CSS entrypoint importing directives. So the npm packages are unused; runtime styling depends on the CDN (network access required in production).

### 5.12 Tests

| File | Tests | What it covers |
|---|---|---|
| `test_game_parser.py` | 2 | clock regex (598.0, 15.5, empty), PGN header/move/clock parsing |
| `test_blunder_detector.py` | 4 | centipawn→probability, clock buckets, rating bands, extraction returns list |
| `test_models.py` | 4 | tensor shape/dtype, CNN forward, GRU forward, all three network outputs |
| `test_api.py` | 4 | `/health`, `/user/import`, `/scenario/predict`, `/practice/move` (via TestClient) |
| `test_model_gate.py` | 1 | 10 inference iterations: FEN legality via `chess.Board.is_valid()`, avg latency < 150 ms, head shapes |

Quality assessment:
- Good: tests assert shapes and real chess validity, not just "no exception".
- Weak: `test_blunder_extraction` only asserts `isinstance(list)` — a function returning `[]` passes. `test_api` depends on network/cache fallback behavior. The gate uses `torch.randn` history rather than realistic encodings and excludes candidate-embedding computation from latency.
- `pytest.ini` sets `pythonpath = .` and `testpaths = tests`.

### 5.13 MLOps infrastructure

**DVC:** remote configured to `https://dagshub.com/Shayan-Qadir/Pawnder.dvc`; stage graph and lock hashes are real. `data/*/.gitignore` files are DVC-generated markers; root `.gitignore` ignores `*.pt`, `mlflow.db`, `mlruns/`, `.dvc/cache`, frontend build artifacts — but **duplicates the entire ignore list twice** (lines 3–41 and 43–81), a cosmetic maintenance smell.

**MLflow:** only `settings.MLFLOW_TRACKING_URI` (default DagsHub MLflow URL) and a Compose service. No `import mlflow`, no `mlflow.start_run()`, no registry usage. The README's "logs hyperparameters, loss curves, top-k accuracy, clock MAE" is aspirational; only `metrics.json` is produced.

**CI (`.github/workflows/ci.yml`):** on push to `main`/`shayan-dev` and PRs to `main`: checkout, Python 3.11 with pip cache, `apt install stockfish`, `pip install -r requirements.txt`, four test files, then the model gate as a separate step, then Node 20 with npm cache (`cache-dependency-path: frontend/package.json` — should be `package-lock.json` to actually hash the lockfile), `npm install`, `npm run build`.
- **Fatal:** `requirements.txt` has no `torch`. `src/models/*` imports torch at module import, so the test steps fail at collection. The most likely explanation is that torch was installed manually in the original environment and never added to the file.
- Also missing from requirements relative to claims: `mlflow`, `dagshub`. `pandas` and `python-dotenv` are installed but unused.

**Metrics:** `metrics.json` is tracked in Git and also declared as a DVC metric with `cache: false`. DVC metric history works only if stages actually run — see the missing `__main__` problem.

### 5.14 Containers and deployment

- `Dockerfile.api`: `python:3.11-slim`, apt-installs `stockfish` + `curl`, pip installs requirements, copies `src/`, `data/`, **`models_saved/`**, `params.yaml`, `pytest.ini`, sets env, exposes 8000, runs uvicorn on 0.0.0.0.
  - `models_saved/` is not in Git (only `*.pt` files are ignored by root `.gitignore`, and the directory exists only after `config.py` runs locally or `dvc pull`). A clean clone `docker build` will fail at `COPY models_saved/` with "no such file or directory".
  - Healthcheck in Compose uses `curl`, which is installed — good.
- `Dockerfile.frontend`: multi-stage Node 20 Alpine build → nginx Alpine serving `/app/dist`. It copies only `package.json` before `npm install` (ignoring the committed `package-lock.json` → non-reproducible installs; `npm ci` would be better).
  - **No nginx config proxies `/api` to `backend:8000`.** The browser calls relative `/api/...`; nginx returns 404. The Vite proxy in `vite.config.ts` applies only to `npm run dev`.
- `docker-compose.yml`: three services (backend :8000, frontend :3000→80, mlflow :5000 with sqlite + named volume). Backend mounts `./data` and `./models_saved`. `STOCKFISH_PATH=/usr/games/stockfish` is set but ignored by `StockfishManager` (which finds `/usr/games/stockfish` via PATH anyway on Debian).
- README port table matches Compose (3000/8000/5000).

---

## 6. Risk Register (prioritized)

| # | Severity | Finding | Evidence | Impact |
|---|---|---|---|---|
| 1 | 🔴 Critical | `torch` absent from `requirements.txt` | `requirements.txt` lines 1–11; `test_models.py`, `test_model_gate.py`, all `src/models/*` import it | CI and fresh installs fail immediately |
| 2 | 🔴 Critical | DVC stage entry modules have no `__main__` | `chess_com_client.py` ends at line 78; `blunder_detector.py` ends at line 168 | `dvc repro` produces no artifacts from a clean state |
| 3 | 🔴 Critical | Inference input is all zeros; user history unused | `scenario.py:96` `torch.zeros((1,5,131))`; `USER_CACHE` imported at line 13, never read | "History-conditioned" product claim is false |
| 4 | 🔴 Critical | No real training/eval pair construction; 1-sample smoke data | `train.py:117-132`, `evaluate.py:95-110`, `metrics.json` | Model quality metrics are meaningless |
| 5 | 🔴 Critical | CNN receives no gradients | `dataset.py:102`, `train.py`/`evaluate.py:29` use `torch.no_grad()` around all encoder calls | Random board representation; retrieval signal is arbitrary |
| 6 | 🟠 High | Docker frontend has no API proxy | `Dockerfile.frontend` nginx stage; `vite.config.ts` proxy is dev-only | Containerized app cannot function |
| 7 | 🟠 High | `COPY models_saved/` from a non-existent directory | `Dockerfile.api:18`; no `models_saved/` in repo | `docker compose up --build` fails on clean clone |
| 8 | 🟠 High | Slider value sent as `player_rating`, not engine Elo separation | `App.tsx:65`; `api.ts:43-52` | Engine difficulty and personalization are conflated |
| 9 | 🟠 High | Scenario `player_color` contradicts FEN side-to-move | `scenario.py:34` (black + `w` FEN), `user.py:60-61` (black + `w` FEN) | Board UI deadlock for those scenarios |
| 10 | 🟠 High | No true Stockfish in blunder extraction | `user.py:17` `BlunderDetector()` | Detection is material-only; positional blunders missed |
| 11 | 🟡 Medium | MLflow configured, never used | grep: zero `import mlflow`; only config + Compose | No experiment tracking, no registry, no feedback-driven learning |
| 12 | 🟡 Medium | Candidate bank mutated at runtime, checkpoint not versioned with it | `user.py:79-86`; `train.py` saves only state_dict | Retrieval index semantics drift after imports |
| 13 | 🟡 Medium | In-memory `USER_CACHE` / `SESSION_LOGS` | `user.py:21`, `practice.py:19` | Data lost on restart; not multi-worker safe |
| 14 | 🟡 Medium | CORS wildcard + credentials | `main.py:18-24` | Invalid combination; brittle if cookies are ever added |
| 15 | 🟡 Medium | CPU embedding recomputation per predict request | `scenario.py:93` | Latency grows linearly with bank size; not covered by gate |
| 16 | 🟡 Medium | Eval sign vs player color | `stockfish_manager.py` returns White-perspective; `App.tsx:110,123` assumes player is White | Wrong feedback label and eval bar for Black |
| 17 | 🟢 Low | ClockWidget effect recreated every second | `ClockWidget.tsx:23-38` | Wasted work, timer drift |
| 18 | 🟢 Low | Tailwind CDN + unused npm packages | `index.html:9`, `package.json` | Production runtime depends on CDN |
| 19 | 🟢 Low | Duplicate `.gitignore` blocks; obsolete `time_control` param; unused imports/vars | multiple | Maintenance noise |
| 20 | 🟢 Low | `src/api/` lacks `__init__.py` | tree | Tooling inconsistency |

---

## 7. Concept Review: Is the ML Design Sound?

**The good idea:** Instead of generating a FEN token-by-token (which risks illegal positions), retrieve from a bank of positions that were *already observed in real lost games*. Retrieval guarantees legality by construction, and the auxiliary heads let you describe the scenario (clock pressure, opponent strength). The multi-task loss and the kNN baseline are appropriate choices. Using pre-move FEN as the target and post-move FEN to compute the drop is exactly right.

**What's missing conceptually:**
1. **A history representation that actually contains history.** The current `history` vector is a sequence of board embeddings of *other* positions, but no code assembles a player's chronological blunder timeline. A reasonable construction would be: for each target blunder at time *t*, take blunders from games before *t*, encode each with FEN + clock + rating delta + ply (the dataset already supports this!), and use the target's candidate index as the label. This pairing function is the single most valuable missing piece.
2. **A trained board encoder.** Either unfreeze it with the rest of the model, or replace it with a pretrained chess encoder / handcrafted feature set and state that it's frozen. Right now it is accidentally random.
3. **Proper validation.** With a handful of candidates, top-1 accuracy must be reported against a chronological train/test split (train on older games, test on newer ones) and compared to "always predict most frequent candidate" and the kNN baseline, otherwise the number is not interpretable.
4. **Consistency between the predicted attributes and the retrieved candidate.** Currently the model can return candidate A's FEN with head-predicted clock bucket B and rating band C. Either use the candidate's stored metadata or train the heads on the same target and present them as *diagnostics*, not ground truth.
5. **Feedback loop.** `SESSION_LOGS` is written but never read back into training. If "MLOps" is the theme, the natural next step is a feedback table → new samples → `dvc repro` → gate → redeploy.

---

## 8. Prioritized Remediation Roadmap

### P0 — Make it runnable and honest (hours)
1. Add `torch>=2.0` (and `mlflow` + `dagshub` if the claims are kept) to `requirements.txt`; pin versions. Remove unused `pandas`/`python-dotenv` or use them.
2. Add `if __name__ == "__main__":` blocks to `chess_com_client.py` and `blunder_detector.py` that read `params.yaml`, fetch/extract, and write the DVC-declared outputs.
3. Make `train.py`/`evaluate.py` read `params.yaml` instead of ignoring it; stop using 1-sample smoke data in the DVC stages. Create a `build_training_pairs()` function in `dataset.py` that pairs chronological blunder histories with target candidates.
4. Fix Docker: create a committed `models_saved/.gitkeep` (or make the COPY conditional / generate the dir first), and add an `nginx.conf` with:
   ```nginx
   location /api/ { proxy_pass http://backend:8000; }
   location / { try_files $uri /index.html; }
   ```
5. Fix scenario FEN/color mismatch: when selecting a candidate, verify `chess.Board(fen).turn` matches `player_color`; if not, flip the FEN (`board.mirror().fen()`) or regenerate the side-to-move field.
6. Add `src/api/__init__.py` and `src/api/routes/__init__.py` for consistency.

### P1 — Make the ML real (days)
7. Feed real user history into inference: in `scenario.py`, read `USER_CACHE[username]`, build the `(5,131)` tensor using the same `_encode_step` logic as training (factor it into a shared utility), and use `player_rating` for personalization rather than only `target_elo`.
8. Unfreeze the CNN (or explicitly freeze and pretrain it). Remove `torch.no_grad()` from the encoder paths during training; keep it for candidate embedding refresh, and refresh embeddings per epoch or with a momentum-updated target encoder. Add a temperature parameter to cosine logits.
9. Version the candidate bank with the checkpoint: save `bank_md5` and candidate list (or hashes) inside the `.pt`; validate on load; refuse to serve if they diverge.
10. Wire Stockfish into extraction (`BlunderDetector(engine_evaluator=StockfishManager(...))`) for real evaluation-based deterioration; keep heuristic as an explicit, reported fallback.
11. Persist users and feedback (SQLite via SQLAlchemy is enough) and log runs/metrics to MLflow — this makes the README truthful and enables retraining on feedback.
12. Replace the demo metrics with a chronological split and report baseline comparison, top-1/top-3, clock accuracy, latency measured end-to-end (including embedding computation).

### P2 — Harden and polish
13. Serialize engine access with a lock; distinguish engine-evaluated vs fallback-evaluated results in the response.
14. Replace client-only clocks with server-authoritative timestamps; derive feedback `result` from actual game outcome, not eval sign.
15. Use `npm ci`, add Tailwind config/PostCSS pipeline (drop the CDN), fix CI cache path to `package-lock.json`.
16. Deduplicate `.gitignore`, add `data.time_control` to DVC params, add embedding/hidden dims to tracked params, protect `allow_origins`/`allow_credentials` combination.
17. Strengthen weak tests (assert non-empty blunder extraction on a crafted material-drop PGN; assert API contract fields; add a Docker smoke test in CI).

---

## 9. Suggested Minimal Data-Pairing Function (sketch)

To illustrate P0/P1 item 3 — the missing bridge between extraction and training:

```python
# src/training/dataset.py (sketch, not implemented)
def build_training_pairs(parsed_games: list[dict], blunders_by_game: list[list[dict]],
                         bank: list[dict], max_seq_len: int = 5) -> list[dict]:
    """Chronologically pair a player's earlier blunders with a later target blunder."""
    history: list[dict] = []
    pairs: list[dict] = []
    for game, blunders in zip(parsed_games, blunders_by_game):
        for b in blunders:                      # blunders already in chronological order
            if not history:
                history.append(b)
                continue
            target_idx = next(
                (i for i, c in enumerate(bank) if c["fen"] == b["fen"]), None
            )
            if target_idx is None:
                continue
            pairs.append({
                "history": history[-max_seq_len:],
                "target_candidate_idx": target_idx,
                "target_clock_bucket": b["clock_bucket"],
                "target_rating_band": b["opponent_rating_band"],
            })
            history.append(b)                   # leak-free: only earlier moments become history
    return pairs
```

With a real `parsed_games` list this immediately turns the existing `HistoryScenarioDataset` from a smoke-test prop into a functional training pipeline. A chronological split (e.g. first 80% of games train, last 20% test) then yields interpretable metrics.

---

## 10. Appendix

### 10.1 API endpoints

| Method | Path | Request | Response | Notes |
|---|---|---|---|---|
| GET | `/health` | — | status/service/version | Docker healthcheck target |
| POST | `/api/user/import` | `UserImportRequest` | `UserProfileResponse` | Network fallback → synthetic profile |
| GET | `/api/user/{username}/profile` | — | `UserProfileResponse` | Cache or on-demand import |
| POST | `/api/scenario/predict` | `ScenarioPredictRequest` | `ScenarioResponse` | baseline or neural or fallback-to-candidate-0 |
| POST | `/api/practice/move` | `PracticeMoveRequest` | `PracticeMoveResponse` | engine strength via `target_elo` |
| POST | `/api/practice/feedback` | `SessionFeedbackRequest` | saved count | in-memory only |
| GET | `/api/practice/sessions` | — | session list | in-memory only |

### 10.2 Configuration reference

| Source | Key(s) | Consumed? |
|---|---|---|
| `config.py` Settings | `API_PREFIX`, `PROJECT_NAME` | ✅ main.py |
| | `STOCKFISH_PATH` (env) | ❌ manager auto-discovers |
| | `DEFAULT_ENGINE_DEPTH`, blunder thresholds | ⚠️ thresholds used only as defaults in `BlunderDetector`; DVC uses params.yaml instead |
| | `BOARD_CHANNELS`, `EMBEDDING_DIM`, `HISTORY_HIDDEN_DIM`, buckets/bands | ✅ models/dataset/gate |
| | `MLFLOW_TRACKING_URI` (env) | ❌ |
| `params.yaml` | `data.username`, `data.max_months` | ✅ declared in DVC, but no Python reader |
| | `data.time_control` | ❌ |
| | `blunder.*` | ✅ declared in DVC only |
| | `train.epochs/batch_size/lr` | ✅ declared in DVC only (code hardcodes) |
| | `train.embedding_dim/hidden_dim` | ❌ |
| | `gate.*` | ❌ (gate hardcodes 150 ms / 100%) |
| Env | `STOCKFISH_PATH`, `MLFLOW_TRACKING_URI` | see above |

### 10.3 Glossary / mental model

- **FEN** — Forsyth–Edwards Notation; a single board position string. The pre-move FEN is the "scenario", the post-move FEN lets us measure the damage.
- **Ply** — one half-move (white or black). `ply > 60` is the code's "endgame" proxy.
- **Centipawn (cp)** — 100 cp = 1 pawn. The thresholds (150 cp, 0.20 win-probability drop) define what the project calls a blunder.
- **Candidate bank** — the set of legal, real-game positions the model may retrieve; retrieval makes legality structural.
- **History tensor** — `(batch, 5, 131)`: up to five prior blunder moments, each a 128-d board embedding + `[clock_ratio, rating_delta, ply/100]`.
- **Retrieval head** — GRU-query cosine similarity blended with the most-recent blunder's learned board-embedding similarity; the score ranks CNN-encoded candidates in the private bank.
- **Gate** — CI test asserting every retrieved FEN is valid and mean inference < 150 ms.

### 10.4 Final assessment (historical)

Pawnder is a **credible, well-organized prototype shell** with a genuinely nice product idea (retrieve real high-pressure positions instead of generating them) and surprisingly complete surrounding infrastructure (DVC hashes, quality gate, UI, CI, Compose). Its weak point is the center: the data pairing, history conditioning, and training signal that would make the "ML" in MLOps real. Fixing items P0–P1 above would turn it from a demo into a defensible end-to-end system; until then, the README should be read as a target architecture rather than a description of current behavior.

---

## 11. Final model justification (current working tree)

Sections 1–10 are the original analysis; the following supersedes the model-specific findings now that real data, real history, and a chronological evaluator exist.

**What changed in the model:** `ScenarioNetwork.retrieval_logits` now blends the GRU query with cosine similarity between the *most recent* blunder's learned board embedding and every private-bank candidate. This keeps the learned Lichess board representation as the core retrieval signal while using the 5-step history as context. No new parameters were added, so existing checkpoints remain loadable. `train_scenario_model` explicitly freezes the board encoder so precomputed candidate embeddings and history encodings always use the same weights.

**Evidence (`python -m src.training.evaluate_usefulness` → `metrics_usefulness.json`):**

| Metric | Private ranker | Recent | Random | Hand rule / majority | Global base |
|---|---:|---:|---:|---:|---:|
| Same-failure-category drill | **0.974** | 0.789 | 0.713 | 0.711 | **0.974** |
| Board similarity to later real blunder | **0.984** | 0.924 | 0.865 | 0.940 | — |
| Actual later blunder in top 5 | **0.589** | — | ~0.075 chance | — | — |

Evaluated on `bonc_1` (10 test pairs / 51 candidates) and `1tshassan` (19 / 95), with an **80/20 chronological split** and the evaluated model fine-tuned only on the older 80%. A label-shuffle control collapses exact top-1 to 0.079 (random: 0.076).

**Interpretation:** the ML representation is useful for selecting a *relevant* drill from a user's own bank, which is the product's actual job. It is not evidence that exact future positions are predictable (top-1 is 0.0), and it is not a live effectiveness study. The Lichess-pretrained model matches the privately fine-tuned model on category, so current personalization comes mainly from the private bank plus history-conditioned scoring; the private checkpoint remains the mechanism for future feedback-driven training. Full narrative and limitations are in `MODEL_JUSTIFICATION.md`.
