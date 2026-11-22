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
# Install Python dependencies
pip install -r requirements.txt

# Run pytest test suite (31 automated tests including model gate)
python -m pytest -v

# Train or evaluate scenario model
python -m src.training.train
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
