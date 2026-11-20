# Pawnder (PressureLab) - Chess Practice with MLOps

> **Predict difficult practice scenarios from player game history and train under high-pressure conditions against Stockfish.**

Pawnder analyzes completed games from the Chess.com Published Data API (specifically focusing on lost games and evaluation deterioration) and trains a history-conditioned scenario prediction neural network. Its output is a complete scenario:
- **A legal board position (FEN)**
- **The player's color**
- **Remaining clock times**
- **Time control (10+0 Rapid)**
- **An opponent-rating band**

The player then plays out the scenario against a strength-adjusted Stockfish engine, receives post-practice feedback, and logs performance.

---

## Architecture Overview

```
                      +-----------------------------+
                      |   Chess.com Published API   |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      | Game Parser & Clock Reconst |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      | Stockfish Deterioration &   |
                      | Blunder Scenario Extractor  |
                      +--------------+--------------+
                                     |
                       +-------------+-------------+
                       |                           |
                       v                           v
             +-------------------+       +-------------------+
             | Candidate Bank    |       | Training Pairs    |
             | (DVC Tracked)     |       | (History -> Target|
             +---------+---------+       +---------+---------+
                       |                           |
                       +-------------+-------------+
                                     |
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

---

## MLOps Stack

- **Data Versioning:** DVC (Data Version Control) managing raw PGNs, processed game representations, and legal candidate position banks.
- **Experiment Tracking & Registry:** MLflow logging hyperparameters, loss curves, top-k scenario retrieval accuracy, and clock MAE.
- **Model Quality Gate:** Automated CI/CD regression testing ensuring 100% legal FEN generation, clock boundary compliance, and inference latency under 150ms.
- **Backend Serving:** FastAPI REST API with UCI Stockfish subprocess manager and skill-limiting.
- **Frontend UI:** React + Vite + TailwindCSS interactive chessboard, real-time pressure clocks, and weakness diagnosis dashboard.
- **Containerization & CI/CD:** Docker, Docker Compose, GitHub Actions.

---

## Quick Start (Local Setup)

### 1. Backend & Machine Learning
```bash
# Install Python dependencies
pip install -r requirements.txt

# Run pytest test suite (15 automated tests including model gate)
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
