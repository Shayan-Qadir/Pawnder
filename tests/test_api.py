import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"

def test_user_import():
    res = client.post("/api/user/import", json={"username": "bonc_1", "max_months": 1})
    assert res.status_code == 200
    data = res.json()
    assert data["username"] == "bonc_1"
    assert "weaknesses" in data
    assert isinstance(data["weaknesses"], list)

def test_scenario_predict():
    client.post("/api/user/import", json={"username": "bonc_1", "max_months": 1})
    res = client.post(
        "/api/scenario/predict",
        json={"username": "bonc_1", "player_rating": 1300, "use_baseline": False},
    )
    assert res.status_code == 200
    data = res.json()
    assert "fen" in data
    assert "player_color" in data
    assert "clock_remaining" in data
    # Elo is part of the returned scenario, not a fixed client default.
    assert 1000 <= data["target_elo"] <= 2400


def test_scenario_predict_requires_username():
    res = client.post("/api/scenario/predict", json={"player_rating": 1300})
    assert res.status_code == 422

def test_practice_move_requires_scenario_elo():
    start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    res = client.post("/api/practice/move", json={"fen": start_fen, "move_uci": "e2e4"})
    assert res.status_code == 422


def test_practice_move():
    # Starting position, player plays e2e4
    start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    res = client.post("/api/practice/move", json={"fen": start_fen, "move_uci": "e2e4", "target_elo": 1200})
    assert res.status_code == 200
    data = res.json()
    assert data["is_legal"] is True
    assert data["engine_move_uci"] is not None
    assert "fen_after" in data
