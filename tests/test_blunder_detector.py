import pytest
from src.data.blunder_detector import BlunderDetector, cp_to_win_prob, clock_to_bucket, rating_to_band

def test_cp_to_win_prob():
    # 0 centipawns should be 50%
    assert abs(cp_to_win_prob(0) - 0.5) < 0.001
    # Large positive centipawns -> near 1.0
    assert cp_to_win_prob(1000) > 0.95
    # Large negative centipawns -> near 0.0
    assert cp_to_win_prob(-1000) < 0.05

def test_clock_to_bucket():
    assert clock_to_bucket(15) == "0-30s"
    assert clock_to_bucket(45) == "30-60s"
    assert clock_to_bucket(90) == "60-120s"
    assert clock_to_bucket(300) == "120s+"

def test_rating_to_band():
    assert rating_to_band(850) == "<1000"
    assert rating_to_band(1150) == "1000-1200"
    assert rating_to_band(1350) == "1200-1400"
    assert rating_to_band(1550) == "1400-1600"
    assert rating_to_band(1750) == "1600-1800"
    assert rating_to_band(1950) == "1800+"

def test_blunder_extraction():
    detector = BlunderDetector(cp_drop_threshold=100.0)
    fake_game = {
        "player_color": "white",
        "player_won": 0,
        "black_rating": 1350,
        "moves": [
            {
                "ply": 10,
                "fen_before": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
                "fen_after": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                "turn_color": "white",
                "move_san": "Qh5",
                "white_clock": 35.0,
            }
        ],
    }
    scenarios = detector.extract_scenarios_from_game(fake_game)
    assert isinstance(scenarios, list)
