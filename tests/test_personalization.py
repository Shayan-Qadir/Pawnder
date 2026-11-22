"""Tests for per-user personalization and Lichess data ingestion."""
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.training.dataset import CandidateBankManager

client = TestClient(app)

FEN_A = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
FEN_B = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
FEN_C = "8/5pk1/4p1p1/3pP2p/3P3P/5KP1/8/8 w - - 1 45"


@pytest.fixture
def isolated_storage(monkeypatch, tmp_path):
    """Redirect all per-user artifacts to a temporary directory."""
    import src.training.dataset as ds
    import src.api.routes.scenario as sc
    import src.api.routes.user as ur

    banks = tmp_path / "banks"
    profiles = tmp_path / "profiles"
    models = tmp_path / "models"
    raw = tmp_path / "raw"
    for d in (banks, profiles, models, raw):
        d.mkdir()

    monkeypatch.setattr(ds, "USER_BANK_DIR", banks)
    monkeypatch.setattr(ur, "USER_PROFILE_DIR", profiles)
    monkeypatch.setattr(ur, "RAW_DATA_DIR", raw)
    monkeypatch.setattr(ur, "USER_MODEL_DIR", models)
    monkeypatch.setattr(sc, "USER_MODEL_DIR", models)

    ur.USER_CACHE.clear()
    ur.LIVE_FETCH_CACHE.clear()
    sc._EMBEDDING_CACHE.clear()
    sc._RECENT_SCENARIOS.clear()
    yield
    ur.USER_CACHE.clear()
    ur.LIVE_FETCH_CACHE.clear()
    sc._EMBEDDING_CACHE.clear()
    sc._RECENT_SCENARIOS.clear()


def test_user_banks_are_separate_paths(isolated_storage):
    alice = CandidateBankManager.for_user("alice")
    bob = CandidateBankManager.for_user("bob")
    assert alice.bank_path != bob.bank_path
    assert alice.embeddings_path != bob.embeddings_path
    assert alice.username == "alice"


def test_named_users_only_see_their_own_scenarios(isolated_storage):
    for name, fen in (("alice", FEN_A), ("bob", FEN_B)):
        CandidateBankManager.for_user(name).save_candidates(
            [
                {
                    "fen": fen,
                    "player_color": "white",
                    "clock_remaining": 60.0,
                    "clock_bucket": "60-120s",
                    "opponent_rating_band": "1200-1400",
                    "owner": name,
                    "reason": "personal test blunder",
                }
            ]
        )

    res_a = client.post("/api/scenario/predict", json={"username": "alice"})
    res_b = client.post("/api/scenario/predict", json={"username": "bob"})

    assert res_a.status_code == 200
    assert res_b.status_code == 200
    assert res_a.json()["fen"] == FEN_A
    assert res_b.json()["fen"] == FEN_B
    assert "Personalized" in res_a.json()["reason"]


def test_prediction_without_import_is_rejected(isolated_storage):
    res = client.post("/api/scenario/predict", json={"username": "never_imported"})
    assert res.status_code == 422
    assert "Import this player first" in res.json()["detail"]


def test_import_trains_private_model_from_raw_pgns(isolated_storage, monkeypatch):
    """Regression: the pair builder needs raw PGNs, not parsed game dicts."""
    import src.api.routes.user as ur
    import src.training.dataset as ds

    pgn = """[Event "Live Chess"]
[Site "Chess.com"]
[White "carol"]
[Black "opponent"]
[Result "0-1"]
[WhiteElo "1350"]
[BlackElo "1410"]
[TimeControl "600"]

1. e4 {[%clk 0:09:58]} 1... e5 {[%clk 0:09:55]} 2. Nf3 {[%clk 0:09:40]} 2... Nc6 {[%clk 0:09:30]} 3. Bc4 {[%clk 0:09:12]} 3... Bc5 {[%clk 0:09:00]} 4. Nxe5 {[%clk 0:00:41]} 4... Nxe5 {[%clk 0:00:38]} 0-1
"""
    raw = [
        {
            "pgn": pgn,
            "white": {"username": "carol", "rating": 1350},
            "black": {"username": "opponent", "rating": 1410},
            "result": "0-1",
        }
    ]
    monkeypatch.setattr(ur.client, "load_cached_games", lambda u: raw if u == "carol" else None)
    monkeypatch.setattr(ur, "_get_analysis_engine", lambda: None)

    class FakeDetector:
        def __init__(self, **kwargs):
            pass

        def extract_scenarios_from_game(self, game):
            fens = [
                m["fen_before"]
                for m in game["moves"]
                if m["turn_color"] == game["player_color"]
            ]
            return [
                {
                    "fen": fen,
                    "player_color": game["player_color"],
                    "clock_remaining": 30.0,
                    "clock_bucket": "0-30s",
                    "opponent_rating": 1410,
                    "opponent_rating_band": "1200-1400",
                    "reason": "Time scramble blunder (<45s)",
                    "ply": i + 1,
                }
                for i, fen in enumerate(fens[:3])
            ]

    monkeypatch.setattr(ur, "BlunderDetector", FakeDetector)

    seen = {}
    real_builder = ds.build_training_pairs_from_bank

    def spy(games, username, bank, **kwargs):
        seen["games"] = games
        return real_builder(games, username, bank, **kwargs)

    monkeypatch.setattr(ds, "build_training_pairs_from_bank", spy)

    res = client.post("/api/user/import", json={"username": "carol", "max_months": 1})

    assert res.status_code == 200
    assert seen["games"] and "pgn" in seen["games"][0]
    assert res.json()["personalized_model"] is True


def test_reimport_scans_new_games_and_retrains_fresh_model(isolated_storage, monkeypatch):
    """Re-import must not reuse the old model or skip newly available games."""
    import src.api.routes.user as ur
    import src.training.train as train_mod

    pgn = """[Event "Live Chess"]
[Site "Chess.com"]
[White "carol2"]
[Black "opponent"]
[Result "0-1"]
[WhiteElo "1350"]
[BlackElo "1410"]
[TimeControl "600"]

1. e4 {[%clk 0:09:58]} 1... e5 {[%clk 0:09:55]} 2. Nf3 {[%clk 0:09:40]} 2... Nc6 {[%clk 0:09:30]} 3. Bc4 {[%clk 0:09:12]} 3... Bc5 {[%clk 0:09:00]} 4. Nxe5 {[%clk 0:00:41]} 4... Nxe5 {[%clk 0:00:38]} 0-1
"""

    def raw_game(game_id: str) -> dict:
        return {
            "url": f"https://www.chess.com/game/live/{game_id}",
            "pgn": pgn,
            "white": {"username": "carol2", "rating": 1350},
            "black": {"username": "opponent", "rating": 1410},
            "result": "0-1",
        }

    # The client cache is a snapshot from BEFORE the fetch; a new game shows up
    # in the live fetch first and is only cached after the import completes.
    state = {
        "cached": [raw_game("1"), raw_game("2")],
        "live": [raw_game("1"), raw_game("2")],
    }
    monkeypatch.setattr(ur.client, "load_cached_games", lambda u: list(state["cached"]))
    monkeypatch.setattr(
        ur.client, "fetch_player_games", lambda u, max_months=3: list(state["live"])
    )
    monkeypatch.setattr(ur, "_get_analysis_engine", lambda: None)

    scanned: list[int] = []
    fens = [FEN_A, FEN_B, FEN_C]

    class FakeDetector:
        def __init__(self, **kwargs):
            pass

        def extract_scenarios_from_game(self, game):
            fen = fens[len(scanned)]
            scanned.append(1)
            return [
                {
                    "fen": fen,
                    "player_color": "white",
                    "clock_remaining": 30.0,
                    "clock_bucket": "0-30s",
                    "opponent_rating": 1410,
                    "opponent_rating_band": "1200-1400",
                    "reason": "Time scramble blunder (<45s)",
                    "ply": 8,
                }
            ]

    monkeypatch.setattr(ur, "BlunderDetector", FakeDetector)

    train_calls: list[tuple[str, int, int]] = []

    def fake_train(username, candidates, pairs):
        train_calls.append((username, len(candidates), len(pairs)))
        # Simulate the checkpoint saved by the real trainer.
        (ur.USER_MODEL_DIR / f"{username}.pt").write_bytes(b"fake-checkpoint")
        return True

    monkeypatch.setattr(train_mod, "train_personalized_model", fake_train)

    first = client.post("/api/user/import", json={"username": "carol2", "max_months": 1})
    assert first.status_code == 200
    first_profile = first.json()
    assert first_profile["games_analyzed"] == 2
    assert first_profile["training_pairs"] > 0
    assert len(scanned) == 2

    # A new game appears on Chess.com; the cache still holds the old snapshot.
    state["live"].append(raw_game("3"))
    ur.LIVE_FETCH_CACHE.clear()  # simulate a new fetch after the TTL window
    second = client.post("/api/user/import", json={"username": "carol2", "max_months": 1})
    assert second.status_code == 200
    second_profile = second.json()

    # Only the newly available game is scanned by Stockfish.
    assert len(scanned) == 3
    assert second_profile["games_analyzed"] == 3
    assert second_profile["blunders_extracted"] == 3
    state["cached"] = list(state["live"])

    # A repeat import with no new games must not re-scan, inflate the bank, or
    # pay for an identical retrain: the cached private model is reused.
    third = client.post("/api/user/import", json={"username": "carol2", "max_months": 1})
    assert third.status_code == 200
    third_profile = third.json()
    assert len(scanned) == 3
    assert third_profile["games_analyzed"] == 3
    assert third_profile["blunders_extracted"] == 3
    assert third_profile["model_reused"] is True
    assert len(train_calls) == 2
    assert third_profile["personalized_model"] is True

    # New data triggers a fresh model; unchanged data does not.
    assert train_calls[0][0] == "carol2"
    assert train_calls[0][1] == 2
    assert train_calls[1][0] == "carol2"
    assert train_calls[1][1] == 3
    assert first.json()["model_reused"] is False
    assert second_profile["model_reused"] is False
    assert second_profile["personalized_model"] is True


def test_generate_new_scenario_avoids_recent_positions(isolated_storage, monkeypatch):
    """Each Generate click must serve a different scenario, even with baseline."""
    import src.api.routes.scenario as sc

    candidates = [
        {
            "fen": fen,
            "player_color": "white",
            "clock_remaining": 45.0,
            "clock_bucket": "30-60s",
            "opponent_rating_band": "1200-1400",
            "owner": "dave",
            "reason": "personal test blunder",
        }
        for fen in (FEN_A, FEN_B, FEN_C)
    ]
    CandidateBankManager.for_user("dave").save_candidates(candidates)
    sc._RECENT_SCENARIOS.clear()

    # Force the matcher to recommend the same (first) candidate every time;
    # the recent-FEN exclusion is what has to produce the variation.
    monkeypatch.setattr(sc.baseline, "fit", lambda candidates: None)
    monkeypatch.setattr(sc.baseline, "predict", lambda **kwargs: candidates[0])

    fens = []
    for _ in range(3):
        res = client.post(
            "/api/scenario/predict", json={"username": "dave", "use_baseline": True}
        )
        assert res.status_code == 200
        fens.append(res.json()["fen"])

    assert len(set(fens)) == 3


def test_neural_retrieval_excludes_recent_fen(isolated_storage):
    """The neural path samples from the ranking instead of reusing the argmax."""
    import torch
    import src.api.routes.scenario as sc

    sc._RECENT_SCENARIOS.clear()
    candidates = [{"fen": fen} for fen in (FEN_A, FEN_B, FEN_C)]
    logits = torch.tensor([[1000.0, 0.0, 0.0]])  # first candidate always wins initially

    first = sc._pick_varied_index(logits, candidates, "erin")
    second = sc._pick_varied_index(logits, candidates, "erin")

    assert first == 0
    assert second in (1, 2)
    assert sc._RECENT_SCENARIOS["erin"] == [candidates[first]["fen"], candidates[second]["fen"]]


def test_scenario_target_elo_comes_from_scenario_metadata():
    """Stockfish strength is derived from opponent band + clock, not a fixed 1400."""
    from src.api.routes.scenario import _scenario_target_elo

    time_scramble = _scenario_target_elo({}, 1400, "0-30s", "1200-1400")
    calm_endgame = _scenario_target_elo({}, 1400, "120s+", "1800+")

    assert time_scramble == 1200
    assert calm_endgame == 1800
    assert time_scramble != calm_endgame

    # A real opponent rating from the bank wins over the coarse band midpoint.
    stronger_opponent = _scenario_target_elo(
        {"opponent_rating": 1750}, 1400, "60-120s", "<1000"
    )
    assert stronger_opponent > time_scramble


def test_lichess_pgn_normalization_keeps_clock_annotations():
    from src.data.lichess_client import LichessClient

    pgn = """[Event "rated rapid game"]
[Site "https://lichess.org/abc123"]
[White "alice"]
[Black "bob"]
[Result "0-1"]
[WhiteElo "1350"]
[BlackElo "1410"]
[TimeControl "600+0"]

1. e4 { [%clk 0:09:58] } 1... e5 { [%clk 0:09:55] } 0-1
"""
    payload = {
        "id": "abc123",
        "perf": "rapid",
        "players": {
            "white": {"user": {"name": "alice"}, "rating": 1350},
            "black": {"user": {"name": "bob"}, "rating": 1410},
        },
        "pgn": pgn,
    }
    normalized = LichessClient._normalize_game(payload)

    assert normalized is not None
    assert "[%clk 0:09:58]" in normalized["pgn"]
    assert normalized["white"]["username"] == "alice"
    assert normalized["black"]["rating"] == 1410
    assert normalized["result"] == "0-1"
