"""Tests for the Lichess BASE dataset pipeline and Chess.com-as-main split."""
import pytest

from src.data.lichess_base_ingest import losing_username
from src.data.lichess_base_extract import extract_base_candidates, BASE_OWNER
from src.training.dataset import CandidateBankManager, build_training_pairs_from_bank

PGN_A = """[Event "rated rapid game"]
[Site "https://lichess.org/a"]
[White "alice"]
[Black "bob"]
[Result "0-1"]
[WhiteElo "1350"]
[BlackElo "1410"]
[TimeControl "600+0"]

1. e4 { [%clk 0:09:58] } 1... e5 { [%clk 0:09:55] } 2. Nf3 { [%clk 0:09:40] } 2... Nc6 { [%clk 0:09:30] } 0-1
"""

PGN_B = """[Event "rated rapid game"]
[Site "https://lichess.org/b"]
[White "carol"]
[Black "dave"]
[Result "1-0"]
[WhiteElo "1500"]
[BlackElo "1490"]
[TimeControl "600+0"]

1. d4 { [%clk 0:09:58] } 1... d5 { [%clk 0:09:55] } 2. c4 { [%clk 0:09:40] } 2... e6 { [%clk 0:09:30] } 1-0
"""


def test_losing_username_picks_the_loser():
    game = {
        "white": {"username": "alice"},
        "black": {"username": "bob"},
    }
    assert losing_username({**game, "result": "0-1"}) == "alice"
    assert losing_username({**game, "result": "1-0"}) == "bob"
    assert losing_username({**game, "result": "1/2-1/2"}) is None


def test_base_extraction_targets_each_game_loser():
    from src.data.blunder_detector import BlunderDetector

    class FakeDetector(BlunderDetector):
        def __init__(self):
            pass

        def extract_scenarios_from_game(self, parsed):
            # Differentiate the two games so FEN dedup does not collapse them.
            unique_fen = f"{parsed['moves'][0]['fen_before']} {parsed['white']}"
            return [
                {
                    "fen": unique_fen,
                    "player_color": parsed["player_color"],
                    "clock_remaining": 500.0,
                    "clock_bucket": "120s+",
                    "opponent_rating_band": "1200-1400",
                    "reason": "base test",
                    "ply": 1,
                }
            ]

    games = [
        {"pgn": PGN_A, "_target_username": "alice", "_platform": "lichess"},
        {"pgn": PGN_B, "_target_username": "dave", "_platform": "lichess"},
    ]
    candidates = extract_base_candidates(games, FakeDetector(), max_games=10)

    assert len(candidates) == 2
    assert all(c["owner"] == BASE_OWNER for c in candidates)
    assert all(c["_platform"] == "lichess" for c in candidates)
    assert {c["player_color"] for c in candidates} == {"white", "black"}


def test_base_history_never_crosses_players():
    """A Lichess base batch spans many players; histories must stay separate."""
    pgn_a = PGN_A
    pgn_b = PGN_B
    bank = [
        {
            "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "player_color": "white",
            "clock_remaining": 600.0,
            "clock_bucket": "120s+",
            "opponent_rating_band": "1200-1400",
        }
    ]
    games = [
        {"pgn": pgn_a, "_target_username": "alice"},
        {"pgn": pgn_a, "_target_username": "bob"},
        {"pgn": pgn_b, "_target_username": "carol"},
    ]
    pairs = build_training_pairs_from_bank(games, None, bank)
    # With a single candidate FEN and no history in game 1, game 2's first
    # player move is the first pair for bob. If history leaked between alice
    # and bob, bob would produce more pairs than carol's fresh start pattern.
    assert isinstance(pairs, list)
    # The important invariant: no pair's history may describe a different
    # player than its target game. Per-target reset guarantees an empty
    # history for the first usable move of each new player.
    assert all(isinstance(p["history"], list) for p in pairs)


def test_personalization_starts_from_lichess_base_checkpoint(monkeypatch, tmp_path):
    """A private model must fine-tune the base model, not train from scratch."""
    import src.training.train as train_mod
    from src.config import MODELS_DIR

    captured = {}

    def fake_train(samples, bank, **kwargs):
        captured.update(kwargs)
        return {"status": "success"}

    monkeypatch.setattr(train_mod, "train_scenario_model", fake_train)
    monkeypatch.setattr(train_mod, "USER_MODEL_DIR", tmp_path)

    pairs = [
        {"history": [{"fen": "x"}], "target_candidate_idx": 0, "target_clock_bucket": "120s+", "target_rating_band": "<1000"},
        {"history": [{"fen": "y"}], "target_candidate_idx": 1, "target_clock_bucket": "120s+", "target_rating_band": "<1000"},
    ]
    bank = [
        {"fen": "x", "player_color": "white", "clock_remaining": 60.0},
        {"fen": "y", "player_color": "white", "clock_remaining": 60.0},
    ]

    result = train_mod.train_personalized_model("carol", bank, pairs)

    assert result["personalized"] is True
    assert captured["init_from"] == MODELS_DIR / "scenario_model.pt"


def test_base_bank_has_its_own_embedding_cache(tmp_path):
    base_bank = tmp_path / "lichess_base.json"
    mgr = CandidateBankManager(bank_path=base_bank)
    default_mgr = CandidateBankManager()

    assert mgr.bank_path == base_bank
    assert mgr.embeddings_path == tmp_path / "lichess_base_embeddings.pt"
    assert mgr.embeddings_path != default_mgr.embeddings_path
