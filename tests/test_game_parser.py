import pytest
from src.data.game_parser import GameParser, parse_clock_string

SAMPLE_PGN = """[Event "Live Chess"]
[Site "Chess.com"]
[Date "2026.11.15"]
[White "bonc_1"]
[Black "opponent_master"]
[Result "0-1"]
[WhiteElo "1350"]
[BlackElo "1410"]
[TimeControl "600"]

1. e4 {[%clk 0:09:58]} 1... e5 {[%clk 0:09:55]} 2. Nf3 {[%clk 0:09:40]} 2... Nc6 {[%clk 0:09:30]} 0-1
"""

def test_parse_clock_string():
    clk1 = parse_clock_string("[%clk 0:09:58]")
    assert clk1 == 598.0
    clk2 = parse_clock_string("[%clk 0:00:15.5]")
    assert clk2 == 15.5
    assert parse_clock_string("") is None

def test_parse_pgn():
    parsed = GameParser.parse_pgn(SAMPLE_PGN, target_username="bonc_1")
    assert parsed is not None
    assert parsed["white"] == "bonc_1"
    assert parsed["black"] == "opponent_master"
    assert parsed["player_color"] == "white"
    assert parsed["player_won"] == 0  # 0-1 means white lost
    assert parsed["white_rating"] == 1350
    assert len(parsed["moves"]) == 4
    assert parsed["moves"][0]["move_san"] == "e4"
    assert parsed["moves"][0]["white_clock"] == 598.0
