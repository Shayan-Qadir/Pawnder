"""
DVC stage: extract blunder scenarios from the Lichess BASE dataset.

Unlike per-user extraction, every base game is parsed with the player who lost
that game as the target. The resulting bank trains the global model; it is
never mixed into a player's private bank.

Output: data/candidate_bank/lichess_base.json
"""
import json
import logging
from typing import Any

from src.config import RAW_DATA_DIR, CANDIDATE_BANK_DIR
from src.data.chess_com_client import load_params
from src.data.game_parser import GameParser
from src.data.blunder_detector import BlunderDetector
from src.engine.stockfish_manager import StockfishManager
from src.data.lichess_base_ingest import BASE_GAMES_FILE

logger = logging.getLogger(__name__)

BASE_BANK_FILE = "lichess_base.json"
BASE_OWNER = "__base__"


def extract_base_candidates(
    raw_games: list[dict[str, Any]], detector: BlunderDetector, max_games: int
) -> list[dict[str, Any]]:
    """Parse each base game for its loser and collect unique blunder positions."""
    all_scenarios: list[dict[str, Any]] = []
    parsed_count = 0

    for raw in raw_games[:max_games]:
        target = raw.get("_target_username")
        if not target:
            continue
        parsed = GameParser.parse_pgn(raw.get("pgn", ""), target_username=target)
        if parsed is None or parsed.get("player_won") != 0:
            continue
        parsed_count += 1
        for scenario in detector.extract_scenarios_from_game(parsed):
            scenario["owner"] = BASE_OWNER
            scenario["_platform"] = "lichess"
            scenario["_source"] = "lichess_base"
            all_scenarios.append(scenario)

    seen: set[str] = set()
    candidates: list[dict[str, Any]] = []
    for scenario in all_scenarios:
        if scenario["fen"] in seen:
            continue
        seen.add(scenario["fen"])
        candidates.append(scenario)

    logger.info("Extracted %d base scenarios from %d lost games", len(candidates), parsed_count)
    return candidates


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    params = load_params()
    cfg = params.get("base_data", {})

    raw_path = RAW_DATA_DIR / BASE_GAMES_FILE
    if not raw_path.exists():
        raise SystemExit(f"Base games not found: {raw_path}. Run the ingest_base stage first.")
    with open(raw_path, "r", encoding="utf-8") as f:
        raw_games = json.load(f)

    engine = StockfishManager(default_depth=int(params.get("engine", {}).get("depth", 6)))
    if engine.process is None:
        logger.warning("Stockfish unavailable; base extraction falls back to material heuristic")

    blunder_params = params.get("blunder", {})
    detector = BlunderDetector(
        cp_drop_threshold=float(blunder_params.get("cp_drop_threshold", 150.0)),
        win_prob_drop_threshold=float(blunder_params.get("win_prob_drop_threshold", 0.20)),
        engine_evaluator=engine,
    )

    candidates = extract_base_candidates(
        raw_games, detector, max_games=int(cfg.get("max_extract_games", 30))
    )

    out_path = CANDIDATE_BANK_DIR / BASE_BANK_FILE
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(candidates, f, indent=2)
    engine.close()
    print(f"Extracted {len(candidates)} Lichess base scenarios -> {out_path}")
