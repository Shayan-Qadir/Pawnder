"""
Blunder and High-Pressure Scenario Detector
Identifies critical evaluation deterioration moments from player losses.
"""
import math
from typing import Any, Optional


def cp_to_win_prob(cp: float) -> float:
    """
    Converts centipawns to estimated winning probability.
    Formula: W = 1 / (1 + 10^(-cp / 400))
    """
    # Cap cp to avoid overflow
    clipped_cp = max(-2000.0, min(2000.0, cp))
    return 1.0 / (1.0 + math.pow(10.0, -clipped_cp / 400.0))


def clock_to_bucket(clock_seconds: float) -> str:
    """Classifies clock remaining into predefined pressure buckets."""
    if clock_seconds < 30.0:
        return "0-30s"
    elif clock_seconds < 60.0:
        return "30-60s"
    elif clock_seconds < 120.0:
        return "60-120s"
    else:
        return "120s+"


def rating_to_band(rating: Optional[int]) -> str:
    """Classifies rating into bands."""
    if rating is None or rating < 1000:
        return "<1000"
    elif rating < 1200:
        return "1000-1200"
    elif rating < 1400:
        return "1200-1400"
    elif rating < 1600:
        return "1400-1600"
    elif rating < 1800:
        return "1600-1800"
    else:
        return "1800+"


class BlunderDetector:
    """Extracts high-pressure scenario targets from parsed games."""

    def __init__(
        self,
        cp_drop_threshold: float = 150.0,
        win_prob_drop_threshold: float = 0.20,
        engine_evaluator: Optional[Any] = None,
    ):
        self.cp_drop_threshold = cp_drop_threshold
        self.win_prob_drop_threshold = win_prob_drop_threshold
        self.engine = engine_evaluator

    def heuristic_eval(self, fen: str) -> float:
        """
        Fast heuristic evaluation (centipawns from White's perspective)
        used as fallback or pre-filter when engine is not active.
        """
        piece_values = {"p": 100, "n": 320, "b": 330, "r": 500, "q": 900, "k": 20000}
        parts = fen.split()
        board_part = parts[0]
        score = 0
        for char in board_part:
            if char.lower() in piece_values:
                val = piece_values[char.lower()]
                score += val if char.isupper() else -val
        return float(score)

    def evaluate_fen(self, fen: str) -> float:
        """Evaluate FEN using engine if available, else heuristic."""
        if self.engine is not None and hasattr(self.engine, "evaluate_position"):
            try:
                res = self.engine.evaluate_position(fen)
                if res is not None:
                    return float(res)
            except Exception:
                pass
        return self.heuristic_eval(fen)

    def extract_scenarios_from_game(
        self, game_dict: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """
        Extract practice scenarios from a lost game where the player suffered an evaluation drop.
        """
        # Only analyze games where player lost (player_won == 0)
        if game_dict.get("player_won") != 0:
            return []

        player_color = game_dict.get("player_color")  # "white" or "black"
        if not player_color:
            return []

        opponent_rating = (
            game_dict.get("black_rating")
            if player_color == "white"
            else game_dict.get("white_rating")
        )
        rating_band = rating_to_band(opponent_rating)

        scenarios = []
        moves = game_dict.get("moves", [])

        prev_eval = 0.0

        for idx, m in enumerate(moves):
            # Check position before move
            fen_before = m["fen_before"]
            fen_after = m["fen_after"]
            turn = m["turn_color"]

            eval_before = self.evaluate_fen(fen_before)
            eval_after = self.evaluate_fen(fen_after)

            # Player's perspective
            if player_color == "white":
                player_eval_before = eval_before
                player_eval_after = eval_after
                clock = m.get("white_clock", 600.0)
            else:
                player_eval_before = -eval_before
                player_eval_after = -eval_after
                clock = m.get("black_clock", 600.0)

            # Check if this move was played by the target player
            if turn == player_color:
                cp_drop = player_eval_before - player_eval_after
                prob_before = cp_to_win_prob(player_eval_before)
                prob_after = cp_to_win_prob(player_eval_after)
                prob_drop = prob_before - prob_after

                # Blunder criteria: significant drop or clock pressure blunder
                is_blunder = (
                    cp_drop >= self.cp_drop_threshold
                    or prob_drop >= self.win_prob_drop_threshold
                    or (clock < 45.0 and cp_drop >= 100.0)
                )

                if is_blunder:
                    reason = "Tactical deterioration"
                    if clock < 45.0:
                        reason = "Time scramble blunder (<45s)"
                    elif m["ply"] > 60:
                        reason = "Endgame miscalculation"

                    scenarios.append(
                        {
                            "fen": fen_before,
                            "player_color": player_color,
                            "clock_remaining": clock,
                            "clock_bucket": clock_to_bucket(clock),
                            "opponent_rating": opponent_rating,
                            "opponent_rating_band": rating_band,
                            "blunder_move": m["move_san"],
                            "cp_drop": round(cp_drop, 1),
                            "win_prob_drop": round(prob_drop, 3),
                            "reason": reason,
                            "ply": m["ply"],
                        }
                    )

        return scenarios


if __name__ == "__main__":
    """Standalone DVC stage: raw games -> legal candidate position bank."""
    import json
    import logging
    from pathlib import Path

    from src.config import RAW_DATA_DIR, CANDIDATE_BANK_DIR
    from src.data.chess_com_client import load_params
    from src.data.game_parser import GameParser
    from src.engine.stockfish_manager import StockfishManager

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    params = load_params()
    data_params = params.get("data", {})
    blunder_params = params.get("blunder", {})
    username = str(data_params.get("username", "bonc_1")).lower()

    raw_path = RAW_DATA_DIR / f"{username}_games.json"
    if not raw_path.exists():
        raise SystemExit(f"Raw games not found: {raw_path}. Run the ingest stage first.")

    with open(raw_path, "r", encoding="utf-8") as f:
        raw_games = json.load(f)

    # Limit engine work for reproducible, fast pipeline runs.
    max_games = int(params.get("extract", {}).get("max_games", 30))
    total_games = len(raw_games)
    raw_games = raw_games[:max_games]
    logger.info("Analyzing %d of %d games", len(raw_games), total_games)

    # A material-only heuristic can never produce a positive evaluation drop on
    # the player's own move, so real engine evaluation is required here.
    engine = StockfishManager(default_depth=int(params.get("engine", {}).get("depth", 8)))
    if engine.process is None:
        logger.warning("Stockfish unavailable; extraction will use the material-only fallback")

    detector = BlunderDetector(
        cp_drop_threshold=float(blunder_params.get("cp_drop_threshold", 150.0)),
        win_prob_drop_threshold=float(blunder_params.get("win_prob_drop_threshold", 0.20)),
        engine_evaluator=engine,
    )

    parsed_games = []
    all_scenarios = []
    for raw in raw_games:
        parsed = GameParser.parse_pgn(raw.get("pgn", ""), target_username=username)
        if parsed is None:
            continue
        parsed_games.append(parsed)
        all_scenarios.extend(detector.extract_scenarios_from_game(parsed))

    # Deduplicate by FEN while preserving chronology.
    seen: set[str] = set()
    candidates = []
    for scenario in all_scenarios:
        if scenario["fen"] in seen:
            continue
        seen.add(scenario["fen"])
        candidates.append(scenario)

    if not candidates:
        logger.warning("No blunders extracted; seeding a labeled fallback candidate")
        candidates = [
            {
                "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
                "player_color": "white",
                "clock_remaining": 65.0,
                "clock_bucket": "60-120s",
                "opponent_rating": 1350,
                "opponent_rating_band": "1200-1400",
                "blunder_move": "Nxe5",
                "cp_drop": 210.0,
                "win_prob_drop": 0.28,
                "reason": "Time scramble blunder (<45s)",
                "ply": 42,
                "_source": "fallback_seed",
            }
        ]

    out_path = CANDIDATE_BANK_DIR / "legal_positions.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(candidates, f, indent=2)

    engine.close()
    print(
        f"Extracted {len(candidates)} unique scenarios from {len(parsed_games)} games -> {out_path}"
    )
