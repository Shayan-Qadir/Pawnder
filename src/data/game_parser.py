"""
Game Parser and Clock/Board Reconstruction using python-chess
"""
import io
import re
from typing import Any, Optional
import chess
import chess.pgn


def parse_clock_string(comment: str) -> Optional[float]:
    """
    Extract remaining clock seconds from Chess.com PGN comment annotations.
    Format: [%clk 0:08:42.5] or [%clk 0:00:15]
    """
    if not comment:
        return None
    match = re.search(r"\[%clk\s+(\d+):(\d+):(\d+(?:\.\d+)?)\]", comment)
    if match:
        hours = int(match.group(1))
        minutes = int(match.group(2))
        seconds = float(match.group(3))
        return hours * 3600 + minutes * 60 + seconds
    return None


class ParsedMoveInfo:
    def __init__(
        self,
        ply: int,
        fen_before: str,
        move_san: str,
        move_uci: str,
        fen_after: str,
        turn_color: str,
        white_clock: Optional[float] = None,
        black_clock: Optional[float] = None,
    ):
        self.ply = ply
        self.fen_before = fen_before
        self.move_san = move_san
        self.move_uci = move_uci
        self.fen_after = fen_after
        self.turn_color = turn_color
        self.white_clock = white_clock
        self.black_clock = black_clock

    def to_dict(self) -> dict[str, Any]:
        return {
            "ply": self.ply,
            "fen_before": self.fen_before,
            "move_san": self.move_san,
            "move_uci": self.move_uci,
            "fen_after": self.fen_after,
            "turn_color": self.turn_color,
            "white_clock": self.white_clock,
            "black_clock": self.black_clock,
        }


class GameParser:
    """Parses PGN and reconstructs game states, clocks, and outcomes."""

    @staticmethod
    def parse_pgn(pgn_text: str, target_username: Optional[str] = None) -> Optional[dict[str, Any]]:
        """
        Parses PGN string into structured game dictionary with all move-by-move states.
        """
        pgn_io = io.StringIO(pgn_text)
        game = chess.pgn.read_game(pgn_io)
        if game is None:
            return None

        headers = game.headers
        white_player = headers.get("White", "").strip()
        black_player = headers.get("Black", "").strip()
        result = headers.get("Result", "*")
        time_control = headers.get("TimeControl", "600")

        # Determine target player role & outcome
        player_color = None
        player_won = None
        if target_username:
            uname = target_username.lower()
            if white_player.lower() == uname:
                player_color = "white"
                player_won = 1 if result == "1-0" else (0 if result == "0-1" else 0.5)
            elif black_player.lower() == uname:
                player_color = "black"
                player_won = 1 if result == "0-1" else (0 if result == "1-0" else 0.5)

        white_rating = int(headers.get("WhiteElo", 0)) if headers.get("WhiteElo", "").isdigit() else None
        black_rating = int(headers.get("BlackElo", 0)) if headers.get("BlackElo", "").isdigit() else None

        board = game.board()
        moves_list = []
        ply = 0

        # Initial clock times
        last_white_clock = 600.0
        last_black_clock = 600.0

        for node in game.mainline():
            move = node.move
            fen_before = board.fen()
            turn = "white" if board.turn == chess.WHITE else "black"
            
            # Extract clock from comment
            clock_val = parse_clock_string(node.comment)
            if turn == "white":
                if clock_val is not None:
                    last_white_clock = clock_val
            else:
                if clock_val is not None:
                    last_black_clock = clock_val

            san = board.san(move)
            uci = move.uci()
            board.push(move)
            fen_after = board.fen()
            ply += 1

            moves_list.append(
                ParsedMoveInfo(
                    ply=ply,
                    fen_before=fen_before,
                    move_san=san,
                    move_uci=uci,
                    fen_after=fen_after,
                    turn_color=turn,
                    white_clock=last_white_clock,
                    black_clock=last_black_clock,
                ).to_dict()
            )

        return {
            "white": white_player,
            "black": black_player,
            "white_rating": white_rating,
            "black_rating": black_rating,
            "result": result,
            "time_control": time_control,
            "date": headers.get("Date", ""),
            "player_color": player_color,
            "player_won": player_won,
            "total_plies": ply,
            "moves": moves_list,
        }
