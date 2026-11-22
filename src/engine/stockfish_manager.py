"""
Stockfish Engine Manager (UCI Subprocess + Fallback Engine)
Manages chess engine evaluation and opponent moves with skill and Elo controls.
"""
import logging
import os
import shutil
import subprocess
import time
from typing import Optional, Tuple
import chess

logger = logging.getLogger(__name__)


class StockfishManager:
    """Manages Stockfish UCI engine interactions or falls back to internal alpha-beta."""

    def __init__(
        self,
        binary_path: Optional[str] = None,
        default_depth: int = 12,
        default_elo: int = 1500,
    ):
        self.default_depth = default_depth
        self.default_elo = default_elo
        self.binary_path = binary_path or self._find_stockfish()
        self.process: Optional[subprocess.Popen] = None
        if self.binary_path:
            self._start_engine()

    def _find_stockfish(self) -> Optional[str]:
        """Locates stockfish via env var, PATH, or project dirs."""
        configured = os.getenv("STOCKFISH_PATH", "")
        if configured and os.path.exists(configured):
            return configured
        path = shutil.which("stockfish") or shutil.which("stockfish.exe")
        if not path:
            candidate_paths = [
                os.path.join(os.getcwd(), "bin", "stockfish.exe"),
                os.path.join(os.getcwd(), "stockfish.exe"),
            ]
            for p in candidate_paths:
                if os.path.exists(p):
                    return p
        return path

    def _start_engine(self) -> None:
        if not self.binary_path:
            return
        try:
            self.process = subprocess.Popen(
                [self.binary_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            self._send_command("uci")
            self._wait_for("uciok")
            self._send_command("isready")
            self._wait_for("readyok")
            self.set_elo(self.default_elo)
            logger.info(f"Stockfish engine initialized at: {self.binary_path}")
        except Exception as e:
            logger.warning(f"Could not launch Stockfish binary ({e}). Using internal engine.")
            self.process = None

    def _send_command(self, cmd: str) -> None:
        if self.process and self.process.stdin:
            self.process.stdin.write(f"{cmd}\n")
            self.process.stdin.flush()

    def _wait_for(self, target: str, timeout: float = 3.0) -> list[str]:
        lines = []
        if not self.process or not self.process.stdout:
            return lines
        start = time.time()
        while time.time() - start < timeout:
            line = self.process.stdout.readline()
            if not line:
                break
            line_str = line.strip()
            lines.append(line_str)
            if target in line_str:
                break
        return lines

    def set_elo(self, elo: int) -> None:
        """Sets engine strength via UCI options."""
        if not self.process:
            return
        self._send_command("setoption name UCI_LimitStrength value true")
        self._send_command(f"setoption name UCI_Elo value {max(1000, min(2800, elo))}")
        skill_level = max(0, min(20, int((elo - 1000) / 70)))
        self._send_command(f"setoption name Skill Level value {skill_level}")
        self._send_command("isready")
        self._wait_for("readyok")

    def evaluate_position(self, fen: str, depth: Optional[int] = None) -> float:
        """
        Returns centipawn evaluation from White's perspective.
        """
        d = depth or self.default_depth
        if self.process:
            self._send_command(f"position fen {fen}")
            self._send_command(f"go depth {d}")
            lines = self._wait_for("bestmove", timeout=3.0)
            score_cp = 0.0
            for line in reversed(lines):
                if "score cp" in line:
                    parts = line.split()
                    try:
                        idx = parts.index("cp")
                        score_cp = float(parts[idx + 1])
                        # Check turn
                        board = chess.Board(fen)
                        if board.turn == chess.BLACK:
                            score_cp = -score_cp
                        return score_cp
                    except (ValueError, IndexError):
                        pass
                elif "score mate" in line:
                    parts = line.split()
                    try:
                        idx = parts.index("mate")
                        mate_in = int(parts[idx + 1])
                        score_cp = 10000.0 if mate_in > 0 else -10000.0
                        board = chess.Board(fen)
                        if board.turn == chess.BLACK:
                            score_cp = -score_cp
                        return score_cp
                    except (ValueError, IndexError):
                        pass

        # Fallback evaluation
        return self._fallback_eval(fen)

    def get_best_move(self, fen: str, movetime_ms: int = 400, elo: Optional[int] = None) -> str:
        """
        Returns best move in UCI format (e.g., 'e2e4').
        """
        if elo:
            self.set_elo(elo)

        if self.process:
            self._send_command(f"position fen {fen}")
            self._send_command(f"go movetime {movetime_ms}")
            lines = self._wait_for("bestmove", timeout=3.0)
            for line in lines:
                if line.startswith("bestmove"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return parts[1]

        # Fallback minimax move
        return self._fallback_best_move(fen)

    def _fallback_eval(self, fen: str) -> float:
        """Heuristic evaluation for fallback mode."""
        board = chess.Board(fen)
        piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 0,
        }
        score = 0
        for sq, piece in board.piece_map().items():
            val = piece_values.get(piece.piece_type, 0)
            score += val if piece.color == chess.WHITE else -val
        return float(score)

    def _fallback_best_move(self, fen: str) -> str:
        """Simple legal move chooser with capture preference."""
        board = chess.Board(fen)
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return "0000"

        # Prefer captures or checks
        captures = [m for m in legal_moves if board.is_capture(m)]
        if captures:
            return captures[0].uci()
        return legal_moves[0].uci()

    def close(self) -> None:
        if self.process:
            try:
                self._send_command("quit")
                self.process.terminate()
            except Exception:
                pass
            self.process = None
