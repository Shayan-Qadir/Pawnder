"""
Practice Routes: Interactive Stockfish Engine Continuation & Feedback Logging
"""
import logging
from typing import Any
import chess
from fastapi import APIRouter, HTTPException
from src.api.schemas import (
    PracticeMoveRequest,
    PracticeMoveResponse,
    SessionFeedbackRequest,
)
from src.engine.stockfish_manager import StockfishManager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/practice", tags=["Practice"])

engine = StockfishManager()
SESSION_LOGS: list[dict[str, Any]] = []


@router.post("/move", response_model=PracticeMoveResponse)
def play_move(req: PracticeMoveRequest):
    """
    Validates player move, pushes to board, and requests Stockfish counter-move
    with strength limited to the scenario's opponent rating band.
    """
    try:
        board = chess.Board(req.fen)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid board FEN")

    # Parse and validate player move
    try:
        move = chess.Move.from_uci(req.move_uci)
        if move not in board.legal_moves:
            return PracticeMoveResponse(
                is_legal=False,
                fen_after=board.fen(),
                evaluation_cp=0.0,
                game_over=board.is_game_over(),
                status="Illegal move",
            )
    except Exception:
        return PracticeMoveResponse(
            is_legal=False,
            fen_after=board.fen(),
            evaluation_cp=0.0,
            game_over=board.is_game_over(),
            status="Invalid move UCI format",
        )

    # Push player move
    board.push(move)

    if board.is_game_over():
        return PracticeMoveResponse(
            is_legal=True,
            fen_after=board.fen(),
            evaluation_cp=0.0,
            game_over=True,
            status=f"Game Over: {board.result()}",
        )

    # Request Stockfish response move
    engine_move_uci = engine.get_best_move(
        board.fen(), movetime_ms=350, elo=req.target_elo
    )
    engine_move_san = None

    if engine_move_uci and engine_move_uci != "0000":
        try:
            eng_move = chess.Move.from_uci(engine_move_uci)
            if eng_move in board.legal_moves:
                engine_move_san = board.san(eng_move)
                board.push(eng_move)
        except Exception:
            pass

    eval_score = engine.evaluate_position(board.fen())

    return PracticeMoveResponse(
        is_legal=True,
        engine_move_uci=engine_move_uci,
        engine_move_san=engine_move_san,
        fen_after=board.fen(),
        evaluation_cp=eval_score,
        game_over=board.is_game_over(),
        status="In progress" if not board.is_game_over() else f"Game Over: {board.result()}",
    )


@router.post("/feedback")
def submit_session_feedback(req: SessionFeedbackRequest):
    """Logs practice session completion and user feedback."""
    entry = req.model_dump()
    SESSION_LOGS.append(entry)
    logger.info(f"Recorded practice session: {entry}")
    return {"status": "saved", "total_sessions_logged": len(SESSION_LOGS)}


@router.get("/sessions")
def list_session_feedback():
    return {"sessions": SESSION_LOGS}
