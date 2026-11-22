"""
Pydantic Schemas for Pawnder REST API
"""
from typing import Any, Optional
from pydantic import BaseModel, Field


class UserImportRequest(BaseModel):
    username: str = Field(..., description="Chess.com handle (main platform)")
    max_months: int = Field(default=3, description="Months of game archives to fetch")


class UserProfileResponse(BaseModel):
    username: str
    source: str = "chesscom"
    player_rating: float = 1300.0
    total_games: int
    losses_analyzed: int
    blunders_extracted: int
    games_analyzed: int = 0
    training_pairs: int = 0
    weaknesses: list[str]
    sample_blunders: list[dict[str, Any]]
    personalized_model: bool = False
    model_reused: bool = False


class ScenarioPredictRequest(BaseModel):
    username: Optional[str] = None
    player_rating: int = 1200
    use_baseline: bool = False
    target_elo: Optional[int] = None  # legacy override; scenario now supplies its own Elo


class ScenarioResponse(BaseModel):
    scenario_id: str
    fen: str
    player_color: str
    clock_remaining: float
    clock_bucket: str
    opponent_rating_band: str
    target_elo: int
    reason: str


class PracticeMoveRequest(BaseModel):
    fen: str
    move_uci: str
    # Engine strength must come from the loaded scenario, never a fixed default.
    target_elo: int = Field(...)


class PracticeMoveResponse(BaseModel):
    is_legal: bool
    engine_move_uci: Optional[str] = None
    engine_move_san: Optional[str] = None
    fen_after_player: Optional[str] = None  # board right after the player's move
    fen_after: str
    evaluation_cp: float
    game_over: bool
    status: str
    result: Optional[str] = None  # board.result() when the game is over


class SessionFeedbackRequest(BaseModel):
    scenario_id: str
    username: Optional[str] = "guest"
    result: str  # "win", "loss", "draw"
    user_feedback_score: Optional[int] = Field(default=5, ge=1, le=5)
    notes: Optional[str] = None
