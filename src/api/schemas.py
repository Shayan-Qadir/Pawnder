"""
Pydantic Schemas for Pawnder REST API
"""
from typing import Any, Optional
from pydantic import BaseModel, Field


class UserImportRequest(BaseModel):
    username: str = Field(..., description="Chess.com handle")
    max_months: int = Field(default=3, description="Months of game archives to fetch")


class UserProfileResponse(BaseModel):
    username: str
    total_games: int
    losses_analyzed: int
    blunders_extracted: int
    weaknesses: list[str]
    sample_blunders: list[dict[str, Any]]


class ScenarioPredictRequest(BaseModel):
    username: Optional[str] = None
    player_rating: int = 1200
    use_baseline: bool = False


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
    target_elo: int = 1400


class PracticeMoveResponse(BaseModel):
    is_legal: bool
    engine_move_uci: Optional[str] = None
    engine_move_san: Optional[str] = None
    fen_after: str
    evaluation_cp: float
    game_over: bool
    status: str


class SessionFeedbackRequest(BaseModel):
    scenario_id: str
    username: Optional[str] = "guest"
    result: str  # "win", "loss", "draw"
    user_feedback_score: Optional[int] = Field(default=5, ge=1, le=5)
    notes: Optional[str] = None
