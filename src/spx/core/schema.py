"""Data schemas and models using Pydantic."""

from datetime import date as Date, datetime
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class MatchOutcome(str, Enum):
    """Match outcome enum."""
    HOME_WIN = "H"
    DRAW = "D"
    AWAY_WIN = "A"


class Match(BaseModel):
    """Match data schema."""
    
    # Core identifiers
    date: Date = Field(..., description="Match date")
    season: str = Field(..., description="Season (e.g., '2023-24')")
    home_team: str = Field(..., description="Home team name")
    away_team: str = Field(..., description="Away team name")
    
    # Results
    home_goals: int = Field(..., ge=0, description="Home team goals")
    away_goals: int = Field(..., ge=0, description="Away team goals")
    outcome: MatchOutcome = Field(..., description="Match outcome")
    
    # Expected goals
    home_xg: Optional[float] = Field(None, ge=0, description="Home team expected goals")
    away_xg: Optional[float] = Field(None, ge=0, description="Away team expected goals")
    
    # Additional stats
    home_shots: Optional[int] = Field(None, ge=0, description="Home team shots")
    away_shots: Optional[int] = Field(None, ge=0, description="Away team shots")
    home_shots_on_target: Optional[int] = Field(None, ge=0, description="Home team shots on target")
    away_shots_on_target: Optional[int] = Field(None, ge=0, description="Away team shots on target")
    
    # Cards and fouls
    home_red_cards: Optional[int] = Field(None, ge=0, description="Home team red cards")
    away_red_cards: Optional[int] = Field(None, ge=0, description="Away team red cards")
    home_yellow_cards: Optional[int] = Field(None, ge=0, description="Home team yellow cards")
    away_yellow_cards: Optional[int] = Field(None, ge=0, description="Away team yellow cards")
    
    # Optional metadata
    referee: Optional[str] = Field(None, description="Referee name")
    attendance: Optional[int] = Field(None, ge=0, description="Match attendance")
    
    @field_validator('outcome', mode='before')
    @classmethod
    def determine_outcome(cls, v: str, info) -> str:
        """Automatically determine outcome from goals if not provided."""
        if v is not None:
            return v
            
        values = info.data if hasattr(info, 'data') else {}
        home_goals = values.get('home_goals')
        away_goals = values.get('away_goals')
        
        if home_goals is None or away_goals is None:
            raise ValueError("Cannot determine outcome without goals")
            
        if home_goals > away_goals:
            return MatchOutcome.HOME_WIN
        elif home_goals < away_goals:
            return MatchOutcome.AWAY_WIN
        else:
            return MatchOutcome.DRAW


class Fixture(BaseModel):
    """Future fixture schema."""
    
    date: Date = Field(..., description="Fixture date")
    season: str = Field(..., description="Season")
    home_team: str = Field(..., description="Home team name")
    away_team: str = Field(..., description="Away team name")
    
    # Optional scheduling info
    gameweek: Optional[int] = Field(None, description="Gameweek number")
    kick_off_time: Optional[str] = Field(None, description="Kick-off time")


class TeamRating(BaseModel):
    """Team rating schema."""
    
    team: str = Field(..., description="Team name")
    date: Date = Field(..., description="Rating date")
    elo_rating: float = Field(..., description="Elo rating")
    games_played: int = Field(..., ge=0, description="Games played")
    
    # Optional form metrics
    form_points: Optional[float] = Field(None, description="Recent form points")
    form_goals_for: Optional[float] = Field(None, description="Recent goals for")
    form_goals_against: Optional[float] = Field(None, description="Recent goals against")


class MatchPrediction(BaseModel):
    """Match prediction schema."""
    
    # Match info
    date: Date = Field(..., description="Match date")
    home_team: str = Field(..., description="Home team")
    away_team: str = Field(..., description="Away team")
    
    # Outcome probabilities
    prob_home_win: float = Field(..., ge=0, le=1, description="Home win probability")
    prob_draw: float = Field(..., ge=0, le=1, description="Draw probability") 
    prob_away_win: float = Field(..., ge=0, le=1, description="Away win probability")
    
    # Most likely scoreline
    most_likely_score: str = Field(..., description="Most likely scoreline (e.g., '2-1')")
    most_likely_prob: float = Field(..., ge=0, le=1, description="Probability of most likely score")
    
    # Expected goals
    expected_home_goals: float = Field(..., ge=0, description="Expected home goals")
    expected_away_goals: float = Field(..., ge=0, description="Expected away goals")
    
    # Model metadata
    model_version: str = Field(..., description="Model version used")
    prediction_timestamp: datetime = Field(..., description="When prediction was made")
    
    @field_validator('prob_home_win', 'prob_draw', 'prob_away_win')
    @classmethod
    def validate_probabilities(cls, v: float) -> float:
        """Validate probability values."""
        if not 0 <= v <= 1:
            raise ValueError("Probabilities must be between 0 and 1")
        return v
    
    @field_validator('most_likely_score')
    @classmethod
    def validate_score_format(cls, v: str) -> str:
        """Validate scoreline format."""
        try:
            home, away = v.split('-')
            int(home)
            int(away)
        except ValueError:
            raise ValueError("Score must be in format 'X-Y' where X and Y are integers")
        return v


class SeasonPrediction(BaseModel):
    """Season simulation prediction schema."""
    
    season: str = Field(..., description="Season")
    team: str = Field(..., description="Team name")
    
    # League position probabilities
    position_probs: Dict[int, float] = Field(..., description="Position probability distribution")
    expected_position: float = Field(..., description="Expected final position")
    
    # Key outcome probabilities
    prob_title: float = Field(..., ge=0, le=1, description="Title probability")
    prob_top4: float = Field(..., ge=0, le=1, description="Top 4 probability")
    prob_relegation: float = Field(..., ge=0, le=1, description="Relegation probability")
    
    # Expected season stats
    expected_points: float = Field(..., ge=0, description="Expected points")
    expected_goal_difference: float = Field(..., description="Expected goal difference")
    
    # Simulation metadata
    n_simulations: int = Field(..., description="Number of simulations run")
    simulation_timestamp: datetime = Field(..., description="When simulation was run")


class CalibrationMetrics(BaseModel):
    """Model calibration metrics."""
    
    model_name: str = Field(..., description="Model name")
    
    # Calibration scores
    brier_score: float = Field(..., description="Brier score")
    log_loss: float = Field(..., description="Logarithmic loss")
    expected_calibration_error: float = Field(..., description="Expected calibration error")
    
    # Reliability metrics
    reliability_bins: List[float] = Field(..., description="Reliability plot bin centers")
    reliability_frequencies: List[float] = Field(..., description="Observed frequencies per bin")
    reliability_confidences: List[float] = Field(..., description="Predicted probabilities per bin")
    
    # Evaluation period
    evaluation_start: Date = Field(..., description="Evaluation period start")
    evaluation_end: Date = Field(..., description="Evaluation period end")
    n_predictions: int = Field(..., description="Number of predictions evaluated")
