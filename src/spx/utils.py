"""Utility functions and helpers."""

from datetime import date as Date, datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np


def parse_season_string(season: str) -> tuple[int, int]:
    """Parse season string into start and end years.
    
    Args:
        season: Season string like '2023-24'
        
    Returns:
        Tuple of (start_year, end_year)
    """
    if '-' in season:
        start_str, end_str = season.split('-')
        start_year = int(start_str)
        
        # Handle 2-digit end year
        if len(end_str) == 2:
            end_year = int(f"20{end_str}")
        else:
            end_year = int(end_str)
    else:
        # Single year format
        start_year = int(season)
        end_year = start_year + 1
    
    return start_year, end_year


def get_season_from_date(match_date: Date) -> str:
    """Get season string from match date.
    
    Args:
        match_date: Date of the match
        
    Returns:
        Season string like '2023-24'
    """
    year = match_date.year
    
    # Football seasons typically start in August
    if match_date.month >= 8:
        start_year = year
        end_year = year + 1
    else:
        start_year = year - 1
        end_year = year
    
    return f"{start_year}-{end_year % 100:02d}"


def calculate_rest_days(last_match_date: Date, current_match_date: Date) -> int:
    """Calculate rest days between matches.
    
    Args:
        last_match_date: Date of previous match
        current_match_date: Date of current match
        
    Returns:
        Number of rest days
    """
    return (current_match_date - last_match_date).days


def normalize_probabilities(probs: np.ndarray) -> np.ndarray:
    """Normalize probabilities to sum to 1.
    
    Args:
        probs: Array of probabilities
        
    Returns:
        Normalized probabilities
    """
    total = probs.sum()
    if total == 0:
        return np.ones_like(probs) / len(probs)
    return probs / total


def outcome_to_numeric(outcome: str) -> int:
    """Convert outcome string to numeric.
    
    Args:
        outcome: Outcome string ('H', 'D', 'A')
        
    Returns:
        Numeric outcome (0=away win, 1=draw, 2=home win)
    """
    mapping = {'A': 0, 'D': 1, 'H': 2}
    return mapping[outcome]


def numeric_to_outcome(numeric: int) -> str:
    """Convert numeric outcome to string.
    
    Args:
        numeric: Numeric outcome (0, 1, 2)
        
    Returns:
        Outcome string ('A', 'D', 'H')
    """
    mapping = {0: 'A', 1: 'D', 2: 'H'}
    return mapping[numeric]


def calculate_points(home_goals: int, away_goals: int) -> tuple[int, int]:
    """Calculate points for home and away teams.
    
    Args:
        home_goals: Home team goals
        away_goals: Away team goals
        
    Returns:
        Tuple of (home_points, away_points)
    """
    if home_goals > away_goals:
        return 3, 0  # Home win
    elif home_goals < away_goals:
        return 0, 3  # Away win
    else:
        return 1, 1  # Draw


def moving_average(values: List[float], window: int) -> List[float]:
    """Calculate moving average.
    
    Args:
        values: List of values
        window: Window size
        
    Returns:
        List of moving averages
    """
    if len(values) < window:
        return [np.mean(values)] * len(values)
    
    result = []
    for i in range(len(values)):
        start_idx = max(0, i - window + 1)
        window_values = values[start_idx:i + 1]
        result.append(np.mean(window_values))
    
    return result


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division with default value.
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if division by zero
        
    Returns:
        Division result or default
    """
    if denominator == 0:
        return default
    return numerator / denominator


def create_team_mapping(teams: List[str]) -> Dict[str, int]:
    """Create team name to index mapping.
    
    Args:
        teams: List of team names
        
    Returns:
        Dictionary mapping team names to indices
    """
    return {team: idx for idx, team in enumerate(sorted(teams))}


def validate_probability_distribution(probs: np.ndarray, tolerance: float = 1e-6) -> bool:
    """Validate that probabilities form a valid distribution.
    
    Args:
        probs: Array of probabilities
        tolerance: Tolerance for sum check
        
    Returns:
        True if valid probability distribution
    """
    # Check non-negative
    if np.any(probs < 0):
        return False
    
    # Check sums to 1
    if abs(probs.sum() - 1.0) > tolerance:
        return False
    
    return True


def format_scoreline(home_goals: int, away_goals: int) -> str:
    """Format scoreline as string.
    
    Args:
        home_goals: Home team goals
        away_goals: Away team goals
        
    Returns:
        Formatted scoreline (e.g., '2-1')
    """
    return f"{home_goals}-{away_goals}"


def parse_scoreline(scoreline: str) -> tuple[int, int]:
    """Parse scoreline string.
    
    Args:
        scoreline: Scoreline string (e.g., '2-1')
        
    Returns:
        Tuple of (home_goals, away_goals)
    """
    try:
        home_str, away_str = scoreline.split('-')
        return int(home_str), int(away_str)
    except ValueError:
        raise ValueError(f"Invalid scoreline format: {scoreline}")


def get_current_gameweek(fixtures: List[Any], current_date: Date) -> int:
    """Get current gameweek number.
    
    Args:
        fixtures: List of fixtures
        current_date: Current date
        
    Returns:
        Current gameweek number
    """
    # Simple implementation - count completed gameweeks
    gameweek = 1
    
    for fixture in fixtures:
        if fixture.date <= current_date:
            if hasattr(fixture, 'gameweek') and fixture.gameweek:
                gameweek = max(gameweek, fixture.gameweek)
    
    return gameweek
