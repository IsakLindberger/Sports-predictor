"""Test Elo rating system."""

from datetime import date

import pytest

from spx.core.ratings import EloRatingSystem
from spx.core.schema import Match, MatchOutcome


class TestEloRatingSystem:
    """Test Elo rating system."""
    
    def test_initialization(self):
        """Test Elo system initialization."""
        elo = EloRatingSystem(
            k_factor=20.0,
            base_rating=1500.0,
            home_advantage=30.0
        )
        
        assert elo.k_factor == 20.0
        assert elo.base_rating == 1500.0
        assert elo.home_advantage == 30.0
        assert len(elo.ratings) == 0
    
    def test_team_initialization(self):
        """Test new team initialization."""
        elo = EloRatingSystem()
        elo._initialize_team("Arsenal")
        
        assert "Arsenal" in elo.ratings
        assert elo.ratings["Arsenal"] == 1500.0
        assert elo.games_played["Arsenal"] == 0
    
    def test_rating_update_home_win(self):
        """Test rating update for home win."""
        elo = EloRatingSystem(k_factor=20.0, base_rating=1500.0, home_advantage=30.0)
        
        match = Match(
            date=date(2024, 8, 11),
            season="2024-25",
            home_team="Arsenal",
            away_team="Wolves",
            home_goals=2,
            away_goals=0,
            outcome=MatchOutcome.HOME_WIN
        )
        
        ratings = elo.update_ratings([match])
        
        assert len(ratings) == 2  # One for each team
        assert elo.ratings["Arsenal"] > 1500.0  # Home team should gain rating
        assert elo.ratings["Wolves"] < 1500.0   # Away team should lose rating
    
    def test_rating_update_away_win(self):
        """Test rating update for away win."""
        elo = EloRatingSystem(k_factor=20.0, base_rating=1500.0, home_advantage=30.0)
        
        match = Match(
            date=date(2024, 8, 11),
            season="2024-25",
            home_team="Arsenal",
            away_team="Wolves",
            home_goals=0,
            away_goals=2,
            outcome=MatchOutcome.AWAY_WIN
        )
        
        ratings = elo.update_ratings([match])
        
        assert elo.ratings["Arsenal"] < 1500.0  # Home team should lose more (had advantage)
        assert elo.ratings["Wolves"] > 1500.0   # Away team should gain more
    
    def test_rating_update_draw(self):
        """Test rating update for draw."""
        elo = EloRatingSystem(k_factor=20.0, base_rating=1500.0, home_advantage=30.0)
        
        match = Match(
            date=date(2024, 8, 11),
            season="2024-25",
            home_team="Arsenal",
            away_team="Wolves",
            home_goals=1,
            away_goals=1,
            outcome=MatchOutcome.DRAW
        )
        
        ratings = elo.update_ratings([match])
        
        # In a draw, home team typically loses rating (had advantage)
        assert elo.ratings["Arsenal"] < 1500.0
        assert elo.ratings["Wolves"] > 1500.0
    
    def test_multiple_matches(self):
        """Test rating updates over multiple matches."""
        elo = EloRatingSystem()
        
        matches = [
            Match(
                date=date(2024, 8, 11),
                season="2024-25",
                home_team="Arsenal",
                away_team="Wolves",
                home_goals=2,
                away_goals=0,
                outcome=MatchOutcome.HOME_WIN
            ),
            Match(
                date=date(2024, 8, 18),
                season="2024-25",
                home_team="Wolves",
                away_team="Arsenal",
                home_goals=1,
                away_goals=2,
                outcome=MatchOutcome.AWAY_WIN
            )
        ]
        
        ratings = elo.update_ratings(matches)
        
        assert len(ratings) == 4  # 2 teams × 2 matches
        assert elo.games_played["Arsenal"] == 2
        assert elo.games_played["Wolves"] == 2
    
    def test_predict_match_outcome(self):
        """Test match outcome prediction."""
        elo = EloRatingSystem()
        
        # Initialize teams with different ratings
        elo.ratings["Strong Team"] = 1600.0
        elo.ratings["Weak Team"] = 1400.0
        
        home_prob, draw_prob, away_prob = elo.predict_match_outcome(
            "Strong Team", "Weak Team"
        )
        
        # Strong home team should be favored
        assert home_prob > away_prob
        assert 0 <= home_prob <= 1
        assert 0 <= draw_prob <= 1
        assert 0 <= away_prob <= 1
        assert abs(home_prob + draw_prob + away_prob - 1.0) < 1e-6
