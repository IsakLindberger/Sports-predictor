"""Test schema validation."""

from datetime import date
import pytest

from spx.core.schema import Match, Fixture, MatchOutcome, MatchPrediction


class TestMatch:
    """Test Match schema."""
    
    def test_valid_match_creation(self):
        """Test creating a valid match."""
        match = Match(
            date=date(2024, 8, 11),
            season="2024-25",
            home_team="Arsenal",
            away_team="Wolves", 
            home_goals=2,
            away_goals=0,
            outcome=MatchOutcome.HOME_WIN
        )
        
        assert match.home_team == "Arsenal"
        assert match.away_team == "Wolves"
        assert match.home_goals == 2
        assert match.away_goals == 0
        assert match.outcome == MatchOutcome.HOME_WIN
    
    def test_automatic_outcome_determination(self):
        """Test that outcome is automatically determined from goals."""
        # Home win
        match = Match(
            date=date(2024, 8, 11),
            season="2024-25",
            home_team="Arsenal",
            away_team="Wolves",
            home_goals=2,
            away_goals=0,
            outcome='H'  # Will be validated
        )
        assert match.outcome == MatchOutcome.HOME_WIN
        
        # Away win  
        match = Match(
            date=date(2024, 8, 11),
            season="2024-25",
            home_team="Arsenal", 
            away_team="Wolves",
            home_goals=0,
            away_goals=2,
            outcome='A'
        )
        assert match.outcome == MatchOutcome.AWAY_WIN
        
        # Draw
        match = Match(
            date=date(2024, 8, 11),
            season="2024-25", 
            home_team="Arsenal",
            away_team="Wolves",
            home_goals=1,
            away_goals=1,
            outcome='D'
        )
        assert match.outcome == MatchOutcome.DRAW
    
    def test_invalid_negative_goals(self):
        """Test that negative goals are rejected."""
        with pytest.raises(ValueError):
            Match(
                date=date(2024, 8, 11),
                season="2024-25",
                home_team="Arsenal",
                away_team="Wolves",
                home_goals=-1,
                away_goals=0,
                outcome='H'
            )


class TestFixture:
    """Test Fixture schema."""
    
    def test_valid_fixture_creation(self):
        """Test creating a valid fixture."""
        fixture = Fixture(
            date=date(2024, 9, 7),
            season="2024-25",
            home_team="Arsenal",
            away_team="Man City"
        )
        
        assert fixture.home_team == "Arsenal"
        assert fixture.away_team == "Man City"
        assert fixture.date == date(2024, 9, 7)


class TestMatchPrediction:
    """Test MatchPrediction schema."""
    
    def test_valid_prediction_creation(self):
        """Test creating a valid prediction."""
        from datetime import datetime
        
        prediction = MatchPrediction(
            date=date(2024, 9, 7),
            home_team="Arsenal",
            away_team="Man City",
            prob_home_win=0.4,
            prob_draw=0.3,
            prob_away_win=0.3,
            most_likely_score="1-1",
            most_likely_prob=0.12,
            expected_home_goals=1.2,
            expected_away_goals=1.1,
            model_version="test_v1.0",
            prediction_timestamp=datetime.now()
        )
        
        assert prediction.prob_home_win == 0.4
        assert prediction.prob_draw == 0.3
        assert prediction.prob_away_win == 0.3
        assert prediction.most_likely_score == "1-1"
    
    def test_invalid_probabilities(self):
        """Test that invalid probabilities are rejected."""
        from datetime import datetime
        
        with pytest.raises(ValueError):
            MatchPrediction(
                date=date(2024, 9, 7),
                home_team="Arsenal",
                away_team="Man City",
                prob_home_win=1.5,  # Invalid: > 1
                prob_draw=0.3,
                prob_away_win=0.3,
                most_likely_score="1-1",
                most_likely_prob=0.12,
                expected_home_goals=1.2,
                expected_away_goals=1.1,
                model_version="test_v1.0",
                prediction_timestamp=datetime.now()
            )
    
    def test_invalid_scoreline_format(self):
        """Test that invalid scoreline formats are rejected."""
        from datetime import datetime
        
        with pytest.raises(ValueError):
            MatchPrediction(
                date=date(2024, 9, 7),
                home_team="Arsenal",
                away_team="Man City",
                prob_home_win=0.4,
                prob_draw=0.3,
                prob_away_win=0.3,
                most_likely_score="invalid",  # Invalid format
                most_likely_prob=0.12,
                expected_home_goals=1.2,
                expected_away_goals=1.1,
                model_version="test_v1.0",
                prediction_timestamp=datetime.now()
            )
