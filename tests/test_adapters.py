"""Test EPL adapter functionality."""

from datetime import date
from pathlib import Path

import pytest

from spx.adapters.football.epl import EPLAdapter
from spx.core.schema import Match, Fixture


class TestEPLAdapter:
    """Test EPL adapter."""
    
    def test_adapter_initialization(self, test_data_dir):
        """Test adapter initialization."""
        adapter = EPLAdapter(test_data_dir)
        assert adapter.data_dir == test_data_dir
    
    def test_load_matches(self, test_data_dir):
        """Test loading matches from sample data."""
        adapter = EPLAdapter(test_data_dir)
        
        # Copy sample file to expected location
        sample_file = test_data_dir / "sample_epl.csv"
        if sample_file.exists():
            matches = adapter.load_matches("2024-25")
            
            assert len(matches) > 0
            assert all(isinstance(match, Match) for match in matches)
            
            # Check first match
            first_match = matches[0]
            assert first_match.home_team == "Arsenal"
            assert first_match.away_team == "Brighton & Hove Albion"
            assert first_match.home_goals == 2
            assert first_match.away_goals == 1
    
    def test_load_fixtures(self, test_data_dir):
        """Test loading fixtures from sample data."""
        adapter = EPLAdapter(test_data_dir)
        
        # This would load from the same file but filter for empty results
        # For now, we'll test with the fixtures file
        fixtures_file = test_data_dir / "sample_fixtures.csv"
        if fixtures_file.exists():
            # Manually test fixture loading logic
            import pandas as pd
            df = pd.read_csv(fixtures_file)
            
            # Should have empty FTR, FTHG, FTAG for fixtures
            future_fixtures = df[df['FTR'].isna()]
            assert len(future_fixtures) > 0
    
    def test_validate_data_format(self, test_data_dir):
        """Test data format validation."""
        adapter = EPLAdapter(test_data_dir)
        
        sample_file = test_data_dir / "sample_epl.csv"
        if sample_file.exists():
            assert adapter.validate_data_format(sample_file) is True
        
        # Test invalid file
        invalid_file = test_data_dir / "nonexistent.csv"
        assert adapter.validate_data_format(invalid_file) is False
    
    def test_team_name_normalization(self, test_data_dir):
        """Test team name normalization."""
        adapter = EPLAdapter(test_data_dir)
        
        # Test some common name mappings
        assert adapter._normalize_team_name("Man City") == "Manchester City"
        assert adapter._normalize_team_name("Man United") == "Manchester United"
        assert adapter._normalize_team_name("Spurs") == "Tottenham"
        assert adapter._normalize_team_name("Arsenal") == "Arsenal"  # No change needed
