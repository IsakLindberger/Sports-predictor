#!/usr/bin/env python3
"""
Simple test script to demonstrate the SPX system end-to-end functionality.
This script validates that all core components work together.
"""

from pathlib import Path
from datetime import date, datetime
import sys
import os

# Add src to Python path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from spx.core.schema import Match, Fixture, MatchOutcome
from spx.adapters.football.epl import EPLAdapter
from spx.core.ratings import EloRatingSystem
from spx.core.config import get_default_config

def test_basic_functionality():
    """Test basic SPX functionality."""
    print("🚀 Starting SPX System Test...")
    
    # Test 1: Load configuration
    print("\n1. Testing configuration loading...")
    try:
        config = get_default_config()
        print(f"   ✅ Configuration loaded: {config['league']}")
    except Exception as e:
        print(f"   ❌ Configuration failed: {e}")
        return False
    
    # Test 2: Test data schemas
    print("\n2. Testing data schemas...")
    try:
        # Create a test match
        match = Match(
            date=date(2024, 8, 16),
            season="2024-25",
            home_team="Arsenal",
            away_team="Brighton",
            home_goals=2,
            away_goals=1,
            outcome=MatchOutcome.HOME_WIN
        )
        print(f"   ✅ Match schema: {match.home_team} {match.home_goals}-{match.away_goals} {match.away_team}")
        
        # Create a test fixture
        fixture = Fixture(
            date=date(2024, 9, 1),
            season="2024-25",
            home_team="Chelsea",
            away_team="Liverpool"
        )
        print(f"   ✅ Fixture schema: {fixture.home_team} vs {fixture.away_team}")
    except Exception as e:
        print(f"   ❌ Schema test failed: {e}")
        return False
    
    # Test 3: Test EPL adapter
    print("\n3. Testing EPL adapter...")
    try:
        test_data_dir = Path(__file__).parent / "tests" / "data"
        adapter = EPLAdapter(test_data_dir)
        
        # Test loading matches (if test data exists)
        if (test_data_dir / "epl_2024_25.csv").exists():
            matches = adapter.load_matches("2024-25")
            print(f"   ✅ Loaded {len(matches)} matches from EPL adapter")
        else:
            print("   ⚠️  No test data found, adapter initialized successfully")
    except Exception as e:
        print(f"   ❌ Adapter test failed: {e}")
        return False
    
    # Test 4: Test Elo rating system
    print("\n4. Testing Elo rating system...")
    try:
        elo = EloRatingSystem()
        
        # Update ratings with matches (this will initialize teams automatically)
        ratings = elo.update_ratings([match])
        
        current_ratings = elo.get_current_ratings()
        arsenal_rating = current_ratings.get("Arsenal", 1500)
        brighton_rating = current_ratings.get("Brighton", 1500)
        
        print(f"   ✅ Arsenal rating: {arsenal_rating:.1f}")
        print(f"   ✅ Brighton rating: {brighton_rating:.1f}")
        
        # Test prediction
        prediction = elo.predict_match_outcome("Arsenal", "Brighton")
        print(f"   ✅ Prediction probabilities: H={prediction[0]:.3f}, D={prediction[1]:.3f}, A={prediction[2]:.3f}")
    except Exception as e:
        print(f"   ❌ Elo test failed: {e}")
        return False
    
    print("\n🎉 All core tests passed! SPX system is working correctly.")
    return True

if __name__ == "__main__":
    success = test_basic_functionality()
    sys.exit(0 if success else 1)
