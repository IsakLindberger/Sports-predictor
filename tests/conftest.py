"""Test configuration and setup."""

import pytest
from pathlib import Path


@pytest.fixture
def test_data_dir():
    """Test data directory fixture."""
    return Path(__file__).parent / "data"


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "league": "epl",
        "adapter": "EPLAdapter",
        "data": {
            "raw_dir": "tests/data",
            "interim_dir": "tests/interim",
            "processed_dir": "tests/processed",
            "source_files": ["sample_epl.csv"]
        },
        "features": {
            "lookback_games": 3,
            "min_games_for_rating": 5,
            "home_advantage": True,
            "elo_k_factor": 20.0,
            "elo_base_rating": 1500.0,
            "elo_home_boost": 30.0,
            "elo_margin_multiplier": 0.1
        },
        "model": {
            "xgb_n_estimators": 10,
            "xgb_learning_rate": 0.3,
            "xgb_max_depth": 3,
            "calibration_method": "isotonic",
            "max_goals": 4,
            "dixon_coles_rho": -0.1
        },
        "simulation": {
            "n_simulations": 100,
            "random_seed": 42,
            "use_head_to_head": True
        },
        "training": {
            "train_start_date": "2024-08-01",
            "train_end_date": "2024-08-31",
            "test_start_date": "2024-09-01", 
            "test_end_date": "2024-09-30"
        }
    }
