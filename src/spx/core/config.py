"""Configuration management using Pydantic and OmegaConf."""

from datetime import date as Date
from pathlib import Path
from typing import Any, Dict, List, Optional

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, Field, validator


class DataConfig(BaseModel):
    """Data configuration."""
    
    raw_dir: Path = Field(..., description="Raw data directory")
    interim_dir: Path = Field(..., description="Interim data directory")
    processed_dir: Path = Field(..., description="Processed data directory")
    source_files: List[str] = Field(..., description="Source data files")
    
    @validator('raw_dir', 'interim_dir', 'processed_dir', pre=True)
    def convert_to_path(cls, v: Any) -> Path:
        """Convert string paths to Path objects."""
        return Path(v)


class FeatureConfig(BaseModel):
    """Feature engineering configuration."""
    
    lookback_games: int = Field(6, description="Number of games for rolling features")
    min_games_for_rating: int = Field(10, description="Minimum games before using Elo")
    home_advantage: bool = Field(True, description="Include home advantage feature")
    
    # Elo rating parameters
    elo_k_factor: float = Field(20.0, description="Elo K-factor")
    elo_base_rating: float = Field(1500.0, description="Base Elo rating")
    elo_home_boost: float = Field(30.0, description="Home team Elo boost")
    elo_margin_multiplier: float = Field(0.1, description="Margin of victory multiplier")


class ModelConfig(BaseModel):
    """Model configuration."""
    
    # XGBoost parameters
    xgb_n_estimators: int = Field(100, description="Number of XGBoost estimators")
    xgb_learning_rate: float = Field(0.1, description="XGBoost learning rate")
    xgb_max_depth: int = Field(6, description="XGBoost max depth")
    xgb_min_child_weight: int = Field(1, description="XGBoost min child weight")
    xgb_subsample: float = Field(0.8, description="XGBoost subsample ratio")
    xgb_colsample_bytree: float = Field(0.8, description="XGBoost feature subsample ratio")
    
    # Calibration
    calibration_method: str = Field("isotonic", description="Calibration method")
    
    # Bivariate Poisson parameters
    max_goals: int = Field(6, description="Maximum goals to model")
    dixon_coles_decay: float = Field(0.01, description="Dixon-Coles time decay")
    dixon_coles_rho: float = Field(-0.1, description="Dixon-Coles correlation parameter")


class SimulationConfig(BaseModel):
    """Season simulation configuration."""
    
    n_simulations: int = Field(20000, description="Number of Monte Carlo runs")
    random_seed: int = Field(42, description="Random seed for reproducibility")
    use_head_to_head: bool = Field(True, description="Use head-to-head for tie-breaking")


class TrainingConfig(BaseModel):
    """Training configuration."""
    
    train_start_date: Date = Field(..., description="Training data start date")
    train_end_date: Date = Field(..., description="Training data end date")
    test_start_date: Date = Field(..., description="Test data start date")
    test_end_date: Date = Field(..., description="Test data end date")
    
    @validator('train_start_date', 'train_end_date', 'test_start_date', 'test_end_date', pre=True)
    def parse_date(cls, v: Any) -> Date:
        """Parse date from string if needed."""
        if isinstance(v, str):
            return Date.fromisoformat(v)
        return v


class Config(BaseModel):
    """Main configuration class."""
    
    league: str = Field(..., description="League identifier")
    adapter: str = Field(..., description="Adapter class name")
    
    data: DataConfig
    features: FeatureConfig
    model: ModelConfig
    simulation: SimulationConfig
    training: TrainingConfig
    
    class Config:
        """Pydantic config."""
        arbitrary_types_allowed = True


def load_config(config_path: str) -> Config:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Validated configuration object
    """
    omega_config = OmegaConf.load(config_path)
    config_dict = OmegaConf.to_container(omega_config, resolve=True)
    return Config(**config_dict)


def get_default_config() -> Dict[str, Any]:
    """Get default configuration dictionary.
    
    Returns:
        Default configuration as dictionary
    """
    return {
        "league": "epl",
        "adapter": "EPLAdapter",
        "data": {
            "raw_dir": "data/raw",
            "interim_dir": "data/interim", 
            "processed_dir": "data/processed",
            "source_files": ["epl_2023_24.csv", "epl_2024_25.csv"]
        },
        "features": {
            "lookback_games": 6,
            "min_games_for_rating": 10,
            "home_advantage": True,
            "elo_k_factor": 20.0,
            "elo_base_rating": 1500.0,
            "elo_home_boost": 30.0,
            "elo_margin_multiplier": 0.1
        },
        "model": {
            "xgb_n_estimators": 100,
            "xgb_learning_rate": 0.1,
            "xgb_max_depth": 6,
            "xgb_min_child_weight": 1,
            "xgb_subsample": 0.8,
            "xgb_colsample_bytree": 0.8,
            "calibration_method": "isotonic",
            "max_goals": 6,
            "dixon_coles_decay": 0.01,
            "dixon_coles_rho": -0.1
        },
        "simulation": {
            "n_simulations": 20000,
            "random_seed": 42,
            "use_head_to_head": True
        },
        "training": {
            "train_start_date": "2020-08-01",
            "train_end_date": "2024-05-31", 
            "test_start_date": "2024-08-01",
            "test_end_date": "2025-05-31"
        }
    }
