"""Data input/output utilities."""

from pathlib import Path
from typing import Any, List, Union

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger

from .schema import Fixture, Match, MatchPrediction, TeamRating


def save_matches_parquet(matches: List[Match], filepath: Union[str, Path]) -> None:
    """Save matches to Parquet format.
    
    Args:
        matches: List of Match objects
        filepath: Output file path
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to DataFrame
    df = pd.DataFrame([match.dict() for match in matches])
    
    # Ensure date column is properly typed
    df['date'] = pd.to_datetime(df['date']).dt.date
    
    # Save to Parquet
    df.to_parquet(filepath, index=False, engine='pyarrow')
    logger.info(f"Saved {len(matches)} matches to {filepath}")


def load_matches_parquet(filepath: Union[str, Path]) -> List[Match]:
    """Load matches from Parquet format.
    
    Args:
        filepath: Input file path
        
    Returns:
        List of Match objects
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Match data not found: {filepath}")
    
    df = pd.read_parquet(filepath)
    
    # Convert back to Match objects
    matches = []
    for _, row in df.iterrows():
        match_data = row.to_dict()
        # Handle NaN values
        for key, value in match_data.items():
            if pd.isna(value):
                match_data[key] = None
        matches.append(Match(**match_data))
    
    logger.info(f"Loaded {len(matches)} matches from {filepath}")
    return matches


def save_fixtures_parquet(fixtures: List[Fixture], filepath: Union[str, Path]) -> None:
    """Save fixtures to Parquet format.
    
    Args:
        fixtures: List of Fixture objects
        filepath: Output file path
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame([fixture.dict() for fixture in fixtures])
    df['date'] = pd.to_datetime(df['date']).dt.date
    
    df.to_parquet(filepath, index=False, engine='pyarrow')
    logger.info(f"Saved {len(fixtures)} fixtures to {filepath}")


def load_fixtures_parquet(filepath: Union[str, Path]) -> List[Fixture]:
    """Load fixtures from Parquet format.
    
    Args:
        filepath: Input file path
        
    Returns:
        List of Fixture objects
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Fixture data not found: {filepath}")
    
    df = pd.read_parquet(filepath)
    
    fixtures = []
    for _, row in df.iterrows():
        fixture_data = row.to_dict()
        for key, value in fixture_data.items():
            if pd.isna(value):
                fixture_data[key] = None
        fixtures.append(Fixture(**fixture_data))
    
    logger.info(f"Loaded {len(fixtures)} fixtures from {filepath}")
    return fixtures


def save_ratings_parquet(ratings: List[TeamRating], filepath: Union[str, Path]) -> None:
    """Save team ratings to Parquet format.
    
    Args:
        ratings: List of TeamRating objects
        filepath: Output file path
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if not ratings:
        # Create empty DataFrame with correct columns
        df = pd.DataFrame(columns=['team', 'date', 'elo_rating', 'games_played'])
    else:
        df = pd.DataFrame([rating.dict() for rating in ratings])
        df['date'] = pd.to_datetime(df['date']).dt.date
    
    df.to_parquet(filepath, index=False, engine='pyarrow')
    logger.info(f"Saved {len(ratings)} ratings to {filepath}")


def load_ratings_parquet(filepath: Union[str, Path]) -> List[TeamRating]:
    """Load team ratings from Parquet format.
    
    Args:
        filepath: Input file path
        
    Returns:
        List of TeamRating objects
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Rating data not found: {filepath}")
    
    df = pd.read_parquet(filepath)
    
    ratings = []
    for _, row in df.iterrows():
        rating_data = row.to_dict()
        for key, value in rating_data.items():
            if pd.isna(value):
                rating_data[key] = None
        ratings.append(TeamRating(**rating_data))
    
    logger.info(f"Loaded {len(ratings)} ratings from {filepath}")
    return ratings


def save_predictions_parquet(predictions: List[MatchPrediction], filepath: Union[str, Path]) -> None:
    """Save predictions to Parquet format.
    
    Args:
        predictions: List of MatchPrediction objects
        filepath: Output file path
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame([pred.dict() for pred in predictions])
    df['date'] = pd.to_datetime(df['date']).dt.date
    
    df.to_parquet(filepath, index=False, engine='pyarrow')
    logger.info(f"Saved {len(predictions)} predictions to {filepath}")


def load_predictions_parquet(filepath: Union[str, Path]) -> List[MatchPrediction]:
    """Load predictions from Parquet format.
    
    Args:
        filepath: Input file path
        
    Returns:
        List of MatchPrediction objects
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Prediction data not found: {filepath}")
    
    df = pd.read_parquet(filepath)
    
    predictions = []
    for _, row in df.iterrows():
        pred_data = row.to_dict()
        for key, value in pred_data.items():
            if pd.isna(value):
                pred_data[key] = None
        predictions.append(MatchPrediction(**pred_data))
    
    logger.info(f"Loaded {len(predictions)} predictions from {filepath}")
    return predictions


def ensure_data_dirs(data_config: Any) -> None:
    """Ensure all data directories exist.
    
    Args:
        data_config: Data configuration object
    """
    for dir_attr in ['raw_dir', 'interim_dir', 'processed_dir']:
        dir_path = getattr(data_config, dir_attr)
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured directory exists: {dir_path}")
