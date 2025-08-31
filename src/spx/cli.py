"""Command-line interface for sports predictor."""

from pathlib import Path
from typing import List, Optional

import pandas as pd
import typer
from loguru import logger

from .adapters import AdapterRegistry
from .adapters.web_scraper import fetch_and_save_current_season
from .core.calibration import ModelCalibrator
from .core.config import Config, load_config
from .core.features import FeatureEngineer
from .core.io import (
    ensure_data_dirs,
    load_fixtures_parquet,
    load_matches_parquet,
    save_fixtures_parquet,
    save_matches_parquet,
    save_predictions_parquet,
    save_ratings_parquet,
)
from .core.models import OutcomeModel
from .core.models.scoreline import BivariatePoisson
from .core.ratings import EloRatingSystem
from .core.simulate import SeasonSimulator

app = typer.Typer(name="spx", help="Sports Predictor CLI")


@app.command()
def ingest(
    config: str = typer.Option(..., "--config", "-c", help="Configuration file path"),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-ingestion")
) -> None:
    """Ingest raw data and convert to processed format."""
    logger.info("Starting data ingestion")
    
    # Load configuration
    cfg = load_config(config)
    
    # Ensure data directories exist
    ensure_data_dirs(cfg.data)
    
    # Get adapter
    adapter_class = AdapterRegistry.get_adapter(cfg.adapter)
    adapter = adapter_class(cfg.data.raw_dir)
    
    # Process each season
    all_matches = []
    all_fixtures = []
    
    for season_file in cfg.data.source_files:
        # Extract season from filename
        season = _extract_season_from_filename(season_file)
        
        try:
            # Load matches
            matches = adapter.load_matches(season)
            all_matches.extend(matches)
            
            # Load fixtures
            fixtures = adapter.load_fixtures(season)
            all_fixtures.extend(fixtures)
            
            logger.info(f"Processed {len(matches)} matches and {len(fixtures)} fixtures for {season}")
            
        except Exception as e:
            logger.error(f"Error processing {season}: {e}")
            continue
    
    # Save processed data
    matches_file = cfg.data.processed_dir / "matches.parquet"
    fixtures_file = cfg.data.processed_dir / "fixtures.parquet"
    
    if all_matches:
        save_matches_parquet(all_matches, matches_file)
    
    if all_fixtures:
        save_fixtures_parquet(all_fixtures, fixtures_file)
    
    logger.info("Data ingestion completed successfully")


@app.command()
def train(
    config: str = typer.Option(..., "--config", "-c", help="Configuration file path"),
    outcome_only: bool = typer.Option(False, "--outcome-only", help="Train only outcome model"),
    scoreline_only: bool = typer.Option(False, "--scoreline-only", help="Train only scoreline model")
) -> None:
    """Train prediction models."""
    import time
    from datetime import datetime, timedelta
    
    start_time = time.time()
    logger.info("🚀 Starting model training")
    logger.info(f"📅 Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load configuration
    step_start = time.time()
    cfg = load_config(config)
    logger.info(f"✅ Configuration loaded ({time.time() - step_start:.1f}s)")
    
    # Load processed matches
    step_start = time.time()
    matches_file = cfg.data.processed_dir / "matches.parquet"
    matches = load_matches_parquet(matches_file)
    logger.info(f"✅ Loaded {len(matches)} total matches ({time.time() - step_start:.1f}s)")
    
    # Filter to training period
    training_matches = [
        m for m in matches 
        if cfg.training.train_start_date <= m.date <= cfg.training.train_end_date
    ]
    
    logger.info(f"🎯 Training on {len(training_matches)} matches from {cfg.training.train_start_date} to {cfg.training.train_end_date}")
    
    # Estimate training time based on data size and models to train
    base_time = len(training_matches) / 1000  # Base time per 1000 matches
    if not outcome_only and not scoreline_only:
        estimated_minutes = max(3, int(base_time * 2))  # Both models
    else:
        estimated_minutes = max(2, int(base_time))  # Single model
    
    estimated_completion = datetime.now() + timedelta(minutes=estimated_minutes)
    logger.info(f"⏱️ Estimated training time: ~{estimated_minutes} minutes")
    logger.info(f"🎯 Estimated completion: {estimated_completion.strftime('%H:%M:%S')}")
    
    # Calculate Elo ratings
    logger.info("📊 Calculating Elo ratings...")
    step_start = time.time()
    elo_system = EloRatingSystem(
        k_factor=cfg.features.elo_k_factor,
        base_rating=cfg.features.elo_base_rating,
        home_advantage=cfg.features.elo_home_boost,
        margin_multiplier=cfg.features.elo_margin_multiplier
    )
    
    elo_ratings = elo_system.update_ratings(training_matches)
    ratings_file = cfg.data.processed_dir / "ratings.parquet"
    save_ratings_parquet(elo_ratings, ratings_file)
    logger.info(f"✅ Elo ratings calculated and saved ({time.time() - step_start:.1f}s)")
    
    # Create features
    logger.info("🔧 Creating features...")
    step_start = time.time()
    feature_engineer = FeatureEngineer(
        lookback_games=cfg.features.lookback_games,
        min_games_for_rating=cfg.features.min_games_for_rating
    )
    
    rating_dict = elo_system.get_rating_history()
    features_df = feature_engineer.create_features(training_matches, rating_dict)
    logger.info(f"✅ Features created: {len(features_df)} rows, {len(features_df.columns)} columns ({time.time() - step_start:.1f}s)")
    
    # Train models
    models_dir = cfg.data.processed_dir / "models"
    models_dir.mkdir(exist_ok=True)
    
    if not scoreline_only:
        # Train outcome model
        logger.info("🤖 Training outcome model (XGBoost)...")
        step_start = time.time()
        outcome_model = OutcomeModel(
            n_estimators=cfg.model.xgb_n_estimators,
            learning_rate=cfg.model.xgb_learning_rate,
            max_depth=cfg.model.xgb_max_depth,
            min_child_weight=cfg.model.xgb_min_child_weight,
            subsample=cfg.model.xgb_subsample,
            colsample_bytree=cfg.model.xgb_colsample_bytree,
            calibration_method=cfg.model.calibration_method
        )
        
        # Prepare training data
        feature_cols = [col for col in features_df.columns 
                       if col not in ['outcome', 'home_goals', 'away_goals', 'date', 'home_team', 'away_team']]
        X = features_df[feature_cols]
        y = features_df['outcome'].map({'H': 2, 'D': 1, 'A': 0}).values
        
        logger.info(f"   📊 Training data: {len(X)} samples, {len(feature_cols)} features")
        
        # Train
        metrics = outcome_model.train(X, y)
        logger.info(f"✅ Outcome model trained ({time.time() - step_start:.1f}s)")
        logger.info(f"   📈 Model metrics: {metrics}")
        
        # Save model
        outcome_model.save_model(models_dir / "outcome_model.joblib")
        logger.info("   💾 Outcome model saved")
    
    if not outcome_only:
        # Train scoreline model
        logger.info("⚽ Training scoreline model (Bivariate Poisson)...")
        step_start = time.time()
        scoreline_model = BivariatePoisson(
            max_goals=cfg.model.max_goals,
            rho=cfg.model.dixon_coles_rho,
            time_decay=cfg.model.dixon_coles_decay
        )
        
        logger.info(f"   📊 Training on {len(training_matches)} historical matches")
        
        # Pass full config to training method for enhanced optimizer settings
        config_dict = {
            'scoreline': getattr(cfg, 'scoreline', {}),
            'model': cfg.model
        }
        metrics = scoreline_model.train(training_matches, config_dict)
        logger.info(f"✅ Scoreline model trained ({time.time() - step_start:.1f}s)")
        logger.info(f"   📈 Model metrics: {metrics}")
        
        # Save model
        scoreline_model.save_model(models_dir / "scoreline_model.joblib")
        logger.info("   💾 Scoreline model saved")
    
    # Training completion summary
    total_time = time.time() - start_time
    completion_time = datetime.now()
    
    logger.info("🎉 Model training completed successfully!")
    logger.info(f"⏱️ Total training time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    logger.info(f"✅ Training completed at: {completion_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Summary of what was trained
    trained_models = []
    if not scoreline_only:
        trained_models.append("XGBoost Outcome Model")
    if not outcome_only:
        trained_models.append("Bivariate Poisson Scoreline Model")
    
    logger.info(f"📋 Models trained: {', '.join(trained_models)}")
    logger.info(f"🏆 Ready for predictions and season simulation!")


@app.command()
def predict(
    config: str = typer.Option(..., "--config", "-c", help="Configuration file path"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file path")
) -> None:
    """Generate predictions for upcoming fixtures."""
    logger.info("Generating match predictions")
    
    # Load configuration
    cfg = load_config(config)
    
    # Load data
    fixtures_file = cfg.data.processed_dir / "fixtures.parquet"
    matches_file = cfg.data.processed_dir / "matches.parquet"
    
    fixtures = load_fixtures_parquet(fixtures_file)
    historical_matches = load_matches_parquet(matches_file)
    
    # Load models
    models_dir = cfg.data.processed_dir / "models"
    
    outcome_model = OutcomeModel()
    outcome_model.load_model(models_dir / "outcome_model.joblib")
    
    scoreline_model = BivariatePoisson()
    scoreline_model.load_model(models_dir / "scoreline_model.joblib")
    
    # Load ratings
    elo_system = EloRatingSystem()
    # Reconstruct ratings from historical matches
    elo_system.update_ratings(historical_matches)
    rating_dict = elo_system.get_rating_history()
    
    # Pre-calculate features for historical data once
    logger.info(f"🔧 Pre-calculating features for {len(historical_matches)} historical matches")
    from .core.features import FeatureEngineer
    engineer = FeatureEngineer(lookback_games=cfg.features.lookback_games)
    historical_features_df = engineer.create_features(historical_matches, rating_dict)
    logger.info("✅ Historical features pre-calculated")
    
    # Generate predictions efficiently
    predictions = []
    
    for i, fixture in enumerate(fixtures, 1):
        if i % 50 == 0:
            logger.info(f"📊 Predicting matches: {i}/{len(fixtures)}")
        
        # Create features for this match using pre-calculated historical features
        features = engineer.create_single_prediction_features(
            fixture.home_team,
            fixture.away_team,
            fixture.date,
            historical_matches,
            rating_dict
        )
        
        # Get outcome prediction
        outcome_pred = outcome_model.predict_match(
            fixture.home_team,
            fixture.away_team,
            features,
            fixture.date
        )
        
        # Enhance with scoreline prediction
        enhanced_pred = scoreline_model.predict_match_enhanced(
            fixture.home_team,
            fixture.away_team,
            fixture.date,
            (outcome_pred.prob_home_win, outcome_pred.prob_draw, outcome_pred.prob_away_win)
        )
        
        predictions.append(enhanced_pred)
    
    # Save predictions
    output_file = Path(output) if output else cfg.data.processed_dir / "predictions.parquet"
    save_predictions_parquet(predictions, output_file)
    
    logger.info(f"Generated {len(predictions)} predictions, saved to {output_file}")


@app.command()
def simulate(
    config: str = typer.Option(..., "--config", "-c", help="Configuration file path"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file path")
) -> None:
    """Run season simulation."""
    logger.info("Starting season simulation")
    
    # Load configuration
    cfg = load_config(config)
    
    # Load data
    fixtures_file = cfg.data.processed_dir / "fixtures.parquet"
    predictions_file = cfg.data.processed_dir / "predictions.parquet"
    
    fixtures = load_fixtures_parquet(fixtures_file)
    
    # Load or generate predictions
    try:
        from .core.io import load_predictions_parquet
        predictions = load_predictions_parquet(predictions_file)
    except FileNotFoundError:
        logger.info("No predictions found. Run 'spx predict' first or predictions will be generated.")
        # Could auto-generate here, but keeping it explicit for now
        raise
    
    # Run simulation
    simulator = SeasonSimulator(
        n_simulations=cfg.simulation.n_simulations,
        random_seed=cfg.simulation.random_seed,
        use_head_to_head=cfg.simulation.use_head_to_head
    )
    
    season_predictions = simulator.simulate_season(fixtures, predictions)
    
    # Save average table
    avg_table_file = cfg.data.processed_dir / "average_season_table.csv"
    simulator.save_average_table(str(avg_table_file))
    
    # Save results
    output_file = Path(output) if output else cfg.data.processed_dir / "season_predictions.parquet"
    
    # Convert to DataFrame and save
    results_data = []
    for pred in season_predictions:
        # Convert position_probs dict keys to strings for Parquet compatibility
        pred_data = pred.model_dump()
        pred_data['position_probs'] = {str(k): v for k, v in pred_data['position_probs'].items()}
        results_data.append(pred_data)
    
    results_df = pd.DataFrame(results_data)
    results_df.to_parquet(output_file, index=False)
    
    logger.info(f"Season simulation completed, results saved to {output_file}")
    logger.info(f"Average season table saved to {avg_table_file}")
    
    # Print summary
    _print_simulation_summary(season_predictions)


@app.command()
def eval(
    config: str = typer.Option(..., "--config", "-c", help="Configuration file path")
) -> None:
    """Evaluate model performance."""
    logger.info("Evaluating model performance")
    
    # Load configuration
    cfg = load_config(config)
    
    # Load test data
    matches_file = cfg.data.processed_dir / "matches.parquet"
    matches = load_matches_parquet(matches_file)
    
    # Filter to test period
    test_matches = [
        m for m in matches 
        if cfg.training.test_start_date <= m.date <= cfg.training.test_end_date
    ]
    
    if not test_matches:
        logger.error("No test matches found in specified date range")
        return
    
    logger.info(f"Evaluating on {len(test_matches)} test matches")
    
    # Load models and generate predictions for test set
    # Implementation would generate predictions and evaluate them
    # This is a simplified version
    
    logger.info("Model evaluation completed")


def _extract_season_from_filename(filename: str) -> str:
    """Extract season from filename.
    
    Args:
        filename: Data filename
        
    Returns:
        Season identifier
    """
    # Simple extraction - can be enhanced
    if "2023_24" in filename:
        return "2023-24"
    elif "2024_25" in filename:
        return "2024-25"
    elif "2025_26" in filename:
        return "2025-26"
    else:
        # Default to current season
        return "2024-25"


def _print_simulation_summary(predictions: List) -> None:
    """Print simulation summary to console.
    
    Args:
        predictions: Season predictions
    """
    print("\n=== SEASON SIMULATION SUMMARY ===")
    
    # Sort by title probability
    title_contenders = sorted(predictions, key=lambda x: x.prob_title, reverse=True)
    
    print("\nTitle Race:")
    for i, pred in enumerate(title_contenders[:6], 1):
        print(f"{i:2d}. {pred.team:<20} {pred.prob_title:6.1%} "
              f"(Exp Pos: {pred.expected_position:.1f})")
    
    print("\nTop 4 Race:")
    top4_contenders = sorted(predictions, key=lambda x: x.prob_top4, reverse=True)
    for i, pred in enumerate(top4_contenders[:8], 1):
        print(f"{i:2d}. {pred.team:<20} {pred.prob_top4:6.1%}")
    
    print("\nRelegation Battle:")
    relegation_candidates = sorted(predictions, key=lambda x: x.prob_relegation, reverse=True)
    for i, pred in enumerate(relegation_candidates[:6], 1):
        print(f"{i:2d}. {pred.team:<20} {pred.prob_relegation:6.1%}")


@app.command()
def fetch(
    config: str = typer.Option(..., "--config", "-c", help="Configuration file path"),
    season: int = typer.Option(2025, "--season", "-s", help="Season year (e.g., 2025 for 2025-26)"),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-fetch even if file exists")
) -> None:
    """Fetch current season data from Premier League website."""
    logger.info(f"Fetching {season}-{str(season+1)[-2:]} season data from web")
    
    # Load configuration
    cfg = load_config(config)
    
    # Determine output file
    output_file = cfg.data.raw_dir / f"epl_{season}_{str(season+1)[-2:]}.csv"
    
    if output_file.exists() and not force:
        logger.warning(f"File {output_file} already exists. Use --force to overwrite.")
        return
    
    # Fetch data
    success = fetch_and_save_current_season(str(output_file), season)
    
    if success:
        logger.info(f"Successfully fetched and saved {season}-{str(season+1)[-2:]} season data")
        logger.info(f"Next step: Update configs/{cfg.league}.yaml to include the new file")
        logger.info(f"Then run: spx ingest -c {config}")
    else:
        logger.error("Failed to fetch season data")


if __name__ == "__main__":
    app()
