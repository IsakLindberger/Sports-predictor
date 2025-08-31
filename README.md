# Sports Predictor

A production-ready system for predicting football match outcomes, scorelines, and simulating full seasons. Designed for extensibility across leagues and sports.

## Features

- **Match Outcome Prediction**: Win/Draw/Loss probabilities using XGBoost
- **Scoreline Prediction**: Bivariate Poisson with Dixon-Coles correlation
- **Season Simulation**: Monte Carlo simulation for league tables
- **Elo Rating System**: Dynamic team strength ratings
- **League Extensibility**: Easy adapter pattern for new leagues/sports

## Quick Start

```bash
# Install dependencies
make setup

# Ingest sample data
make ingest

# Train models
make train

# Run season simulation
make simulate
```

## Project Structure

```
sports-predictor/
├── src/spx/           # Main package
│   ├── core/          # Core functionality
│   ├── adapters/      # League-specific adapters
│   ├── cli.py         # Command-line interface
│   └── utils.py       # Utilities
├── configs/           # Configuration files
├── data/             # Data storage (excluded from git)
├── notebooks/        # Jupyter notebooks for EDA
├── tests/            # Test suite
└── docs/             # Documentation
```

## Models

### Outcome Prediction
- **Primary**: XGBoost classifier for W/D/L outcomes
- **Features**: Rolling form, Elo ratings, home advantage
- **Evaluation**: Log loss, Brier score, calibration plots

### Scoreline Prediction
- **Model**: Bivariate Poisson with Dixon-Coles correlation
- **Output**: Full probability grid (0-6 goals per team)
- **Calibration**: Isotonic regression for reliability

### Season Simulation
- **Method**: Monte Carlo (20,000 runs)
- **Rules**: Points, goal difference, goals scored, head-to-head
- **Outputs**: Title, top-4, relegation probabilities

## Adding New Leagues

1. Create adapter in `src/spx/adapters/`
2. Add configuration in `configs/`
3. Register in adapter registry
4. No core code changes required

## Configuration

All settings are managed via YAML configs with Pydantic validation:
- Data sources and paths
- Feature engineering parameters
- Model hyperparameters
- Simulation settings

## Data Sources

Current support:
- **EPL**: football-data.co.uk CSV format
- **Extensible**: Adapter pattern for new sources

## Development

```bash
# Setup development environment
make setup-dev

# Run tests
make test

# Run linting
make lint

# Run type checking
make mypy
```

## License

MIT License - see LICENSE file for details.
