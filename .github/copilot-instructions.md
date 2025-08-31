# Sports Predictor Project Instructions

This is a production-ready sports prediction system focused on football match outcomes, scorelines, and season simulations.

## Project Structure
- **Core**: Configuration, schemas, data processing, features, ratings, models
- **Adapters**: League-specific data loaders (starting with EPL)
- **CLI**: Commands for ingestion, training, prediction, simulation
- **Tests**: Comprehensive coverage of all components

## Key Technologies
- Python 3.11+, Poetry, pytest
- Typer CLI, OmegaConf/Hydra, Pydantic
- pandas/pyarrow, scikit-learn, xgboost
- Custom bivariate Poisson with Dixon-Coles correlation
- Pre-commit hooks (ruff, black, isort, mypy)

## Development Guidelines
- Use strict typing and comprehensive docstrings
- All data paths must be configurable
- Maintain league-agnostic design for extensibility
- Follow test-driven development practices

✅ Created workspace structure and instructions
