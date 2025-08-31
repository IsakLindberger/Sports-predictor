.PHONY: help setup setup-dev install clean test lint mypy format check ingest train predict simulate

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup: ## Install production dependencies
	poetry install --only=main

setup-dev: ## Install all dependencies including dev tools
	poetry install
	poetry run pre-commit install

install: setup ## Alias for setup

clean: ## Clean build artifacts and cache
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

test: ## Run tests
	poetry run pytest

test-cov: ## Run tests with coverage report
	poetry run pytest --cov-report=html

lint: ## Run linting
	poetry run ruff check src/ tests/
	poetry run black --check src/ tests/
	poetry run isort --check-only src/ tests/

mypy: ## Run type checking
	poetry run mypy src/

format: ## Format code
	poetry run black src/ tests/
	poetry run isort src/ tests/
	poetry run ruff check --fix src/ tests/

check: lint mypy test ## Run all checks

# Data pipeline commands
ingest: ## Ingest raw data
	poetry run spx ingest --config configs/epl.yaml

train: ## Train prediction models
	poetry run spx train --config configs/epl.yaml

predict: ## Generate predictions for fixtures
	poetry run spx predict --config configs/epl.yaml

simulate: ## Run season simulation
	poetry run spx simulate --config configs/epl.yaml

eval: ## Evaluate model performance
	poetry run spx eval --config configs/epl.yaml

# Development workflow
dev-setup: setup-dev ## Complete development setup
	@echo "Development environment ready!"
	@echo "Run 'make check' to verify everything works."

ci: check ## Run CI checks locally
	@echo "All CI checks passed!"
