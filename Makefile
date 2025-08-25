# Makefile for Diabetic Retinopathy Detection Project

.PHONY: help install test test-fast test-slow test-unit test-integration coverage clean lint format

# Default target
help:
	@echo "Available commands:"
	@echo "  install          - Install all dependencies"
	@echo "  test             - Run all tests with coverage"
	@echo "  test-fast        - Run only fast tests"
	@echo "  test-slow        - Run only slow tests"
	@echo "  test-unit        - Run only unit tests"
	@echo "  test-integration - Run only integration tests"
	@echo "  coverage         - Generate coverage report"
	@echo "  clean            - Clean temporary files"
	@echo "  lint             - Run linting checks"
	@echo "  format           - Format code with black"
	@echo "  train            - Train the model"
	@echo "  evaluate         - Evaluate trained model"
	@echo "  serve            - Start the API server"
	@echo "  frontend         - Start the frontend development server"

# Installation
install:
	pip install -r requirements.txt
	pip install -r backend/requirements.txt

# Testing
test:
	python run_tests.py --type all

test-fast:
	python run_tests.py --type fast

test-slow:
	python run_tests.py --type slow

test-unit:
	python run_tests.py --type unit

test-integration:
	python run_tests.py --type integration

# Coverage
coverage:
	python -m pytest --cov=ml --cov=backend --cov-report=html --cov-report=term-missing

# Cleaning
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/
	rm -rf ml/evaluation_results/

# Code quality
lint:
	flake8 ml/ backend/ tests/
	pylint ml/ backend/ tests/

format:
	black ml/ backend/ tests/
	isort ml/ backend/ tests/

# Model operations
train:
	python -m ml.train --labels-csv ml/data/train.csv --img-dir ml/data/train_images --epochs 20

evaluate:
	python -m ml.evaluate_model --model-path ml/models/best_model.pth --test-csv ml/data/test.csv --img-dir ml/data/test_images

# Server operations
serve:
	cd backend && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

frontend:
	cd frontend && npm run dev

# Docker operations
docker-build:
	docker build -t diabetic-retinopathy-api .

docker-run:
	docker run -p 8000:8000 diabetic-retinopathy-api

# Setup development environment
setup-dev: install
	pip install black flake8 pylint isort
	pre-commit install