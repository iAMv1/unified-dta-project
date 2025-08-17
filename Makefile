# Makefile for Unified DTA System

.PHONY: help install install-dev test lint format clean build upload verify

# Default target
help:
	@echo "Available targets:"
	@echo "  install      - Install package in production mode"
	@echo "  install-dev  - Install package in development mode"
	@echo "  test         - Run tests"
	@echo "  lint         - Run linting checks"
	@echo "  format       - Format code with black and isort"
	@echo "  clean        - Clean build artifacts"
	@echo "  build        - Build package"
	@echo "  upload       - Upload to PyPI"
	@echo "  verify       - Verify installation"
	@echo "  release      - Create a new release"

# Installation targets
install:
	python -m pip install .

install-dev:
	python -m pip install -e .[dev,docs,visualization,notebooks]
	python -m pip install -r requirements-dev.txt

# Testing
test:
	python -m pytest tests/ -v --cov=unified_dta --cov-report=html --cov-report=term

test-fast:
	python -m pytest tests/ -x -v

# Code quality
lint:
	python -m flake8 unified_dta tests scripts
	python -m mypy unified_dta --ignore-missing-imports

format:
	python -m black unified_dta tests scripts
	python -m isort unified_dta tests scripts

# Build and distribution
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean
	python scripts/build_package.py build

upload:
	python scripts/build_package.py upload --repository pypi

upload-test:
	python scripts/build_package.py upload --repository testpypi

# Verification
verify:
	python scripts/verify_installation.py

# Release management
release-patch:
	python scripts/release.py --bump patch

release-minor:
	python scripts/release.py --bump minor

release-major:
	python scripts/release.py --bump major

release-dry-run:
	python scripts/release.py --bump patch --dry-run

# Documentation
docs:
	cd docs && make html

docs-clean:
	cd docs && make clean

# Development setup
setup-dev: install-dev
	pre-commit install
	@echo "Development environment setup complete!"

# Quick development cycle
dev: format lint test
	@echo "Development cycle complete!"