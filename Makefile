.PHONY: setup format lint typecheck test coverage build train monitor verify-audit demo serve clean

PYTHON ?= python

setup:
	$(PYTHON) -m pip install -e ".[dev]"
	$(PYTHON) -m pre_commit install || echo "pre-commit hook not installed (optional)"

format:
	$(PYTHON) -m ruff format src tests
	$(PYTHON) -m ruff check --fix src tests

lint:
	$(PYTHON) -m ruff format --check src tests
	$(PYTHON) -m ruff check src tests

typecheck:
	$(PYTHON) -m mypy

test:
	$(PYTHON) -m pytest

coverage:
	$(PYTHON) -m pytest --cov --cov-report=term-missing --cov-fail-under=85

build: clean-dist
	$(PYTHON) -m build

train:
	$(PYTHON) -m feg_mlops.pipeline.cli train

monitor:
	$(PYTHON) -m feg_mlops.pipeline.cli monitor

verify-audit:
	$(PYTHON) -m feg_mlops.pipeline.cli audit-verify

demo: train monitor verify-audit

serve:
	$(PYTHON) -m uvicorn feg_mlops.serving.app:app --host 0.0.0.0 --port 8000

clean-dist:
	rm -rf dist

clean:
	rm -rf dist build src/*.egg-info .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
