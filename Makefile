VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
BLACK := $(VENV)/bin/black
RUFF := $(VENV)/bin/ruff
PYTEST := $(VENV)/bin/pytest
LINT_PATHS := portfolio_analysis.py tests
FORMAT_PATHS := portfolio_analysis.py tests

$(PYTHON):
	python3 -m venv $(VENV)

$(VENV)/.deps: $(PYTHON)
	$(PIP) install --upgrade pip
	$(PIP) install black ruff pytest
	touch $(VENV)/.deps

.PHONY: lint format test

lint: $(VENV)/.deps
	$(RUFF) check $(LINT_PATHS)
	$(BLACK) --check $(FORMAT_PATHS)

format: $(VENV)/.deps
	$(BLACK) $(FORMAT_PATHS)
	$(RUFF) check $(LINT_PATHS) --fix

test: $(VENV)/.deps
	$(PYTEST)
