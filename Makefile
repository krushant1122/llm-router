.PHONY: setup test lint format data train eval report clean all

VENV_BIN := $(shell if [ -f .venv/bin/python ]; then echo ".venv/bin/"; else echo ""; fi)
RUN := $(shell if command -v uv >/dev/null 2>&1; then echo "uv run "; elif [ -f .venv/bin/python ]; then echo ".venv/bin/"; else echo "python3 -m "; fi)

setup:
	@if command -v uv >/dev/null 2>&1; then uv sync --extra dev; else python3 -m pip install -e ".[dev]"; fi

test:
	$(RUN)pytest tests/ -v --tb=short

lint:
	$(RUN)ruff check src/ tests/

format:
	$(RUN)ruff format src/ tests/

data:
	PYTHONPATH=src $(RUN)router data build

train:
	PYTHONPATH=src $(RUN)router train-difficulty

eval:
	PYTHONPATH=src $(RUN)router eval --all

report:
	@echo "Usage: router report <results_dir>"
	@echo "Example: router report results/eval_20261017_120000"

clean:
	rm -rf .cache results/ models/ __pycache__ .pytest_cache *.egg-info build dist

all: setup lint test
