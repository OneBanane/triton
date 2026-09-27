.PHONY: install check format lint

install:
	uv sync --dev
	uv run pre-commit install

check:
	uv run pre-commit run --all-files

format:
	uv run ruff format .
	uv run ruff check . --fix

lint:
	uv run ruff check .
