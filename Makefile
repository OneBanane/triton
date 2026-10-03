.PHONY: install check format lint clean train load-test

LOCUST_HOST ?= http://localhost:8003
LOCUST_USERS ?= 10
LOCUST_SPAWN_RATE ?= 2
LOCUST_RUN_TIME ?= 1m
LOCUST_ARGS ?=

install:
	uv sync --dev
	uv run pre-commit install

check:
	uv run pre-commit run --all-files

load-test:
	uv run --group load locust -f tests/load/locustfile.py --headless \
		--host $(LOCUST_HOST) --users $(LOCUST_USERS) \
		--spawn-rate $(LOCUST_SPAWN_RATE) --run-time $(LOCUST_RUN_TIME) \
		$(LOCUST_ARGS)

format:
	uv run ruff format .
	uv run ruff check . --fix

lint:
	uv run ruff check .

clean:
	rm -rf .ruff_cache .pytest_cache .mypy_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +

train:
	PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python -m triton.training.main
