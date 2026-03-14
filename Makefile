.DEFAULT_GOAL := help
.PHONY: help install lint format format-check typecheck interrogate vulture \
        test cov test-full mutant check ci prerelease clean

## ─── Setup ───────────────────────────────────────────────────────────────────

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install project + dev deps
	uv sync
	uv run pre-commit install

## ─── Quality ─────────────────────────────────────────────────────────────────

lint: ## Run ruff linter
	uv run ruff check src/ tests/

format: ## Format code with ruff
	uv run ruff format src/ tests/

format-check: ## Check formatting without modifying
	uv run ruff format --check src/ tests/

typecheck: ## Run pyright + mypy (strict)
	uv run pyright src/
	uv run mypy src/

interrogate: ## Check docstring coverage (95% minimum)
	uv run interrogate src/sqlearn/

vulture: ## Detect dead code
	uv run vulture src/sqlearn/ vulture_whitelist.py

## ─── Testing (3-tier) ────────────────────────────────────────────────────────

test: ## Tier 1: Fast tests (failed first, then new)
	uv run pytest --failed-first -x

cov: ## Tier 2: Full test suite with coverage report
	uv run pytest --cov --cov-report=term-missing

mutant: ## Tier 3: Mutation testing (verifies test quality)
	uv run mutmut run --paths-to-mutate=src/sqlearn/

test-full: cov mutant ## Tier 2+3: Coverage + mutation testing

## ─── Combined ────────────────────────────────────────────────────────────────

check: lint format-check typecheck interrogate vulture test ## Run all checks (lint + type + docstrings + dead code + test)

ci: check ## Alias for check (mirrors CI pipeline)

## ─── Multi-Python ────────────────────────────────────────────────────────────

prerelease: ## Test across Python 3.10-3.14
	@for ver in 3.10 3.11 3.12 3.13 3.14; do \
		echo "\n\033[36m──── Python $$ver ────\033[0m"; \
		UV_PROJECT_ENVIRONMENT=.venv-$$ver uv run --python $$ver pytest || exit 1; \
	done
	@echo "\n\033[32m✓ All Python versions passed\033[0m"

## ─── Cleanup ─────────────────────────────────────────────────────────────────

clean: ## Remove build artifacts and caches
	rm -rf dist/ build/ *.egg-info/
	rm -rf .pytest_cache/ .mypy_cache/ .pyright/ .ruff_cache/
	rm -rf htmlcov/ .coverage coverage.xml
	rm -rf .mutmut-cache/
	rm -rf .venv-3.1*
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
