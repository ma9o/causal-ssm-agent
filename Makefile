# Makefile for dsem-agent
# Orchestrates uv (Python) and renv (R) environments

.PHONY: all setup setup-python setup-r test test-all test-ctsem-parity test-ctsem-parity-full clean clean-r help

# Default target
all: setup

# ============================================================================
# Setup targets
# ============================================================================

setup: setup-python setup-r ## Set up both Python and R environments

setup-python: ## Set up Python environment with uv
	uv sync --all-groups

setup-r: renv/.installed ## Set up R environment with renv

renv/.installed: renv.lock
	@echo "Setting up R environment with renv..."
	@mkdir -p renv
	R --quiet -e "if (!requireNamespace('renv', quietly = TRUE)) install.packages('renv', repos='https://cloud.r-project.org/')"
	R --quiet -e "renv::restore(prompt = FALSE)"
	@touch renv/.installed

renv.lock: ## Initialize renv.lock if it doesn't exist
	@if [ ! -f renv.lock ]; then \
		echo "Initializing renv..."; \
		R --quiet -e "renv::init(bare = TRUE)"; \
		R --quiet -e "renv::install(c('ctsem', 'Matrix'))"; \
		R --quiet -e "renv::snapshot()"; \
	fi

# ============================================================================
# Test targets
# ============================================================================

test: setup-python ## Run Python tests (skips R parity tests if R not set up)
	uv run pytest tests/ -v --ignore=tests/test_ctsem_parity.py

test-all: setup ## Run all tests including R parity tests
	uv run pytest tests/ -v

test-ctsem-parity: setup ## Run ctsem parity tests (excluding slow recovery tests)
	uv run pytest tests/test_ctsem.py::TestParityWithCtsem \
		tests/test_ctsem.py::TestParityDimensionScaling \
		tests/test_ctsem.py::TestParityEdgeCases \
		tests/test_ctsem.py::TestParityMultiSubject -v

test-ctsem-parity-full: setup ## Run all ctsem parity tests including slow recovery
	uv run pytest tests/test_ctsem.py::TestParityWithCtsem \
		tests/test_ctsem.py::TestParityDimensionScaling \
		tests/test_ctsem.py::TestParityEdgeCases \
		tests/test_ctsem.py::TestParityMultiSubject \
		tests/test_ctsem.py::TestParityModelRecovery -v

# ============================================================================
# Clean targets
# ============================================================================

clean: ## Clean Python cache files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

clean-r: ## Clean R environment (forces reinstall)
	rm -rf renv/library renv/.installed
	@echo "R environment cleaned. Run 'make setup-r' to reinstall."

clean-all: clean clean-r ## Clean both Python and R artifacts

# ============================================================================
# Help
# ============================================================================

help: ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
