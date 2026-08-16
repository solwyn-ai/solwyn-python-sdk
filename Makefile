.PHONY: install install-hooks lint fmt fmt-check typecheck test test-unit test-integration test-frameworks check capture-provider-surfaces check-provider-surfaces check-surface-canary export-surface-contract check-surface-contract

##@ Setup

install: ## Install package in dev mode
	uv pip install -e ".[dev]"

install-hooks: ## Install git pre-commit hook
	@hook_dir=$$(git rev-parse --git-common-dir)/hooks && \
	mkdir -p "$$hook_dir" && \
	if [ -f "$$hook_dir/pre-commit" ]; then \
		echo "Backing up existing pre-commit hook"; \
		cp "$$hook_dir/pre-commit" "$$hook_dir/pre-commit.bak"; \
	fi && \
	cp scripts/pre-commit "$$hook_dir/pre-commit" && \
	chmod +x "$$hook_dir/pre-commit" && \
	echo "Pre-commit hook installed"

##@ Quality

lint: ## Run ruff linter
	uv run ruff check src/ tests/ scripts/

fmt: ## Auto-format code
	uv run ruff format src/ tests/ scripts/
	uv run ruff check --fix src/ tests/ scripts/

fmt-check: ## Check formatting without changes
	uv run ruff format --check src/ tests/ scripts/

typecheck: ## Run mypy strict type checking
	uv run mypy src/solwyn/

check: lint fmt-check typecheck ## Full quality gate (pre-commit hook)

##@ Testing

test: test-unit check-surface-canary ## Run unit tests and the real-SDK canary (default)

test-unit: ## Run unit tests
	uv run pytest tests/ -m unit -v --tb=short --ignore=tests/unit/test_surface_canary.py --ignore=tests/integration/frameworks

test-integration: ## Run integration tests (requires API at localhost:8080)
	uv run pytest tests/ -m integration -v --tb=short --ignore=tests/integration/frameworks

test-frameworks: ## Run offline smoke tests against real framework packages
	uv run --extra dev --group frameworks pytest tests/integration/frameworks/ -m framework_smoke -v --tb=short

capture-provider-surfaces: ## Refresh latest real-SDK surface fingerprints
	uv run --extra dev --with 'aioboto3>=13.0' python scripts/capture_surface_inventory.py --interval latest --output-dir build/provider_surface_inventory

check-provider-surfaces: ## Verify latest real-SDK surface fingerprints
	uv run --extra dev --with 'aioboto3>=13.0' python scripts/capture_surface_inventory.py --check --interval latest --output-dir build/provider_surface_inventory

check-surface-canary: ## Verify every latest real-SDK graph against the contextual contract
	uv run --extra dev --with 'aioboto3>=13.0' pytest tests/unit/test_surface_canary.py -v --tb=short

export-surface-contract: ## Generate the contextual surface contract artifact under build/
	uv run python scripts/export_surface_contract.py

check-surface-contract: ## Generate and verify the contextual surface contract artifact
	uv run python scripts/export_surface_contract.py --check
	uv run python scripts/embed_surface_rules.py --input build/surface_contract/surface-classification.json --check
