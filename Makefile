.PHONY: run status serve lint format typecheck check

# Run the full pipeline (only rebuilds what's out of date)
run:
	uv run dvc repro

# Show what's out of date without running
status:
	uv run dvc status

# Serve the book locally with live-reload
serve:
	cd book && uv run myst start

# Lint source files
lint:
	uv run ruff check vpp/ pipeline/

# Format source files in place
format:
	uv run ruff format vpp/ pipeline/

# Type check the package
typecheck:
	uv run pyright vpp/

# Lint + typecheck (no auto-fix)
check: lint typecheck
