.PHONY: run dry-run serve

# Run the full pipeline (only rebuilds what's out of date)
run:
	uv run snakemake -j4

# Preview what would be executed without running anything
dry-run:
	uv run snakemake -n

# Serve the book locally with live-reload
serve:
	cd book && uv run myst start
