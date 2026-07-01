# VPP Learning

A learning project covering end-to-end Virtual Power Plant (VPP) operations: day-ahead price forecasting, load/generation forecasting, battery dispatch optimization, and intraday trading. The project is structured as an interactive book backed by a reproducible data pipeline.

## Browsing the book

The rendered book is published to GitHub Pages. To preview it locally:

```bash
make serve
```

## Environment setup

Requires [uv](https://docs.astral.sh/uv/). Install it once, then:

```bash
uv sync
```

This creates `.venv/` and installs all dependencies including dev tools.

## Running the pipeline

```bash
make run       # run all outdated pipeline stages
make status    # check what's out of date without running
```

To run or force-rerun a single stage:

```bash
uv run dvc repro download_smard_prices
uv run dvc repro --force download_smard_prices
```

## Code quality

```bash
make lint        # ruff lint check
make format      # ruff format (writes fixes in place)
make typecheck   # pyright type check
make check       # lint + typecheck (no auto-fix)
```

## Contributing

See [CONVENTIONS.md](CONVENTIONS.md) for pipeline patterns, coding conventions, DVC usage, and git conventions.

See [PROJECT.md](PROJECT.md) for current goals, project state, and next steps.
