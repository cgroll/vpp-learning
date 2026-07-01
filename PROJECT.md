# Project State

## Goals

End-to-end Virtual Power Plant (VPP) learning project. Planned stages, roughly in order:

1. **Day-ahead price forecasting** — EPEX Spot / Nord Pool prices for DE-LU zone. Starting data source: SMARD (Bundesnetzagentur).
2. **Zone-level load/generation forecasting** — aggregate load and renewables for DE-LU. Also SMARD.
3. **Household/prosumer forecasting** — net energy (production − consumption) per geographic cluster, with PV + weather features. Dataset: Enefit/Kaggle (Estonian counties).
4. **Battery / VPP optimization** — LP/stochastic dispatch against day-ahead prices. Reference: `/home/chris/research/flexa-challenge/` Task 2.
5. **Intraday trading** (later) — synthetic order book anchored to ex-post prices; RL / threshold policies.

## Current state (2026-07-01)

Infrastructure set up, first data source wired in:

- DVC pipeline configured (replaced earlier Snakemake setup)
- MyST Jupyter Book skeleton in place with table of contents
- `vpp` package with centralized path configuration (`ProjPaths`)
- SMARD price download: script written (`pipeline/01_download_smard_prices.py`), DVC stage defined, DE-LU hourly prices downloaded (`data/downloads/smard/prices_de_lu.parquet`, ~810K)

## Next steps

1. Run `make run` to execute the `download_smard_prices` stage and fetch DE-LU hourly prices
2. Write `pipeline/02_eda_smard_prices.py` — EDA of the price time series (distribution, seasonality, spike characterization)
3. Add the notebook to `book/myst.yml` TOC and run `make serve` to verify the book renders
