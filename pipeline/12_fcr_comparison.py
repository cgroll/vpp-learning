# ---
# jupytext:
#   text_representation:
#     format_name: percent
# kernelspec:
#   display_name: Python 3
#   language: python
#   name: python3
# ---

# %% [markdown]
# # Stage 3a — FCR vs Day-Ahead Arbitrage
#
# Compares two pure strategies for a 100 kWh / 50 kW battery with η=1 and no
# degradation cost over the FCR data period (2021-01-01 onward):
#
# - **Scenario A — DA only:** optimise day-ahead price arbitrage with perfect
#   foresight (LP, daily reset, SoC=0 at midnight).
# - **Scenario B — FCR only:** commit all six 4-hour blocks to FCR every day;
#   collect capacity payments; no arbitrage. Optimistic assumption: FCR
#   activations cancel within each block, so SoC stays at 50%.
#
# Settlement for scenario A is at actual prices. FCR revenue is the capacity
# payment only (no energy activation revenue modelled).

# %% [markdown]
# ## 1. Setup

# %%
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from vpp.paths import ProjPaths

paths = ProjPaths()
paths.ensure_directories()

plt.rcParams["figure.dpi"] = 100
sns.set_theme(style="whitegrid")

DA_SCENARIO = "actual__lp_dr__eta100__deg000"
FCR_SCENARIO = "fcr_only__eta100"

# %% [markdown]
# ## 2. Load Data

# %%
prices_raw = pd.read_parquet(paths.smard_prices_file)
prices_eur_mwh = prices_raw["price_de_lu"].dropna().sort_index()
prices_berlin = prices_eur_mwh.copy()
prices_berlin.index = prices_berlin.index.tz_convert("Europe/Berlin")

fcr_raw = pd.read_parquet(paths.fcr_prices_file)

dispatch_all = pd.read_parquet(paths.dispatch_schedules_file)

print(f"DA prices: {prices_berlin.index[0].date()} → {prices_berlin.index[-1].date()}")
print(f"FCR prices: {fcr_raw.index[0].date()} → {fcr_raw.index[-1].date()}")

# %% [markdown]
# ## 3. Prepare FCR Prices (long format)
#
# `fcr_raw` has one row per delivery date and one column per 4-hour block.
# Melt to `(local_date, block, fcr_price_eur_mw_h)` for hourly joins.

# %%
_BLOCK_COLS = [
    "negpos_00_04",
    "negpos_04_08",
    "negpos_08_12",
    "negpos_12_16",
    "negpos_16_20",
    "negpos_20_24",
]
_BLOCK_NUM = {col: i for i, col in enumerate(_BLOCK_COLS)}
_BLOCK_LABEL = {
    0: "00–04",
    1: "04–08",
    2: "08–12",
    3: "12–16",
    4: "16–20",
    5: "20–24",
}

fcr_long = fcr_raw.reset_index().melt(
    id_vars="delivery_date",
    var_name="block_col",
    value_name="fcr_price_eur_mw_h",
)
fcr_long["block"] = fcr_long["block_col"].map(_BLOCK_NUM)
fcr_long["local_date"] = fcr_long["delivery_date"].dt.date
fcr_long = fcr_long[["local_date", "block", "fcr_price_eur_mw_h"]]

# %% [markdown]
# ## 4. Compute Revenue

# %%
# Filter to the two comparison scenarios
dispatch = dispatch_all[
    dispatch_all["scenario_id"].isin([DA_SCENARIO, FCR_SCENARIO])
].copy()

prices_df = (
    prices_berlin.rename("price_eur_mwh")
    .reset_index()
    .rename(columns={"index": "timestamp"})
)

# Join DA prices for DA revenue
dispatch = dispatch.merge(prices_df, on="timestamp", how="left")
dispatch["da_revenue_eur"] = (
    dispatch["price_eur_mwh"] * (dispatch["d"] - dispatch["c"]) / 1000
)

# Join FCR prices for FCR revenue
dispatch["local_date"] = dispatch["timestamp"].dt.date
dispatch["block"] = dispatch["timestamp"].dt.hour // 4
dispatch = dispatch.merge(fcr_long, on=["local_date", "block"], how="left")
dispatch["fcr_revenue_eur"] = (
    dispatch["fcr_price_eur_mw_h"].fillna(0) * dispatch["fcr_mw"]
)

dispatch["total_revenue_eur"] = dispatch["da_revenue_eur"] + dispatch["fcr_revenue_eur"]

# Restrict to FCR data period for the comparison
FCR_START = pd.Timestamp("2021-01-01", tz="Europe/Berlin")
dispatch = dispatch[dispatch["timestamp"] >= FCR_START].copy()

dispatch["year"] = dispatch["timestamp"].dt.year
dispatch["month"] = dispatch["timestamp"].dt.month

print(
    f"Comparison period: {dispatch['timestamp'].min().date()} "
    f"→ {dispatch['timestamp'].max().date()}"
)

# %% [markdown]
# ## 5. Annual Revenue Comparison

# %%
annual = (
    dispatch.groupby(["scenario_id", "year"])["total_revenue_eur"]
    .sum()
    .reset_index()
    .rename(columns={"total_revenue_eur": "revenue_eur"})
)

# Annualise partial years by day count
day_counts = (
    dispatch.groupby(["scenario_id", "year"])["local_date"]
    .nunique()
    .reset_index()
    .rename(columns={"local_date": "n_days"})
)
annual = annual.merge(day_counts, on=["scenario_id", "year"])
annual["annual_rev_eur"] = annual["revenue_eur"] / annual["n_days"] * 365.25

print("\nAnnual revenue (EUR/yr):")
print(
    annual.pivot(index="year", columns="scenario_id", values="annual_rev_eur")
    .rename(columns={DA_SCENARIO: "DA only", FCR_SCENARIO: "FCR only"})
    .round(0)
    .to_string()
)

# %%
years = sorted(annual["year"].unique())
da_rev = annual[annual["scenario_id"] == DA_SCENARIO].set_index("year")[
    "annual_rev_eur"
]
fcr_rev = annual[annual["scenario_id"] == FCR_SCENARIO].set_index("year")[
    "annual_rev_eur"
]

x = np.arange(len(years))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(
    x - width / 2,
    [da_rev.get(y, 0) for y in years],
    width,
    label="DA only (η=1, perfect foresight)",
    color="#1f77b4",
    alpha=0.85,
)
ax.bar(
    x + width / 2,
    [fcr_rev.get(y, 0) for y in years],
    width,
    label="FCR only – all blocks",
    color="#ff7f0e",
    alpha=0.85,
)

ax.set_xticks(x)
ax.set_xticklabels(years)
ax.set_ylabel("Annualised revenue (EUR/yr)")
ax.set_title("DA arbitrage vs FCR capacity payments — 100 kWh / 50 kW battery, η=1")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
ax.legend()
fig.tight_layout()
fig.savefig(
    paths.images_path / "12_annual_revenue_comparison.png", dpi=150, bbox_inches="tight"
)
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/12_annual_revenue_comparison.png
# :name: fig-12-annual-revenue-comparison
# Annual revenue for pure DA arbitrage (perfect foresight, LP, daily reset) vs
# pure FCR commitment (all six 4-hour blocks, capacity payments only). The FCR
# revenue reflects the actual clearing prices from regelleistung.net. DA revenue
# is the theoretical upper bound under perfect price foresight.
# ```

# %% [markdown]
# ## 6. FCR Block Price Pattern

# %%
# Monthly mean FCR price per block — shows the intraday and seasonal patterns
fcr_monthly = fcr_raw.copy()
fcr_monthly.index = fcr_monthly.index.tz_convert("Europe/Berlin")
fcr_monthly = fcr_monthly[fcr_monthly.index >= FCR_START]
fcr_monthly["year_month"] = fcr_monthly.index.to_period("M")

fcr_heatmap = fcr_monthly.groupby("year_month")[_BLOCK_COLS].mean()
fcr_heatmap.columns = [_BLOCK_LABEL[_BLOCK_NUM[c]] for c in fcr_heatmap.columns]

fig, ax = plt.subplots(figsize=(14, 6))
sns.heatmap(
    fcr_heatmap.T,
    ax=ax,
    cmap="YlOrRd",
    cbar_kws={"label": "EUR/MW/h"},
    linewidths=0,
)

# Reduce x-tick density: show every 6th label
xtick_positions = list(range(len(fcr_heatmap)))
xtick_labels = [str(p) if i % 6 == 0 else "" for i, p in enumerate(fcr_heatmap.index)]
ax.set_xticks(xtick_positions)
ax.set_xticklabels(xtick_labels, rotation=45, ha="right", fontsize=8)
ax.set_xlabel("")
ax.set_ylabel("4-hour block")
ax.set_title("FCR clearing price by block and month (EUR/MW/h)")
fig.tight_layout()
fig.savefig(paths.images_path / "12_fcr_block_prices.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/12_fcr_block_prices.png
# :name: fig-12-fcr-block-prices
# Monthly mean FCR clearing price per 4-hour block. The midday block (12–16)
# consistently commands the highest price because batteries would otherwise
# prefer DA arbitrage in those hours — FCR must compensate for the opportunity
# cost. The 2022 energy crisis drove exceptional DA spreads (and hence higher
# FCR prices in the midday block) as batteries demanded more to forego arbitrage.
# ```

# %% [markdown]
# ## 7. Summary Table

# %%
overall = (
    dispatch.groupby("scenario_id")
    .agg(
        total_rev=("total_revenue_eur", "sum"),
        n_days=("local_date", "nunique"),
    )
    .assign(annual_rev=lambda df: df["total_rev"] / df["n_days"] * 365.25)
    .rename(
        index={DA_SCENARIO: "DA only (η=1)", FCR_SCENARIO: "FCR only – all blocks"}
    )[["annual_rev"]]
    .rename(columns={"annual_rev": "Annual revenue (EUR/yr)"})
)
overall["vs DA"] = (
    overall["Annual revenue (EUR/yr)"]
    / overall.loc["DA only (η=1)", "Annual revenue (EUR/yr)"]
    - 1
) * 100
print(overall.round(1).to_string())
