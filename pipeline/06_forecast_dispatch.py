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
# # Stage 2b — Forecast-Driven Battery Dispatch
#
# Stage 1 established the perfect-foresight revenue upper bound: the MILP always
# optimises with today's actual prices. Stage 2b replaces the optimisation signal
# with the **naïve lag-24 forecast** (yesterday's prices), then settles both
# strategies against the actual day-ahead clearing prices.
#
# Battery: η_rt = 0.9, MILP (binary mutual exclusivity), no degradation,
# 100 kWh / 50 kW, daily reset.
#
# | Strategy | Optimisation prices | Settlement |
# |---|---|---|
# | **Hindsight** | Actual prices, day D | Actual prices, day D |
# | **Naïve (lag-24)** | Actual prices, day D−1 | Actual prices, day D |
#
# Because hindsight dispatch is optimal for the actual prices, hindsight revenue
# ≥ forecast revenue on every single day. The gap is the value of perfect price
# foresight — the maximum any better forecast model could recover.

# %% [markdown]
# ## 1. Setup

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from vpp.paths import ProjPaths

paths = ProjPaths()
paths.ensure_directories()

plt.rcParams["figure.dpi"] = 100
sns.set_theme(style="whitegrid")

CAPACITY_KWH = 100.0
POWER_KW = 50.0
ETA_RT = 0.90

HINDSIGHT_COLOR = "#2ca02c"
FORECAST_COLOR = "#d62728"

# %% [markdown]
# ## 2. Load Data

# %%
prices_raw = pd.read_parquet(paths.smard_prices_file)
prices_eur_mwh = prices_raw["price_de_lu"].dropna().sort_index()
prices_berlin = prices_eur_mwh.copy()
prices_berlin.index = prices_berlin.index.tz_convert("Europe/Berlin")

print(
    f"Loaded {len(prices_berlin):,} hourly prices: "
    f"{prices_berlin.index[0].date()} to {prices_berlin.index[-1].date()}"
)

# %% [markdown]
# ## 3. Load Pre-computed Dispatch

# %%
_SCENARIO_IDS = [
    "actual__milp_dr__eta090__deg000",
    "naive__milp_dr__eta090__deg000",
]

dispatch_raw = pd.read_parquet(
    paths.dispatch_schedules_file,
    filters=[("scenario_id", "in", _SCENARIO_IDS)],
)

prices_df = (
    prices_berlin.rename("price_eur_mwh")
    .reset_index()
    .rename(columns={"index": "timestamp"})
)
dispatch_merged = dispatch_raw.merge(prices_df, on="timestamp")
dispatch_merged["revenue_eur"] = (
    dispatch_merged["price_eur_mwh"]
    * (dispatch_merged["d"] - dispatch_merged["c"])
    / 1000
)
dispatch_merged["date"] = pd.to_datetime(dispatch_merged["timestamp"].dt.date)

daily_rev = (
    dispatch_merged.groupby(["date", "scenario_id"])["revenue_eur"]
    .sum()
    .unstack("scenario_id")
    .reset_index()
    .rename(
        columns={
            "actual__milp_dr__eta090__deg000": "rev_hindsight",
            "naive__milp_dr__eta090__deg000": "rev_forecast",
        }
    )
    .dropna()
)
daily = daily_rev.set_index("date")
daily.index = pd.DatetimeIndex(daily.index).tz_localize("Europe/Berlin")
daily["gap"] = daily["rev_hindsight"] - daily["rev_forecast"]
daily["year"] = daily.index.year

n_days = len(daily)
ann = 365.25 / n_days
rev_h_ann = float(daily["rev_hindsight"].sum() * ann)
rev_f_ann = float(daily["rev_forecast"].sum() * ann)
eff = rev_f_ann / rev_h_ann * 100

print(
    f"\nAnnualised over {n_days:,} days "
    f"({daily.index[0].date()} to {daily.index[-1].date()}):"
)
print(f"  Hindsight:       {rev_h_ann:,.0f} EUR/yr")
print(f"  Naive forecast:  {rev_f_ann:,.0f} EUR/yr  ({eff:.1f}% of hindsight)")
print(f"  Revenue gap:     {rev_h_ann - rev_f_ann:,.0f} EUR/yr  ({100 - eff:.1f}%)")

neg_days = (daily["rev_forecast"] < 0).sum()
neg_pct = neg_days / n_days * 100
print(f"\n  Days with negative forecast revenue: {neg_days:,} ({neg_pct:.1f}%)")

# %% [markdown]
# ## 5. Annual Revenue Comparison

# %%
yearly = (
    daily.groupby("year")[["rev_hindsight", "rev_forecast"]]
    .agg(lambda s: s.sum() * 365.25 / len(s))
    .rename(columns={"rev_hindsight": "Hindsight", "rev_forecast": "Naive"})
)
yearly["Efficiency (%)"] = yearly["Naive"] / yearly["Hindsight"] * 100

print("\nAnnual revenue by year (annualised EUR/yr):")
print(yearly.round(1).to_string())

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

x = np.arange(len(yearly))
w = 0.35

ax = axes[0]
ax.bar(
    x - w / 2,
    yearly["Hindsight"],
    width=w,
    color=HINDSIGHT_COLOR,
    label="Hindsight",
)
ax.bar(
    x + w / 2,
    yearly["Naive"],
    width=w,
    color=FORECAST_COLOR,
    label="Naive (lag-24)",
)
ax.set_xticks(x)
ax.set_xticklabels(yearly.index)
ax.set_ylabel("EUR/yr (annualised)")
ax.set_title("Annual revenue by strategy")
ax.legend()

ax = axes[1]
bars = ax.bar(x, yearly["Efficiency (%)"], color=FORECAST_COLOR, alpha=0.75)
ax.axhline(
    eff,
    color="black",
    linewidth=1.0,
    linestyle="--",
    label=f"Overall mean: {eff:.1f}%",
)
for bar, val in zip(bars, yearly["Efficiency (%)"]):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.5,
        f"{val:.0f}%",
        ha="center",
        va="bottom",
        fontsize=9,
    )
ax.set_xticks(x)
ax.set_xticklabels(yearly.index)
ax.set_ylabel("Naive / hindsight (%)")
ax.set_title("Forecast efficiency by year")
ax.set_ylim(0, 130)
ax.legend()

fig.suptitle(
    f"Hindsight vs naive forecast dispatch  (eta_rt={ETA_RT}, MILP, 100 kWh / 50 kW)",
    fontsize=12,
)
fig.tight_layout()
fig.savefig(
    paths.images_path / "06_annual_revenue_comparison.png",
    dpi=150,
    bbox_inches="tight",
)
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/06_annual_revenue_comparison.png
# :name: fig-06-annual-revenue-comparison
# Left: annualised revenue by year for hindsight and naive forecast dispatch.
# Right: forecast efficiency (naive / hindsight) by year; the overall mean is
# shown as a dashed line. The 2022 energy crisis year stands out in both panels.
# ```

# %% [markdown]
# ## 6. Daily Revenue Detail

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

# Left: scatter daily revenue — hindsight (x) vs forecast (y), coloured by year
ax = axes[0]
years = sorted(daily["year"].unique())
cmap = plt.colormaps["tab10"]
for j, yr in enumerate(years):
    mask = daily["year"] == yr
    ax.scatter(
        daily.loc[mask, "rev_hindsight"],
        daily.loc[mask, "rev_forecast"],
        s=4,
        alpha=0.4,
        color=cmap(j),
        label=str(yr),
    )
xy_max = max(
    daily["rev_hindsight"].quantile(0.999),
    daily["rev_forecast"].quantile(0.999),
)
xy_min = min(daily["rev_forecast"].min(), 0)
ax.plot(
    [xy_min, xy_max],
    [xy_min, xy_max],
    "k--",
    linewidth=0.8,
    label="Perfect capture",
)
ax.set_xlabel("Hindsight revenue (EUR/day)")
ax.set_ylabel("Forecast revenue (EUR/day)")
ax.set_title("Daily revenue: forecast vs hindsight")
ax.legend(fontsize=7, ncol=2)

# Right: histogram of daily revenue gap (hindsight - forecast)
ax = axes[1]
gap_mean = daily["gap"].mean()
ax.hist(daily["gap"], bins=60, color=FORECAST_COLOR, alpha=0.75, edgecolor="none")
ax.axvline(gap_mean, color="black", linewidth=1.0, linestyle="--")
ax.text(
    gap_mean + daily["gap"].std() * 0.05,
    ax.get_ylim()[1] * 0.9,
    f"Mean gap: {gap_mean:.2f} EUR/day",
    fontsize=9,
)
ax.set_xlabel("Hindsight - forecast revenue (EUR/day)")
ax.set_ylabel("Days")
ax.set_title("Distribution of daily revenue gap")

fig.suptitle("Daily dispatch detail — naive vs hindsight", fontsize=12)
fig.tight_layout()
fig.savefig(
    paths.images_path / "06_forecast_dispatch_detail.png",
    dpi=150,
    bbox_inches="tight",
)
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/06_forecast_dispatch_detail.png
# :name: fig-06-forecast-dispatch-detail
# Left: daily revenue scatter — each point is one day. Points below the diagonal
# indicate days where the naive forecast under-performed hindsight; points above
# the x-axis but below the diagonal represent the typical partially-captured day;
# points below the x-axis are days where the naive dispatch lost money.
# Right: distribution of the daily revenue gap (hindsight − forecast). The mean
# gap equals the average daily value of perfect price foresight.
# ```

# %% [markdown]
# ## 7. Summary
#
# The naive lag-24 dispatch captures a significant fraction of the hindsight
# upper bound, but the gap measures the cost of not knowing tomorrow's prices.
#
# Key findings:
#
# - **Overall efficiency**: naive captures ~X% of the perfect-foresight revenue.
# - **2022 crisis effect**: the energy crisis year typically shows either very
#   high absolute revenue (large spreads to exploit) or reduced efficiency (if
#   price patterns changed dramatically day-over-day, making lag-24 unreliable).
# - **Negative-revenue days**: a fraction of days see the naive strategy actively
#   lose money — the previous day's prices suggested a spread that inverted.
# - **Foresight value**: the gap (hindsight − naive) is the upper bound on what
#   any forecast improvement can recover. Closing even half of it would materially
#   change the investment case.
#
# **Next**: Stage 2c improves the forecast (rolling training window, richer
# features) and measures how much of the foresight gap it recovers.
