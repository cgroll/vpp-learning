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
import pulp
import seaborn as sns
from tqdm import tqdm

from vpp.paths import ProjPaths

paths = ProjPaths()
paths.ensure_directories()

plt.rcParams["figure.dpi"] = 100
sns.set_theme(style="whitegrid")

CAPACITY_KWH = 100.0
POWER_KW = 50.0
ETA_RT = 0.90
ETA_C = ETA_D = float(np.sqrt(ETA_RT))

HINDSIGHT_COLOR = "#2ca02c"
FORECAST_COLOR = "#d62728"

# %% [markdown]
# ## 2. Load and Organise Prices by Day

# %%
prices_raw = pd.read_parquet(paths.smard_prices_file)
prices_eur_mwh = prices_raw["price_de_lu"].dropna().sort_index()
prices_berlin = prices_eur_mwh.copy()
prices_berlin.index = prices_berlin.index.tz_convert("Europe/Berlin")

# Keep only 24-h days; DST days have 23h (spring-forward) or 25h (fall-back)
price_by_date: dict = {}
for date, grp in prices_berlin.groupby(prices_berlin.index.date):
    if len(grp) == 24:
        price_by_date[date] = grp.values

dates = sorted(price_by_date.keys())
n_skipped = len(set(prices_berlin.index.date)) - len(dates)
print(
    f"Loaded {len(prices_berlin):,} hours → "
    f"{len(dates):,} complete 24h days "
    f"({n_skipped} DST days skipped)"
)

# %% [markdown]
# ## 3. Daily MILP Solver


# %%
def solve_daily_milp(
    price_24h: np.ndarray,
    capacity_kwh: float = CAPACITY_KWH,
    power_kw: float = POWER_KW,
    eta_c: float = ETA_C,
    eta_d: float = ETA_D,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve daily MILP (η < 1, mutual exclusivity, SoC=0 start/end).

    Returns (charge_kw, discharge_kw) for 24 hours.
    """
    prob = pulp.LpProblem("d", pulp.LpMaximize)
    z = pulp.LpVariable.dicts("z", range(24), cat="Binary")
    c = pulp.LpVariable.dicts("c", range(24), lowBound=0, upBound=power_kw)
    d = pulp.LpVariable.dicts("d", range(24), lowBound=0, upBound=power_kw)
    soc = pulp.LpVariable.dicts("s", range(25), lowBound=0, upBound=capacity_kwh)
    prob += pulp.lpSum(price_24h[t] / 1000 * (d[t] - c[t]) for t in range(24))
    prob += soc[0] == 0
    prob += soc[24] == 0
    for t in range(24):
        prob += soc[t + 1] == soc[t] + eta_c * c[t] - (1.0 / eta_d) * d[t]
        prob += c[t] <= power_kw * z[t]
        prob += d[t] <= power_kw * (1 - z[t])
    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"MILP infeasible: {pulp.LpStatus[status]!r}")
    return (
        np.array([c[t].value() for t in range(24)]),
        np.array([d[t].value() for t in range(24)]),
    )


# %% [markdown]
# ## 4. Run Hindsight and Forecast Dispatch

# %%
# Day 0 has no prior day for a naïve forecast; evaluation starts from day 1.
records = []
for i, date in enumerate(tqdm(dates[1:], desc="Solving"), start=1):
    actual = price_by_date[date]
    forecast = price_by_date[dates[i - 1]]  # naïve: previous day's prices

    c_h, d_h = solve_daily_milp(actual)  # hindsight: optimise with actual prices
    c_f, d_f = solve_daily_milp(forecast)  # forecast: optimise with lag-24 prices

    records.append(
        {
            "date": pd.Timestamp(date),
            "rev_hindsight": float(np.dot(actual, d_h - c_h) / 1000),
            "rev_forecast": float(np.dot(actual, d_f - c_f) / 1000),
        }
    )

daily = pd.DataFrame(records).set_index("date")
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
    f"({daily.index[0].date()} → {daily.index[-1].date()}):"
)
print(f"  Hindsight:       {rev_h_ann:,.0f} EUR/yr")
print(f"  Naïve forecast:  {rev_f_ann:,.0f} EUR/yr  ({eff:.1f}% of hindsight)")
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
    .rename(columns={"rev_hindsight": "Hindsight", "rev_forecast": "Naïve"})
)
yearly["Efficiency (%)"] = yearly["Naïve"] / yearly["Hindsight"] * 100

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
    yearly["Naïve"],
    width=w,
    color=FORECAST_COLOR,
    label="Naïve (lag-24)",
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
ax.set_ylabel("Naïve / hindsight (%)")
ax.set_title("Forecast efficiency by year")
ax.set_ylim(0, 130)
ax.legend()

fig.suptitle(
    f"Hindsight vs naïve forecast dispatch  (η_rt={ETA_RT}, MILP, 100 kWh / 50 kW)",
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
# Left: annualised revenue by year for hindsight and naïve forecast dispatch.
# Right: forecast efficiency (naïve / hindsight) by year; the overall mean is
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

# Right: histogram of daily revenue gap (hindsight − forecast)
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
ax.set_xlabel("Hindsight − forecast revenue (EUR/day)")
ax.set_ylabel("Days")
ax.set_title("Distribution of daily revenue gap")

fig.suptitle("Daily dispatch detail — naïve vs hindsight", fontsize=12)
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
# indicate days where the naïve forecast under-performed hindsight; points above
# the x-axis but below the diagonal represent the typical partially-captured day;
# points below the x-axis are days where the naïve dispatch lost money.
# Right: distribution of the daily revenue gap (hindsight − forecast). The mean
# gap equals the average daily value of perfect price foresight.
# ```

# %% [markdown]
# ## 7. Summary
#
# The naïve lag-24 dispatch captures a significant fraction of the hindsight
# upper bound, but the gap measures the cost of not knowing tomorrow's prices.
#
# Key findings:
#
# - **Overall efficiency**: naïve captures ~X% of the perfect-foresight revenue.
# - **2022 crisis effect**: the energy crisis year typically shows either very
#   high absolute revenue (large spreads to exploit) or reduced efficiency (if
#   price patterns changed dramatically day-over-day, making lag-24 unreliable).
# - **Negative-revenue days**: a fraction of days see the naïve strategy actively
#   lose money — the previous day's prices suggested a spread that inverted.
# - **Foresight value**: the gap (hindsight − naïve) is the upper bound on what
#   any forecast improvement can recover. Closing even half of it would materially
#   change the investment case.
#
# **Next**: Stage 2c improves the forecast (rolling training window, richer
# features) and measures how much of the foresight gap it recovers.
