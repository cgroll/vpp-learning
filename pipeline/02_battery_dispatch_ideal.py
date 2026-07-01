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
# # Stage 1a — Ideal Battery Dispatch (Perfect Price Foresight)
#
# This notebook solves a linear program (LP) that maximises day-ahead price
# arbitrage revenue for a 100 kWh / 50 kW battery with perfect hindsight —
# it sees the full price curve before committing to a schedule. The result is
# the **theoretical upper bound**: what a clairvoyant price-taker could earn
# before adding forecast error, round-trip losses, or transaction costs.
#
# Two constraint variants are compared:
#
# | Scenario | SoC reset condition | Effect |
# |---|---|---|
# | `daily_reset` | $SoC = 0$ at end of each calendar day (Berlin) | Days decouple |
# | `free_horizon` | $SoC = 0$ at horizon start/end only | Carry-over allowed |
#
# The gap between them is the **value of the daily-reset assumption**: how much
# revenue is lost by forcing the battery to return to empty each night.

# %% [markdown]
# ## 1. LP Formulation
#
# **Decision variable**: $f_t \in [-50, 50]$ kW per hour.
# $f_t > 0$ = net charging from grid; $f_t < 0$ = net discharging to grid.
#
# A single signed variable is valid here because $\eta_{rt} = 1$: with no
# round-trip loss, a variable cannot simultaneously earn revenue as both
# charging and discharging, so the LP will never set it to simultaneously
# positive and negative (structurally impossible for one variable).
#
# **State of charge** (start-of-hour convention):
# $$SoC_t = SoC_0 + \sum_{s=0}^{t-1} f_s, \quad SoC_0 = 0,
# \quad 0 \le SoC_t \le 100 \text{ kWh}$$
#
# **Objective** (maximise revenue):
# $$\max \sum_t -P_t \cdot f_t \,/\, 1000
# \quad [\text{EUR/h, with } P_t \text{ in EUR/MWh}]$$
#
# The end constraint $SoC = 0$ is applied at every calendar day boundary
# (`daily_reset`) or only at the end of the full dataset (`free_horizon`).

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pulp
import seaborn as sns

from vpp.paths import ProjPaths

paths = ProjPaths()
paths.ensure_directories()

plt.rcParams["figure.dpi"] = 100
sns.set_theme(style="whitegrid")

CAPACITY_KWH = 100.0
POWER_KW = 50.0

CHARGE_COLOR = "#1f77b4"
DISCHARGE_COLOR = "#d62728"
SOC_COLOR = "#2ca02c"
SCENARIO_COLORS = {"daily_reset": "#1f77b4", "free_horizon": "#ff7f0e"}

# %% [markdown]
# ## 2. Load Data

# %%
prices_raw = pd.read_parquet(paths.smard_prices_file)
prices_eur_mwh = prices_raw["price_de_lu"].dropna().sort_index()

# Convert to Berlin local time for calendar-day boundary detection
prices_berlin = prices_eur_mwh.copy()
prices_berlin.index = prices_berlin.index.tz_convert("Europe/Berlin")

price_array = prices_berlin.to_numpy(dtype=float)
n = len(price_array)

print(
    f"Loaded {n:,} hourly prices: "
    f"{prices_berlin.index[0].date()} → {prices_berlin.index[-1].date()}"
)

# %% [markdown]
# ## 3. Solve LP


# %%
def _day_boundary_positions(index: pd.DatetimeIndex) -> list[int]:
    """Integer positions (0..n) where SoC must be 0 under daily reset.

    Returns the cumulative-length positions at each day boundary, including 0
    and n, so both the start and end of each day are forced to SoC=0.
    """
    dates = index.date
    lengths = [int((np.array(dates) == d).sum()) for d in sorted(set(dates))]
    boundaries = np.concatenate([[0], np.cumsum(lengths)]).tolist()
    return [int(b) for b in boundaries]


def solve_battery_lp(
    price_array: np.ndarray,
    soc_zero_positions: list[int],
    capacity_kwh: float = CAPACITY_KWH,
    power_kw: float = POWER_KW,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve net-power LP with explicit SoC state variables.

    Uses explicit SoC variables (O(n) sparse constraints) rather than the
    cumulative-sum trick (which produces O(n²) expression sizes for large n).

    soc_zero_positions: positions in 0..n where SoC is forced to 0.
      daily_reset → every day boundary (including 0 and n).
      free_horizon → [0, n] only.

    Returns (f_kw, soc_kwh) arrays of length n, where soc_kwh[t] is the
    start-of-hour SoC (before hour t's dispatch).
    """
    n = len(price_array)
    prob = pulp.LpProblem("battery_ideal", pulp.LpMaximize)

    f = pulp.LpVariable.dicts("f", range(n), lowBound=-power_kw, upBound=power_kw)
    soc = pulp.LpVariable.dicts("soc", range(n + 1), lowBound=0, upBound=capacity_kwh)

    prob += pulp.lpSum(-price_array[t] * f[t] / 1000 for t in range(n))

    for t in range(n):
        prob += soc[t + 1] == soc[t] + f[t]

    for pos in soc_zero_positions:
        prob += soc[pos] == 0

    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"LP status: {pulp.LpStatus[status]!r}")

    f_val = np.array([f[t].value() for t in range(n)])
    soc_val = np.array([soc[t].value() for t in range(n)])
    return f_val, soc_val


# %%
reset_daily = _day_boundary_positions(prices_berlin.index)
reset_free = [0, n]

print(f"Solving daily_reset LP ({len(reset_daily) - 1} days, {n:,} hours) ...")
f_daily, soc_daily = solve_battery_lp(price_array, reset_daily)
print("Done.")

print(f"Solving free_horizon LP ({n:,} hours, single period) ...")
f_free, soc_free = solve_battery_lp(price_array, reset_free)
print("Done.")


# %%
def _build_schedule(
    index: pd.DatetimeIndex,
    price_array: np.ndarray,
    f_kw: np.ndarray,
    soc_kwh: np.ndarray,
    scenario: str,
) -> pd.DataFrame:
    f_clean = np.where(np.abs(f_kw) < 1e-9, 0.0, f_kw)
    return pd.DataFrame(
        {
            "datetime": index,
            "date": pd.to_datetime(index.date),
            "hour": index.hour,
            "price_eur_mwh": price_array,
            "charge_kw": np.clip(f_clean, 0, None).round(6),
            "discharge_kw": np.clip(-f_clean, 0, None).round(6),
            "soc_kwh": np.clip(soc_kwh, 0, None).round(6),
            "revenue_eur": -price_array * f_clean / 1000,
            "scenario": scenario,
        }
    )


schedule = pd.concat(
    [
        _build_schedule(
            prices_berlin.index, price_array, f_daily, soc_daily, "daily_reset"
        ),
        _build_schedule(
            prices_berlin.index, price_array, f_free, soc_free, "free_horizon"
        ),
    ],
    ignore_index=True,
)
schedule["scenario"] = schedule["scenario"].astype("category")

daily = (
    schedule.groupby(["scenario", "date"], observed=True)
    .agg(
        revenue_eur=("revenue_eur", "sum"),
        charge_kwh=("charge_kw", "sum"),
        discharge_kwh=("discharge_kw", "sum"),
    )
    .reset_index()
)
daily["cycles"] = (daily["charge_kwh"] + daily["discharge_kwh"]) / (2 * CAPACITY_KWH)
daily["year"] = daily["date"].dt.year
daily["month"] = daily["date"].dt.month
daily["year_month"] = daily["date"].dt.to_period("M").astype(str)
daily["day_of_month"] = daily["date"].dt.day

# %% [markdown]
# ## 4. Headline Results

# %%
summary = (
    daily.groupby("scenario", observed=True)
    .agg(
        n_days=("revenue_eur", "count"),
        total_revenue_eur=("revenue_eur", "sum"),
        avg_daily_revenue_eur=("revenue_eur", "mean"),
        avg_daily_cycles=("cycles", "mean"),
    )
    .reset_index()
    .sort_values("total_revenue_eur", ascending=False)
)
summary["annual_revenue_eur"] = summary["avg_daily_revenue_eur"] * 365.25
free_total = summary.loc[
    summary["scenario"] == "free_horizon", "total_revenue_eur"
].values[0]
summary["delta_pct"] = (summary["total_revenue_eur"] / free_total - 1) * 100

print(
    summary[
        [
            "scenario",
            "n_days",
            "total_revenue_eur",
            "annual_revenue_eur",
            "delta_pct",
            "avg_daily_cycles",
        ]
    ].to_string(index=False)
)

# %% [markdown]
# ## 5. Diurnal Dispatch Profile

# %%
dr_hourly = schedule[schedule["scenario"] == "daily_reset"].sort_values("datetime")

diurnal = dr_hourly.groupby("hour")[["charge_kw", "discharge_kw", "soc_kwh"]].mean()
midnight = pd.DataFrame(
    {"charge_kw": [np.nan], "discharge_kw": [np.nan], "soc_kwh": [0.0]},
    index=pd.Index([24], name="hour"),
)
diurnal = pd.concat([diurnal, midnight])

fig, ax1 = plt.subplots(figsize=(10, 4.5))
bars = diurnal.iloc[:24]
ax1.bar(
    bars.index,
    bars["charge_kw"],
    width=1.0,
    align="edge",
    label="Avg charge (kW)",
    color=CHARGE_COLOR,
    edgecolor="white",
    linewidth=0.5,
)
ax1.bar(
    bars.index,
    -bars["discharge_kw"],
    width=1.0,
    align="edge",
    label="Avg discharge (kW)",
    color=DISCHARGE_COLOR,
    edgecolor="white",
    linewidth=0.5,
)
ax1.axhline(0, color="black", linewidth=0.6)
ax1.set_xlabel("Hour of day (Europe/Berlin)")
ax1.set_ylabel("Avg power (kW)\ncharge (+) / discharge (−)")
ax1.set_xticks(range(0, 25, 2))

ax2 = ax1.twinx()
ax2.plot(
    diurnal.index,
    diurnal["soc_kwh"],
    color=SOC_COLOR,
    marker="o",
    markersize=3,
    label="Avg SoC (kWh)",
)
ax2.set_ylabel("Avg state of charge (kWh)")
ax2.set_ylim(0, CAPACITY_KWH * 1.1)
ax2.grid(False)

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="upper left")
ax1.set_title("Average diurnal dispatch profile — daily_reset (100 kWh / 50 kW, η=1)")

fig.tight_layout()
fig.savefig(
    paths.images_path / "02_diurnal_dispatch_profile.png", dpi=150, bbox_inches="tight"
)
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/02_diurnal_dispatch_profile.png
# :name: fig-02-diurnal-dispatch-profile
# Average charge power, discharge power, and state of charge (start-of-hour)
# by hour of day under the `daily_reset` constraint. The canonical arbitrage
# pattern: charge during low-price overnight hours, discharge into the morning
# and evening price peaks. SoC starts and ends each day at 0 (hours 0 and 24).
# ```

# %% [markdown]
# ## 6. Revenue Seasonality

# %%
dr_daily = daily[daily["scenario"] == "daily_reset"]
monthly_avg = dr_daily.groupby(["year", "month"])["revenue_eur"].mean().reset_index()
heatmap_data = monthly_avg.pivot(index="year", columns="month", values="revenue_eur")
heatmap_data.columns = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]

fig, ax = plt.subplots(figsize=(12, max(4, len(heatmap_data) * 0.5 + 1)))
sns.heatmap(
    heatmap_data,
    ax=ax,
    cmap="RdYlGn",
    center=float(dr_daily["revenue_eur"].median()),
    cbar_kws={"label": "Avg daily revenue (EUR)"},
    linewidths=0.3,
    linecolor="white",
    annot=True,
    fmt=".0f",
    annot_kws={"size": 7},
)
ax.set_xlabel("")
ax.set_ylabel("Year")
ax.set_title(
    "Average daily arbitrage revenue by year and month — daily_reset (100 kWh / 50 kW)"
)
fig.tight_layout()
fig.savefig(
    paths.images_path / "02_daily_revenue_heatmap.png", dpi=150, bbox_inches="tight"
)
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/02_daily_revenue_heatmap.png
# :name: fig-02-daily-revenue-heatmap
# Average daily arbitrage revenue by calendar year and month. The 2022 energy
# crisis stands out: DE-LU day-ahead price spreads were unusually wide, with
# average daily revenues several times the long-run baseline. Summer months
# (high solar generation, negative-price episodes) tend to show elevated
# spreads.
# ```

# %% [markdown]
# ## 7. Impact of Daily-Reset Constraint

# %%
scenarios = ["free_horizon", "daily_reset"]
annual_rev = {
    s: float(summary.loc[summary["scenario"] == s, "annual_revenue_eur"].values[0])
    for s in scenarios
}
delta = float(summary.loc[summary["scenario"] == "daily_reset", "delta_pct"].values[0])

fig, axes = plt.subplots(
    1, 2, figsize=(13, 4.5), gridspec_kw={"width_ratios": [1, 2.2]}
)

colors = [SCENARIO_COLORS[s] for s in scenarios]
bars_obj = axes[0].bar(
    scenarios, [annual_rev[s] for s in scenarios], color=colors, edgecolor="white"
)
axes[0].set_ylabel("Annualized revenue (EUR/year)")
axes[0].set_title("Annual revenue by scenario")
max_rev = max(annual_rev.values())
axes[0].set_ylim(0, max_rev * 1.18)
for bar, s in zip(bars_obj, scenarios):
    axes[0].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + max_rev * 0.01,
        f"{annual_rev[s]:,.0f} EUR",
        ha="center",
        va="bottom",
        fontsize=9,
    )
axes[0].text(
    0.5,
    0.93,
    f"Daily reset costs {delta:.1f}%",
    ha="center",
    transform=axes[0].transAxes,
    fontsize=10,
    bbox={"boxstyle": "round,pad=0.3", "facecolor": "lightyellow", "edgecolor": "gray"},
)

for s in scenarios:
    d = daily[daily["scenario"] == s].sort_values("date")
    cumrev = d["revenue_eur"].cumsum()
    axes[1].plot(d["date"], cumrev, color=SCENARIO_COLORS[s], linewidth=1.2, label=s)
axes[1].set_xlabel("Date")
axes[1].set_ylabel("Cumulative revenue (EUR)")
axes[1].set_title("Cumulative revenue — both scenarios")
axes[1].legend()

fig.tight_layout()
fig.savefig(
    paths.images_path / "02_annual_revenue_comparison.png", dpi=150, bbox_inches="tight"
)
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/02_annual_revenue_comparison.png
# :name: fig-02-annual-revenue-comparison
# Left: annualized revenue under each constraint variant. Right: cumulative
# revenue over time — the gap between the two lines is the value surrendered
# by requiring the battery to return to empty at local midnight each day.
# ```

# %% [markdown]
# ## 8. SoC Trajectory

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

# Panel 1: diurnal SoC profile for daily_reset
axes[0].plot(
    diurnal.index,
    diurnal["soc_kwh"],
    color=SOC_COLOR,
    marker="o",
    markersize=3,
    linewidth=1.5,
)
axes[0].axhline(
    CAPACITY_KWH,
    color="gray",
    linestyle="--",
    linewidth=0.8,
    label="Capacity (100 kWh)",
)
axes[0].set_xlabel("Hour of day (Europe/Berlin)")
axes[0].set_ylabel("Avg SoC (kWh)")
axes[0].set_title("Diurnal SoC profile — daily_reset")
axes[0].set_xticks(range(0, 25, 2))
axes[0].set_ylim(0, CAPACITY_KWH * 1.1)
axes[0].legend(fontsize=8)

# Panel 2: rolling average of daily mean SoC for free_horizon
fh_hourly = schedule[schedule["scenario"] == "free_horizon"].sort_values("datetime")
fh_daily_soc = fh_hourly.groupby("date")["soc_kwh"].mean().reset_index()
fh_daily_soc["soc_ma30"] = (
    fh_daily_soc["soc_kwh"].rolling(30, center=True, min_periods=1).mean()
)

color = SCENARIO_COLORS["free_horizon"]
axes[1].fill_between(
    fh_daily_soc["date"], fh_daily_soc["soc_kwh"], alpha=0.15, color=color
)
axes[1].plot(
    fh_daily_soc["date"],
    fh_daily_soc["soc_ma30"],
    color=color,
    linewidth=1.5,
    label="30-day rolling avg",
)
axes[1].axhline(
    CAPACITY_KWH,
    color="gray",
    linestyle="--",
    linewidth=0.8,
    label="Capacity (100 kWh)",
)
axes[1].set_xlabel("Date")
axes[1].set_ylabel("Avg daily SoC (kWh)")
axes[1].set_title("Daily-average SoC over time — free_horizon")
axes[1].set_ylim(0, CAPACITY_KWH * 1.1)
axes[1].legend(fontsize=8)

fig.tight_layout()
fig.savefig(paths.images_path / "02_soc_trajectory.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/02_soc_trajectory.png
# :name: fig-02-soc-trajectory
# Left: average state of charge by hour of day under `daily_reset`. The battery
# charges overnight and early morning, discharges into morning and evening price
# peaks, and returns to 0 by hour 24. Right: daily-average SoC over time under
# `free_horizon`, showing how the battery carries inventory across days — most
# visible during periods of highly variable multi-day prices (e.g. the 2022
# energy crisis).
# ```

# %% [markdown]
# ## 9. Revenue from Negative Prices
#
# When day-ahead prices go negative the LP charges the battery — effectively
# getting paid to consume electricity. This section quantifies how much of the
# total arbitrage revenue comes from these negative-price events versus
# conventional spread arbitrage (buy cheap, sell expensive).

# %%
NEG_PRICE_COLOR = "#9b59b6"

dr_hourly = schedule[schedule["scenario"] == "daily_reset"].copy()
dr_hourly["year"] = pd.to_datetime(dr_hourly["datetime"]).dt.year

# Revenue contribution during negative-price hours (charging = positive revenue)
dr_hourly["neg_price_rev"] = np.where(
    dr_hourly["price_eur_mwh"] < 0,
    dr_hourly["revenue_eur"],
    0.0,
)
dr_hourly["pos_price_rev"] = np.where(
    dr_hourly["price_eur_mwh"] >= 0,
    dr_hourly["revenue_eur"],
    0.0,
)

total_rev_all = dr_hourly["revenue_eur"].sum()
neg_rev_all = dr_hourly["neg_price_rev"].sum()
neg_hours = int((dr_hourly["price_eur_mwh"] < 0).sum())

print(f"Total revenue (daily_reset):              {total_rev_all:>10,.0f} EUR")
print(
    f"  from negative-price hours:              {neg_rev_all:>10,.0f} EUR  "
    f"({neg_rev_all / total_rev_all * 100:.1f}% of total)"
)
pos_rev_all = total_rev_all - neg_rev_all
print(
    f"  from positive-price hours:              {pos_rev_all:>10,.0f} EUR  "
    f"({pos_rev_all / total_rev_all * 100:.1f}% of total)"
)
print(
    f"Negative-price hours: {neg_hours:,} "
    f"({neg_hours / len(dr_hourly) * 100:.1f}% of all hours)"
)

# %%
yearly_neg = (
    dr_hourly.groupby("year")
    .agg(
        total_rev=("revenue_eur", "sum"),
        neg_price_rev=("neg_price_rev", "sum"),
        pos_price_rev=("pos_price_rev", "sum"),
        neg_hours=("price_eur_mwh", lambda x: (x < 0).sum()),
    )
    .reset_index()
)
yearly_neg["neg_price_pct"] = (
    yearly_neg["neg_price_rev"] / yearly_neg["total_rev"] * 100
)

print("\nNegative-price revenue by year:")
print(
    yearly_neg[["year", "total_rev", "neg_price_rev", "neg_price_pct", "neg_hours"]]
    .rename(
        columns={
            "total_rev": "Total (EUR)",
            "neg_price_rev": "Neg-price (EUR)",
            "neg_price_pct": "Neg-price (%)",
            "neg_hours": "Neg-price hours",
        }
    )
    .to_string(index=False, float_format="{:,.1f}".format)
)

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

years = yearly_neg["year"].astype(str)
x = np.arange(len(years))

# Stacked bar: pos-price vs neg-price revenue
axes[0].bar(
    x, yearly_neg["pos_price_rev"], label="Positive-price arbitrage", color=CHARGE_COLOR
)
axes[0].bar(
    x,
    yearly_neg["neg_price_rev"],
    bottom=yearly_neg["pos_price_rev"],
    label="Negative-price charging",
    color=NEG_PRICE_COLOR,
)
axes[0].set_xticks(x)
axes[0].set_xticklabels(years)
axes[0].set_ylabel("Annual revenue (EUR)")
axes[0].set_title("Revenue by source")
axes[0].legend(fontsize=8)

# % of total from negative prices
axes[1].bar(x, yearly_neg["neg_price_pct"], color=NEG_PRICE_COLOR)
avg_pct = yearly_neg["neg_price_pct"].mean()
axes[1].axhline(
    avg_pct,
    color="gray",
    linestyle="--",
    linewidth=0.9,
    label=f"Avg {avg_pct:.1f}%",
)
axes[1].set_xticks(x)
axes[1].set_xticklabels(years)
axes[1].set_ylabel("Share of total revenue (%)")
axes[1].set_title("Negative-price charging as % of annual revenue")
axes[1].legend(fontsize=8)

fig.tight_layout()
fig.savefig(
    paths.images_path / "02_negative_price_breakdown.png", dpi=150, bbox_inches="tight"
)
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/02_negative_price_breakdown.png
# :name: fig-02-negative-price-breakdown
# Left: annual revenue stacked by source — positive-price spread arbitrage
# (buy low, sell high) vs negative-price charging (getting paid to consume).
# Right: the negative-price component as a percentage of total annual revenue.
# The share spikes in years with frequent solar-driven negative-price events
# and post-2022 renewable expansion.
# ```

# %% [markdown]
# ## 10. Annual Revenue Table

# %%
dr_daily_dr = daily[daily["scenario"] == "daily_reset"].copy()
annual_table = (
    dr_daily_dr.groupby("year")
    .agg(
        n_days=("revenue_eur", "count"),
        total_revenue_eur=("revenue_eur", "sum"),
        avg_daily_revenue_eur=("revenue_eur", "mean"),
        avg_daily_cycles=("cycles", "mean"),
    )
    .reset_index()
)
annual_table["avg_annual_cycles"] = annual_table["avg_daily_cycles"] * 365.25
annual_table["full_year"] = annual_table["n_days"] >= 360

print("Annual revenue — daily_reset (100 kWh / 50 kW, η=1):")
print(
    annual_table[
        [
            "year",
            "n_days",
            "total_revenue_eur",
            "avg_daily_revenue_eur",
            "avg_annual_cycles",
        ]
    ]
    .rename(
        columns={
            "n_days": "Days",
            "total_revenue_eur": "Revenue (EUR)",
            "avg_daily_revenue_eur": "Avg daily (EUR)",
            "avg_annual_cycles": "Est. annual cycles",
        }
    )
    .to_string(index=False, float_format="{:.1f}".format)
)

# %%
fig, ax = plt.subplots(figsize=(10, 4.5))

x = np.arange(len(annual_table))
bar_colors = [CHARGE_COLOR if fy else "#aec7e8" for fy in annual_table["full_year"]]
bars = ax.bar(x, annual_table["total_revenue_eur"], color=bar_colors, edgecolor="white")

full_yr_avg = annual_table.loc[annual_table["full_year"], "total_revenue_eur"].mean()
ax.axhline(
    full_yr_avg,
    color="gray",
    linestyle="--",
    linewidth=0.9,
    label=f"Full-year avg {full_yr_avg:,.0f} EUR",
)
ax.set_xticks(x)
ax.set_xticklabels(annual_table["year"])
ax.set_ylabel("Annual revenue (EUR)")
ax.set_title("Annual arbitrage revenue — daily_reset (100 kWh / 50 kW, η=1)")
ax.legend(fontsize=8)
for bar, rev, fy in zip(
    bars, annual_table["total_revenue_eur"], annual_table["full_year"]
):
    suffix = "" if fy else "*"
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + full_yr_avg * 0.01,
        f"{rev:,.0f}{suffix}",
        ha="center",
        va="bottom",
        fontsize=8,
    )
ax.text(0.01, 0.02, "* partial year", transform=ax.transAxes, fontsize=7, color="gray")

fig.tight_layout()
fig.savefig(
    paths.images_path / "02_annual_revenue_table.png", dpi=150, bbox_inches="tight"
)
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/02_annual_revenue_table.png
# :name: fig-02-annual-revenue-table
# Annual arbitrage revenue under the `daily_reset` constraint. Lighter bars
# mark partial calendar years (2018, 2026). The 2022 energy crisis produced
# revenues roughly 3–5× the long-run average due to exceptionally wide
# day-ahead price spreads. The dashed line is the average across full years.
# ```

# %% [markdown]
# ## 11. Battery Investment Economics & IRR
#
# How profitable is the battery as a capital investment?
#
# **Cost assumptions** (all-in, 2024 commercial Li-ion BESS):
#
# | Component | Unit cost | 100 kWh / 50 kW system |
# |---|---|---|
# | Storage (cells + BMS) | €200–400 / kWh | €20,000–40,000 |
# | Power conversion (inverter) | €100–200 / kW | €5,000–10,000 |
# | Installation + grid connection | — | €5,000–10,000 |
# | **Total CAPEX** | **€300–600 / kWh** | **€30,000–60,000** |
#
# | Parameter | Value |
# |---|---|
# | O&M | 1.5 % of CAPEX / year |
# | Project lifetime | 12 years (LFP capacity-retention horizon) |
# | Salvage value | 0 EUR (conservative) |
#
# **Revenue projection**: full-calendar-year observed revenues for years
# available; long-run average fills remaining years. Two variants:
# *including* the 2022 outlier and *excluding* it.

# %%
HURDLE_RATE = 0.08
PROJECT_LIFETIME = 12
OAM_RATE = 0.015

capex_scenarios = {
    "Optimistic (€300/kWh)": 300.0 * CAPACITY_KWH,
    "Base case (€450/kWh)": 450.0 * CAPACITY_KWH,
    "Conservative (€600/kWh)": 600.0 * CAPACITY_KWH,
}

full_year_rows = annual_table[annual_table["full_year"]]
full_year_rev = full_year_rows["total_revenue_eur"].values
full_year_labels = full_year_rows["year"].values

lr_avg_incl = float(full_year_rev.mean())
lr_avg_excl = float(
    full_year_rows.loc[full_year_rows["year"] != 2022, "total_revenue_eur"].mean()
)

print(f"Full years: {full_year_labels}")
print(f"Long-run average (incl. 2022): {lr_avg_incl:,.0f} EUR/yr")
print(f"Long-run average (excl. 2022): {lr_avg_excl:,.0f} EUR/yr")


# %%
def _compute_irr(
    cash_flows: np.ndarray, tol: float = 1e-8, max_iter: int = 300
) -> float:
    """IRR by bisection: find rate where NPV = 0."""
    t = np.arange(len(cash_flows), dtype=float)

    def npv(r: float) -> float:
        return float(np.sum(cash_flows / (1.0 + r) ** t))

    lo, hi = -0.999, 20.0
    if npv(lo) * npv(hi) > 0:
        return np.nan
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        if npv(mid) > 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return (lo + hi) / 2.0


def _build_cash_flows(
    capex: float,
    oam_rate: float,
    lifetime: int,
    observed: np.ndarray,
    long_run_avg: float,
) -> np.ndarray:
    annual_oam = capex * oam_rate
    rev = list(observed) + [long_run_avg] * max(0, lifetime - len(observed))
    net = [r - annual_oam for r in rev[:lifetime]]
    return np.array([-capex] + net)


def _simple_payback(cash_flows: np.ndarray, capex: float) -> float:
    """Years until cumulative net revenues recover CAPEX."""
    cumnet = np.cumsum(cash_flows[1:])
    if cumnet[-1] < capex:
        return np.nan
    idx = int(np.argmax(cumnet >= capex))
    prev = cumnet[idx - 1] if idx > 0 else 0.0
    frac = (capex - prev) / (cumnet[idx] - prev)
    return idx + frac


# %%
irr_results = {}
for name, capex in capex_scenarios.items():
    row = {}
    for label, avg in [("incl. 2022", lr_avg_incl), ("excl. 2022", lr_avg_excl)]:
        cf = _build_cash_flows(capex, OAM_RATE, PROJECT_LIFETIME, full_year_rev, avg)
        irr = _compute_irr(cf)
        pb = _simple_payback(cf, capex)
        t = np.arange(len(cf), dtype=float)
        npv_at_hurdle = float(np.sum(cf / (1 + HURDLE_RATE) ** t))
        row[label] = {"irr": irr, "payback": pb, "npv_hurdle": npv_at_hurdle}
    irr_results[name] = row

oam_pct = OAM_RATE * 100
print(
    f"\nBattery investment economics"
    f"  (lifetime={PROJECT_LIFETIME} yr, O&M={oam_pct:.1f}% CAPEX/yr)"
)
col_hdr = f"{'Scenario':<32} {'CAPEX':>8}  {'Variant':<13}"
col_hdr += f" {'IRR':>7}  {'Payback':>9}  {'NPV@8% (EUR)':>13}"
print(col_hdr)
print("-" * 90)
for name, capex in capex_scenarios.items():
    for label in ("incl. 2022", "excl. 2022"):
        r = irr_results[name][label]
        irr_str = f"{r['irr'] * 100:.1f}%" if not np.isnan(r["irr"]) else "N/A"
        pb_str = f"{r['payback']:.1f} yr" if not np.isnan(r["payback"]) else ">15 yr"
        prefix = f"{name:<32} {capex:>8,.0f}" if label == "incl. 2022" else " " * 41
        npv_str = f"{r['npv_hurdle']:>13,.0f}"
        print(f"{prefix}  {label:<13} {irr_str:>7}  {pb_str:>9}  {npv_str}")

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

# Left: NPV profile (incl. 2022 variant only, for readability)
rates = np.linspace(-0.02, 0.60, 300)
cmap_colors = ["#2ca02c", "#1f77b4", "#d62728"]
for (name, capex), col in zip(capex_scenarios.items(), cmap_colors):
    cf = _build_cash_flows(
        capex, OAM_RATE, PROJECT_LIFETIME, full_year_rev, lr_avg_incl
    )
    t = np.arange(len(cf), dtype=float)
    npvs = [float(np.sum(cf / (1 + r) ** t)) for r in rates]
    irr = irr_results[name]["incl. 2022"]["irr"]
    axes[0].plot(
        rates * 100, npvs, color=col, linewidth=1.5, label=name.split("(")[0].strip()
    )
    if not np.isnan(irr):
        axes[0].axvline(irr * 100, color=col, linestyle=":", linewidth=0.9, alpha=0.7)
axes[0].axhline(0, color="black", linewidth=0.7)
axes[0].axvline(
    HURDLE_RATE * 100,
    color="gray",
    linestyle="--",
    linewidth=0.9,
    label=f"{HURDLE_RATE * 100:.0f}% hurdle",
)
axes[0].set_xlabel("Discount rate (%)")
axes[0].set_ylabel("NPV (EUR)")
axes[0].set_title(f"NPV profile — {PROJECT_LIFETIME}-yr project (incl. 2022)")
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.4)

# Right: IRR bar chart comparing both revenue variants
bar_labels = [n.split("(")[0].strip() for n in capex_scenarios]
x = np.arange(len(bar_labels))
w = 0.38
for i, (variant, col) in enumerate(
    [("incl. 2022", "#1f77b4"), ("excl. 2022", "#ff7f0e")]
):
    irrs = [irr_results[name][variant]["irr"] * 100 for name in capex_scenarios]
    offset = (i - 0.5) * w
    bars = axes[1].bar(
        x + offset,
        irrs,
        width=w,
        color=col,
        alpha=0.85,
        label=f"Revenue {variant}",
        edgecolor="white",
    )
    for bar, v in zip(bars, irrs):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            v + 0.3,
            f"{v:.1f}%",
            ha="center",
            va="bottom",
            fontsize=7.5,
        )
axes[1].axhline(
    HURDLE_RATE * 100,
    color="gray",
    linestyle="--",
    linewidth=0.9,
    label=f"{HURDLE_RATE * 100:.0f}% hurdle rate",
)
axes[1].set_xticks(x)
axes[1].set_xticklabels(bar_labels, fontsize=8.5)
axes[1].set_ylabel("IRR (%)")
axes[1].set_title(f"IRR by CAPEX scenario — {PROJECT_LIFETIME}-yr project")
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3, axis="y")

fig.tight_layout()
fig.savefig(paths.images_path / "02_irr_analysis.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/02_irr_analysis.png
# :name: fig-02-irr-analysis
# Left: NPV profile (NPV vs discount rate) for each CAPEX scenario using
# observed annual revenues including the 2022 spike. Vertical dotted lines mark
# each scenario's IRR; the dashed grey line is an 8 % hurdle rate. Right: IRR
# for each CAPEX scenario under two revenue projections — including the
# exceptional 2022 revenues (optimistic) and excluding them (conservative
# long-run estimate).
# ```
