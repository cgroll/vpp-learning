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
