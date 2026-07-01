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
# # Stage 1b — Realistic Battery Dispatch (Efficiency + Degradation)
#
# Stage 1a established the theoretical upper bound: perfect-foresight arbitrage with
# no physical losses. Stage 1b adds two sources of revenue erosion:
#
# 1. **Round-trip efficiency η_rt = 90 %** — 10 % of stored energy is lost on every
#    charge/discharge cycle. A single signed net-flow variable is no longer valid once
#    η < 1; separate charge and discharge variables are required.
#
# 2. **Degradation cost** — cycling wears the battery. Modelled as a linear EUR/kWh
#    penalty on energy stored, parameterised by a cost per equivalent full cycle.
#
# Three fixed scenarios are compared, then the degradation cost is swept to answer:
# *at what per-cycle cost does day-ahead arbitrage stop clearing its own wear?*
#
# | Scenario | η_rt | Deg cost | Purpose |
# |---|---|---|---|
# | `ideal` | 1.0 | 0 | Stage 1a reference (daily reset) |
# | `eta90` | 0.9 | 0 | Isolate round-trip loss impact |
# | `eta90_deg` | 0.9 | 10 EUR/cycle | Combined physical realism |

# %% [markdown]
# ## 1. LP Formulation
#
# With $\eta_{rt} < 1$, separate **charge** $c_t \ge 0$ and **discharge** $d_t \ge 0$
# variables (both in kW at the grid boundary) are needed:
#
# $$\text{SoC}_{t+1} = \text{SoC}_t + \eta_c \, c_t - \frac{d_t}{\eta_d}$$
#
# where $\eta_c = \eta_d = \sqrt{\eta_{rt}}$ (symmetric split of round-trip losses).
# Physical constraints:
#
# $$0 \le c_t \le P_{\max}, \quad 0 \le d_t \le P_{\max},
# \quad 0 \le \text{SoC}_t \le C$$
#
# Objective — maximise revenue minus degradation cost:
#
# $$\max \sum_t \left[\frac{P_t}{1000}(d_t - c_t)
#   - \kappa_{\text{deg}} \cdot \eta_c \cdot c_t\right]$$
#
# $\kappa_{\text{deg}}$ (EUR/kWh stored) relates to a per-cycle cost $\Delta$ via
# $\kappa_{\text{deg}} = \Delta / C$.
#
# **Break-even spread** (approximate, ignoring asymmetric buy/sell price levels):
# a cycle is profitable when
# $$\eta_{rt} \cdot P_{\text{sell}} - P_{\text{buy}}
#   \;>\; \frac{\kappa_{\text{deg}} \cdot \eta_c \cdot 1000}{1} \text{ EUR/MWh}$$
# For $\eta_{rt} = 0.9$, $\Delta = 10$ EUR/cycle this hurdle is roughly
# **95 EUR/MWh** — present in the 2022 crisis but rare in normal years.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from vpp.paths import ProjPaths
from vpp.scenarios import SCENARIO_BY_ID

paths = ProjPaths()
paths.ensure_directories()

plt.rcParams["figure.dpi"] = 100
sns.set_theme(style="whitegrid")

CAPACITY_KWH = 100.0
POWER_KW = 50.0
ETA_RT = 0.90
ETA_C = ETA_D = float(np.sqrt(ETA_RT))
DEG_COST_EUR_PER_CYCLE = 10.0

CHARGE_COLOR = "#1f77b4"
DISCHARGE_COLOR = "#d62728"
SOC_COLOR = "#2ca02c"
SCENARIO_COLORS = {
    "ideal": "#2ca02c",
    "eta90": "#1f77b4",
    "eta90_deg": "#d62728",
    "eta90_free": "#aec7e8",
    "eta90_deg_free": "#ffbb78",
}
SCENARIO_LABELS = {
    "ideal": "ideal (eta=1) [DR]",
    "eta90": f"eta={ETA_RT}, no deg [DR]",
    "eta90_deg": f"eta={ETA_RT}, {DEG_COST_EUR_PER_CYCLE:.0f} EUR/cyc [DR]",
    "eta90_free": f"eta={ETA_RT}, no deg [FH]",
    "eta90_deg_free": f"eta={ETA_RT}, {DEG_COST_EUR_PER_CYCLE:.0f} EUR/cyc [FH]",
}

_SCENARIO_IDS_5 = [
    "actual__lp_dr__eta100__deg000",
    "actual__lp_dr__eta090__deg000",
    "actual__lp_dr__eta090__deg010",
    "actual__lp_fh__eta090__deg000",
    "actual__lp_fh__eta090__deg010",
]
_SCENARIO_NAME_5 = {
    "actual__lp_dr__eta100__deg000": "ideal",
    "actual__lp_dr__eta090__deg000": "eta90",
    "actual__lp_dr__eta090__deg010": "eta90_deg",
    "actual__lp_fh__eta090__deg000": "eta90_free",
    "actual__lp_fh__eta090__deg010": "eta90_deg_free",
}

# %% [markdown]
# ## 2. Load Data

# %%
prices_raw = pd.read_parquet(paths.smard_prices_file)
prices_eur_mwh = prices_raw["price_de_lu"].dropna().sort_index()
prices_berlin = prices_eur_mwh.copy()
prices_berlin.index = prices_berlin.index.tz_convert("Europe/Berlin")

price_array = prices_berlin.to_numpy(dtype=float)
n = len(price_array)

print(
    f"Loaded {len(price_array):,} hourly prices: "
    f"{prices_berlin.index[0].date()} to {prices_berlin.index[-1].date()}"
)

prices_df = (
    prices_berlin.rename("price_eur_mwh")
    .reset_index()
    .rename(columns={"index": "timestamp"})
)

# %% [markdown]
# ## 3. Load Pre-computed Dispatch


# %%
dispatch_raw = pd.read_parquet(
    paths.dispatch_schedules_file,
    filters=[("scenario_id", "in", _SCENARIO_IDS_5)],
)
dispatch_raw["scenario"] = dispatch_raw["scenario_id"].map(_SCENARIO_NAME_5)

merged = dispatch_raw.merge(prices_df, on="timestamp")
merged["revenue_eur"] = merged["price_eur_mwh"] * (merged["d"] - merged["c"]) / 1000
eta_c_map = {sid: SCENARIO_BY_ID[sid].battery.eta_c for sid in _SCENARIO_IDS_5}
merged["eta_c"] = merged["scenario_id"].map(eta_c_map)
merged["kwh_stored"] = merged["eta_c"] * merged["c"]

schedule = merged.rename(
    columns={"c": "charge_kw", "d": "discharge_kw", "soc": "soc_kwh"}
).assign(
    datetime=merged["timestamp"],
    date=pd.to_datetime(merged["timestamp"].dt.date),
    hour=merged["timestamp"].dt.hour,
)
schedule["scenario"] = schedule["scenario"].astype("category")

daily = (
    schedule.groupby(["scenario", "date"], observed=True)
    .agg(
        revenue_eur=("revenue_eur", "sum"),
        charge_kwh=("charge_kw", "sum"),
        discharge_kwh=("discharge_kw", "sum"),
        kwh_stored=("kwh_stored", "sum"),
    )
    .reset_index()
)
daily["cycles"] = daily["kwh_stored"] / CAPACITY_KWH
daily["year"] = daily["date"].dt.year

# %% [markdown]
# ## 6. Scenario Comparison

# %%
ideal_total = float(daily.loc[daily["scenario"] == "ideal", "revenue_eur"].sum())

summary = (
    daily.groupby("scenario", observed=True)
    .agg(
        n_days=("revenue_eur", "count"),
        total_revenue_eur=("revenue_eur", "sum"),
        avg_daily_revenue_eur=("revenue_eur", "mean"),
        avg_daily_cycles=("cycles", "mean"),
    )
    .reset_index()
)
summary["annual_revenue_eur"] = summary["avg_daily_revenue_eur"] * 365.25
summary["delta_pct"] = (summary["total_revenue_eur"] / ideal_total - 1) * 100
summary["avg_annual_cycles"] = summary["avg_daily_cycles"] * 365.25

_order = {"ideal": 0, "eta90": 1, "eta90_deg": 2, "eta90_free": 3, "eta90_deg_free": 4}
summary = (
    summary.assign(_o=summary["scenario"].map(_order))
    .sort_values("_o")
    .drop(columns="_o")
    .reset_index(drop=True)
)

print(
    "Scenario comparison (100 kWh / 50 kW, daily reset, "
    f"η_rt={ETA_RT}, deg={DEG_COST_EUR_PER_CYCLE:.0f} EUR/cycle):\n"
)
print(
    summary[
        [
            "scenario",
            "annual_revenue_eur",
            "delta_pct",
            "avg_daily_cycles",
            "avg_annual_cycles",
        ]
    ]
    .rename(
        columns={
            "scenario": "Scenario",
            "annual_revenue_eur": "Annual rev (EUR)",
            "delta_pct": "Δ% vs ideal",
            "avg_daily_cycles": "Avg cycles/day",
            "avg_annual_cycles": "Est cycles/year",
        }
    )
    .to_string(index=False, float_format="{:.2f}".format)
)

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.2, 1]})

# Left: revenue bar chart with Δ% annotations
colors = [SCENARIO_COLORS[s] for s in summary["scenario"]]
bars = axes[0].bar(
    [SCENARIO_LABELS[s] for s in summary["scenario"]],
    summary["annual_revenue_eur"],
    color=colors,
    edgecolor="white",
    width=0.55,
)
ideal_rev = float(
    summary.loc[summary["scenario"] == "ideal", "annual_revenue_eur"].values[0]
)
axes[0].set_ylim(0, ideal_rev * 1.28)
axes[0].set_ylabel("Annualized revenue (EUR/year)")
axes[0].set_title("Annual revenue by scenario")
axes[0].tick_params(axis="x", labelsize=8.5)
for bar, row in zip(bars, summary.itertuples()):
    label = f"{row.annual_revenue_eur:,.0f}\n({row.delta_pct:+.1f}%)"
    axes[0].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + ideal_rev * 0.01,
        label,
        ha="center",
        va="bottom",
        fontsize=9,
    )

# Right: summary table
table_rows = [
    [
        SCENARIO_LABELS[row.scenario],
        f"{row.annual_revenue_eur:,.0f}",
        f"{row.delta_pct:+.1f}%",
        f"{row.avg_daily_cycles:.3f}",
        f"{row.avg_annual_cycles:.0f}",
    ]
    for row in summary.itertuples()
]
col_labels = [
    "Scenario",
    "Annual rev\n(EUR)",
    "Δ% vs\nideal",
    "Avg cyc\n/day",
    "Est cyc\n/year",
]
t = axes[1].table(
    cellText=table_rows,
    colLabels=col_labels,
    loc="center",
    cellLoc="center",
)
t.auto_set_font_size(False)
t.set_fontsize(9)
t.scale(1, 1.7)
axes[1].axis("off")
axes[1].set_title("Comparison table (DR=daily reset, FH=free horizon)", pad=14)

fig.suptitle(
    "Stage 1b — scenario comparison (100 kWh / 50 kW)",
    fontsize=11,
)
fig.tight_layout()
fig.savefig(
    paths.images_path / "03_scenario_comparison.png", dpi=150, bbox_inches="tight"
)
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/03_scenario_comparison.png
# :name: fig-03-scenario-comparison
# Left: annualized revenue for each dispatch scenario. Right: summary table.
# Adding round-trip efficiency (η_rt = 0.9) cuts revenue by the percentage shown;
# adding a 10 EUR/cycle degradation cost further reduces it by suppressing
# low-spread cycling.
# ```

# %% [markdown]
# ## 7. Diurnal Dispatch Comparison

# %%
scenarios_ordered = ["ideal", "eta90", "eta90_deg"]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

for ax, scenario in zip(axes, scenarios_ordered):
    hourly = schedule[schedule["scenario"] == scenario].sort_values("datetime")
    diurnal = hourly.groupby("hour")[["charge_kw", "discharge_kw", "soc_kwh"]].mean()
    midnight = pd.DataFrame(
        {"charge_kw": [np.nan], "discharge_kw": [np.nan], "soc_kwh": [0.0]},
        index=pd.Index([24], name="hour"),
    )
    diurnal = pd.concat([diurnal, midnight])
    bars = diurnal.iloc[:24]

    ax.bar(
        bars.index,
        bars["charge_kw"],
        width=1.0,
        align="edge",
        label="Avg charge",
        color=CHARGE_COLOR,
        edgecolor="white",
        linewidth=0.4,
    )
    ax.bar(
        bars.index,
        -bars["discharge_kw"],
        width=1.0,
        align="edge",
        label="Avg discharge",
        color=DISCHARGE_COLOR,
        edgecolor="white",
        linewidth=0.4,
    )
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(range(0, 25, 4))
    ax.set_xlabel("Hour (Berlin)")
    ax.set_title(SCENARIO_LABELS[scenario], fontsize=9.5)

    ax2 = ax.twinx()
    ax2.plot(
        diurnal.index,
        diurnal["soc_kwh"],
        color=SOC_COLOR,
        marker="o",
        markersize=2.5,
        linewidth=1.2,
        label="Avg SoC",
    )
    ax2.set_ylim(0, CAPACITY_KWH * 1.15)
    ax2.grid(False)
    if scenario == "eta90_deg":
        ax2.set_ylabel("Avg SoC (kWh)")
    else:
        ax2.set_yticklabels([])

axes[0].set_ylabel("Avg power (kW)\ncharge (+) / discharge (−)")

h1, l1 = axes[0].get_legend_handles_labels()
axes[1].legend(h1, l1, loc="upper left", fontsize=8)

fig.suptitle(
    "Average diurnal dispatch profiles — Stage 1a vs 1b (100 kWh / 50 kW)",
    fontsize=11,
)
fig.tight_layout()
fig.savefig(
    paths.images_path / "03_diurnal_comparison.png", dpi=150, bbox_inches="tight"
)
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/03_diurnal_comparison.png
# :name: fig-03-diurnal-comparison
# Average diurnal dispatch profile (charge bars up, discharge bars down) and SoC
# trajectory for each scenario. Round-trip efficiency (centre panel) shifts optimal
# dispatch slightly but preserves the overnight-charge/peak-discharge pattern.
# Degradation cost (right panel) visibly suppresses low-spread hours where the
# margin no longer covers wear.
# ```

# %% [markdown]
# ## 8. Degradation Cost Sensitivity
#
# How much revenue survives as the degradation cost rises? And at what cost does
# cycling itself become unprofitable? We sweep `DEG_COST` from 0 to 50 EUR/cycle,
# solving the η=0.9 LP at each value.

# %%
_SWEEP_COSTS = [0, 5, 10, 15, 20, 30, 40, 50]
_SWEEP_IDS = [
    f"actual__lp_{h}__eta090__deg{cost:03d}"
    for h in ["dr", "fh"]
    for cost in _SWEEP_COSTS
]

sweep_raw = pd.read_parquet(
    paths.dispatch_schedules_file,
    filters=[("scenario_id", "in", _SWEEP_IDS)],
)
sweep_raw = sweep_raw.merge(prices_df, on="timestamp")
sweep_raw["revenue_eur"] = (
    sweep_raw["price_eur_mwh"] * (sweep_raw["d"] - sweep_raw["c"]) / 1000
)
sweep_raw["kwh_stored"] = ETA_C * sweep_raw["c"]
sweep_raw["horizon"] = sweep_raw["scenario_id"].str.extract(r"lp_(dr|fh)__")[0]
sweep_raw["deg_cost"] = (
    sweep_raw["scenario_id"].str.extract(r"deg(\d+)$")[0].astype(int)
)
sweep_raw["date"] = sweep_raw["timestamp"].dt.date

sweep_stats = (
    sweep_raw.groupby(["horizon", "deg_cost"])
    .agg(
        total_revenue_eur=("revenue_eur", "sum"),
        n_days=("date", "nunique"),
        total_kwh_stored=("kwh_stored", "sum"),
    )
    .reset_index()
)
sweep_stats["annual_revenue_eur"] = (
    sweep_stats["total_revenue_eur"] / sweep_stats["n_days"] * 365.25
)
sweep_stats["avg_annual_cycles"] = (
    sweep_stats["total_kwh_stored"] / CAPACITY_KWH / sweep_stats["n_days"] * 365.25
)

sweep_dr = (
    sweep_stats[sweep_stats["horizon"] == "dr"]
    .drop(columns="horizon")
    .reset_index(drop=True)
)
sweep_fh = (
    sweep_stats[sweep_stats["horizon"] == "fh"]
    .drop(columns="horizon")
    .reset_index(drop=True)
)
sweep = sweep_dr  # backward-compat alias for existing chart code below

# %%
# Break-even spread annotation: approximate minimum price spread for a profitable cycle
# Condition: eta_rt * p_sell - p_buy > kappa_deg * eta_c * 1000 (EUR/MWh)
# ≈ eta_c / eta_rt * deg_cost / CAPACITY_KWH * 1000 EUR/MWh (ignoring buy price level)
sweep["breakeven_spread_eur_mwh"] = (
    ETA_C / ETA_RT * sweep["deg_cost"] / CAPACITY_KWH * 1000
)

print("\nDegradation sensitivity (η_rt=0.9):")
print(
    sweep[
        [
            "deg_cost",
            "annual_revenue_eur",
            "avg_annual_cycles",
            "breakeven_spread_eur_mwh",
        ]
    ]
    .rename(
        columns={
            "deg_cost": "Deg cost (EUR/cyc)",
            "annual_revenue_eur": "Annual rev (EUR)",
            "avg_annual_cycles": "Cycles/year",
            "breakeven_spread_eur_mwh": "Break-even spread (EUR/MWh)",
        }
    )
    .to_string(index=False, float_format="{:.1f}".format)
)

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

# Left: annual revenue vs degradation cost — DR and FH
for df, lbl, col, marker in [
    (sweep_dr, "Daily reset", "#1f77b4", "o"),
    (sweep_fh, "Free horizon", "#ff7f0e", "s"),
]:
    axes[0].plot(
        df["deg_cost"],
        df["annual_revenue_eur"],
        color=col,
        marker=marker,
        markersize=5,
        linewidth=1.8,
        label=lbl,
    )
axes[0].axhline(
    ideal_rev,
    color=SCENARIO_COLORS["ideal"],
    linestyle="--",
    linewidth=1.0,
    label=f"Ideal (η=1): {ideal_rev:,.0f} EUR",
)
axes[0].set_xlabel("Degradation cost (EUR / equivalent full cycle)")
axes[0].set_ylabel("Annualized revenue (EUR/year)")
axes[0].set_title(f"Revenue vs degradation cost (η_rt={ETA_RT})")
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.4)

# Right: annual cycles for DR and FH + break-even spread secondary axis
ax2r = axes[1].twinx()
color_cycles_dr = "#1f77b4"
color_cycles_fh = "#ff7f0e"
color_spread = "#d62728"

axes[1].plot(
    sweep_dr["deg_cost"],
    sweep_dr["avg_annual_cycles"],
    color=color_cycles_dr,
    marker="o",
    markersize=5,
    linewidth=1.8,
    label="Cycles/yr — DR (left)",
)
axes[1].plot(
    sweep_fh["deg_cost"],
    sweep_fh["avg_annual_cycles"],
    color=color_cycles_fh,
    marker="s",
    markersize=5,
    linewidth=1.8,
    linestyle="-.",
    label="Cycles/yr — FH (left)",
)
ax2r.plot(
    sweep_dr["deg_cost"],
    sweep_dr["breakeven_spread_eur_mwh"],
    color=color_spread,
    marker="^",
    markersize=4,
    linewidth=1.5,
    linestyle="--",
    label="Break-even spread (right)",
)

axes[1].set_xlabel("Degradation cost (EUR / equivalent full cycle)")
axes[1].set_ylabel("Estimated full-cycle equivalents per year")
ax2r.set_ylabel("Min profitable price spread (EUR/MWh)", color=color_spread)
ax2r.tick_params(axis="y", labelcolor=color_spread)
ax2r.grid(False)
axes[1].set_title("Cycling activity vs degradation cost (DR vs FH)")

h1, l1 = axes[1].get_legend_handles_labels()
h2, l2 = ax2r.get_legend_handles_labels()
axes[1].legend(h1 + h2, l1 + l2, fontsize=8)
axes[1].grid(True, alpha=0.3)

fig.suptitle(
    f"Stage 1b — degradation cost sensitivity (100 kWh / 50 kW, η_rt={ETA_RT})",
    fontsize=11,
)
fig.tight_layout()
fig.savefig(
    paths.images_path / "03_degradation_sensitivity.png", dpi=150, bbox_inches="tight"
)
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/03_degradation_sensitivity.png
# :name: fig-03-degradation-sensitivity
# Left: annualized revenue as a function of degradation cost for daily-reset (DR)
# and free-horizon (FH) constraint variants. The FH premium grows with degradation
# cost — at high per-cycle costs, multi-day carry-over lets the LP cherry-pick only
# the best spread opportunities. Right: annual cycles for both constraint variants
# plus the break-even price spread (right axis).
# ```

# %% [markdown]
# ## 9. Free Horizon vs Daily Reset
#
# The daily-reset constraint forces SoC = 0 at midnight every day. Idle days have
# SoC = 0 throughout because there is no carry-over from the previous day. With high
# degradation costs — where only exceptional price spreads justify a cycle — the
# free-horizon formulation can accumulate charge over multiple days and wait for the
# best opportunity, while daily-reset must start fresh each morning with an empty
# battery. We expect the FH premium to grow with degradation cost.

# %%
fh_premium = {}
for dr_key, fh_key in [("eta90", "eta90_free"), ("eta90_deg", "eta90_deg_free")]:
    dr_rev = float(
        summary.loc[summary["scenario"] == dr_key, "annual_revenue_eur"].values[0]
    )
    fh_rev = float(
        summary.loc[summary["scenario"] == fh_key, "annual_revenue_eur"].values[0]
    )
    fh_premium[dr_key] = {
        "dr": dr_rev,
        "fh": fh_rev,
        "premium_pct": (fh_rev / dr_rev - 1) * 100,
    }

print("Free horizon premium over daily reset:")
for key, v in fh_premium.items():
    print(
        f"  {SCENARIO_LABELS[key]:35s}  DR={v['dr']:,.0f}  FH={v['fh']:,.0f}"
        f"  premium={v['premium_pct']:+.2f}%"
    )

sweep_fh["fh_premium_pct"] = (
    sweep_fh["annual_revenue_eur"] / sweep_dr["annual_revenue_eur"].values - 1
) * 100
print("\nFH premium vs degradation cost:")
print(
    sweep_fh[["deg_cost", "fh_premium_pct"]]
    .rename(
        columns={"deg_cost": "Deg cost (EUR/cyc)", "fh_premium_pct": "FH premium (%)"}
    )
    .to_string(index=False, float_format="{:.2f}".format)
)

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

# Left: grouped bars — DR vs FH for eta90 and eta90_deg
group_keys = [("eta90", "eta90_free"), ("eta90_deg", "eta90_deg_free")]
group_labels = [
    f"η={ETA_RT}, no deg",
    f"η={ETA_RT}, {DEG_COST_EUR_PER_CYCLE:.0f} EUR/cyc",
]
x = np.array([0.0, 2.0])
w = 0.7

for i, (dr_key, fh_key) in enumerate(group_keys):
    dr_rev = fh_premium[dr_key]["dr"]
    fh_rev = fh_premium[dr_key]["fh"]
    premium = fh_premium[dr_key]["premium_pct"]
    axes[0].bar(
        x[i] - w / 2,
        dr_rev,
        width=w,
        color=SCENARIO_COLORS[dr_key],
        label="Daily reset" if i == 0 else None,
    )
    axes[0].bar(
        x[i] + w / 2,
        fh_rev,
        width=w,
        color=SCENARIO_COLORS[fh_key],
        label="Free horizon" if i == 0 else None,
    )
    top = max(dr_rev, fh_rev)
    axes[0].annotate(
        f"FH +{premium:.1f}%",
        xy=(x[i] + w / 2, fh_rev),
        xytext=(x[i] + w / 2, top * 1.06),
        ha="center",
        fontsize=8.5,
        arrowprops={"arrowstyle": "->", "lw": 0.7},
    )

axes[0].axhline(
    ideal_rev,
    color=SCENARIO_COLORS["ideal"],
    linestyle="--",
    linewidth=0.9,
    label=f"Ideal η=1: {ideal_rev:,.0f} EUR",
)
axes[0].set_xticks(x)
axes[0].set_xticklabels(group_labels, fontsize=9)
axes[0].set_ylabel("Annualized revenue (EUR/year)")
axes[0].set_title("Daily reset vs free horizon — baseline scenarios")
axes[0].legend(fontsize=8.5)
axes[0].set_ylim(0, ideal_rev * 1.25)

# Right: FH premium (%) vs degradation cost
axes[1].plot(
    sweep_fh["deg_cost"],
    sweep_fh["fh_premium_pct"],
    color="#ff7f0e",
    marker="o",
    markersize=5,
    linewidth=1.8,
)
axes[1].axhline(0, color="gray", linewidth=0.7)
axes[1].set_xlabel("Degradation cost (EUR / equivalent full cycle)")
axes[1].set_ylabel("Free-horizon premium over daily reset (%)")
axes[1].set_title("FH premium grows with degradation cost")
axes[1].grid(True, alpha=0.4)

# Annotate baseline case
base_fh = sweep_fh[sweep_fh["deg_cost"] == int(DEG_COST_EUR_PER_CYCLE)].iloc[0]
axes[1].annotate(
    f"{int(DEG_COST_EUR_PER_CYCLE)} EUR/cyc\n+{base_fh['fh_premium_pct']:.1f}%",
    xy=(base_fh["deg_cost"], base_fh["fh_premium_pct"]),
    xytext=(base_fh["deg_cost"] + 5, base_fh["fh_premium_pct"] - 3),
    fontsize=8,
    arrowprops={"arrowstyle": "->", "lw": 0.7},
)

fig.suptitle(
    f"Stage 1b — free horizon vs daily reset (100 kWh / 50 kW, η_rt={ETA_RT})",
    fontsize=11,
)
fig.tight_layout()
fig.savefig(
    paths.images_path / "03_free_horizon_comparison.png", dpi=150, bbox_inches="tight"
)
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/03_free_horizon_comparison.png
# :name: fig-03-free-horizon-comparison
# Left: annual revenue for daily-reset (DR) and free-horizon (FH) constraint
# variants at two efficiency/degradation scenarios. The free-horizon premium is
# small when degradation cost is zero (battery cycles nearly every day anyway),
# but grows when degradation is high (FH can selectively wait for the best spreads
# across days, while DR must reset to zero each midnight). Right: the FH premium
# as a percentage of DR revenue across the full degradation cost sweep — confirming
# the intuition that multi-day carry-over becomes increasingly valuable when only
# occasional large spreads justify a cycle.
# ```
