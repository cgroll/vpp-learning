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
# # Daily Reset vs Free Horizon — Constraint Comparison
#
# This notebook compares the two horizon constraint variants across the full
# degradation cost sweep (0–50 EUR/cycle at η_rt = 0.9):
#
# - **Daily reset (DR)**: SoC forced to zero at every calendar-day midnight.
#   The battery starts each day empty and must end empty.
# - **Free horizon (FH)**: SoC=0 only at the very start and end of the full
#   dataset. The battery can carry charge across day boundaries.
#
# The key insight: when degradation cost is high, the daily-reset constraint
# forces the battery to cycle on marginal days where the spread barely covers
# wear. The free-horizon formulation can skip those days and wait for the rare
# high-spread opportunities, growing the FH premium with degradation cost.

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
ETA_RT = 0.90
_SWEEP_COSTS = [0, 5, 10, 15, 20, 30, 40, 50]

# %% [markdown]
# ## 2. Load Data

# %%
prices_raw = pd.read_parquet(paths.smard_prices_file)
prices_eur_mwh = prices_raw["price_de_lu"].dropna().sort_index()
prices_berlin = prices_eur_mwh.copy()
prices_berlin.index = prices_berlin.index.tz_convert("Europe/Berlin")

_SWEEP_IDS = [
    f"actual__lp_{h}__eta090__deg{cost:03d}"
    for h in ["dr", "fh"]
    for cost in _SWEEP_COSTS
]

dispatch_raw = pd.read_parquet(
    paths.dispatch_schedules_file,
    filters=[("scenario_id", "in", _SWEEP_IDS)],
)

prices_df = (
    prices_berlin.rename("price_eur_mwh")
    .reset_index()
    .rename(columns={"index": "timestamp"})
)

dispatch_raw = dispatch_raw.merge(prices_df, on="timestamp")
dispatch_raw["revenue_eur"] = (
    dispatch_raw["price_eur_mwh"] * (dispatch_raw["d"] - dispatch_raw["c"]) / 1000
)
dispatch_raw["kwh_stored"] = np.sqrt(ETA_RT) * dispatch_raw["c"]
dispatch_raw["horizon"] = dispatch_raw["scenario_id"].str.extract(r"lp_(dr|fh)__")[0]
dispatch_raw["deg_cost"] = (
    dispatch_raw["scenario_id"].str.extract(r"deg(\d+)$")[0].astype(int)
)
dispatch_raw["date"] = dispatch_raw["timestamp"].dt.date

sweep_stats = (
    dispatch_raw.groupby(["horizon", "deg_cost"])
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
    .sort_values("deg_cost")
)
sweep_fh = (
    sweep_stats[sweep_stats["horizon"] == "fh"]
    .drop(columns="horizon")
    .reset_index(drop=True)
    .sort_values("deg_cost")
)

sweep_fh = sweep_fh.assign(
    fh_premium_pct=(
        sweep_fh["annual_revenue_eur"].values / sweep_dr["annual_revenue_eur"].values
        - 1
    )
    * 100
)

print("Degradation sweep — DR vs FH annual revenue (EUR/yr):")
comp = sweep_dr[["deg_cost", "annual_revenue_eur"]].merge(
    sweep_fh[["deg_cost", "annual_revenue_eur", "fh_premium_pct"]],
    on="deg_cost",
    suffixes=("_dr", "_fh"),
)
print(comp.to_string(index=False, float_format="{:.1f}".format))

# %% [markdown]
# ## 3. Revenue Comparison Charts

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

# Left: revenue lines — DR and FH across sweep
axes[0].plot(
    sweep_dr["deg_cost"],
    sweep_dr["annual_revenue_eur"],
    color="#1f77b4",
    marker="o",
    markersize=5,
    linewidth=1.8,
    label="Daily reset",
)
axes[0].plot(
    sweep_fh["deg_cost"],
    sweep_fh["annual_revenue_eur"],
    color="#ff7f0e",
    marker="s",
    markersize=5,
    linewidth=1.8,
    label="Free horizon",
)
axes[0].set_xlabel("Degradation cost (EUR / equivalent full cycle)")
axes[0].set_ylabel("Annualised revenue (EUR/year)")
axes[0].set_title(f"Revenue vs degradation cost (eta_rt={ETA_RT})")
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.4)

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

fig.suptitle(
    f"Daily reset vs free horizon — deg. sweep (100 kWh / 50 kW, eta_rt={ETA_RT})",
    fontsize=11,
)
fig.tight_layout()
fig.savefig(
    paths.images_path / "10_constraint_comparison.png", dpi=150, bbox_inches="tight"
)
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/10_constraint_comparison.png
# :name: fig-10-constraint-comparison
# Left: annualised revenue vs degradation cost for daily-reset (DR) and free-horizon
# (FH) constraint variants. Right: FH premium (%) grows with degradation cost —
# at high per-cycle costs, multi-day carry-over lets the optimiser cherry-pick only
# the best spread opportunities.
# ```

# %% [markdown]
# ## 4. Cycling Activity Comparison

# %%
fig, ax = plt.subplots(figsize=(9, 4.5))

ax.plot(
    sweep_dr["deg_cost"],
    sweep_dr["avg_annual_cycles"],
    color="#1f77b4",
    marker="o",
    markersize=5,
    linewidth=1.8,
    label="Daily reset",
)
ax.plot(
    sweep_fh["deg_cost"],
    sweep_fh["avg_annual_cycles"],
    color="#ff7f0e",
    marker="s",
    markersize=5,
    linewidth=1.8,
    linestyle="-.",
    label="Free horizon",
)
ax.set_xlabel("Degradation cost (EUR / equivalent full cycle)")
ax.set_ylabel("Estimated full-cycle equivalents per year")
ax.set_title(f"Annual cycles vs degradation cost (eta_rt={ETA_RT})")
ax.legend()
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(
    paths.images_path / "10_cycling_comparison.png", dpi=150, bbox_inches="tight"
)
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/10_cycling_comparison.png
# :name: fig-10-cycling-comparison
# Annual full-cycle equivalents for DR and FH across the degradation cost sweep.
# Both curves decline sharply as higher degradation costs price out marginal cycling
# opportunities. The FH formulation preserves slightly more cycles at low cost
# (carry-over days) but drops faster at high cost (waits for fewer, larger spreads).
# ```

# %% [markdown]
# ## 5. Summary
#
# - **Low degradation cost** (0–5 EUR/cycle): DR and FH deliver nearly identical
#   revenue. The battery cycles almost every day regardless, so carry-over adds
#   little value.
# - **Medium degradation cost** (5–20 EUR/cycle): FH premium grows from ~0.5% to
#   several percent. Fewer opportunities clear the cost hurdle, and FH can
#   concentrate them across day boundaries.
# - **High degradation cost** (20–50 EUR/cycle): FH premium can reach 20–100%+
#   of DR revenue. DR forces the battery to start empty every morning even when
#   no profitable opportunity exists that day; FH accumulates charge over several
#   days waiting for the right spread.
#
# For practical dispatch (daily gate closure), the daily-reset constraint is
# natural and unavoidable. But the FH premium quantifies the value of allowing
# multi-day strategies — relevant for longer-horizon contracts or aggregated fleets.
