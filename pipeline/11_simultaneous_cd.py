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
# # LP vs MILP vs LP+Price Floor
#
# This notebook compares three dispatch formulations for handling simultaneous
# charge/discharge in the η < 1 case:
#
# | Approach | scenario_id | Mechanism |
# |---|---|---|
# | LP (baseline) | `actual__lp_dr__eta090__deg000` | No constraint; c and d may both be > 0 |  # noqa: E501
# | MILP (correct) | `actual__milp_dr__eta090__deg000` | Binary z[t] enforces mutual exclusivity |  # noqa: E501
# | LP + price floor | `actual__lp_floor_dr__eta090__deg000` | Clip prices to 0 before LP |  # noqa: E501
#
# Revenue is always settled at actual prices. The LP "cheat" — simultaneously
# charging and discharging at negative-price hours — is only worth a fraction of a
# percent, so the LP is a safe approximation for revenue forecasting.

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
_MIN_ACTIVE = 0.01  # kW — below this is numerical noise

# %% [markdown]
# ## 2. Load Data

# %%
prices_raw = pd.read_parquet(paths.smard_prices_file)
prices_eur_mwh = prices_raw["price_de_lu"].dropna().sort_index()
prices_berlin = prices_eur_mwh.copy()
prices_berlin.index = prices_berlin.index.tz_convert("Europe/Berlin")

_SCENARIO_IDS_3 = [
    "actual__lp_dr__eta090__deg000",
    "actual__milp_dr__eta090__deg000",
    "actual__lp_floor_dr__eta090__deg000",
]
_SCENARIO_LABEL = {
    "actual__lp_dr__eta090__deg000": "LP (baseline)",
    "actual__milp_dr__eta090__deg000": "MILP (correct)",
    "actual__lp_floor_dr__eta090__deg000": "LP + price floor",
}

dispatch_raw = pd.read_parquet(
    paths.dispatch_schedules_file,
    filters=[("scenario_id", "in", _SCENARIO_IDS_3)],
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
dispatch_merged["simul"] = (dispatch_merged["c"] > _MIN_ACTIVE) & (
    dispatch_merged["d"] > _MIN_ACTIVE
)
dispatch_merged["neg_price"] = dispatch_merged["price_eur_mwh"] < 0
dispatch_merged["date"] = dispatch_merged["timestamp"].dt.date
dispatch_merged["year"] = dispatch_merged["timestamp"].dt.year

n_days = dispatch_merged[
    dispatch_merged["scenario_id"] == "actual__lp_dr__eta090__deg000"
]["date"].nunique()
ann = 365.25 / n_days

# Annual revenue per scenario
rev_by_scenario = (
    dispatch_merged.groupby("scenario_id")["revenue_eur"].sum() * ann
).to_dict()

lp_rev = rev_by_scenario["actual__lp_dr__eta090__deg000"]
milp_rev = rev_by_scenario["actual__milp_dr__eta090__deg000"]
floor_rev = rev_by_scenario["actual__lp_floor_dr__eta090__deg000"]

milp_gap = (milp_rev / lp_rev - 1) * 100
floor_gap = (floor_rev / lp_rev - 1) * 100

print(f"Annual revenue (eta_rt={ETA_RT}, daily reset, 100 kWh / 50 kW):")
print(f"  LP (baseline):       {lp_rev:,.0f} EUR/yr")
print(f"  MILP (correct):      {milp_rev:,.0f} EUR/yr")
print(f"  LP + price floor:    {floor_rev:,.0f} EUR/yr")
print(f"  MILP vs LP gap:      {milp_gap:+.3f}%")
print(f"  Price floor vs LP:   {floor_gap:+.2f}%")

# Simultaneous C+D stats (LP only)
lp_dispatch = dispatch_merged[
    dispatch_merged["scenario_id"] == "actual__lp_dr__eta090__deg000"
]
simul_mask = lp_dispatch["simul"]
neg_and_simul = lp_dispatch["neg_price"] & simul_mask
n_total = len(lp_dispatch)
n_simul = simul_mask.sum()
n_neg_simul = neg_and_simul.sum()

print("\nLP simultaneous C+D:")
print(f"  Total: {n_simul:,} / {n_total:,} hours ({n_simul / n_total * 100:.2f}%)")
print(
    f"  At negative prices: {n_neg_simul:,} "
    f"({n_neg_simul / max(n_simul, 1) * 100:.0f}% of simultaneous)"
)

# %% [markdown]
# ## 3. Revenue and Simultaneous C+D by Year

# %%
yearly_stats = (
    dispatch_merged.groupby(["scenario_id", "year"])
    .agg(
        annual_rev=(
            "revenue_eur",
            lambda x: (
                x.sum()
                * 365.25
                / dispatch_merged.loc[
                    dispatch_merged["scenario_id"] == x.name[0], "date"
                ].nunique()
            ),
        ),
        n_hours=("simul", "count"),
        simul_hours=("simul", "sum"),
    )
    .reset_index()
)
# Simpler: compute annual revenue from daily aggregation
daily_rev = (
    dispatch_merged.groupby(["scenario_id", "year", "date"])["revenue_eur"]
    .sum()
    .reset_index()
)
yearly_rev = (
    daily_rev.groupby(["scenario_id", "year"])["revenue_eur"]
    .agg(lambda x: x.sum() * 365.25 / len(x))
    .reset_index()
    .rename(columns={"revenue_eur": "annual_rev_eur"})
)

lp_yearly = dispatch_merged[
    dispatch_merged["scenario_id"] == "actual__lp_dr__eta090__deg000"
]
simul_by_year = (
    lp_yearly.groupby("year")
    .agg(
        simul_pct=("simul", lambda x: x.mean() * 100),
        neg_simul_pct=(
            "simul",
            lambda x: (x & lp_yearly.loc[x.index, "neg_price"]).mean() * 100,
        ),
    )
    .reset_index()
)

print("\nAnnual revenue by scenario and year:")
yearly_pivot = yearly_rev.pivot(
    index="year", columns="scenario_id", values="annual_rev_eur"
)
yearly_pivot.columns = [_SCENARIO_LABEL.get(c, c) for c in yearly_pivot.columns]
print(yearly_pivot.round(0).to_string())

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

# Left: annual revenue by scenario (bar chart)
approaches = ["LP\n(baseline)", "MILP\n(correct)", "LP +\nprice floor"]
revenues = [lp_rev, milp_rev, floor_rev]
colors = ["#1f77b4", "#2ca02c", "#ff7f0e"]

bars = axes[0].bar(approaches, revenues, color=colors, edgecolor="white", width=0.5)
axes[0].set_ylabel("Annualised revenue (EUR/year)")
axes[0].set_title("Revenue comparison by approach")
ymax = max(revenues) * 1.18
axes[0].set_ylim(0, ymax)
for bar, rev in zip(bars, revenues):
    axes[0].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + ymax * 0.01,
        f"{rev:,.0f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )
axes[0].text(
    0.5,
    0.92,
    f"MILP vs LP: {milp_gap:+.3f}%\nFloor vs LP: {floor_gap:+.2f}%",
    ha="center",
    transform=axes[0].transAxes,
    fontsize=8.5,
    bbox={"boxstyle": "round,pad=0.3", "facecolor": "lightyellow", "edgecolor": "gray"},
)

# Right: simultaneous C+D by year (LP only)
x = np.arange(len(simul_by_year))
axes[1].bar(
    x,
    simul_by_year["simul_pct"],
    color="#1f77b4",
    alpha=0.8,
    label="All simultaneous",
)
axes[1].bar(
    x,
    simul_by_year["neg_simul_pct"],
    color="#d62728",
    alpha=0.8,
    label="Simul. at neg. price",
)
axes[1].set_xticks(x)
axes[1].set_xticklabels(simul_by_year["year"])
axes[1].set_ylabel("% of all hours")
axes[1].set_title("Simultaneous C+D hours by year (LP)")
axes[1].legend(fontsize=8)

fig.suptitle(
    f"LP vs MILP vs LP+floor (eta_rt={ETA_RT}, daily reset, 100 kWh / 50 kW)",
    fontsize=11,
)
fig.tight_layout()
fig.savefig(paths.images_path / "11_simultaneous_cd.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/11_simultaneous_cd.png
# :name: fig-11-simultaneous-cd
# Left: annualised revenue for each approach; the LP–MILP gap (the value of the
# physically-infeasible simultaneous dispatch) is annotated as a percentage.
# Right: fraction of hours with simultaneous C+D in the LP by year, split by
# negative-price vs positive-price hours.
# ```

# %% [markdown]
# ## 4. Summary
#
# Key findings replicated from this cross-scenario view:
#
# - The LP simultaneous C+D "cheat" is worth only ~0.27% of annual revenue.
#   The LP is therefore a safe proxy for revenue forecasting, even though it
#   produces physically invalid dispatch schedules in ~2% of hours.
# - All simultaneous-dispatch hours in the LP occur at negative-price hours,
#   confirming the mechanism: the LP "burns" SoC through efficiency losses to
#   create headroom for additional gross charging when prices are negative.
# - The price-floor fix is counterproductive: it forfeits genuine negative-price
#   charging revenue (worth ~2.5% of total), far exceeding the LP cheat it removes.
# - MILP is the correct and practical solution. With daily-reset sub-problems of
#   24 binary variables, CBC solves it as fast as the LP relaxation.
