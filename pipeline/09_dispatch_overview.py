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
# # All-Scenarios Dispatch Overview
#
# This notebook provides a cross-scenario comparison of all dispatch strategies
# defined in the scenario registry. For each scenario it computes:
#
# - Annualised revenue (EUR/yr)
# - Average daily cycles (kWh stored / capacity)
# - Simultaneous charge/discharge rate (% of hours with c > 0 and d > 0)
# - Feasibility flag (MILP-based scenarios are physically valid; LP may have
#   simultaneous C+D at negative-price hours)
#
# Settlement is always at **actual prices** regardless of which signal drove
# the optimisation. This means the naive forecast scenario is settled against
# actual prices, making its revenue directly comparable to the hindsight scenarios.

# %% [markdown]
# ## 1. Setup

# %%
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

from vpp.paths import ProjPaths
from vpp.scenarios import SCENARIOS

paths = ProjPaths()
paths.ensure_directories()

plt.rcParams["figure.dpi"] = 100
sns.set_theme(style="whitegrid")

CAPACITY_KWH = 100.0
_MIN_ACTIVE = 0.01  # kW threshold for simultaneous C+D detection

# %% [markdown]
# ## 2. Load Data

# %%
prices_raw = pd.read_parquet(paths.smard_prices_file)
prices_eur_mwh = prices_raw["price_de_lu"].dropna().sort_index()
prices_berlin = prices_eur_mwh.copy()
prices_berlin.index = prices_berlin.index.tz_convert("Europe/Berlin")

dispatch_all = pd.read_parquet(paths.dispatch_schedules_file)

prices_df = (
    prices_berlin.rename("price_eur_mwh")
    .reset_index()
    .rename(columns={"index": "timestamp"})
)

print(
    f"Loaded {len(prices_berlin):,} hourly prices: "
    f"{prices_berlin.index[0].date()} to {prices_berlin.index[-1].date()}"
)
n_sc = dispatch_all["scenario_id"].nunique()
print(f"Dispatch: {len(dispatch_all):,} rows, {n_sc} scenarios")

# %% [markdown]
# ## 3. Compute Per-Scenario Metrics

# %%
dispatch_merged = dispatch_all.merge(prices_df, on="timestamp")
dispatch_merged["revenue_eur"] = (
    dispatch_merged["price_eur_mwh"]
    * (dispatch_merged["d"] - dispatch_merged["c"])
    / 1000
)
dispatch_merged["kwh_stored"] = dispatch_merged["c"]  # kWh entering storage per hour
dispatch_merged["simul"] = (dispatch_merged["c"] > _MIN_ACTIVE) & (
    dispatch_merged["d"] > _MIN_ACTIVE
)
dispatch_merged["date"] = dispatch_merged["timestamp"].dt.date

metrics = (
    dispatch_merged.groupby("scenario_id")
    .agg(
        total_revenue_eur=("revenue_eur", "sum"),
        n_days=("date", "nunique"),
        total_kwh_stored=("kwh_stored", "sum"),
        n_hours=("revenue_eur", "count"),
        simul_hours=("simul", "sum"),
    )
    .reset_index()
)

metrics["annual_revenue_eur"] = (
    metrics["total_revenue_eur"] / metrics["n_days"] * 365.25
)
metrics["avg_annual_cycles"] = (
    metrics["total_kwh_stored"] / CAPACITY_KWH / metrics["n_days"] * 365.25
)
metrics["simul_pct"] = metrics["simul_hours"] / metrics["n_hours"] * 100
metrics["feasible"] = metrics["simul_pct"] < 0.001

# Add human-readable name from registry (only for scenarios in SCENARIOS)
name_map = {sc.scenario_id: sc.name for sc in SCENARIOS}
metrics["name"] = metrics["scenario_id"].map(name_map).fillna(metrics["scenario_id"])

# Sort by annual revenue descending
metrics = metrics.sort_values("annual_revenue_eur", ascending=False).reset_index(
    drop=True
)

print("\nAll-scenario metrics:")
print(
    metrics[
        ["name", "annual_revenue_eur", "avg_annual_cycles", "simul_pct", "feasible"]
    ]
    .rename(
        columns={
            "name": "Scenario",
            "annual_revenue_eur": "Annual rev (EUR)",
            "avg_annual_cycles": "Cycles/yr",
            "simul_pct": "Simul C+D (%)",
            "feasible": "Feasible",
        }
    )
    .to_string(index=False, float_format="{:.1f}".format)
)

# %% [markdown]
# ## 4. Revenue Overview Chart

# %%
# Filter to the main "actual" price scenarios for plotting (exclude naive forecast)
actual_metrics = metrics[metrics["scenario_id"].str.startswith("actual__")].copy()

fig, ax = plt.subplots(figsize=(14, 5))

colors = ["#2ca02c" if f else "#d62728" for f in actual_metrics["feasible"]]
bars = ax.barh(
    range(len(actual_metrics)),
    actual_metrics["annual_revenue_eur"],
    color=colors,
    edgecolor="white",
    height=0.7,
)
ax.set_yticks(range(len(actual_metrics)))
ax.set_yticklabels(actual_metrics["name"], fontsize=8)
ax.set_xlabel("Annualised revenue (EUR/yr)")
ax.set_title("Cross-scenario annual revenue (settled at actual prices)")

# Legend
legend_elements = [
    Patch(facecolor="#2ca02c", label="Feasible (no simul. C+D)"),
    Patch(facecolor="#d62728", label="Infeasible (simul. C+D detected)"),
]
ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

fig.tight_layout()
fig.savefig(
    paths.images_path / "09_dispatch_overview.png", dpi=150, bbox_inches="tight"
)
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/09_dispatch_overview.png
# :name: fig-09-dispatch-overview
# Cross-scenario annual revenue for all "actual" price scenarios. Green bars are
# physically feasible (MILP or price-floor LP); red bars contain simultaneous
# charge/discharge hours (LP without complementarity constraint). The degradation
# cost sweep shows how revenue falls as per-cycle costs rise.
# ```

# %% [markdown]
# ## 5. Scenario Comparison Table

# %%
display_cols = [
    "name",
    "annual_revenue_eur",
    "avg_annual_cycles",
    "simul_pct",
    "feasible",
]

fig, ax = plt.subplots(figsize=(13, max(4, 0.3 * len(metrics) + 1.5)))
ax.axis("off")

table_data = [
    [
        row["name"],
        f"{row['annual_revenue_eur']:,.0f}",
        f"{row['avg_annual_cycles']:.0f}",
        f"{row['simul_pct']:.2f}%",
        "Yes" if row["feasible"] else "No",
    ]
    for _, row in metrics.iterrows()
]

col_labels = ["Scenario", "Annual rev (EUR)", "Cycles/yr", "Simul C+D", "Feasible"]
t = ax.table(
    cellText=table_data,
    colLabels=col_labels,
    loc="center",
    cellLoc="left",
)
t.auto_set_font_size(False)
t.set_fontsize(8)
t.scale(1.0, 1.4)

# Colour feasibility column
n_cols = len(col_labels)
for i, row in enumerate(metrics.itertuples()):
    cell = t[i + 1, n_cols - 1]
    cell.set_facecolor("#d4edda" if row.feasible else "#f8d7da")

ax.set_title("All-scenario comparison table", pad=12, fontsize=11)
fig.tight_layout()
fig.savefig(paths.images_path / "09_scenario_table.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/09_scenario_table.png
# :name: fig-09-scenario-table
# Full scenario comparison table. The feasibility column (green = valid) flags
# scenarios where the LP solution is physically realizable. Infeasible scenarios
# still provide useful revenue estimates but would need MILP to produce valid dispatch.
# ```
