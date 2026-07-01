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
# # Simultaneous Charge/Discharge in LP Battery Dispatch
#
# With η_rt < 1 and separate charge/discharge variables, the LP may set both
# c[t] > 0 and d[t] > 0 simultaneously at the same hour. This is physically
# impossible — a real battery cannot charge and discharge at the same time — but
# the LP exploits it to "burn" energy through losses, creating SoC headroom for
# additional charging when prices are negative.
#
# The pathological case: at a negative-price hour when SoC is near capacity,
# gross-charging more earns negative-price revenue. Running d[t] > 0 simultaneously
# loses energy through η losses, freeing capacity for additional gross charging c[t].
# Revenue = price × (d − c) / 1000; with price < 0 and c > d, this is positive.
#
# Three remedies are compared:
#
# | Approach | Mechanism | Trade-off |
# |---|---|---|
# | **MILP** | Binary z[t]: c[t] ≤ P·z[t], d[t] ≤ P·(1−z[t]) | Correct; slow |
# | **LP + price floor** | Clip prices at 0 before LP | Fast; forfeits neg-price rev |
# | **LP (baseline)** | No constraint | Fastest; physically infeasible dispatch |

# %% [markdown]
# ## 1. LP Formulation Recap
#
# The MILP adds one binary variable per hour:
#
# $$z_t \in \{0, 1\}, \quad c_t \le P_{\max} z_t, \quad d_t \le P_{\max}(1 - z_t)$$
#
# This enforces mutual exclusivity: if $z_t = 1$ charging is active and discharging
# is blocked; if $z_t = 0$ the reverse. With daily-reset sub-problems of only 24
# hours each, CBC solves the MILP nearly as fast as the LP relaxation.
#
# The **price-floor LP** simply clips prices to $\max(P_t, 0)$ in the objective.
# At zero price the LP has no incentive for any dispatch and any simultaneous C+D
# would yield zero revenue — so the LP degeneracy that produces simultaneous C+D
# (exploitation of negative prices) is removed. Revenue is then computed against the
# *original* prices.

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
ETA_RT = 0.90
ETA_C = ETA_D = float(np.sqrt(ETA_RT))

CHARGE_COLOR = "#1f77b4"
DISCHARGE_COLOR = "#d62728"

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
    f"Loaded {n:,} hourly prices: "
    f"{prices_berlin.index[0].date()} → {prices_berlin.index[-1].date()}"
)


def _day_boundary_positions(index: pd.DatetimeIndex) -> list[int]:
    dates = index.date
    lengths = [int((np.array(dates) == d).sum()) for d in sorted(set(dates))]
    return [int(b) for b in np.concatenate([[0], np.cumsum(lengths)]).tolist()]


reset_daily = _day_boundary_positions(prices_berlin.index)

# %% [markdown]
# ## 3. LP Solvers


# %%
def _solve_lp(
    price_array: np.ndarray,
    soc_zero_positions: list[int],
    capacity_kwh: float = CAPACITY_KWH,
    power_kw: float = POWER_KW,
    eta_c: float = ETA_C,
    eta_d: float = ETA_D,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standard LP — separate c/d, no complementarity constraint."""
    n = len(price_array)
    prob = pulp.LpProblem("battery_lp", pulp.LpMaximize)
    c = pulp.LpVariable.dicts("c", range(n), lowBound=0, upBound=power_kw)
    d = pulp.LpVariable.dicts("d", range(n), lowBound=0, upBound=power_kw)
    soc = pulp.LpVariable.dicts("soc", range(n + 1), lowBound=0, upBound=capacity_kwh)
    prob += pulp.lpSum(price_array[t] / 1000 * (d[t] - c[t]) for t in range(n))
    for t in range(n):
        prob += soc[t + 1] == soc[t] + eta_c * c[t] - (1.0 / eta_d) * d[t]
    for pos in soc_zero_positions:
        prob += soc[pos] == 0
    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"LP status: {pulp.LpStatus[status]!r}")
    c_val = np.array([c[t].value() for t in range(n)])
    d_val = np.array([d[t].value() for t in range(n)])
    soc_val = np.array([soc[t].value() for t in range(n)])
    return c_val, d_val, soc_val


def _solve_milp(
    price_array: np.ndarray,
    soc_zero_positions: list[int],
    capacity_kwh: float = CAPACITY_KWH,
    power_kw: float = POWER_KW,
    eta_c: float = ETA_C,
    eta_d: float = ETA_D,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MILP with binary z[t] enforcing c[t]*d[t] = 0 (mutual exclusivity)."""
    n = len(price_array)
    prob = pulp.LpProblem("battery_milp", pulp.LpMaximize)
    z = pulp.LpVariable.dicts("z", range(n), cat="Binary")
    c = pulp.LpVariable.dicts("c", range(n), lowBound=0, upBound=power_kw)
    d = pulp.LpVariable.dicts("d", range(n), lowBound=0, upBound=power_kw)
    soc = pulp.LpVariable.dicts("soc", range(n + 1), lowBound=0, upBound=capacity_kwh)
    prob += pulp.lpSum(price_array[t] / 1000 * (d[t] - c[t]) for t in range(n))
    for t in range(n):
        prob += soc[t + 1] == soc[t] + eta_c * c[t] - (1.0 / eta_d) * d[t]
        prob += c[t] <= power_kw * z[t]
        prob += d[t] <= power_kw * (1 - z[t])
    for pos in soc_zero_positions:
        prob += soc[pos] == 0
    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"MILP status: {pulp.LpStatus[status]!r}")
    c_val = np.array([c[t].value() for t in range(n)])
    d_val = np.array([d[t].value() for t in range(n)])
    soc_val = np.array([soc[t].value() for t in range(n)])
    return c_val, d_val, soc_val


# %% [markdown]
# ## 4. Solve All Three Approaches

# %%
print("Solving LP (no complementarity constraint) …")
c_lp, d_lp, soc_lp = _solve_lp(price_array, reset_daily)
print("Done.")

print("Solving MILP (binary mutual exclusivity) …")
c_milp, d_milp, soc_milp = _solve_milp(price_array, reset_daily)
print("Done.")

print("Solving LP with price floor at 0 …")
c_floor, d_floor, soc_floor = _solve_lp(np.maximum(price_array, 0.0), reset_daily)
print("Done.")

# Revenue computed against ORIGINAL prices for all approaches
eps = 1e-9
for arr in (c_lp, d_lp, c_milp, d_milp, c_floor, d_floor):
    arr[np.abs(arr) < eps] = 0.0

rev_lp = float(np.dot(price_array, d_lp - c_lp) / 1000)
rev_milp = float(np.dot(price_array, d_milp - c_milp) / 1000)
rev_floor = float(np.dot(price_array, d_floor - c_floor) / 1000)

n_days = len(reset_daily) - 1
ann = 365.25 / n_days

print(f"\nAnnual revenue (η_rt={ETA_RT}, daily reset, 100 kWh / 50 kW):")
print(f"  LP (baseline):       {rev_lp * ann:,.0f} EUR/yr")
print(f"  MILP (correct):      {rev_milp * ann:,.0f} EUR/yr")
print(f"  LP + price floor:    {rev_floor * ann:,.0f} EUR/yr")
print(f"  MILP vs LP gap:      {(rev_milp / rev_lp - 1) * 100:+.3f}%")
print(f"  Price floor vs LP:   {(rev_floor / rev_lp - 1) * 100:+.3f}%")
print(f"  Price floor vs MILP: {(rev_floor / rev_milp - 1) * 100:+.3f}%")

# %% [markdown]
# ## 5. Detect Simultaneous Charge/Discharge

# %%
_MIN_ACTIVE = 0.01  # kW threshold — numerical noise below this is ignored

simul_lp = (c_lp > _MIN_ACTIVE) & (d_lp > _MIN_ACTIVE)
simul_milp = (c_milp > _MIN_ACTIVE) & (d_milp > _MIN_ACTIVE)
simul_floor = (c_floor > _MIN_ACTIVE) & (d_floor > _MIN_ACTIVE)

print(f"Hours with simultaneous C and D (threshold {_MIN_ACTIVE} kW):")
print(f"  LP:          {simul_lp.sum():6,} / {n:,}  ({simul_lp.mean() * 100:.2f}%)")
print(f"  MILP:        {simul_milp.sum():6,} / {n:,}  ({simul_milp.mean() * 100:.2f}%)")
pct_floor = simul_floor.mean() * 100
print(f"  Price floor: {simul_floor.sum():6,} / {n:,}  ({pct_floor:.2f}%)")

# Breakdown by price regime
neg_mask = price_array < 0
print(f"\nOf LP simultaneous hours: {simul_lp.sum():,} total")
n_neg = (simul_lp & neg_mask).sum()
n_pos = (simul_lp & ~neg_mask).sum()
total_simul = max(simul_lp.sum(), 1)
print(f"  During negative-price hours: {n_neg:,} ({n_neg / total_simul * 100:.1f}%)")
print(f"  During positive-price hours: {n_pos:,} ({n_pos / total_simul * 100:.1f}%)")

# Revenue attributable to simultaneous dispatch hours
rev_from_simul = float(np.dot(price_array[simul_lp], (d_lp - c_lp)[simul_lp]) / 1000)
print(
    f"\nRevenue in simultaneous-dispatch hours (LP): "
    f"{rev_from_simul * ann:,.0f} EUR/yr annualized"
)
print("  (This is the revenue at those hours; actual C*D impact is the LP-MILP gap)")

# %% [markdown]
# ## 6. Comparison Charts

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# --- Panel 1: Annual revenue comparison bar chart ---
approaches = ["LP\n(baseline)", "MILP\n(correct)", "LP +\nprice floor"]
revenues = [rev_lp * ann, rev_milp * ann, rev_floor * ann]
colors = ["#1f77b4", "#2ca02c", "#ff7f0e"]

bars = axes[0].bar(approaches, revenues, color=colors, edgecolor="white", width=0.5)
axes[0].set_ylabel("Annualized revenue (EUR/year)")
axes[0].set_title("Revenue comparison by approach")
ymax = max(revenues) * 1.2
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
# Annotate gaps vs LP
milp_gap = (rev_milp / rev_lp - 1) * 100
floor_gap = (rev_floor / rev_lp - 1) * 100
axes[0].text(
    0.5,
    0.93,
    f"MILP vs LP: {milp_gap:+.3f}%\nFloor vs LP: {floor_gap:+.2f}%",
    ha="center",
    transform=axes[0].transAxes,
    fontsize=8.5,
    bbox={"boxstyle": "round,pad=0.3", "facecolor": "lightyellow", "edgecolor": "gray"},
)

# --- Panel 2: Simultaneous C+D by year ---
df_lp = pd.DataFrame(
    {
        "datetime": prices_berlin.index,
        "price": price_array,
        "c_lp": c_lp,
        "d_lp": d_lp,
        "simul": simul_lp,
        "neg_price": neg_mask,
    }
)
df_lp["year"] = df_lp["datetime"].dt.year
yearly_simul = (
    df_lp.groupby("year")
    .agg(
        simul_hours=("simul", "sum"),
        neg_and_simul=("simul", lambda x: (x & df_lp.loc[x.index, "neg_price"]).sum()),
        total_hours=("simul", "count"),
    )
    .reset_index()
)
yearly_simul["simul_pct"] = (
    yearly_simul["simul_hours"] / yearly_simul["total_hours"] * 100
)
yearly_simul["neg_pct"] = (
    yearly_simul["neg_and_simul"] / yearly_simul["total_hours"] * 100
)

x = np.arange(len(yearly_simul))
axes[1].bar(
    x,
    yearly_simul["simul_pct"],
    color="#1f77b4",
    alpha=0.8,
    label="All simultaneous",
)
axes[1].bar(
    x,
    yearly_simul["neg_pct"],
    color="#d62728",
    alpha=0.8,
    label="Simul. at neg. price",
)
axes[1].set_xticks(x)
axes[1].set_xticklabels(yearly_simul["year"])
axes[1].set_ylabel("% of all hours")
axes[1].set_title("Simultaneous C+D hours by year (LP)")
axes[1].legend(fontsize=8)

# --- Panel 3: Example day with simultaneous C+D ---
# Find a day with the most simultaneous dispatch in LP
simul_by_date = (
    df_lp[df_lp["simul"]]
    .groupby(df_lp["datetime"].dt.date)
    .size()
    .reset_index(name="simul_hours")
    .sort_values("simul_hours", ascending=False)
)

if len(simul_by_date) > 0:
    example_date = simul_by_date.iloc[0]["datetime"]
    day_lp = df_lp[df_lp["datetime"].dt.date == example_date].copy()

    # Get MILP dispatch for the same day
    day_mask = prices_berlin.index.date == example_date
    day_prices = price_array[day_mask]
    day_c_milp = c_milp[day_mask]
    day_d_milp = d_milp[day_mask]

    hours = np.arange(len(day_lp))
    ax3a = axes[2]
    ax3b = ax3a.twinx()

    # Price on secondary axis
    ax3b.plot(
        hours,
        day_lp["price"].values,
        color="gray",
        linewidth=1.0,
        linestyle="--",
        alpha=0.6,
        label="Price (right)",
    )
    ax3b.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    ax3b.set_ylabel("Price (EUR/MWh)", color="gray")
    ax3b.tick_params(axis="y", labelcolor="gray")
    ax3b.grid(False)

    # LP dispatch
    ax3a.bar(
        hours - 0.2,
        day_lp["c_lp"].values,
        width=0.35,
        color=CHARGE_COLOR,
        alpha=0.8,
        label="LP charge",
    )
    ax3a.bar(
        hours - 0.2,
        -day_lp["d_lp"].values,
        width=0.35,
        color=DISCHARGE_COLOR,
        alpha=0.8,
        label="LP discharge",
    )
    # MILP dispatch
    ax3a.bar(
        hours + 0.2,
        day_c_milp,
        width=0.35,
        color=CHARGE_COLOR,
        alpha=0.4,
        hatch="///",
        label="MILP charge",
    )
    ax3a.bar(
        hours + 0.2,
        -day_d_milp,
        width=0.35,
        color=DISCHARGE_COLOR,
        alpha=0.4,
        hatch="///",
        label="MILP discharge",
    )

    # Mark simultaneous hours
    for h, is_simul in enumerate(day_lp["simul"].values):
        if is_simul:
            ax3a.axvspan(h - 0.5, h + 0.5, alpha=0.12, color="purple", zorder=0)

    ax3a.axhline(0, color="black", linewidth=0.5)
    ax3a.set_xlabel("Hour of day")
    ax3a.set_ylabel("Power (kW)")
    ax3a.set_title(f"Example day: {example_date}\n(purple = simultaneous C+D in LP)")
    ax3a.legend(fontsize=7, loc="upper left")
    ax3a.set_xticks(range(0, 24, 4))
else:
    axes[2].text(
        0.5,
        0.5,
        "No simultaneous\nC+D found",
        ha="center",
        va="center",
        transform=axes[2].transAxes,
        fontsize=12,
    )
    axes[2].axis("off")

fig.suptitle(
    f"Simultaneous charge/discharge in LP dispatch (η_rt={ETA_RT}, daily reset)",
    fontsize=11,
)
fig.tight_layout()
fig.savefig(
    paths.images_path / "04_simultaneous_cd_analysis.png", dpi=150, bbox_inches="tight"
)
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/04_simultaneous_cd_analysis.png
# :name: fig-04-simultaneous-cd-analysis
# Left: annualized revenue for each approach, computed against original prices.
# The LP-MILP gap measures the true benefit of the physically-infeasible
# simultaneous dispatch in the LP; if it is small, the LP approximation is
# acceptable. Centre: fraction of hours with simultaneous C+D in the LP by year
# (total and at negative-price hours specifically). Right: an example day
# exhibiting simultaneous dispatch — purple shading marks the affected hours,
# left bars = LP dispatch, hatched bars = MILP dispatch.
# ```

# %% [markdown]
# ## 7. Summary and Implications
#
# Three observations follow from this analysis:
#
# 1. **Magnitude of LP "cheat"**: the MILP–LP revenue gap shows how much extra
#    revenue the LP earns from physically-infeasible simultaneous dispatch. If the
#    gap is a fraction of a percent, the LP is a safe approximation for revenue
#    forecasting.
#
# 2. **Negative-price trigger**: virtually all simultaneous-dispatch hours occur
#    during negative-price windows, confirming the user's intuition. The LP exploits
#    negative prices to charge as much as possible by burning SoC through η losses —
#    something a real battery cannot do.
#
# 3. **Price-floor vs MILP**: the price-floor LP avoids simultaneous dispatch
#    entirely but also forfeits all negative-price charging revenue (treating those
#    hours as price = 0). The MILP correctly models negative-price charging without
#    simultaneous dispatch. The revenue gap between price-floor and MILP equals the
#    value of negative-price charging under the complementarity constraint.

# %%
print("\nSummary:")
print(f"  LP annual revenue:           {rev_lp * ann:>8,.0f} EUR/yr")
print(
    f"  MILP annual revenue:         {rev_milp * ann:>8,.0f} EUR/yr"
    "  (physically correct)"
)
print(
    f"  LP + price floor revenue:    {rev_floor * ann:>8,.0f} EUR/yr"
    "  (forfeits neg-price charging)"
)
print()
cheat_pct = (rev_lp / rev_milp - 1) * 100
print(
    f"  LP 'cheat' (LP − MILP):      {(rev_lp - rev_milp) * ann:>8,.0f} EUR/yr"
    f"  ({cheat_pct:+.3f}%)"
)
print(f"  Neg-price value (MILP − floor): {(rev_milp - rev_floor) * ann:>8,.0f} EUR/yr")
simul_pct = simul_lp.mean() * 100
print(f"  Simul. C+D hours: {simul_lp.sum():,} ({simul_pct:.2f}% of all hours)")
n_neg_final = (simul_lp & neg_mask).sum()
neg_of_simul = n_neg_final / max(simul_lp.sum(), 1) * 100
print(f"    of which at negative prices: {n_neg_final:,} ({neg_of_simul:.0f}%)")
