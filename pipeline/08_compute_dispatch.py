"""Compute dispatch schedules for all scenarios and write to dispatch_schedules.parquet.

Iterates over SCENARIOS in vpp/scenarios.py, runs the appropriate solver for
each one, and concatenates results into a single wide-format Parquet file.

Output: data/processed/dispatch_schedules.parquet
Columns: timestamp, c (kW), d (kW), soc (kWh), fcr_mw (MW), scenario_id
  - fcr_mw: MW of FCR capacity committed in that hour; 0 for DA-only scenarios.
    FCR revenue = fcr_price_eur_mw_h * fcr_mw (computed in analysis scripts).
"""

import pandas as pd
from tqdm import tqdm

from vpp.dispatch import solve
from vpp.paths import ProjPaths
from vpp.scenarios import SCENARIOS

paths = ProjPaths()
paths.ensure_directories()

# ---------------------------------------------------------------------------
# Load price data
# ---------------------------------------------------------------------------

prices_raw = pd.read_parquet(paths.smard_prices_file)
prices_actual = prices_raw["price_de_lu"].dropna().sort_index()
prices_actual.index = prices_actual.index.tz_convert("Europe/Berlin")

forecasts = pd.read_parquet(paths.forecasts_file)

print(
    f"Actual prices: {len(prices_actual):,} hours "
    f"({prices_actual.index[0].date()} → {prices_actual.index[-1].date()})"
)
print(
    f"Forecasts:     {len(forecasts):,} hours "
    f"({forecasts.index[0].date()} → {forecasts.index[-1].date()})"
)
print(f"Scenarios to run: {len(SCENARIOS)}")


def _load_price_signal(price_signal: str) -> pd.Series:
    if price_signal == "actual":
        return prices_actual
    elif price_signal == "naive_forecast":
        # lag-24 requires no fitting; compute from actual prices for the full range
        return prices_actual.shift(24).dropna()
    elif price_signal == "ridge_forecast":
        return forecasts["ridge"]
    elif price_signal == "lgbm_forecast":
        return forecasts["lgbm"]
    else:
        raise ValueError(f"Unknown price_signal: {price_signal!r}")


# ---------------------------------------------------------------------------
# Run all scenarios
# ---------------------------------------------------------------------------

results = []
for sc in tqdm(SCENARIOS, desc="Solving scenarios"):
    prices = _load_price_signal(sc.price_signal)
    df = solve(prices, sc.battery, sc.method_id)
    df["scenario_id"] = sc.scenario_id

    # FCR committed MW per hour — non-zero only for FCR scenarios.
    # block = hour_of_day // 4 (0–5); power_kw / 1000 converts kW → MW.
    if sc.fcr_blocks:
        hour_of_day = df["timestamp"].dt.hour
        in_fcr_block = (hour_of_day // 4).isin(sc.fcr_blocks)
        fcr_kw = sc.battery.power_kw / 1000.0
        df["fcr_mw"] = in_fcr_block.map({True: fcr_kw, False: 0.0})
    else:
        df["fcr_mw"] = 0.0

    results.append(df)
    tqdm.write(f"  {sc.scenario_id}: {len(df):,} rows")

dispatch = pd.concat(results, ignore_index=True)

# ---------------------------------------------------------------------------
# Write dispatch_schedules.parquet
# ---------------------------------------------------------------------------

dispatch["scenario_id"] = dispatch["scenario_id"].astype("category")
dispatch.to_parquet(paths.dispatch_schedules_file, index=False)

n_scenarios = dispatch["scenario_id"].nunique()
print(f"\nWrote {len(dispatch):,} rows x {n_scenarios} scenarios")
print(f"  -> {paths.dispatch_schedules_file}")
