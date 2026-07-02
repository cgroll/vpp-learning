"""Scenario registry for battery dispatch backtest.

Each ScenarioConfig defines:
  - scenario_id   machine key stored in dispatch_schedules.parquet
  - name          human-readable label for plots and tables
  - battery       physical parameters (BatteryParams)
  - price_signal  which price series drives the optimiser
  - method_id     which solve_* function to call (see vpp/dispatch.py)

Price signals:
  "actual"          realized DE-LU day-ahead clearing prices (full dataset)
  "naive_forecast"  lag-24 naïve forecast (full dataset, computed as shift(24))
  "ridge_forecast"  Ridge regression forecast (test period only)
  "lgbm_forecast"   LightGBM forecast (test period only)
"""

from pydantic import BaseModel

from vpp.battery import BatteryParams


class ScenarioConfig(BaseModel, frozen=True):
    scenario_id: str
    name: str
    battery: BatteryParams
    price_signal: str
    method_id: str
    fcr_blocks: frozenset[int] = frozenset()  # blocks 0–5 committed to FCR


# ---------------------------------------------------------------------------
# Shared battery configurations
# ---------------------------------------------------------------------------

_B_ETA100 = BatteryParams(capacity_kwh=100.0, power_kw=50.0)
_B_ETA090 = BatteryParams.from_eta_rt(capacity_kwh=100.0, power_kw=50.0, eta_rt=0.90)

_SWEEP_COSTS = [0, 5, 10, 15, 20, 30, 40, 50]

# ---------------------------------------------------------------------------
# Scenario registry
# ---------------------------------------------------------------------------

SCENARIOS: list[ScenarioConfig] = [
    # ------------------------------------------------------------------
    # Stage 1a: ideal LP (η=1, perfect foresight)
    # ------------------------------------------------------------------
    ScenarioConfig(
        scenario_id="actual__lp_dr__eta100__deg000",
        name="Ideal LP – daily reset",
        battery=_B_ETA100,
        price_signal="actual",
        method_id="lp_dr",
    ),
    ScenarioConfig(
        scenario_id="actual__lp_fh__eta100__deg000",
        name="Ideal LP – free horizon",
        battery=_B_ETA100,
        price_signal="actual",
        method_id="lp_fh",
    ),
    # ------------------------------------------------------------------
    # Stage 1b: η=0.9 LP, degradation cost sweep × both horizons
    # 8 costs × 2 horizons = 16 scenarios
    # ------------------------------------------------------------------
    *[
        ScenarioConfig(
            scenario_id=f"actual__lp_{horizon}__{eta_tag}__deg{cost:03d}",
            name=f"η=0.9 LP – {horizon_label}, {cost} EUR/cycle",
            battery=BatteryParams.from_eta_rt(
                capacity_kwh=100.0, power_kw=50.0, eta_rt=0.90, deg_cost=float(cost)
            ),
            price_signal="actual",
            method_id=f"lp_{horizon}",
        )
        for horizon, horizon_label, eta_tag in [
            ("dr", "daily reset", "eta090"),
            ("fh", "free horizon", "eta090"),
        ]
        for cost in _SWEEP_COSTS
    ],
    # ------------------------------------------------------------------
    # Stage 1b addendum: MILP and LP+floor for simultaneous C/D analysis
    # (LP daily reset at deg000 reuses actual__lp_dr__eta090__deg000 above)
    # ------------------------------------------------------------------
    ScenarioConfig(
        scenario_id="actual__milp_dr__eta090__deg000",
        name="η=0.9 MILP – daily reset",
        battery=_B_ETA090,
        price_signal="actual",
        method_id="milp_dr",
    ),
    ScenarioConfig(
        scenario_id="actual__lp_floor_dr__eta090__deg000",
        name="η=0.9 LP price-floor – daily reset",
        battery=_B_ETA090,
        price_signal="actual",
        method_id="lp_floor_dr",
    ),
    # ------------------------------------------------------------------
    # Stage 2b: forecast-driven dispatch (naïve lag-24, test period only)
    # ------------------------------------------------------------------
    ScenarioConfig(
        scenario_id="naive__milp_dr__eta090__deg000",
        name="Naïve forecast → MILP dispatch",
        battery=_B_ETA090,
        price_signal="naive_forecast",
        method_id="milp_dr",
    ),
    # ------------------------------------------------------------------
    # Stage 3a: FCR-only baseline (all 6 blocks, η=1, perfect foresight)
    # Revenue = FCR capacity payments only; no DA dispatch.
    # Optimistic assumption: FCR activations cancel within each 4h block,
    # so SoC is held constant at 50% (no energy exchanged).
    # ------------------------------------------------------------------
    ScenarioConfig(
        scenario_id="fcr_only__eta100",
        name="FCR only – all blocks, η=1",
        battery=_B_ETA100,
        price_signal="actual",  # not used by FCR solver; kept for schema consistency
        method_id="fcr_all",
        fcr_blocks=frozenset(range(6)),
    ),
]

# Lookup by scenario_id for convenience
SCENARIO_BY_ID: dict[str, ScenarioConfig] = {sc.scenario_id: sc for sc in SCENARIOS}
