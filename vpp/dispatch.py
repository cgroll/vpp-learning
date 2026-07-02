"""Battery dispatch solvers.

Public API:
  solve(prices, battery, method_id) -> DataFrame[timestamp, c, d, soc]

Named per-methodology functions (one elif per new method added here):
  solve_lp_daily_reset       LP, SoC forced to 0 at each calendar day boundary
  solve_lp_free_horizon      LP, SoC=0 only at start and end of full horizon
  solve_milp_daily_reset     MILP (binary mutual exclusivity), solved day by day
  solve_lp_floor_daily_reset LP daily reset, prices clipped to 0 before optimising
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pulp

from vpp.battery import BatteryParams


def solve(
    prices: pd.Series,
    battery: BatteryParams,
    method_id: str,
) -> pd.DataFrame:
    """Dispatcher — maps method_id to the appropriate named solver function."""
    if method_id == "lp_dr":
        return solve_lp_daily_reset(prices, battery)
    elif method_id == "lp_fh":
        return solve_lp_free_horizon(prices, battery)
    elif method_id == "milp_dr":
        return solve_milp_daily_reset(prices, battery)
    elif method_id == "lp_floor_dr":
        return solve_lp_floor_daily_reset(prices, battery)
    elif method_id == "fcr_all":
        return solve_fcr_all_blocks(prices, battery)
    else:
        raise ValueError(f"Unknown method_id: {method_id!r}")


# ---------------------------------------------------------------------------
# Public named solver functions
# ---------------------------------------------------------------------------


def solve_lp_daily_reset(prices: pd.Series, battery: BatteryParams) -> pd.DataFrame:
    """LP with SoC=0 enforced at every calendar-day boundary (Europe/Berlin).

    Solves the full horizon as one LP with sparse SoC=0 constraints. DST days
    (23h or 25h) are included naturally — _day_boundary_positions handles any
    day length.
    """
    positions = _day_boundary_positions(prices.index)
    c, d, soc = _solve_segment_lp(prices.to_numpy(dtype=float), battery, positions)
    return _to_df(prices.index, c, d, soc)


def solve_lp_free_horizon(prices: pd.Series, battery: BatteryParams) -> pd.DataFrame:
    """LP over the full horizon; SoC=0 only at start and end."""
    n = len(prices)
    c, d, soc = _solve_segment_lp(prices.to_numpy(dtype=float), battery, [0, n])
    return _to_df(prices.index, c, d, soc)


def solve_milp_daily_reset(prices: pd.Series, battery: BatteryParams) -> pd.DataFrame:
    """MILP with binary mutual exclusivity, solved one calendar day at a time.

    The day-by-day approach keeps each sub-problem at ~24 binary variables,
    making it as fast as the LP relaxation. Handles DST days (23h/25h) by
    solving the correct-length MILP for each day.
    """
    result_ts: list = []
    result_c: list = []
    result_d: list = []
    result_soc: list = []

    dti = pd.DatetimeIndex(prices.index)
    for date, grp in prices.groupby(dti.date):  # type: ignore[attr-defined]
        c_day, d_day, soc_day = _solve_segment_milp(grp.to_numpy(dtype=float), battery)
        result_ts.extend(grp.index.tolist())
        result_c.extend(c_day.tolist())
        result_d.extend(d_day.tolist())
        result_soc.extend(soc_day.tolist())

    return _to_df(
        pd.DatetimeIndex(result_ts),
        np.array(result_c),
        np.array(result_d),
        np.array(result_soc),
    )


def solve_fcr_all_blocks(prices: pd.Series, battery: BatteryParams) -> pd.DataFrame:
    """FCR committed for all 24 hours: no DA dispatch, SoC held at 50%.

    Under the optimistic assumption that FCR activations cancel within each
    4-hour block, the battery's SoC is unchanged by FCR. No optimisation is
    needed. Revenue (capacity payments) is added by the caller via the fcr_mw
    column written in 08_compute_dispatch.py.
    """
    n = len(prices)
    soc_target = battery.capacity_kwh * 0.5
    return _to_df(prices.index, np.zeros(n), np.zeros(n), np.full(n, soc_target))


def solve_lp_floor_daily_reset(
    prices: pd.Series, battery: BatteryParams
) -> pd.DataFrame:
    """LP daily reset with prices clipped to 0 before optimising.

    Revenue is always computed at original prices by the caller; the price
    clipping only affects what the optimiser sees. Clipping removes the incentive
    for simultaneous C+D at negative-price hours, but forfeits genuine
    negative-price charging revenue.
    """
    positions = _day_boundary_positions(prices.index)
    c, d, soc = _solve_segment_lp(
        prices.to_numpy(dtype=float), battery, positions, clip_prices=True
    )
    return _to_df(prices.index, c, d, soc)


# ---------------------------------------------------------------------------
# Private core solvers
# ---------------------------------------------------------------------------


def _solve_segment_lp(
    price_array: np.ndarray,
    battery: BatteryParams,
    soc_zero_positions: list[int],
    clip_prices: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """LP over an arbitrary contiguous segment.

    Returns (c, d, soc) arrays of length n. soc[t] is the start-of-hour SoC.
    deg_cost_per_cycle is converted to EUR/kWh-stored internally.
    """
    prices = np.maximum(price_array, 0.0) if clip_prices else price_array
    n = len(prices)
    deg_cost_kwh = battery.deg_cost_per_cycle / battery.capacity_kwh

    prob = pulp.LpProblem("battery_lp", pulp.LpMaximize)
    c = pulp.LpVariable.dicts("c", range(n), lowBound=0, upBound=battery.power_kw)
    d = pulp.LpVariable.dicts("d", range(n), lowBound=0, upBound=battery.power_kw)
    soc = pulp.LpVariable.dicts(
        "soc", range(n + 1), lowBound=0, upBound=battery.capacity_kwh
    )

    prob += pulp.lpSum(
        prices[t] / 1000 * (d[t] - c[t]) - deg_cost_kwh * battery.eta_c * c[t]
        for t in range(n)
    )
    for t in range(n):
        prob += (
            soc[t + 1] == soc[t] + battery.eta_c * c[t] - (1.0 / battery.eta_d) * d[t]
        )
    for pos in soc_zero_positions:
        prob += soc[pos] == 0

    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"LP infeasible: {pulp.LpStatus[status]!r}")

    return _extract_c_d_soc(c, d, soc, n)


def _solve_segment_milp(
    price_array: np.ndarray,
    battery: BatteryParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MILP for one segment with SoC=0 at start and end (binary mutual exclusivity).

    Works for any segment length n (24h standard, 23h/25h for DST days).
    Returns (c, d, soc) arrays of length n.
    """
    n = len(price_array)
    prob = pulp.LpProblem("battery_milp", pulp.LpMaximize)
    z = pulp.LpVariable.dicts("z", range(n), cat="Binary")
    c = pulp.LpVariable.dicts("c", range(n), lowBound=0, upBound=battery.power_kw)
    d = pulp.LpVariable.dicts("d", range(n), lowBound=0, upBound=battery.power_kw)
    soc = pulp.LpVariable.dicts(
        "soc", range(n + 1), lowBound=0, upBound=battery.capacity_kwh
    )

    prob += pulp.lpSum(price_array[t] / 1000 * (d[t] - c[t]) for t in range(n))
    prob += soc[0] == 0
    prob += soc[n] == 0
    for t in range(n):
        prob += (
            soc[t + 1] == soc[t] + battery.eta_c * c[t] - (1.0 / battery.eta_d) * d[t]
        )
        prob += c[t] <= battery.power_kw * z[t]
        prob += d[t] <= battery.power_kw * (1 - z[t])

    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"MILP infeasible: {pulp.LpStatus[status]!r}")

    return _extract_c_d_soc(c, d, soc, n)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _day_boundary_positions(index: pd.Index) -> list[int]:
    """Cumulative-length positions (0..n) at each calendar-day boundary."""
    dti = pd.DatetimeIndex(index)
    dates = dti.date  # type: ignore[attr-defined]
    lengths = [int((np.array(dates) == day).sum()) for day in sorted(set(dates))]
    return [int(b) for b in np.concatenate([[0], np.cumsum(lengths)]).tolist()]


def _extract_c_d_soc(
    c: dict,
    d: dict,
    soc: dict,
    n: int,
    eps: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    c_val = np.array([c[t].value() for t in range(n)])
    d_val = np.array([d[t].value() for t in range(n)])
    soc_val = np.array([soc[t].value() for t in range(n)])
    c_val[np.abs(c_val) < eps] = 0.0
    d_val[np.abs(d_val) < eps] = 0.0
    soc_val[np.abs(soc_val) < eps] = 0.0
    return c_val, d_val, soc_val


def _to_df(
    index: pd.Index,
    c: np.ndarray,
    d: np.ndarray,
    soc: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame({"timestamp": index, "c": c, "d": d, "soc": soc})
