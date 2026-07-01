# Project State

## Goals

End-to-end Virtual Power Plant (VPP) learning project, structured in progressive stages from single-asset hindsight optimization to a full VPP with household prosumers and intraday trading.

---

## Stage 1 — Single-battery dispatch with perfect price knowledge

Understand how much value a battery can capture from day-ahead price arbitrage under ideal conditions (perfect foresight, no friction). This sets the theoretical upper bound before adding forecast error and physical constraints.

**Phase 1a — Ideal LP (no losses, no degradation)**

- Simple LP: single net charge/discharge variable per hour, capacity and power limits only.
  Reference implementation: `/home/chris/research/flexa-challenge/` Task 2.
- Battery assumptions (realistic but simple):
  - Capacity: 100 kWh
  - Max charge/discharge power: 50 kW (2-hour C-rate)
  - Round-trip efficiency: 100% (ideal, added later)
  - Degradation cost: 0 (added later)
- Constraint variants:
  - **With daily reset** (`SoC_T=0 = SoC_T=24`): splits cleanly into independent daily problems → easy to solve, parallelize, and interpret. Evaluate: how much value is lost vs. unconstrained?
  - **Unconstrained across full horizon**: allows carry-over between days; assess sensitivity to the end-condition assumption.
- Metrics: daily/annual revenue, dispatch heatmaps (charge/discharge by hour-of-day × month), SoC trajectory.

**Phase 1b — Add physical realism**

- Round-trip efficiency (e.g. 90%): separate charge/discharge variables needed (can no longer net them). Evaluate impact on revenue vs. ideal case.
- Degradation cost: linear $/cycle term in objective. Evaluate: at what price spread does cycling stop being profitable?
- Re-run scenario comparison table (name, revenue, Δ% vs ideal, feasibility flag) for each lever added.

---

## Stage 2 — Price forecasting and realistic dispatch

Move from perfect foresight to auction-time forecasts; use optimized quantities as actual bids.

**Phase 2a — Baseline price forecast**

- Simple baseline model available at auction time (day-ahead = gate closure ~12:00 for delivery next day).
- Candidate features: lagged prices, hour-of-day, day-of-week, month, load/wind/solar forecasts (SMARD).
- Models: naïve (yesterday's prices), linear regression, gradient boosting. Track MAE / pinball loss per hour.

**Phase 2b — Forecast-driven dispatch**

- Plug forecast prices into the Stage 1 LP → get bid quantities.
- Settle against realized prices.
- Revenue analysis:
  - Revenue under perfect foresight (upper bound from Stage 1)
  - Revenue under forecast (realized P&L)
  - Gap decomposition: forecast error × dispatch sensitivity
- Risk analysis: distribution of daily P&L, worst-day scenarios, correlation of errors with price spikes.

**Phase 2c — Forecast improvement loop**

- Iterate: better features (weather, load forecasts) → better prices → more revenue.
- Track revenue improvement alongside forecast accuracy to understand the revenue–accuracy frontier.

---

## Stage 3 — Additional revenue streams

Extend beyond pure day-ahead arbitrage.

- **Frequency containment reserve (FCR) / aFRR capacity market** — bid symmetric capacity to TSO; understand availability constraints vs. arbitrage conflict.
- **Intraday continuous trading** — trade against intraday price signals; RL or threshold policies. (Later.)
- Combined revenue stack: quantify value of each stream and how they conflict (e.g. FCR capacity reservation reduces arbitrage flexibility).

---

## Stage 4 — VPP with household prosumers

Scale from a single asset to an aggregated fleet.

- **Household behavior modelling** — understand net load profiles (PV + consumption) per household type. Dataset: Enefit/Kaggle (Estonian counties).
- **VPP contract structure** — research how current VPP contracts handle the household/aggregator split: what flexibility does the aggregator control, what must be reserved for personal consumption (e.g. SoC floor), how is revenue shared?
- **Aggregated dispatch** — optimize fleet-level dispatch subject to per-household constraints; compare to single-battery case.
- **Forecasting at household level** — net energy forecasts per cluster; how much individual uncertainty cancels at the fleet level?

---

## Current state (2026-07-01)

Stage 1a complete. SMARD DE-LU hourly prices 2018-10-01 → 2026-06-29.

Ideal LP dispatch (100 kWh / 50 kW, η=1, PuLP/CBC):
- Daily-reset constraint costs only −0.9% vs free horizon (4,205 vs 4,244 EUR/year annualized)
- Diurnal pattern: charge overnight 0–6h, discharge into 8–10h and 18–20h peaks
- 2022 energy crisis produces ~3–5× the long-run average daily revenue
- ~17% of total revenue comes from negative-price charging (getting paid to charge);
  share rising post-2022 with renewable expansion
- Investment economics: pure DA arbitrage does not clear an 8% hurdle at any realistic
  CAPEX (IRR: +6.2% optimistic incl. 2022 → −7.2% conservative excl. 2022; base case IRR ≈ −1%)
  → revenue stacking (FCR/aFRR) required to justify the investment

## Next steps

1. Stage 1b: `pipeline/03_battery_dispatch_realistic.py`
   - Add round-trip efficiency (η=90%): requires separate charge/discharge variables
   - Add degradation cost: linear EUR/cycle term in objective
   - Scenario comparison table: ideal vs η=90% vs η=90%+degradation (revenue, Δ%, avg cycles)
2. Stage 2a: `pipeline/04_price_forecast_baseline.py` — baseline day-ahead price forecast at auction time
