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

Stages 1a, 1b, 1b addendum, and 2a complete.

SMARD DE-LU hourly prices 2018-10-01 → 2026-06-29.

**Stage 1a — Ideal LP (`pipeline/02_battery_dispatch_ideal.py`, η=1, PuLP/CBC):**
- Daily-reset costs only −0.9% vs free horizon (4,205 vs 4,244 EUR/year annualized)
- Diurnal pattern: charge overnight 0–6h, discharge into 8–10h and 18–20h peaks
- 2022 energy crisis produces ~3–5× the long-run average daily revenue
- ~17% of total revenue from negative-price charging; share rising post-2022
- Investment economics: IRR ≈ −1% base case → revenue stacking (FCR/aFRR) required

**Stage 1b — Realistic LP + degradation sweep (`pipeline/03_battery_dispatch_realistic.py`):**

Scenario comparison (100 kWh / 50 kW):

| Scenario | Annual rev (EUR) | Δ% vs ideal | Cycles/year |
|---|---|---|---|
| ideal (η=1, no deg) — DR | 4,205 | 0% | 835 |
| η_rt=0.9, no deg — DR | 3,490 | −17% | 731 |
| η_rt=0.9, 10 EUR/cycle — DR | 1,846 | −56% | 114 |
| η_rt=0.9, no deg — FH | 3,505 | −17% | — |
| η_rt=0.9, 10 EUR/cycle — FH | 1,951 | −54% | — |

- Round-trip loss alone cuts revenue by 17%; break-even spread ≈ 10.5 EUR/MWh per EUR/cycle
- At 10 EUR/cycle degradation (≈ €45k CAPEX / 4,500 cycles), revenue halves and cycling
  drops from 835 → 114 cycles/year
- Free-horizon (FH) vs daily-reset (DR) premium is tiny without degradation (+0.4%) but
  grows sharply with degradation cost: +5.7% at 10 EUR/cycle, +20% at 15, +47% at 20,
  +76% at 30, +145% at 40 EUR/cycle
- Intuition: daily reset forces SoC=0 each midnight so idle days are always empty; FH
  allows multi-day carry-over to wait for rare high-spread opportunities

**Stage 1b addendum — Simultaneous charge/discharge analysis (`pipeline/04_simultaneous_dispatch.py`):**

Three dispatch formulations compared (η_rt=0.9, daily reset, 100 kWh / 50 kW):

| Approach | Annual rev (EUR) | vs LP | States valid? |
|---|---|---|---|
| LP baseline | 3,490 | — | No — c·d > 0 in 2% of hours |
| MILP (binary mutual exclusivity) | 3,481 | −0.27% | Yes |
| LP + price floor at 0 | 3,401 | −2.5% | Yes, but sub-optimal |

- Simultaneous C+D in the LP occurs in 1,390 hours (2.05%) — **100% at negative-price hours**,
  confirming the mechanism: the LP "burns" SoC through η losses to create headroom for
  additional gross charging when prices are negative
- The LP "cheat" is worth only 9 EUR/yr (+0.27%): LP is a safe approximation for revenue
  forecasting but physically invalid
- The price-floor fix is counterproductive: it forfeits 79 EUR/yr of legitimate negative-price
  charging (9× the size of the LP cheat it corrects)
- MILP is the correct and practical solution: with daily-reset sub-problems of 24 binary
  variables each, CBC solves it as fast as the LP
- Revenue is always evaluated at real prices regardless of which prices were used for
  optimization (distorted objective ≠ distorted settlement)

**Stage 2a — Baseline price forecast (`pipeline/05_price_forecast_baseline.py`):**

Train/test split: 2018–2022 train (includes 2022 crisis), 2023–2026 test.
Feature set: lag-24/48/168, previous-day mean/std, 7-day rolling mean, cyclical
hour/dow/month encoding.

| Model | MAE (EUR/MWh) | vs naïve |
|---|---|---|
| Naïve (lag-24) | 27.41 | — |
| Ridge | 26.72 | −2.5% |
| LightGBM | 28.49 | +4.0% |

- Naïve (lag-24) is a very hard baseline: strong diurnal and daily autocorrelation
  means yesterday's same-hour price is already highly informative
- Ridge marginally improves by blending calendar and weekly lag features
- LightGBM slightly *underperforms* the naïve baseline — driven by distribution
  shift: the model learns 2022-crisis spike patterns during training that do not
  transfer to the calmer 2023–2026 test period
- The 4% LightGBM penalty vs naïve is small in absolute terms (≈1 EUR/MWh) but
  reveals a structural issue: a longer training window amplifies crisis-era
  patterns that hurt out-of-sample calibration

## Next steps

1. Stage 2b: `pipeline/06_forecast_dispatch.py` — plug forecast prices into the
   Stage 1 LP, settle at realized prices, decompose the gap to perfect-foresight
   revenue
2. Stage 2c: improve the forecast — rolling training window, additional features
   (weather, load forecasts), or model ensembling

## Future ideas (not yet planned)

- **Separate batch computation from analysis**: a data pipeline stage that runs battery
  optimization across many configurations (capacity, power, η, degradation cost) upfront
  and writes all results to a single structured file. Analysis notebooks would then read
  this file rather than re-running solvers, cleanly separating computation from
  visualization/reporting.
