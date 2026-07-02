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

## Stage 3 — Value stacking: additional revenue streams

Extend beyond pure day-ahead arbitrage. Investment economics on DA alone yield IRR ≈ −1%;
ancillary services are required for viability. A case-study (1 MW / 2 MWh BESS, June 2025)
shows value stacking produces +90% uplift over a DA-only strategy, with FCR and aFRR
contributing far more than DA arbitrage itself (see `bess_research.md` for details).

Market gate closures determine the optimization sequence each day:
FCR (08:00) → aFRR (09:00) → Day-ahead (12:00) → Intraday auction (15:00) → Continuous intraday.

**Phase 3a — FCR (Frequency Containment Reserve)**

- Data: regelleistung.net publishes weekly tender results (capacity price, cleared volume)
  as free downloadable CSVs. Write `pipeline/01_download_fcr_prices.py`.
- Model: optimize FCR capacity committed per 4-hour block (≤ battery power). SoC must stay
  near 50% during the window to allow both upward and downward response.
- Key output: FCR capacity price threshold at which FCR beats DA arbitrage; SoC reservation
  cost (EUR of foregone arbitrage per MW of FCR committed).

**Phase 3b — aFRR (Automatic Frequency Restoration Reserve)**

- Data: regelleistung.net daily aFRR capacity and energy clearing prices. Same download
  script can cover FCR + aFRR + mFRR in one pass.
- Model: joint FCR + aFRR + DA optimization. aFRR adds a stochastic activation component;
  start with a fixed 25% energy activation rate assumption, then sensitivity analysis.
- Products: separate up/down capacity bids; must sustain 60-minute delivery.

**Phase 3c — Intraday trading**

- Data: Open Power System Data (OPSD) 15-min intraday continuous index (ID3 VWAP) as a
  free price proxy. Full EPEX Spot order-book data is commercial.
- Model: rolling-horizon dispatch — after DA results are known, re-optimize residual
  capacity against 15-min intraday prices to correct positions.
- Reference: `PortfolioEnergy/rtc-tools-bess-demo` (RTC-Tools + HiGHS MILP, rolling
  horizon, "rolling intrinsic" valuation).

**Combined revenue stack analysis**

- Quantify marginal value of each added stream (FCR alone, FCR + aFRR, full stack).
- Quantify conflicts: how much DA arbitrage revenue is sacrificed per MW of FCR/aFRR
  capacity committed?
- Scenario comparison table extended with ancillary-service scenarios.

---

## Stage 4 — VPP with household prosumers

Scale from a single asset to an aggregated fleet.

- **Household behavior modelling** — understand net load profiles (PV + consumption) per household type. Dataset: Enefit/Kaggle (Estonian counties).
- **VPP contract structure** — research how current VPP contracts handle the household/aggregator split: what flexibility does the aggregator control, what must be reserved for personal consumption (e.g. SoC floor), how is revenue shared?
- **Aggregated dispatch** — optimize fleet-level dispatch subject to per-household constraints; compare to single-battery case.
- **Forecasting at household level** — net energy forecasts per cluster; how much individual uncertainty cancels at the fleet level?

---

## Current state (2026-07-02)

Stages 1a, 1b, 1b addendum, 2a, 2b, and 3a (FCR baseline) complete.

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

**Stage 2b — Forecast-driven dispatch (`pipeline/06_forecast_dispatch.py`):**

Battery: η_rt=0.9, MILP, no degradation, 100 kWh / 50 kW, daily reset.
Settlement always at actual prices; only the optimisation signal differs.

| Year | Hindsight (EUR/yr) | Naïve (EUR/yr) | Efficiency |
|---|---|---|---|
| 2018 | 1,290 | 863 | 66.9% |
| 2019 | 1,043 | 755 | 72.5% |
| 2020 | 1,149 | 848 | 73.8% |
| 2021 | 2,833 | 2,188 | 77.2% |
| 2022 | 6,881 | 5,543 | 80.6% |
| 2023 | 3,574 | 2,923 | 81.8% |
| 2024 | 4,097 | 3,447 | 84.1% |
| 2025 | 4,594 | 3,947 | 85.9% |
| **Overall** | **3,486** | **2,848** | **81.7%** |

- Naïve lag-24 captures 81.7% of perfect-foresight revenue; gap is 639 EUR/yr
- Efficiency improves over time (67% in 2018 → 87% in 2026): post-crisis
  day-ahead prices are more auto-correlated (renewable-dominated diurnal shape
  repeats more reliably day-over-day)
- 2022 energy crisis: highest absolute revenues but not lowest efficiency,
  because even during the crisis, large day-over-day spreads persisted
- 55 days (2.0%) see negative forecast revenue: naïve dispatch charges on a high-
  price day (because yesterday was high) and discharges on a low-price day

**Backtest framework implemented (pipeline not yet re-run):**

A centralised dispatch computation layer separates solver logic from analysis:

- `vpp/battery.py` — `BatteryParams` (Pydantic, frozen), `from_eta_rt` factory
- `vpp/dispatch.py` — `solve()` dispatcher + named per-methodology functions:
  `solve_lp_daily_reset`, `solve_lp_free_horizon`, `solve_milp_daily_reset`,
  `solve_lp_floor_daily_reset`
- `vpp/scenarios.py` — `ScenarioConfig` (Pydantic) + `SCENARIOS` registry (21 scenarios)
- `pipeline/07_compute_forecasts.py` — fits naive/Ridge/LightGBM, writes
  `data/processed/forecasts.parquet` (test period 2023+)
- `pipeline/08_compute_dispatch.py` — runs all 21 scenarios, writes
  `data/processed/dispatch_schedules.parquet` (schema: timestamp, c, d, soc, scenario_id)

Analysis scripts 02–06 refactored: inline solvers removed, all read pre-computed
dispatch from parquet and compute revenues via merge with actual prices.

Three new cross-scenario comparison notebooks (not yet generated):
- `pipeline/09_dispatch_overview.py` — all-scenarios metrics table + revenue chart
- `pipeline/10_constraint_comparison.py` — DR vs FH premium across degradation sweep
- `pipeline/11_simultaneous_cd.py` — LP vs MILP vs LP+floor revenue + simultaneous C+D

**Stage 3 data download complete (`pipeline/01_download_regelleistung.py`):**

- `vpp/regelleistung.py` — CRDS API client using regelleistung.net public Excel bulk files
- Downloads via `GET /apps/crds/api/v2/tenders/files/{filename}` (monthly result Excel per product)
- Queries one month at a time to avoid the API's 100-item cap
- Handles 3 FCR format variants: pre-2021 24h product (skipped), 2021–2022-08
  abbreviated country columns ("Sheet0"), 2022-09+ full names ("001")
- Handles 2 aFRR format variants: pre-2022 EUR/MW per block (÷4 → EUR/MW/h), 2022+
  EUR/MW/h directly

Output files (in `data/downloads/regelleistung/`):

| File | Coverage | Rows | Schema |
|------|----------|------|--------|
| `fcr_prices.parquet` | 2021-01-01 → 2026-06-30 | 2,006 | negpos_00_04…negpos_20_24 (EUR/MW/h) |
| `afrr_capacity_prices.parquet` | 2018-10-01 → 2026-06-30 | 2,830 | neg_00_04…pos_20_24 (EUR/MW/h) |

FCR 2024 mean prices by block (EUR/MW/h): 10.1 (00-04) → 25.9 (12-16) → 11.5 (20-24).
Midday peak reflects high DA arbitrage competition reducing FCR clearing prices at low-spread hours.

**Stage 3a — FCR vs DA arbitrage (`pipeline/12_fcr_comparison.py`):**

Scenario B (FCR-only): all 6 blocks committed daily, optimistic cancellation
assumption (FCR activations cancel within each 4h block → SoC frozen at 50%),
η=1, no degradation. Revenue = FCR clearing price × 4h × 0.05 MW per block.

| Year | DA only (EUR/yr) | FCR only (EUR/yr) | FCR premium |
|------|-----------------|-------------------|-------------|
| 2021 | 3,577 | 7,504 | +110% |
| 2022 | 8,694 | 10,124 | +16% |
| 2023 | 4,300 | 5,601 | +30% |
| 2024 | 4,720 | 7,130 | +51% |
| 2025 | 5,277 | 6,657 | +26% |
| 2026 | 5,794 | 7,688 | +33% |
| **Overall** | **5,357** | **7,429** | **+39%** |

- FCR beats DA by +39% on average under perfect foresight
- Premium collapses to +16% in 2022 energy crisis (record DA spreads narrowed
  the gap; FCR prices also rose, but couldn't keep pace with extreme arbitrage)
- FCR is more stable: 5,600–10,100 EUR/yr vs DA 3,600–8,700 EUR/yr
- Midday block (12–16) commands highest FCR prices (~26 EUR/MW/h in 2024)
  because batteries prefer DA in those hours — FCR must compensate

Framework additions:
- `ScenarioConfig.fcr_blocks` — frozenset of 4h blocks (0–5) committed to FCR
- `solve_fcr_all_blocks()` in `vpp/dispatch.py` — returns zeros for c/d, SoC at 50%
- `dispatch_schedules.parquet` gains `fcr_mw` column (MW committed per hour; 0
  for DA-only scenarios); FCR revenue = `fcr_price_eur_mw_h × fcr_mw` in analysis

## Next steps

1. **Run `dvc repro`** to re-run 7 stale analysis stages (02–04, 06, 09–11)
   whose outputs are unchanged but whose dvc.lock entries are stale after
   `dispatch_schedules.parquet` was regenerated with the FCR scenario.
2. **Stage 3a continued — optimal block selection (Scenario C)**: MILP with one
   binary variable per 4h block per day (FCR or DA). Key output: for each block,
   the minimum FCR clearing price at which FCR beats DA (the break-even price).
   The 2022 result (FCR premium only +16%) suggests that on high-spread days some
   blocks might flip from FCR to DA — Scenario C will reveal which ones.
3. **Stage 3b** — aFRR capacity bids alongside FCR, using downloaded
   `afrr_capacity_prices.parquet`.
4. **Stage 2c**: improve the forecast — rolling training window, additional
   features, or ensembling to close the 639 EUR/yr DA efficiency gap.

## Future ideas (not yet planned)
