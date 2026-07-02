# BESS Research: Extensions Beyond Day-Ahead Arbitrage

## Current state

The project covers day-ahead price arbitrage with a single 100 kWh / 50 kW battery,
including realistic dispatch (η_rt=0.9, MILP), a naive lag-24 price forecast, and a
backtest framework across 21 scenarios. All analysis uses DE-LU SMARD day-ahead prices.

The naïve forecast captures ~82% of perfect-foresight revenue (2,848 vs 3,486 EUR/yr).
Investment economics require additional revenue streams: IRR ≈ −1% on day-ahead alone.

---

## Value stacking: the opportunity

A [flex-power.energy case study](https://flex-power.energy/energyblog/battery-storage-trading-strategy/)
on a 1 MW / 2 MWh BESS (June 18, 2025) shows how revenue streams stack:

| Revenue stream          | Daily P&L (EUR) |
|-------------------------|----------------|
| FCR capacity            | 126             |
| aFRR positive capacity  | 164             |
| aFRR positive energy    | 207             |
| aFRR negative capacity  | 287             |
| aFRR negative energy    | 19              |
| Day-ahead arbitrage     | 19              |
| Intraday auction        | 119             |
| Continuous intraday     | 12              |
| **Total (stacked)**     | **953**         |
| Wholesale-only strategy | 502             |

**+90% uplift from ancillary services.** Day-ahead arbitrage alone is the smallest
contribution; FCR and aFRR dominate.

Market gate closures determine the optimization sequence:
1. FCR capacity auction (08:00) — 4-hour symmetric blocks
2. aFRR capacity auction (09:00) — 4-hour blocks, separate up/down
3. Day-ahead auction (12:00, EPEX) — hourly products
4. Intraday auction (15:00) — 15-min products
5. Continuous intraday (D-1 15:00 through D H-0:05) — real-time

---

## Extension 1: FCR (Frequency Containment Reserve)

**What it is.** Battery bids symmetric capacity (must respond both up and down
simultaneously). TSO activates within 30 s; must sustain for 15–30 min. Product
windows: 4-hour blocks from 00:00. Tendered weekly on regelleistung.net.

**Market mechanics.**
- Revenue = capacity price (EUR/MW/h) × capacity committed. No separate energy payment.
- Symmetry constraint: SoC must stay near 50% to allow full up and full down response.
- Reduces available capacity for arbitrage — key trade-off.
- German FCR prices ranged roughly 4–30 EUR/MW/h in recent years.

**Data availability.**
- **regelleistung.net Datencenter** publishes weekly tender results (price, volume)
  as downloadable CSVs. Historical data available for several years.
- No SMARD variable for FCR prices. Requires a separate download script.

**Feasibility: High.** Data is public and downloadable. Modeling requires:
- A committed MW variable per 4-hour block (≤ battery power).
- SoC bounds during FCR window: [committed_MW × 15min, capacity − committed_MW × 15min].
- Joint optimization: choose FCR commitment first, then optimize residual capacity
  for DA arbitrage around the SoC reservation.

**Key question to answer:** At what FCR price does it dominate DA arbitrage? How does
the SoC restriction reduce arbitrage revenue?

---

## Extension 2: aFRR (Automatic Frequency Restoration Reserve)

**What it is.** Battery responds to TSO automated signals within 5 min; must sustain
60 min. Products: capacity (EUR/MW/h, committed regardless of activation) + energy
(EUR/MWh, paid only on activation). Symmetric or asymmetric (up only / down only)
bids. Tendered daily.

**Market mechanics.**
- Activation rate is stochastic (typical assumption: 25% energy utilization).
- Higher revenue than FCR per MW when prices are elevated, but more complex modeling.
- SoC constraints similar to FCR but must guarantee 60 min of sustained delivery.

**Data availability.**
- **regelleistung.net** publishes daily aFRR capacity clearing prices and volumes.
- aFRR energy clearing prices (merit order) also published historically.
- Activation volumes (realized energy) available via regelleistung.net / ENTSO-E.

**Feasibility: Medium-High.** Data available but stochastic activation adds complexity:
- Deterministic version: commit capacity, model expected energy at assumed activation rate.
- Stochastic version: scenario tree over activation rates (Stage 3 level of complexity).

---

## Extension 3: mFRR (Manual Frequency Restoration Reserve)

**What it is.** Slower reserve (full activation within 12.5 min), lower availability
payments but larger energy activation. Symmetric or asymmetric. Tendered daily.

**Data availability.** regelleistung.net, same as aFRR.

**Feasibility: High** (simpler than aFRR — slower response, so no SoC-speed constraints).

**Likely conclusion:** Lower value-add vs aFRR for a fast-responding battery; useful
as a building block before implementing full ancillary service stack.

---

## Extension 4: Intraday auction

**What it is.** EPEX Spot intraday auction at D-1 15:00. 15-minute products for the
full next day. Corrects DA positions using more recent forecast information.

**Market mechanics.** Battery uses residual capacity after FCR/aFRR commitments to
improve its position. Expected to add ~100–120 EUR/day per MW at a 1 MW battery (see
stacking example above).

**Data availability.**
- **SMARD has no intraday price variable** in the current Variable enum.
- EPEX Spot sells historical intraday data commercially.
- **ENTSO-E Transparency Platform** (registration required, free API key) provides
  intraday prices via document type A60 (Cross-border flows) and market data.
  Endpoint: `AuctionResultDocument` — but availability and granularity varies by country.
- **Open Power System Data (OPSD)** — historical 15-min German intraday continuous
  index (ID3, VWAP) available as free download. Not the auction price but a proxy.

**Feasibility: Medium.** Data exists but requires a new source (not SMARD). OPSD's
15-min intraday VWAP is the most accessible free option for Germany.

---

## Extension 5: Continuous intraday trading

**What it is.** Real-time LOB (limit-order book) market trading through delivery minus
5 minutes. Battery optimizes against evolving order book and price signals.

**Optimization approach.** The rtc-tools-bess-demo repo (RTC-Tools + HiGHS MILP) uses
a **rolling horizon** approach: repeatedly re-solve a short-horizon sub-problem as new
price information arrives. Referenced method: **rolling intrinsic** valuation — at each
step, solve an LP using current prices to compute the "intrinsic" value of remaining
storage capacity, then decide whether to trade against the order book.

**Data availability.**
- Order book data is commercial (EPEX Spot, ICE).
- OPSD provides a 15-min intraday continuous index (ID3) usable as a price proxy.
- Most accessible approximation: treat the 15-min VWAP as the tradeable price and
  simulate a rolling-horizon dispatch on it.

**Feasibility: Low-Medium** for a first pass (simplified price proxy). Full order-book
simulation is research-grade complexity.

---

## Extension 6: Rolling horizon optimization

**What it is.** Not a new market — a better optimization approach. Instead of solving
the full day in one shot (our current approach), solve a receding-horizon LP/MILP:
at each decision point, optimize over a look-ahead window (e.g. 6 hours) using the
best available forecast; re-solve after each step.

**Benefit:** More realistic decision-making under forecast uncertainty. Our current
setup assumes all 24 hours are decided simultaneously at 12:00; rolling horizon
re-dispatches as better intraday prices arrive.

**Feasibility: High** — no new data needed. Requires restructuring the dispatch solver
to take a window length parameter and loop over time.

---

## Extension 7: Stochastic / robust optimization

**What it is.** Instead of optimizing against a single price forecast, optimize against
a scenario tree (e.g., Monte Carlo price scenarios). Maximizes expected revenue while
controlling downside risk (VaR / CVaR on daily P&L).

**Feasibility: Medium** — requires scenario generation (already have the forecast
residual distribution from Stage 2). Adds complexity without new data.

---

## Extension 8: CAISO / US ancillary markets (reference only)

The `battery-storage-optimization-energy-ancillary` repo (Pyomo + GLPK) optimizes
CAISO markets: energy + RegUp + Spin + RegDown + NonSpin. Data via the Gridstatus API.

This is the **US equivalent** of our DE-LU FCR/aFRR work. Not directly applicable
(different market structure), but useful as a modeling reference:
- Shows how to structure a joint energy + ancillary LP.
- Prohibition on simultaneous C+D is already covered in our MILP formulation.

---

## Data availability summary

| Revenue stream         | Data source          | Freely available? | Effort to add |
|------------------------|----------------------|-------------------|---------------|
| Day-ahead prices       | SMARD API            | Yes ✓             | Done          |
| FCR capacity prices    | regelleistung.net    | Yes ✓             | Low           |
| aFRR cap + energy      | regelleistung.net    | Yes ✓             | Low           |
| mFRR cap + energy      | regelleistung.net    | Yes ✓             | Low           |
| Intraday auction (DA2) | EPEX Spot (commercial) / ENTSO-E (free with key) | Partial | Medium |
| Intraday continuous    | OPSD (VWAP proxy)    | Yes (proxy) ✓     | Medium        |
| Order book depth       | EPEX Spot (commercial) | No ✗            | High          |
| Wind / solar forecasts | SMARD API            | Yes ✓             | Done (enum)   |

---

## Recommended implementation order

1. **Download FCR and aFRR prices** from regelleistung.net into the pipeline.
   This is Stage 3 of the project plan and unblocked — data is free and public.

2. **Model FCR participation** as a Stage 3a notebook:
   - Optimize FCR capacity committed per 4-hour block vs residual DA arbitrage.
   - Key output: FCR capacity price threshold at which FCR beats DA arbitrage.

3. **Model aFRR** as Stage 3b:
   - Joint FCR + aFRR + DA optimization.
   - Use a fixed activation rate assumption first (25%), then sensitivity analysis.

4. **Add intraday prices** (OPSD 15-min VWAP or ENTSO-E API) for Stage 3c:
   - Rolling horizon dispatch using post-DA intraday price corrections.

5. **Rolling horizon framework** can be built independently and used in Stage 3c.

---

## Key open questions

- What is the SoC reservation cost in EUR? I.e., how much DA revenue does a
  unit of FCR capacity commitment sacrifice? This drives the optimal stack.
- How correlated are FCR/aFRR prices with DA price spreads? (If spreads are high,
  both ancillary and DA values are elevated — helpful for SoC management.)
- Does the rolling-intrinsic intraday approach require 15-min granularity throughout,
  or can we use hourly SMARD DA + 15-min intraday for a hybrid model?
