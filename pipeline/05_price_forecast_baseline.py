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
# # Stage 2a — Baseline Day-Ahead Price Forecast
#
# Stage 1 established the revenue upper bound assuming perfect price foresight.
# In practice, bids must be submitted by ~12:00 noon for next-day delivery, so
# only historical prices and calendar features are available at auction time.
#
# Three baseline models are benchmarked on a held-out test set (2023–2026):
#
# | Model | Description |
# |---|---|
# | **Naïve** | Same-hour price from the previous day (lag-24) |
# | **Ridge** | Linear regression on lag + calendar features |
# | **LightGBM** | Gradient-boosted trees (same feature set) |
#
# **Gate-closure rule**: features for day-D+1 forecasts must be observable by
# noon on day D. Day-ahead clearing prices for day D were published after the
# D−1 noon auction, so all 24 hours of day D are always available as lags.

# %% [markdown]
# ## 1. Features
#
# For target hour $h$ of day $D+1$, features available at gate closure (noon, day $D$):
#
# | Feature | Description |
# |---|---|
# | `lag_24` | Price at hour $h$ on day $D$ |
# | `lag_48` | Price at hour $h$ on day $D-1$ |
# | `lag_168` | Price at hour $h$ on day $D-7$ |
# | `prev_day_mean` | Mean of all 24 prices on day $D$ |
# | `prev_day_std` | Std of day $D$ prices (volatility proxy) |
# | `week_mean` | 7-day rolling mean of daily means ending day $D$ |
# | `hour_{sin,cos}` | Cyclical hour-of-day encoding |
# | `dow_{sin,cos}` | Cyclical day-of-week encoding |
# | `month_{sin,cos}` | Cyclical month encoding |

# %%
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from vpp.paths import ProjPaths

paths = ProjPaths()
paths.ensure_directories()

plt.rcParams["figure.dpi"] = 100
sns.set_theme(style="whitegrid")

SPLIT_DATE = "2023-01-01"

# %% [markdown]
# ## 2. Load Data

# %%
prices_raw = pd.read_parquet(paths.smard_prices_file)
prices_eur_mwh = prices_raw["price_de_lu"].dropna().sort_index()
prices_berlin = prices_eur_mwh.copy()
prices_berlin.index = prices_berlin.index.tz_convert("Europe/Berlin")

print(
    f"Loaded {len(prices_berlin):,} hourly prices: "
    f"{prices_berlin.index[0].date()} → {prices_berlin.index[-1].date()}"
)

# %% [markdown]
# ## 3. Feature Engineering


# %%
def build_features(prices: pd.Series) -> pd.DataFrame:
    """Build feature matrix from hourly day-ahead prices."""
    df = prices.to_frame("price")

    # Lag features (row-based; off by ≤1h on DST days — acceptable for baseline)
    df["lag_24"] = prices.shift(24)
    df["lag_48"] = prices.shift(48)
    df["lag_168"] = prices.shift(168)

    # Daily aggregate features (previous day)
    daily = prices.resample("D").agg(day_mean="mean", day_std="std")
    daily["prev_day_mean"] = daily["day_mean"].shift(1)
    daily["prev_day_std"] = daily["day_std"].shift(1)
    daily["week_mean"] = daily["day_mean"].shift(1).rolling(7).mean()

    # Map daily stats back to hourly via floor("D") join
    df["_day"] = prices.index.floor("D")
    df = df.join(
        daily[["prev_day_mean", "prev_day_std", "week_mean"]],
        on="_day",
    ).drop(columns="_day")

    # Cyclical calendar features
    idx = prices.index
    df["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7)
    df["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7)
    df["month_sin"] = np.sin(2 * np.pi * (idx.month - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (idx.month - 1) / 12)

    return df.dropna()


features_df = build_features(prices_berlin)
FEATURE_COLS = [c for c in features_df.columns if c != "price"]

print(f"Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")

# %% [markdown]
# ## 4. Train / Test Split

# %%
split_ts = pd.Timestamp(SPLIT_DATE, tz="Europe/Berlin")
train = features_df[features_df.index < split_ts]
test = features_df[features_df.index >= split_ts].copy()

X_train = train[FEATURE_COLS].values
y_train = train["price"].values
X_test = test[FEATURE_COLS].values
y_test = test["price"].values

print(
    f"Train: {len(train):,} hours"
    f"  ({train.index[0].date()} → {train.index[-1].date()})"
    f"  mean={y_train.mean():.1f} EUR/MWh"
)
print(
    f"Test:  {len(test):,} hours"
    f"  ({test.index[0].date()} → {test.index[-1].date()})"
    f"  mean={y_test.mean():.1f} EUR/MWh"
)
print("Note: training includes the 2022 energy crisis; test is post-crisis.")

# %% [markdown]
# ## 5. Fit Models

# %%
# Naïve: lag-24 extracted directly from test features (no fitting needed)
lag24_idx = FEATURE_COLS.index("lag_24")
y_naive = X_test[:, lag24_idx]

# Ridge regression (features scaled)
ridge_pipe = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0)),
    ]
)
ridge_pipe.fit(X_train, y_train)
y_ridge = ridge_pipe.predict(X_test)

# LightGBM
lgb_model = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=50,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=1,
    random_state=42,
    verbose=-1,
)
lgb_model.fit(X_train, y_train)
y_lgbm = lgb_model.predict(X_test)

print("Training complete.")

# %% [markdown]
# ## 6. Metrics

# %%
mae_naive = mean_absolute_error(y_test, y_naive)
mae_ridge = mean_absolute_error(y_test, y_ridge)
mae_lgbm = mean_absolute_error(y_test, y_lgbm)

ridge_pct = (mae_ridge / mae_naive - 1) * 100
lgbm_pct = (mae_lgbm / mae_naive - 1) * 100

print(f"Overall MAE — test {SPLIT_DATE} → {test.index[-1].date()}:")
print(f"  Naïve (lag-24):  {mae_naive:.2f} EUR/MWh")
print(f"  Ridge:           {mae_ridge:.2f} EUR/MWh  ({ridge_pct:+.1f}% vs naïve)")
print(f"  LightGBM:        {mae_lgbm:.2f} EUR/MWh  ({lgbm_pct:+.1f}% vs naïve)")

# Per-hour MAE
test_hours = test.index.hour.values


def _per_hour_mae(
    y_true: np.ndarray, y_pred: np.ndarray, hours: np.ndarray
) -> list[float]:
    return [
        mean_absolute_error(y_true[hours == h], y_pred[hours == h]) for h in range(24)
    ]


mae_by_hour = pd.DataFrame(
    {
        "naive": _per_hour_mae(y_test, y_naive, test_hours),
        "ridge": _per_hour_mae(y_test, y_ridge, test_hours),
        "lgbm": _per_hour_mae(y_test, y_lgbm, test_hours),
    },
    index=pd.Index(range(24), name="hour"),
)

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

ax = axes[0]
ax.plot(
    mae_by_hour.index,
    mae_by_hour["naive"],
    label="Naïve (lag-24)",
    color="gray",
    linewidth=1.5,
    linestyle="--",
)
ax.plot(
    mae_by_hour.index,
    mae_by_hour["ridge"],
    label="Ridge",
    color="#1f77b4",
    linewidth=1.5,
)
ax.plot(
    mae_by_hour.index,
    mae_by_hour["lgbm"],
    label="LightGBM",
    color="#d62728",
    linewidth=1.5,
)
ax.set_xlabel("Hour of day")
ax.set_ylabel("MAE (EUR/MWh)")
ax.set_title("Forecast error by hour (test set)")
ax.set_xticks(range(0, 24, 2))
ax.legend()

ax = axes[1]
model_names = ["Naïve", "Ridge", "LightGBM"]
model_maes = [mae_naive, mae_ridge, mae_lgbm]
model_colors = ["gray", "#1f77b4", "#d62728"]
bars = ax.bar(model_names, model_maes, color=model_colors, width=0.5)
for bar, mae in zip(bars, model_maes):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.2,
        f"{mae:.1f}",
        ha="center",
        va="bottom",
        fontsize=11,
    )
ax.set_ylabel("MAE (EUR/MWh)")
ax.set_title(f"Overall MAE (test {SPLIT_DATE[:4]}–{test.index[-1].year})")

fig.suptitle("Day-ahead price forecast accuracy — baseline models", fontsize=12)
fig.tight_layout()
fig.savefig(paths.images_path / "05_per_hour_mae.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/05_per_hour_mae.png
# :name: fig-05-per-hour-mae
# Per-hour MAE by model (left) and overall MAE comparison (right) on the
# 2023–2026 test set. The naïve lag-24 baseline is a strong starting point
# due to pronounced diurnal and weekly seasonality in day-ahead prices.
# ```

# %%
# 2-week sample from early 2024 (post-crisis, near-normal conditions)
sample_start = pd.Timestamp("2024-01-08", tz="Europe/Berlin")
sample_end = pd.Timestamp("2024-01-22", tz="Europe/Berlin")
sample_mask = (test.index >= sample_start) & (test.index < sample_end)

test["y_naive"] = y_naive
test["y_lgbm"] = y_lgbm

fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

ax = axes[0]
s = test[sample_mask]
ax.plot(s.index, s["price"], label="Actual", color="black", linewidth=1.2)
ax.plot(
    s.index,
    s["y_naive"],
    label="Naïve",
    color="gray",
    linewidth=1.0,
    linestyle="--",
    alpha=0.9,
)
ax.plot(
    s.index,
    s["y_lgbm"],
    label="LightGBM",
    color="#d62728",
    linewidth=1.0,
    alpha=0.9,
)
ax.set_ylabel("EUR/MWh")
ax.set_title("Forecast sample — 2 weeks Jan 2024")
ax.legend()
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

ax = axes[1]
ax.scatter(y_test, y_lgbm, alpha=0.05, s=3, color="#d62728")
xy_min = min(y_test.min(), y_lgbm.min())
xy_max = max(y_test.max(), y_lgbm.max())
ax.plot(
    [xy_min, xy_max],
    [xy_min, xy_max],
    "k--",
    linewidth=0.8,
    label="Perfect forecast",
)
ax.set_xlabel("Actual (EUR/MWh)")
ax.set_ylabel("Predicted (EUR/MWh)")
ax.set_title("LightGBM: actual vs predicted (test set)")
ax.legend(fontsize=9)

fig.suptitle("Day-ahead price forecast — LightGBM baseline", fontsize=12)
fig.tight_layout()
fig.savefig(paths.images_path / "05_forecast_sample.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/05_forecast_sample.png
# :name: fig-05-forecast-sample
# Left: actual vs naïve and LightGBM forecasts for a two-week sample (Jan 2024).
# Right: scatter of actual vs predicted for the full test set — points above the
# diagonal are over-predictions; points below are under-predictions.
# ```

# %% [markdown]
# ## 7. Summary
#
# The per-hour MAE profile reveals where forecasts are hardest: morning and
# evening ramp hours (roughly 6–9 h and 17–20 h) tend to have higher errors
# because prices are most sensitive to demand peaks and renewable output swings.
#
# Key findings:
#
# - **Naïve (lag-24)** is a competitive baseline due to strong diurnal and weekly
#   seasonality; yesterday's same-hour price is a reasonable first guess.
# - **Ridge** captures linear lag relationships but is constrained by the
#   non-linear price dynamics (spikes, cross-hour interactions).
# - **LightGBM** models non-linear interactions (hour × day-of-week, price level
#   × volatility regime) and substantially reduces error.
#
# **Distribution shift caveat**: the training set includes the 2022 energy crisis
# (extreme prices, high volatility). The test set (2023–2026) is post-crisis with
# lower mean prices. Models trained on 2022 spike dynamics may under-predict the
# frequency of negative prices and over-predict high-price hours in the test set.
#
# **Next**: Stage 2b plugs these forecasts into the Stage 1 LP, computes
# forecast-driven dispatch revenues, and decomposes the gap to the
# perfect-foresight upper bound.
