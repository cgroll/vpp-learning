"""Stage 2a: compute day-ahead price forecasts and write to forecasts.parquet.

Extracts model fitting from 05_price_forecast_baseline.py. Runs once and
caches results so downstream dispatch stages can consume them without
re-fitting models.

Output: data/processed/forecasts.parquet
Columns: timestamp (index, Europe/Berlin), naive, ridge, lgbm
Coverage: test period 2023-01-01 onward
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from vpp.paths import ProjPaths

paths = ProjPaths()
paths.ensure_directories()

SPLIT_DATE = "2023-01-01"

# ---------------------------------------------------------------------------
# Load prices
# ---------------------------------------------------------------------------

prices_raw = pd.read_parquet(paths.smard_prices_file)
prices_eur_mwh = prices_raw["price_de_lu"].dropna().sort_index()
prices_berlin = prices_eur_mwh.copy()
prices_berlin.index = prices_berlin.index.tz_convert("Europe/Berlin")

print(
    f"Loaded {len(prices_berlin):,} hourly prices: "
    f"{prices_berlin.index[0].date()} → {prices_berlin.index[-1].date()}"
)

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def build_features(prices: pd.Series) -> pd.DataFrame:
    """Feature matrix for day-ahead price forecasting.

    All features are observable by gate closure (noon on day D for day D+1).
    """
    df = prices.to_frame("price")

    df["lag_24"] = prices.shift(24)
    df["lag_48"] = prices.shift(48)
    df["lag_168"] = prices.shift(168)

    daily = prices.resample("D").agg(day_mean="mean", day_std="std")
    daily["prev_day_mean"] = daily["day_mean"].shift(1)
    daily["prev_day_std"] = daily["day_std"].shift(1)
    daily["week_mean"] = daily["day_mean"].shift(1).rolling(7).mean()

    df["_day"] = prices.index.floor("D")
    df = df.join(
        daily[["prev_day_mean", "prev_day_std", "week_mean"]],
        on="_day",
    ).drop(columns="_day")

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

# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------

split_ts = pd.Timestamp(SPLIT_DATE, tz="Europe/Berlin")
train = features_df[features_df.index < split_ts]
test = features_df[features_df.index >= split_ts].copy()

X_train = train[FEATURE_COLS].values
y_train = train["price"].values
X_test = test[FEATURE_COLS].values

print(
    f"Train: {len(train):,} hours ({train.index[0].date()} → {train.index[-1].date()})"
)
print(f"Test:  {len(test):,} hours ({test.index[0].date()} → {test.index[-1].date()})")

# ---------------------------------------------------------------------------
# Fit models and produce test-period predictions
# ---------------------------------------------------------------------------

print("Fitting naive (lag-24) ...")
lag24_idx = FEATURE_COLS.index("lag_24")
y_naive = X_test[:, lag24_idx]

print("Fitting Ridge ...")
ridge_pipe = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
ridge_pipe.fit(X_train, y_train)
y_ridge = ridge_pipe.predict(X_test)

print("Fitting LightGBM ...")
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

print("Done.")

# ---------------------------------------------------------------------------
# Write forecasts.parquet
# ---------------------------------------------------------------------------

forecasts = pd.DataFrame(
    {"naive": y_naive, "ridge": y_ridge, "lgbm": y_lgbm},
    index=test.index,
)
forecasts.index.name = "timestamp"
forecasts.to_parquet(paths.forecasts_file)

print(f"Wrote {len(forecasts):,} rows → {paths.forecasts_file}")
