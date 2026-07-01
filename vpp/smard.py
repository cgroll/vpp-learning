"""SMARD API client — enumerations and download function.

Covers the full variable catalogue (generation, consumption, prices, forecasts,
capacity) so later pipeline stages can import from here without changes.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional, Tuple

import pandas as pd
import requests

DEFAULT_START_DATE = datetime(2015, 1, 1, tzinfo=timezone.utc)


class Resolution(str, Enum):
    HOUR = "hour"
    QUARTER_HOUR = "quarterhour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    YEAR = "year"


class Region(str, Enum):
    DE = "DE"
    AT = "AT"
    LU = "LU"
    DE_LU = "DE-LU"
    DE_AT_LU = "DE-AT-LU"
    FIFTY_HERTZ = "50Hertz"
    AMPRION = "Amprion"
    TENNET = "TenneT"
    TRANSNET_BW = "TransnetBW"
    APG = "APG"
    CREOS = "Creos"


class Variable(int, Enum):
    # Power Generation
    BROWN_COAL = 1223
    NUCLEAR = 1224
    WIND_OFFSHORE = 1225
    HYDRO = 1226
    OTHER_CONVENTIONAL = 1227
    OTHER_RENEWABLE = 1228
    BIOMASS = 4066
    WIND_ONSHORE = 4067
    SOLAR = 4068
    HARD_COAL = 4069
    PUMPED_STORAGE = 4070
    NATURAL_GAS = 4071

    # Power Consumption
    TOTAL_LOAD = 410
    RESIDUAL_LOAD = 4359
    PUMPED_STORAGE_LOAD = 4387

    # Market Prices
    PRICE_DE_LU = 4169
    PRICE_DE_LU_NEIGHBORS = 5078
    PRICE_BE = 4996
    PRICE_NO2 = 4997
    PRICE_AT = 4170
    PRICE_DK1 = 252
    PRICE_DK2 = 253
    PRICE_FR = 254
    PRICE_IT_NORTH = 255
    PRICE_NL = 256
    PRICE_PL = 257
    PRICE_PL2 = 258
    PRICE_CH = 259
    PRICE_SI = 260
    PRICE_CZ = 261
    PRICE_HU = 262

    # Forecasts — day-ahead (generation)
    FORECAST_OFFSHORE = 3791
    FORECAST_ONSHORE = 123
    FORECAST_SOLAR = 125
    FORECAST_OTHER = 715
    FORECAST_WIND_SOLAR = 5097
    FORECAST_TOTAL = 122

    # Forecasts — day-ahead (consumption)
    FORECAST_LOAD = 411

    # Forecasts — intraday (generation)
    FORECAST_INTRADAY_SOLAR = 5126
    FORECAST_INTRADAY_ONSHORE = 5127
    FORECAST_INTRADAY_OFFSHORE = 5128
    FORECAST_INTRADAY_WIND_SOLAR = 5129

    # Capacity
    CAPACITY_BIOMASS = 189
    CAPACITY_HYDRO = 3792
    CAPACITY_WIND_OFFSHORE = 4076
    CAPACITY_WIND_ONSHORE = 186
    CAPACITY_SOLAR = 188
    CAPACITY_OTHER_RENEWABLE = 194
    CAPACITY_BROWN_COAL = 4072
    CAPACITY_HARD_COAL = 4075
    CAPACITY_NATURAL_GAS = 198
    CAPACITY_PUMPED_STORAGE = 4074


def download_smard_data(
    region: str,
    resolution: str,
    variable: int,
    variable_name: str,
    start_time: Optional[datetime] = None,
) -> pd.DataFrame:
    """Download data from SMARD API.

    The API is block-based: first fetch block-start timestamps, then fetch
    each block's observations in separate requests.

    Returns:
        DataFrame with UTC-aware DatetimeIndex and one column named variable_name.
        NaN rows are dropped; callers that need a gapless index should reindex.

    Raises:
        RuntimeError: on HTTP errors or when no data is available.
    """
    base_url = "https://www.smard.de/app"

    index_url = f"{base_url}/chart_data/{variable}/{region}/index_{resolution}.json"
    response = requests.get(index_url)
    if response.status_code != 200:
        raise RuntimeError(f"Error fetching index: {response.status_code}")

    block_timestamps: List[int] = response.json()["timestamps"]
    if not block_timestamps:
        raise RuntimeError("No timestamps available for the specified parameters")

    if start_time is not None:
        start_ms = int(start_time.timestamp() * 1000)
        block_timestamps = [ts for ts in block_timestamps if ts >= start_ms]
        if not block_timestamps:
            raise RuntimeError(f"No data available after {start_time}")

    all_timestamps: List[datetime] = []
    all_values: List[float] = []

    for ts_block in block_timestamps:
        data_url = (
            f"{base_url}/chart_data/{variable}/{region}"
            f"/{variable}_{region}_{resolution}_{ts_block}.json"
        )
        data_response = requests.get(data_url)
        if data_response.status_code != 200:
            print(
                f"Warning: skipping block {ts_block}: HTTP {data_response.status_code}"
            )
            continue

        series_data: List[Tuple[int, float]] = data_response.json()["series"]
        all_timestamps.extend(
            datetime.fromtimestamp(ts / 1000, tz=timezone.utc) for ts, _ in series_data
        )
        all_values.extend(val for _, val in series_data)

    df = pd.DataFrame({"timestamp": all_timestamps, variable_name: all_values})
    df = df.sort_values("timestamp").drop_duplicates(subset="timestamp")
    df = df.set_index("timestamp")
    df = df.dropna()
    return df
