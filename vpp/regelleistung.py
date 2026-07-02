"""Regelleistung.net CRDS API client for FCR and aFRR capacity prices.

Downloads monthly result Excel files from the public CRDS API and extracts
Germany settlement capacity prices per 4-hour delivery block.

FCR (PRL) data available from 2019-07-01 (daily tender format).
aFRR (SRL) capacity data available from 2018-07-01.
"""

import io
import warnings
from datetime import date, timedelta
from typing import Literal

import pandas as pd
import requests

_BASE_URL = "https://www.regelleistung.net/apps/crds/api/v2"
_TIMEOUT_SHORT = 30
_TIMEOUT_LONG = 120

# Earliest dates with 4-hour-block tender results available
# (before 2021, FCR used a single daily NEGPOS_00_24 product — not included)
FCR_START_DATE = date(2021, 1, 1)
AFRR_START_DATE = date(2018, 10, 1)  # aligned with SMARD price series

ProductType = Literal["FCR", "aFRR"]


def _list_monthly_result_files(
    product_type: ProductType,
    date_from: date,
    date_to: date,
    market: Literal["CAPACITY", "ENERGY"] = "CAPACITY",
) -> list[dict]:
    """Return metadata dicts for monthly RESULTS files in the given date range."""
    resp = requests.get(
        f"{_BASE_URL}/tenders/files",
        params={
            "productTypes": product_type,
            "from": date_from.isoformat(),
            "to": date_to.isoformat(),
        },
        timeout=_TIMEOUT_SHORT,
    )
    resp.raise_for_status()
    return [
        f
        for f in resp.json()
        if f["fileType"] == "RESULTS"
        and f["dateRangeType"] == "MONTH"
        and f["market"] == market
    ]


def _download_file(filename: str) -> bytes:
    """Download a named file from the CRDS files endpoint."""
    resp = requests.get(
        f"{_BASE_URL}/tenders/files/{filename}",
        timeout=_TIMEOUT_LONG,
    )
    resp.raise_for_status()
    return resp.content


_FCR_DE_PRICE_CANDIDATES = [
    "GERMANY_SETTLEMENTCAPACITY_PRICE_[EUR/MW]",  # 2022-09+
    "DE_SETTLEMENTCAPACITY_PRICE_[EUR/MW]",  # 2021–2022-08
]


def _parse_fcr_results(content: bytes) -> pd.DataFrame:
    """Parse FCR monthly RESULT_OVERVIEW_CAPACITY_MARKET_FCR Excel file.

    Handles both the old (pre-2022-09) abbreviated column names and the newer
    full country-name columns. Files with the old 24-hour NEGPOS_00_24 product
    (pre-2021) are returned as empty DataFrames.

    Returns a wide DataFrame indexed by delivery_date (UTC midnight Timestamps),
    with one column per 4-hour block: negpos_00_04 … negpos_20_24.
    Prices are in EUR/MW/h (raw EUR/MW-per-block divided by 4).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xl = pd.ExcelFile(io.BytesIO(content))
        raw = xl.parse(xl.sheet_names[0], header=0)
    assert isinstance(raw, pd.DataFrame)
    df: pd.DataFrame = raw

    df = df[df["TENDER_NUMBER"] == 1].copy()  # type: ignore[assignment]

    # Skip old 24-hour format (NEGPOS_00_24)
    if "NEGPOS_00_04" not in df["PRODUCTNAME"].values:
        return pd.DataFrame()

    raw_col = next(
        (c for c in _FCR_DE_PRICE_CANDIDATES if c in df.columns),
        None,
    )
    if raw_col is None:
        cols = list(df.columns)
        raise ValueError(f"Germany price column not found. Columns: {cols}")

    df["price"] = pd.to_numeric(df[raw_col], errors="coerce") / 4  # type: ignore[operator]
    df["col"] = df["PRODUCTNAME"].str.lower()  # negpos_00_04, …
    df["delivery_date"] = pd.to_datetime(df["DATE_FROM"], utc=True).dt.normalize()

    return (
        df[["delivery_date", "col", "price"]]
        .pivot(index="delivery_date", columns="col", values="price")
        .rename_axis(None, axis="columns")
        .rename_axis("delivery_date")
    )


_AFRR_DE_MARGINAL_PER_HOUR = "GERMANY_MARGINAL_CAPACITY_PRICE_[(EUR/MW)/h]"  # 2022+
_AFRR_DE_MARGINAL_PER_BLOCK = "GERMANY_MARGINAL_CAPACITY_PRICE_[EUR/MW]"  # pre-2022


def _parse_afrr_capacity_results(content: bytes) -> pd.DataFrame:
    """Parse aFRR monthly RESULT_OVERVIEW_CAPACITY_MARKET_aFRR Excel file.

    Handles both the old format (price in EUR/MW per 4h block, normalised by /4)
    and the new format (price already in EUR/MW/h).

    Returns a wide DataFrame indexed by delivery_date (UTC midnight Timestamps),
    with one column per direction × block: neg_00_04 … pos_20_24.
    Prices are in EUR/MW/h.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xl = pd.ExcelFile(io.BytesIO(content))
        raw = xl.parse(xl.sheet_names[0], header=0)
    assert isinstance(raw, pd.DataFrame)
    df: pd.DataFrame = raw

    if _AFRR_DE_MARGINAL_PER_HOUR in df.columns:
        divisor = 1  # already EUR/MW/h
        raw_col: str = _AFRR_DE_MARGINAL_PER_HOUR
    elif _AFRR_DE_MARGINAL_PER_BLOCK in df.columns:
        divisor = 4  # EUR/MW per 4h block → EUR/MW/h
        raw_col = _AFRR_DE_MARGINAL_PER_BLOCK
    else:
        cols = list(df.columns)
        raise ValueError(f"aFRR Germany price column not found. Columns: {cols}")

    df["price"] = pd.to_numeric(df[raw_col], errors="coerce") / divisor  # type: ignore[operator]
    df["col"] = df["PRODUCT"].str.lower()  # pos_00_04, neg_00_04, …
    df["delivery_date"] = pd.to_datetime(df["DATE_FROM"], utc=True).dt.normalize()

    return (
        df[["delivery_date", "col", "price"]]
        .pivot(index="delivery_date", columns="col", values="price")
        .rename_axis(None, axis="columns")
        .rename_axis("delivery_date")
    )


def _iter_months(date_from: date, date_to: date):
    """Yield (month_start, month_end) pairs covering date_from..date_to."""
    cur = date_from.replace(day=1)
    while cur <= date_to:
        month_end = last_day_of_month(cur)
        yield (max(date_from, cur), min(date_to, month_end))
        cur = (month_end + timedelta(days=1)).replace(day=1)


def _collect_files(
    product_type: ProductType,
    date_from: date,
    date_to: date,
    market: Literal["CAPACITY", "ENERGY"] = "CAPACITY",
) -> list[dict]:
    """Return monthly RESULTS files; queries one month at a time (avoids API cap)."""
    files: list[dict] = []
    seen: set[str] = set()
    for m_from, m_to in _iter_months(date_from, date_to):
        for f in _list_monthly_result_files(product_type, m_from, m_to, market):
            if f["fileName"] not in seen:
                files.append(f)
                seen.add(f["fileName"])
    return sorted(files, key=lambda x: x["dateRange"])


def download_fcr_prices(date_from: date, date_to: date) -> pd.DataFrame:
    """Download FCR capacity prices from regelleistung.net.

    Fetches monthly Excel result files and extracts Germany settlement capacity
    prices per 4-hour block, normalised to EUR/MW/h.

    Args:
        date_from: First delivery date (inclusive).
        date_to: Last delivery date (inclusive).

    Returns:
        DataFrame indexed by delivery_date (UTC-midnight Timestamps) with columns
        negpos_00_04 … negpos_20_24 (EUR/MW/h). Returns empty DataFrame when
        date_from > date_to or no files are available.
    """
    if date_from > date_to:
        return pd.DataFrame()

    files = _collect_files("FCR", date_from, date_to)
    if not files:
        return pd.DataFrame()

    frames = []
    for f in files:
        print(f"  Downloading {f['fileName']} …")
        content = _download_file(f["fileName"])
        frames.append(_parse_fcr_results(content))

    df = pd.concat(frames)
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()

    ts_from = pd.Timestamp(date_from, tz="UTC")
    ts_to = pd.Timestamp(date_to, tz="UTC")
    return df.loc[ts_from:ts_to]


def download_afrr_capacity_prices(date_from: date, date_to: date) -> pd.DataFrame:
    """Download aFRR capacity prices from regelleistung.net.

    Fetches monthly Excel result files and extracts Germany marginal capacity
    prices per 4-hour block and direction (POS/NEG) in EUR/MW/h.

    Args:
        date_from: First delivery date (inclusive).
        date_to: Last delivery date (inclusive).

    Returns:
        DataFrame indexed by delivery_date (UTC-midnight Timestamps) with columns
        neg_00_04 … pos_20_24 (EUR/MW/h). Returns empty DataFrame when
        date_from > date_to or no files are available.
    """
    if date_from > date_to:
        return pd.DataFrame()

    files = _collect_files("aFRR", date_from, date_to)
    if not files:
        return pd.DataFrame()

    frames = []
    for f in files:
        print(f"  Downloading {f['fileName']} …")
        content = _download_file(f["fileName"])
        frames.append(_parse_afrr_capacity_results(content))

    df = pd.concat(frames)
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()

    ts_from = pd.Timestamp(date_from, tz="UTC")
    ts_to = pd.Timestamp(date_to, tz="UTC")
    return df.loc[ts_from:ts_to]


def last_day_of_month(d: date) -> date:
    """Return the last day of the month containing d."""
    next_month = d.replace(day=28) + timedelta(days=4)
    return next_month.replace(day=1) - timedelta(days=1)
