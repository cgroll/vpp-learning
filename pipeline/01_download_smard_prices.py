"""Download DE-LU day-ahead electricity prices from SMARD API.

Incremental: reads the last timestamp from an existing parquet file and fetches
only newer data, then concatenates and writes back. Safe to re-run; DVC will
skip if the output is already current.

Force a full re-download by deleting the output file and running:
    dvc repro --force download_smard_prices
"""

from datetime import datetime, timedelta, timezone

import pandas as pd

from vpp.paths import ProjPaths
from vpp.smard import (
    DEFAULT_START_DATE,
    Region,
    Resolution,
    Variable,
    download_smard_data,
)


def main() -> None:
    paths = ProjPaths()
    paths.ensure_directories()

    output_file = paths.smard_prices_file

    if output_file.exists():
        existing = pd.read_parquet(output_file)
        last_ts = existing.index.max().to_pydatetime()
        start_time = last_ts + timedelta(hours=1)
        print(
            f"Found existing data ({len(existing)} rows). "
            f"Incremental from {start_time}."
        )
    else:
        existing = None
        start_time = DEFAULT_START_DATE
        print(f"No existing data. Full download from {start_time}.")

    if start_time >= datetime.now(tz=timezone.utc):
        print("Data is already up to date. Nothing to download.")
        return

    try:
        new_df = download_smard_data(
            region=Region.DE_LU.value,
            resolution=Resolution.HOUR.value,
            variable=Variable.PRICE_DE_LU.value,
            variable_name="price_de_lu",
            start_time=start_time,
        )
    except RuntimeError as e:
        if "No data available after" in str(e):
            print("No new data available from API.")
            return
        raise

    print(f"Downloaded {len(new_df)} new records.")

    if existing is not None and not existing.empty:
        combined = pd.concat([existing, new_df])
        combined = combined.sort_index()
        combined = combined[~combined.index.duplicated(keep="last")]
    else:
        combined = new_df

    # Reindex to a gapless hourly UTC range; missing hours become NaN
    full_index = pd.date_range(
        start=combined.index.min(),
        end=combined.index.max(),
        freq="h",
        tz="UTC",
    )
    combined = combined.reindex(full_index)
    combined.index.name = "timestamp"

    combined.to_parquet(output_file)
    print(f"Saved {len(combined)} rows to {output_file}.")


if __name__ == "__main__":
    main()
