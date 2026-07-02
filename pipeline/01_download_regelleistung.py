"""Download FCR and aFRR capacity prices from regelleistung.net.

Downloads monthly RESULT_OVERVIEW Excel files from the CRDS API and extracts
Germany settlement/marginal capacity prices per 4-hour delivery block.

Incremental: reads the last delivery_date from an existing parquet file and
fetches only newer months, then re-downloads the current (partial) month.
Safe to re-run; DVC will skip if the output is already current.

Force a full re-download by deleting the output files and running:
    dvc repro --force download_regelleistung
"""

from datetime import date

import pandas as pd

from vpp.paths import ProjPaths
from vpp.regelleistung import (
    AFRR_START_DATE,
    FCR_START_DATE,
    download_afrr_capacity_prices,
    download_fcr_prices,
)


def _first_day_of_month(d: date) -> date:
    return d.replace(day=1)


def _incremental_start(existing_file: object, default_start: date) -> date:
    """Return the start date for the next download run.

    Always re-downloads the last observed month so that partial months get
    refreshed on subsequent runs.
    """
    from pathlib import Path

    path = Path(str(existing_file))
    if not path.exists():
        return default_start

    df = pd.read_parquet(path)
    if df.empty:
        return default_start

    last_date = df.index.max().date()
    return _first_day_of_month(last_date)


def _update_parquet(
    output_file: object,
    new_df: pd.DataFrame,
) -> None:
    """Merge new_df with existing parquet, deduplicate, and write back."""
    from pathlib import Path

    path = Path(str(output_file))
    if path.exists():
        existing = pd.read_parquet(path)
        combined = pd.concat([existing, new_df])
    else:
        combined = new_df

    combined = combined[~combined.index.duplicated(keep="last")]
    combined = combined.sort_index()
    combined.to_parquet(path)
    print(f"Saved {len(combined)} rows to {path}.")


def main() -> None:
    paths = ProjPaths()
    paths.ensure_directories()

    today = date.today()

    # ------------------------------------------------------------------ #
    # FCR                                                                  #
    # ------------------------------------------------------------------ #
    fcr_start = _incremental_start(paths.fcr_prices_file, FCR_START_DATE)
    fcr_end = today

    if fcr_start > fcr_end:
        print("FCR: already up to date.")
    else:
        print(f"FCR: downloading {fcr_start} → {fcr_end} …")
        fcr_df = download_fcr_prices(fcr_start, fcr_end)
        if not fcr_df.empty:
            _update_parquet(paths.fcr_prices_file, fcr_df)
            print(f"FCR: {len(fcr_df)} delivery-day rows added/refreshed.")
        else:
            print("FCR: no new data returned.")

    # ------------------------------------------------------------------ #
    # aFRR capacity                                                        #
    # ------------------------------------------------------------------ #
    afrr_start = _incremental_start(paths.afrr_capacity_prices_file, AFRR_START_DATE)
    afrr_end = today

    if afrr_start > afrr_end:
        print("aFRR capacity: already up to date.")
    else:
        print(f"aFRR capacity: downloading {afrr_start} → {afrr_end} …")
        afrr_df = download_afrr_capacity_prices(afrr_start, afrr_end)
        if not afrr_df.empty:
            _update_parquet(paths.afrr_capacity_prices_file, afrr_df)
            print(f"aFRR capacity: {len(afrr_df)} delivery-day rows added/refreshed.")
        else:
            print("aFRR capacity: no new data returned.")


if __name__ == "__main__":
    main()
