"""Download example dataset.

Pure data script — no visualizations. Snakemake runs this once; the output
is marked ancient() so it is not re-downloaded on subsequent runs unless
explicitly deleted.

Replace this with your actual data acquisition logic.
"""

import pandas as pd
import numpy as np
from pkg.paths import ProjPaths

paths = ProjPaths()
paths.ensure_directories()

# ── Simulate a download ───────────────────────────────────────────────────────
rng = np.random.default_rng(42)
n = 365 * 24  # one year of hourly data
timestamps = pd.date_range("2023-01-01", periods=n, freq="h")
df = pd.DataFrame(
    {
        "timestamp": timestamps,
        "value": rng.normal(loc=50, scale=15, size=n).clip(0),
        "category": rng.choice(["A", "B", "C"], size=n),
    }
)

df.to_parquet(paths.example_raw_file, index=False)
print(f"Saved {len(df):,} rows → {paths.example_raw_file}")
