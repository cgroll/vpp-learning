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
# # Example Analysis
#
# This notebook demonstrates the standard analysis script pattern:
# - load processed data via `ProjPaths`
# - produce figures saved to `output/images/`
# - reference figures with MyST `{figure}` directives in markdown cells
#
# Snakemake runs this script with jupytext to produce the executed `.ipynb`
# in `book/notebooks/`. The book build reads the pre-executed notebook —
# it does **not** re-execute it.

# %%
import pandas as pd
import matplotlib.pyplot as plt
from pkg.paths import ProjPaths

paths = ProjPaths()
paths.ensure_directories()

df = pd.read_parquet(paths.example_raw_file)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.set_index("timestamp")

# %% [markdown]
# ## Daily averages

# %%
daily = df["value"].resample("D").mean()

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(daily.index, daily.values, linewidth=0.8)
ax.set_xlabel("Date")
ax.set_ylabel("Value")
ax.set_title("Daily average value")
fig.tight_layout()
fig.savefig(paths.images_path / "02_daily_average.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/02_daily_average.png
# :name: fig-02-daily-average
# Daily average of the example dataset over the full year.
# ```

# %% [markdown]
# ## Distribution by category

# %%
fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
for ax, (cat, grp) in zip(axes, df.groupby("category")):
    ax.hist(grp["value"], bins=40, edgecolor="white", linewidth=0.3)
    ax.set_title(f"Category {cat}")
    ax.set_xlabel("Value")
axes[0].set_ylabel("Count")
fig.tight_layout()
fig.savefig(paths.images_path / "02_category_dist.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/02_category_dist.png
# :name: fig-02-category-dist
# Value distribution split by category.
# ```
