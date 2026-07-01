# Contribution Conventions

This document describes the project structure and conventions.
It is written for human contributors and AI agents alike.

## Project Structure

```
project-root/
├── pkg/                     # Python package — shared utilities
│   ├── __init__.py
│   └── paths.py             # Centralized path configuration
├── pipeline/                # Data pipeline scripts
│   ├── 01_download_*.py     # Pure data acquisition (no charts)
│   └── 02_analyse_*.py      # Analysis scripts → become book notebooks
├── book/                    # MyST Jupyter Book source
│   ├── notebooks/           # Executed .ipynb files (produced by Snakemake)
│   ├── markdown/            # Static hand-written content
│   └── myst.yml             # Book configuration and table of contents
├── data/
│   ├── downloads/           # Raw downloaded data (git-ignored)
│   └── processed/           # Processed/transformed data (git-ignored)
├── output/
│   ├── images/              # Chart images saved by pipeline scripts
│   └── reports/             # Report files
├── Snakefile                # Pipeline definition (equivalent of dvc.yaml)
└── pyproject.toml           # Dependencies managed by uv
```

## Tools

| Tool | Purpose |
|------|---------|
| **uv** | Package and environment management |
| **Snakemake** | Pipeline orchestration (dependency-aware task runner) |
| **jupytext** | Execute `.py` analysis scripts → `.ipynb` notebooks |
| **MyST / mystmd** | Build the HTML book from notebooks and markdown |

## Snakemake Primer

Snakemake is a Make-inspired workflow manager written in Python.
It reads the `Snakefile` in the project root.

### Core concept: rules

A rule declares *how* to produce an output from inputs:

```python
rule my_stage:
    input:
        data = "data/downloads/raw.parquet",
        script = "pipeline/02_analyse.py",
    output:
        notebook = "book/notebooks/02_analyse.ipynb",
        img      = "output/images/02_chart.png",
    shell:
        "MPLBACKEND=Agg uv run jupytext ..."
```

Snakemake compares **file modification timestamps**: if all outputs are newer
than all inputs, the rule is skipped. This is the key difference from DVC —
there is no explicit lock file; timestamps drive incremental builds.

### `ancient()` for downloaded data

Downloaded files should not be re-fetched every run. Wrap their paths in
`ancient()` so Snakemake treats them as infinitely old and never triggers
re-download as long as the file exists:

```python
rule download_data:
    output:
        ancient("data/downloads/raw.parquet"),
    shell:
        "uv run python pipeline/01_download.py"
```

To force a fresh download: delete the file and re-run `snakemake`.

### Running the pipeline

```bash
snakemake -n          # dry-run: show what would execute
snakemake -j4         # run with up to 4 parallel jobs
snakemake <file>      # build one specific output
snakemake -R <rule>   # force-re-run a specific rule
snakemake --forceall  # re-run everything unconditionally
```

Or via Make shortcuts:

```bash
make run       # snakemake -j4
make dry-run   # snakemake -n
make serve     # myst start (local book preview)
```

### The `rule all` convention

The top of the Snakefile defines a pseudo-rule whose inputs are the final
targets. Running bare `snakemake` builds these:

```python
rule all:
    input:
        "book/notebooks/02_analyse_example.ipynb",
        "book/notebooks/03_another_analysis.ipynb",
```

## Pipeline Conventions

### Two types of scripts

**1. Pure data scripts** (`01_*`, `00_*`, …)
- No charts or visualizations.
- Read/write data files only.
- Registered in Snakemake with `ancient()` outputs to avoid re-downloading.
- Not converted to notebooks.

**2. Analysis scripts** (`02_*`, `03_*`, …)
- Use jupytext `# %%` cell markers and a jupytext/kernelspec header.
- Save all figures to `output/images/` via `fig.savefig()`.
- Use MyST `{figure}` directives in `# %% [markdown]` cells.
- Snakemake runs them via jupytext → produces an executed `.ipynb` in `book/notebooks/`.

### Jupytext header for analysis scripts

```python
# ---
# jupytext:
#   text_representation:
#     format_name: percent
# kernelspec:
#   display_name: Python 3
#   language: python
#   name: python3
# ---
```

### Saving figures and referencing them

```python
# %%
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(...)
fig.savefig(paths.images_path / "03_my_chart.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/03_my_chart.png
# :name: fig-03-my-chart
# Caption describing the figure.
# ```
```

The path `../../output/images/` is relative to `book/notebooks/` where the
generated `.ipynb` lives.

Naming convention: `<script_number>_<descriptive_name>.png`.

### Snakemake rule for analysis scripts

```python
rule process_my_analysis:
    input:
        script = "pipeline/03_my_analysis.py",
        data   = "data/downloads/raw.parquet",
    output:
        notebook = "book/notebooks/03_my_analysis.ipynb",
        img      = "output/images/03_my_chart.png",
    shell:
        """
        MPLBACKEND=Agg uv run jupytext --to notebook --execute \
            --set-kernel python3 \
            --output {output.notebook} {input.script} && \
        uv run python -c "
import nbformat
nb = nbformat.read('{output.notebook}', as_version=4)
nb.cells = [c for c in nb.cells
            if not (c.cell_type == 'raw' and 'jupytext' in c.source)]
nbformat.write(nb, '{output.notebook}')
"
        """
```

The post-processing step strips a raw jupytext metadata cell that MyST does
not recognize. The `MPLBACKEND=Agg` environment variable makes `plt.show()`
a no-op in headless mode.

## Path Conventions

All scripts must be runnable from any working directory. Use `ProjPaths`
from `pkg/paths.py`:

```python
from pkg.paths import ProjPaths

paths = ProjPaths()

df = pd.read_parquet(paths.example_raw_file)
fig.savefig(paths.images_path / "03_chart.png")
```

Key paths:

| Property | Directory |
|----------|-----------|
| `paths.data_path` | `data/` |
| `paths.downloads_path` | `data/downloads/` |
| `paths.processed_data_path` | `data/processed/` |
| `paths.images_path` | `output/images/` |
| `paths.pipeline_path` | `pipeline/` |

## Adding a New Pipeline Stage

1. **Write the script** in `pipeline/`.
2. **Add a property** to `pkg/paths.py` for every new data file:
   ```python
   @property
   def my_new_file(self) -> Path:
       """One-line description."""
       return self.downloads_path / "my_data.parquet"
   ```
3. **Add a rule** to `Snakefile` with `input`, `output`, and `shell`.
4. **Add the notebook** to `ANALYSIS_NOTEBOOKS` in `Snakefile` (if analysis).
5. **Add the notebook** to the `toc` in `book/myst.yml`.

## Git Conventions

| Tracked | Not tracked |
|---------|-------------|
| `pipeline/*.py` source files | `data/downloads/*` |
| `book/notebooks/*.ipynb` generated notebooks | `data/processed/*` |
| `output/images/*.png` generated charts | `.venv/` |
| `book/markdown/*.md` static content | |

The `.ipynb` notebooks and images are tracked so the book can be rebuilt
from git without re-running the pipeline (CI only runs `myst build`).

## Workflow Summary

```
1. Write pipeline script in pipeline/
2. Add Snakemake rule in Snakefile
3. make run          ← execute pipeline
4. make serve        ← preview book locally
5. git add / commit  ← commit notebooks + images
6. git push          ← CI deploys to GitHub Pages
```
