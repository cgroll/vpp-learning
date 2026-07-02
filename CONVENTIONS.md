# VPP Learning — Conventions

## Project structure

```
vpp-learning/
├── vpp/                     # Python package (shared utilities)
│   ├── __init__.py
│   └── paths.py             # Centralized path configuration
├── pipeline/                # Pipeline scripts
│   ├── 01_download_*.py     # Data acquisition (no charts)
│   ├── 07_compute_*.py      # Computation stages (solvers/models → processed parquet)
│   └── 02_*.py, 03_*.py … # Analysis scripts → book notebooks
├── book/                    # MyST Jupyter Book source
│   ├── notebooks/           # Executed .ipynb files (DVC outputs, tracked in git)
│   ├── markdown/            # Static hand-written content
│   └── myst.yml             # Book config and table of contents
├── data/
│   ├── downloads/           # Raw downloaded data (git-ignored, DVC-tracked)
│   └── processed/           # Processed/transformed data (git-ignored, DVC-tracked)
├── output/
│   ├── images/              # Chart images (DVC outputs, tracked in git)
│   └── reports/             # Report files
├── dvc.yaml                 # Pipeline definition
├── dvc.lock                 # DVC state (commit this)
└── pyproject.toml           # Dependencies managed by uv
```

## Tools

| Tool | Purpose |
|------|---------|
| **uv** | Package and environment management |
| **DVC** | Pipeline orchestration (`dvc repro`, `dvc status`) |
| **jupytext** | Execute `.py` analysis scripts → `.ipynb` notebooks |
| **MyST / mystmd** | Build the HTML book from notebooks and markdown |
| **ruff** | Linting and formatting (`make lint`, `make format`) |
| **pyright** | Type checking (`make typecheck`) |

## DVC primer

DVC is a Make-inspired pipeline manager. It reads `dvc.yaml` in the project root.

### Core concept: stages

A stage declares how to produce outputs from inputs:

```yaml
stages:
  process_my_analysis:
    cmd: |
      set -e
      MPLBACKEND=Agg uv run jupytext --to notebook --execute --set-kernel python3 --output book/notebooks/03_my_analysis.ipynb pipeline/03_my_analysis.py
      uv run python -c "import nbformat; nb = nbformat.read('book/notebooks/03_my_analysis.ipynb', as_version=4); nb.cells = [c for c in nb.cells if not (c.cell_type == 'raw' and 'jupytext' in c.source)]; nbformat.write(nb, 'book/notebooks/03_my_analysis.ipynb')"
    deps:
      - pipeline/03_my_analysis.py
      - data/downloads/my_data.parquet
    outs:
      - book/notebooks/03_my_analysis.ipynb:
          cache: false        # tracked in git, not DVC cache
      - output/images/03_my_chart.png:
          cache: false        # tracked in git, not DVC cache
```

DVC compares **file hashes** in `dvc.lock`: if all outputs are current, the stage is skipped.

### Data download stages

Downloaded data should use default `cache: true` (omit the flag). DVC only re-runs a stage when its `deps` change. To force a fresh download: `dvc repro --force download_my_data`.

```yaml
stages:
  download_my_data:
    cmd: uv run python pipeline/01_download_my_data.py
    outs:
      - data/downloads/my_data.parquet   # cached by DVC, not in git
```

### Computation stages

Computation stages (`07_*`, `08_*`, …) run expensive solvers or fit models once and write
the results to `data/processed/`. Analysis notebooks then read from these files rather
than re-running solvers. This cleanly separates computation from visualization and lets
multiple notebooks share the same results without duplication.

- No jupytext header — run directly as `python pipeline/07_compute_*.py`.
- No charts or visualizations — produce parquet files only.
- List all `vpp/` modules they depend on as DVC deps so any solver/model change triggers
  recomputation.
- Output files go to `data/processed/` with DVC caching (default `cache: true`).

```yaml
stages:
  compute_my_results:
    cmd: uv run python pipeline/07_compute_my_results.py
    deps:
      - pipeline/07_compute_my_results.py
      - vpp/dispatch.py          # list any vpp/ modules used
      - vpp/scenarios.py
      - data/downloads/my_data.parquet
    outs:
      - data/processed/my_results.parquet   # cached by DVC, not in git
```

Analysis scripts that consume this file declare it as a dep:

```yaml
  my_analysis:
    cmd: |
      set -e
      MPLBACKEND=Agg uv run jupytext ...
    deps:
      - pipeline/03_my_analysis.py
      - data/processed/my_results.parquet   # ← computation stage output
    outs:
      - book/notebooks/03_my_analysis.ipynb:
          cache: false
```

### Running the pipeline

```bash
dvc status          # show what's out of date (dry-run equivalent)
dvc repro           # run all outdated stages
dvc repro --force   # re-run everything unconditionally
dvc repro <stage>   # run one specific stage and its dependencies
```

Or via Make shortcuts:

```bash
make run       # dvc repro
make status    # dvc status
make serve     # myst start (local book preview)
```

## Pipeline conventions

### Three types of scripts

**1. Download scripts** (`01_*`)
- No charts or visualizations.
- Read/write data files only.
- No jupytext header — run directly as `python pipeline/01_*.py`.

**2. Computation scripts** (`07_*`, `08_*`, …)
- Run solvers or fit models; write results to `data/processed/` as parquet.
- No charts or visualizations.
- No jupytext header — run directly as `python pipeline/07_*.py`.
- Multiple analysis notebooks can read from the same computed file.

**3. Analysis scripts** (`02_*`, `03_*`, …)
- Jupytext `# %%` cell markers and a jupytext/kernelspec header (see below).
- Save all figures to `output/images/` via `fig.savefig()`.
- Use MyST `{figure}` directives in `# %% [markdown]` cells.
- DVC runs them via jupytext → produces an executed `.ipynb` in `book/notebooks/`.
- Read pre-computed data from `data/processed/` rather than running solvers inline.

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

The path `../../output/images/` is relative to `book/notebooks/` where the `.ipynb` lives.

Naming convention: `<script_number>_<descriptive_name>.png`.

### DVC stage for analysis scripts

See "Core concept: stages" above. The post-processing one-liner strips a raw jupytext metadata cell that MyST doesn't recognize. `MPLBACKEND=Agg` makes `plt.show()` a no-op headlessly.

## Path conventions

All scripts must be runnable from any working directory. Use `ProjPaths` from `vpp/paths.py`:

```python
from vpp.paths import ProjPaths

paths = ProjPaths()

df = pd.read_parquet(paths.downloads_path / "my_data.parquet")
fig.savefig(paths.images_path / "03_chart.png")
```

Key paths:

| Property | Directory |
|----------|-----------|
| `paths.downloads_path` | `data/downloads/` |
| `paths.processed_data_path` | `data/processed/` |
| `paths.images_path` | `output/images/` |
| `paths.pipeline_path` | `pipeline/` |

Add a property to `vpp/paths.py` for every new data file:
```python
@property
def my_new_file(self) -> Path:
    """One-line description."""
    return self.downloads_path / "my_data.parquet"
```

## Adding a new pipeline stage

1. Write the script in `pipeline/`.
2. Add a property to `vpp/paths.py` for every new data file.
3. Add a stage to `dvc.yaml` with `cmd`, `deps`, and `outs`.
   - Download stage: outputs in `data/downloads/`, no `cache: false`.
   - Computation stage: outputs in `data/processed/`, no `cache: false`; list all
     `vpp/` modules used as deps.
   - Analysis stage: use the jupytext command template; `cache: false` on notebooks
     and images; declare any `data/processed/` files it reads as deps.
4. Add the notebook to the `toc` in `book/myst.yml` (analysis stages only).

## Git conventions

| Tracked in git | Not tracked (DVC-managed or ignored) |
|----------------|--------------------------------------|
| `pipeline/*.py` source files | `data/downloads/*` |
| `book/notebooks/*.ipynb` generated notebooks | `data/processed/*` |
| `output/images/*.png` generated charts | `.venv/` |
| `book/markdown/*.md` static content | `.dvc/cache/` |
| `dvc.yaml` pipeline definition | |
| `dvc.lock` pipeline state | |

Notebooks and images are tracked in git so the book can be rebuilt from git without re-running the pipeline (CI only runs `myst build`).

## Modeling conventions

Every model variant gets a row in a scenario comparison table (name, MAE, % vs baseline, valid flag). Narrative in a markdown doc explains each lever. Reference implementation: `/home/chris/research/flexa-challenge/`.

## Workflow

```
1. Write pipeline script in pipeline/
2. Add DVC stage in dvc.yaml
3. make run          ← dvc repro (execute pipeline)
4. make serve        ← preview book locally
5. /wrap-up          ← check pipeline, lint, docs; update PROJECT.md
6. git add / commit  ← commit notebooks + images + dvc.lock
7. git push          ← CI deploys to GitHub Pages
```
