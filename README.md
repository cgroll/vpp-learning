# Project Book Template

A template for research projects that publish a [MyST](https://mystmd.org/) Jupyter Book
to GitHub Pages. The data pipeline is managed by [Snakemake](https://snakemake.readthedocs.io/);
dependencies are managed by [uv](https://docs.astral.sh/uv/).

## Starting a new project from this template

### Pick your names up front

You need two names:

| What | Example | Rule |
|------|---------|------|
| **Repository / folder name** | `financial-market-returns` | Chosen on GitHub when you create the repo — becomes the local folder name after cloning |
| **Python package abbreviation** | `fmr` | Short acronym you pick yourself; used in every `import` statement |

The abbreviation is the equivalent of `woe` in `world-of-energy`. It must be a
valid Python identifier (letters, digits, underscores) — shorter is better.

### 1. Create the repository on GitHub

Click **Use this template → Create a new repository** at the top of this page.
Name it (e.g. `financial-market-returns`) and click **Create repository**.

### 2. Clone and initialize

```bash
git clone https://github.com/your-username/financial-market-returns.git
cd financial-market-returns
python init_project.py
```

The script reads the project name from the git remote automatically and asks
only for the package abbreviation. It renames `pkg/`, updates all references
in `pyproject.toml`, `Snakefile`, and the pipeline scripts, commits the
result, and removes itself.

### 3. Set up the environment

```bash
# Install uv if you haven't already — https://docs.astral.sh/uv/
curl -LsSf https://astral.sh/uv/install.sh | sh

uv sync
```

### 4. Verify the example pipeline

```bash
make dry-run   # preview what would run
make run       # execute the example pipeline
make serve     # open http://localhost:3000 — live book preview
```

Once everything works, remove the example files and start your own pipeline:

```bash
rm pipeline/01_download_example.py pipeline/02_analyse_example.py
# Remove the example rule from Snakefile and the notebook entry from book/myst.yml
```

### 5. Enable GitHub Pages

In your repository: **Settings → Pages → Source → GitHub Actions**.

Every push to `main` will build and deploy the book automatically.
Pull requests run only the build check.

## Project layout

```
project-root/
├── <abbrev>/            # Python package — renamed by init_project.py
│   └── paths.py         # Centralized path config
├── pipeline/            # Pipeline scripts
│   ├── 01_download_*    # Data acquisition
│   └── 02_analyse_*     # Analysis → notebook
├── book/                # MyST book source
│   ├── notebooks/       # Executed notebooks (Snakemake output)
│   ├── markdown/        # Static content
│   └── myst.yml         # TOC and site settings
├── data/                # Git-ignored data
├── output/images/       # Figures (tracked in git)
├── Snakefile            # Pipeline DAG
└── contribution_conventions.md   # Detailed conventions for contributors/AI
```

See [contribution_conventions.md](contribution_conventions.md) for full details on
adding pipeline stages, writing analysis scripts, and Snakemake usage.

## Common Snakemake commands

| Command | Effect |
|---------|--------|
| `snakemake -n` | Dry run — show what would execute |
| `snakemake -j4` | Run pipeline (4 parallel jobs) |
| `snakemake -R <rule>` | Force-re-run a specific rule |
| `snakemake <file>` | Build one specific output file |
| `snakemake --forceall` | Re-run everything unconditionally |
