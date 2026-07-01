# Snakefile — project pipeline
#
# Concepts used here:
#
#   ancient(path)   — if the file already exists, treat it as infinitely old so
#                     this rule is skipped. Ideal for downloaded data that
#                     should not be re-fetched every run.
#
#   rule all        — pseudo-rule whose `input` lists the final targets.
#                     `snakemake` (no arguments) builds everything in this list.
#
# Common commands:
#   snakemake -n              dry-run: show what would be executed
#   snakemake -j4             run with 4 parallel jobs
#   snakemake <target>        build one specific output file
#   snakemake --forcerun <rule>   force a rule to re-run even if outputs exist

# ---------------------------------------------------------------------------
# Project-wide settings
# ---------------------------------------------------------------------------

# List every notebook that the book should contain.
# Extend this list when you add a new analysis script.
ANALYSIS_NOTEBOOKS = [
    "book/notebooks/02_analyse_example.ipynb",
]

# ---------------------------------------------------------------------------
# Default target
# ---------------------------------------------------------------------------

rule all:
    input:
        ANALYSIS_NOTEBOOKS

# ---------------------------------------------------------------------------
# Download rules
# ---------------------------------------------------------------------------
# Use ancient() on outputs so that existing downloaded files are never
# re-fetched.  Delete the file manually to force a fresh download.

rule download_example:
    output:
        ancient("data/downloads/example_data.parquet"),
    shell:
        "uv run python pipeline/01_download_example.py"

# ---------------------------------------------------------------------------
# Analysis / notebook rules
# ---------------------------------------------------------------------------
# Pattern for every analysis script:
#   1. jupytext executes the .py script and writes an .ipynb with outputs
#   2. A Python one-liner strips the raw jupytext metadata cell that MyST
#      does not understand
#
# Add one rule per analysis script and list all image outputs explicitly so
# Snakemake can track them as dependencies of downstream rules.

rule process_example:
    input:
        script  = "pipeline/02_analyse_example.py",
        data    = "data/downloads/example_data.parquet",
    output:
        notebook = "book/notebooks/02_analyse_example.ipynb",
        img1     = "output/images/02_daily_average.png",
        img2     = "output/images/02_category_dist.png",
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

# ---------------------------------------------------------------------------
# Utility rules
# ---------------------------------------------------------------------------

# Re-run a single stage by name:  snakemake -R download_example
# Force all:                       snakemake --forceall
