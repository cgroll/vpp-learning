"""Project paths configuration.

All paths are resolved relative to the project root, making scripts runnable
from any working directory. Add a @property for each new data file introduced
in the pipeline.
"""

from pathlib import Path


class ProjPaths:
    """Centralized project paths.

    The root is inferred from the location of this file (pkg/), so scripts
    run correctly regardless of the working directory they are invoked from.
    """

    def __init__(self):
        self._pkg_path = Path(__file__).resolve().parent  # pkg/
        self._project_path = self._pkg_path.parent  # project root

    # ------------------------------------------------------------------ #
    # Top-level directories                                                #
    # ------------------------------------------------------------------ #

    @property
    def project_path(self) -> Path:
        """Root project directory."""
        return self._project_path

    @property
    def pkg_path(self) -> Path:
        """Source package directory (pkg/)."""
        return self._pkg_path

    @property
    def pipeline_path(self) -> Path:
        """Pipeline scripts directory."""
        return self._project_path / "pipeline"

    # ------------------------------------------------------------------ #
    # Data directories                                                     #
    # ------------------------------------------------------------------ #

    @property
    def data_path(self) -> Path:
        """Main data directory."""
        return self._project_path / "data"

    @property
    def downloads_path(self) -> Path:
        """Raw downloaded data."""
        return self.data_path / "downloads"

    @property
    def processed_data_path(self) -> Path:
        """Processed/transformed data."""
        return self.data_path / "processed"

    # ------------------------------------------------------------------ #
    # Output directories                                                   #
    # ------------------------------------------------------------------ #

    @property
    def output_path(self) -> Path:
        """Generated outputs root."""
        return self._project_path / "output"

    @property
    def images_path(self) -> Path:
        """Chart/figure images saved by pipeline scripts."""
        return self.output_path / "images"

    @property
    def reports_path(self) -> Path:
        """Report files."""
        return self.output_path / "reports"

    # ------------------------------------------------------------------ #
    # SMARD data files                                                     #
    # ------------------------------------------------------------------ #

    @property
    def smard_downloads_path(self) -> Path:
        """Directory for all SMARD raw downloads."""
        return self.downloads_path / "smard"

    @property
    def smard_prices_file(self) -> Path:
        """DE-LU day-ahead electricity prices, hourly (EUR/MWh)."""
        return self.smard_downloads_path / "prices_de_lu.parquet"

    @property
    def regelleistung_downloads_path(self) -> Path:
        """Directory for all regelleistung.net raw downloads."""
        return self.downloads_path / "regelleistung"

    @property
    def fcr_prices_file(self) -> Path:
        """FCR capacity prices per 4h block (EUR/MW/h). Six block columns per day."""
        return self.regelleistung_downloads_path / "fcr_prices.parquet"

    @property
    def afrr_capacity_prices_file(self) -> Path:
        """aFRR capacity prices, 4h blocks, EUR/MW/h. Columns: neg_00_04…pos_20_24."""
        return self.regelleistung_downloads_path / "afrr_capacity_prices.parquet"

    # ------------------------------------------------------------------ #
    # Processed data files                                                 #
    # ------------------------------------------------------------------ #

    @property
    def dispatch_schedules_file(self) -> Path:
        """Dispatch schedules for all scenarios: [timestamp, c, d, soc, scenario_id]."""
        return self.processed_data_path / "dispatch_schedules.parquet"

    @property
    def forecasts_file(self) -> Path:
        """Day-ahead price forecasts for test period [timestamp, naive, ridge, lgbm]."""
        return self.processed_data_path / "forecasts.parquet"

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def ensure_directories(self) -> None:
        """Create all standard directories if they do not yet exist."""
        dirs = [
            self.downloads_path,
            self.smard_downloads_path,
            self.regelleistung_downloads_path,
            self.processed_data_path,
            self.images_path,
            self.reports_path,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
