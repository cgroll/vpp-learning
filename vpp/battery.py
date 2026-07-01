"""Battery physical parameters."""

import numpy as np
from pydantic import BaseModel, model_validator


class BatteryParams(BaseModel, frozen=True):
    capacity_kwh: float
    power_kw: float
    eta_c: float = 1.0
    eta_d: float = 1.0
    deg_cost_per_cycle: float = 0.0

    @model_validator(mode="after")
    def _check_ranges(self) -> "BatteryParams":
        assert self.capacity_kwh > 0, "capacity_kwh must be positive"
        assert self.power_kw > 0, "power_kw must be positive"
        assert 0 < self.eta_c <= 1.0, "eta_c must be in (0, 1]"
        assert 0 < self.eta_d <= 1.0, "eta_d must be in (0, 1]"
        assert self.deg_cost_per_cycle >= 0, "deg_cost_per_cycle must be non-negative"
        return self

    @classmethod
    def from_eta_rt(
        cls,
        capacity_kwh: float,
        power_kw: float,
        eta_rt: float,
        deg_cost: float = 0.0,
    ) -> "BatteryParams":
        """Construct from round-trip efficiency, splitting losses symmetrically."""
        eta = float(np.sqrt(eta_rt))
        return cls(
            capacity_kwh=capacity_kwh,
            power_kw=power_kw,
            eta_c=eta,
            eta_d=eta,
            deg_cost_per_cycle=deg_cost,
        )
