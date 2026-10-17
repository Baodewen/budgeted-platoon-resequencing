from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DEFAULT_FIT = {"head": (-0.022, 0.095, 0.503), "middle": (-0.044, 0.236, 0.462), "tail": (-0.045, 0.301, 0.328)}


@dataclass
class EnergyModel:
    battery_kwh: float = 85.0
    soc_start: float = 0.8
    soc_stop: float = 0.2
    delta_kwh_per_km: float = 0.211
    mass_kg: float = 29400.0
    cruise_kmh: float = 105.0
    frontal_area: float = 10.7
    rho_air: float = 1.2
    r0: float = 0.006
    length_m: float = 5.8
    gap_ratio: float = 0.2
    calib_M: float = 0.437
    K1: float = 0.014
    K2: float = 0.366
    g: float = 9.81
    fit: dict = field(default_factory=lambda: dict(DEFAULT_FIT))

    @property
    def usable_kwh(self) -> float:
        return self.battery_kwh * (self.soc_start - self.soc_stop)

    @property
    def gap_m(self) -> float:
        return self.gap_ratio * self.length_m

    def saving_fraction(self, role: str, speed_kmh: float | None = None) -> float:
        a, b, c = self.fit[role]
        v = (speed_kmh if speed_kmh is not None else self.cruise_kmh) / 3.6
        num = self.calib_M * (a * np.log(self.gap_m) + b) * self.length_m ** c
        cd_solo = self.K1 * self.length_m + self.K2
        den = 1.0 + self.r0 * self.mass_kg * self.g / (0.5 * self.rho_air * v ** 2 * self.frontal_area * cd_solo)
        return float(num / den)

    def position_coefficients(self, n: int) -> np.ndarray:
        if n < 2:
            raise ValueError("platoon needs at least two vehicles")
        e = np.full(n, self.saving_fraction("middle"))
        e[0] = self.saving_fraction("head")
        e[-1] = self.saving_fraction("tail")
        return e

    def soc_drop_per_km(self, n: int) -> np.ndarray:
        return self.delta_kwh_per_km / self.usable_kwh * (1.0 - self.position_coefficients(n))

    def fixed_order_distance(self, n: int) -> float:
        return self.usable_kwh / (self.delta_kwh_per_km * (1.0 - self.position_coefficients(n)[0]))

