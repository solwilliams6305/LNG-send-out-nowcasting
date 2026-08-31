"""Physical constants for the LNG inversion layer, with uncertainty ranges.

Every conversion in the pipeline should carry its uncertainty explicitly — the
point of the project is calibrated distributions, and these parameters are where
much of the honest error lives. Values are industry-standard ranges (GIIGNL
Annual Report conventions; cross-check against ALSI's own dtmi/dtrs implied
conversion per facility once data is flowing — cargo composition varies by
origin, so per-terminal empirical calibration in W5 supersedes these defaults).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Bounded:
    """A physical parameter with a central value and honest hard-ish bounds."""

    value: float
    lo: float
    hi: float
    unit: str

    @property
    def half_range(self) -> float:
        return (self.hi - self.lo) / 2


# Density of LNG as loaded (composition-dependent: lean Pacific vs rich Atlantic).
LNG_DENSITY = Bounded(0.450, 0.420, 0.470, "t per m3 LNG")

# Gross calorific value per tonne of LNG.
GCV_MASS = Bounded(15.2, 14.9, 16.0, "MWh per t LNG")

# Energy per cubic metre of LNG — the ALSI inventory (10^3 m^3) -> GWh conversion.
# value = density * GCV; treated as its own parameter so it can be calibrated
# per terminal against observed inventory-delta vs send-out residuals.
ENERGY_PER_M3 = Bounded(6.84, 6.30, 7.30, "MWh per m3 LNG")

# Laden-voyage boil-off, fraction of cargo per day (modern membrane carriers
# with reliquefaction sit at the low end; older steam ships higher).
BOIL_OFF_PER_DAY = Bounded(0.00125, 0.0008, 0.0018, "fraction of cargo per day")

# Heel retained on discharge (cargo kept cold for the ballast leg), fraction.
HEEL_FRACTION = Bounded(0.03, 0.015, 0.06, "fraction of capacity")

# Liquid-to-gas expansion ratio (context only; energy accounting avoids it).
EXPANSION_RATIO = 600.0


def inventory_to_gwh(inventory_1e3m3: float, energy_per_m3: Bounded = ENERGY_PER_M3) -> tuple[float, float]:
    """ALSI tank inventory (10^3 m^3 LNG) -> (GWh central, GWh half-range)."""
    m3 = inventory_1e3m3 * 1e3
    central = m3 * energy_per_m3.value / 1e3  # MWh -> GWh
    half = m3 * energy_per_m3.half_range / 1e3
    return central, half
