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


# ---------------------------------------------------------------------------
# AIS cargo inversion (weeks 3-4)
#
# Two estimators for the energy delivered by a berth call, combined by the
# filter as a jump-size prior:
#
#   1. Capacity-class (PRIMARY): AIS dimensions -> vessel class -> capacity,
#      x fill x voyage boil-off x heel. Tighter than draught for the common
#      full-discharge case.
#   2. Draught-delta (SECONDARY): observed draught change x tonnes-per-cm.
#      Physically subtle: ships take on ballast as cargo comes off (propeller
#      immersion/stability), so the net draught change understates cargo mass:
#          TPC * delta = cargo_out - ballast_in
#      The ballast compensation ratio is large and uncertain (~0.5-0.7 of
#      cargo mass), and AIS draught is hand-entered and often stale. Use it to
#      detect PARTIAL discharges and as a sanity check, not as the primary.
# ---------------------------------------------------------------------------

# Seawater density and waterplane coefficient for TPC = Cw*L*B*rho/100 (t/cm).
SEAWATER_DENSITY = Bounded(1.025, 1.020, 1.028, "t per m3")
WATERPLANE_COEFF = Bounded(0.87, 0.82, 0.92, "dimensionless (LNG carriers)")

# Fraction of cargo mass compensated by ballast intake during discharge.
BALLAST_COMPENSATION = Bounded(0.60, 0.45, 0.75, "fraction of cargo mass")

# Loaded fill fraction of nominal capacity (98% filling limit, minus vapour).
LOADED_FILL = Bounded(0.975, 0.955, 0.985, "fraction of capacity")

# (loa_min, loa_max, beam_min, beam_max, capacity m^3 central, half-range)
# Standard LNG carrier classes; fallback scales by L*B against the 174k class.
VESSEL_CLASSES = (
    (330.0, 360.0, 52.0, 56.0, 265_000.0, 4_000.0),   # Q-Max
    (305.0, 330.0, 48.0, 52.0, 213_000.0, 6_000.0),   # Q-Flex
    (283.0, 305.0, 44.5, 48.0, 172_000.0, 10_000.0),  # modern standard 155-180k
    (265.0, 283.0, 40.0, 44.5, 145_000.0, 9_000.0),   # older conventional 135-150k
    (180.0, 265.0, 26.0, 40.0, 70_000.0, 35_000.0),   # midscale / med-max
)


def capacity_from_dimensions(loa: float, beam: float) -> Bounded | None:
    """Nominal tank capacity (m^3) from AIS dimensions; None if not carrier-sized."""
    if not loa or not beam or loa < 150:
        return None
    for lo, hi, blo, bhi, cap, half in VESSEL_CLASSES:
        if lo <= loa < hi and blo <= beam < bhi:
            return Bounded(cap, cap - half, cap + half, "m3")
    ref_cap, ref_lb = 172_000.0, 295.0 * 46.0  # scale off the standard class
    cap = ref_cap * (loa * beam) / ref_lb
    return Bounded(cap, cap * 0.75, cap * 1.25, "m3")


def full_discharge_energy(
    capacity_m3: Bounded, voyage_days: float = 12.0
) -> tuple[float, float]:
    """(GWh central, GWh half-range) for a full discharge of a given vessel class."""
    cargo_m3 = capacity_m3.value * LOADED_FILL.value * (1 - BOIL_OFF_PER_DAY.value * voyage_days)
    delivered_m3 = cargo_m3 * (1 - HEEL_FRACTION.value)
    central = delivered_m3 * ENERGY_PER_M3.value / 1e3  # MWh -> GWh
    rel = (
        (capacity_m3.half_range / capacity_m3.value) ** 2
        + (LOADED_FILL.half_range / LOADED_FILL.value) ** 2
        + (HEEL_FRACTION.half_range * 1.0) ** 2
        + (ENERGY_PER_M3.half_range / ENERGY_PER_M3.value) ** 2
        + (BOIL_OFF_PER_DAY.half_range * voyage_days) ** 2
    ) ** 0.5
    return central, central * rel


def tonnes_per_cm(loa: float, beam: float) -> Bounded:
    """TPC (t per cm immersion) from waterplane area."""
    tpc = WATERPLANE_COEFF.value * loa * beam * SEAWATER_DENSITY.value / 100
    rel = WATERPLANE_COEFF.half_range / WATERPLANE_COEFF.value
    return Bounded(tpc, tpc * (1 - rel), tpc * (1 + rel), "t per cm")


def draught_delta_to_energy(loa: float, beam: float, delta_m: float) -> tuple[float, float]:
    """(GWh central, GWh half-range) implied by an observed draught DECREASE of
    delta_m during a berth call, after undoing ballast compensation."""
    tpc = tonnes_per_cm(loa, beam)
    net_tonnes = tpc.value * delta_m * 100
    cargo_tonnes = net_tonnes / (1 - BALLAST_COMPENSATION.value)
    central = cargo_tonnes * GCV_MASS.value / 1e3  # MWh -> GWh
    rel = (
        (tpc.half_range / tpc.value) ** 2
        + (BALLAST_COMPENSATION.half_range / (1 - BALLAST_COMPENSATION.value)) ** 2
        + (GCV_MASS.half_range / GCV_MASS.value) ** 2
    ) ** 0.5
    return central, central * rel
