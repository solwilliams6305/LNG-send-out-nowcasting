"""Registry of NW-European LNG terminals and their identifiers across data sources.

ENTSOG identifiers (operator_key, point_key, direction) were verified live against
https://transparency.entsog.eu/api/v1/operatorpointdirections on 2026-09-01:
every (operator, point, entry) triple below returned real daily Physical Flow data,
except where noted.

ALSI facilities are NOT hardcoded: EIC codes are resolved at runtime from
/api/about?show=listing (needs the free key) by name pattern — see
scripts/bootstrap_alsi.py — because GIE warns that EICs change over time.

Coordinates are approximate terminal locations for seeding AIS berth polygons in
weeks 3-4. Verify each against observed ship tracks before using in anger.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PointDirection:
    """One ENTSOG operator-point-direction (a TSO's measurement of one grid point)."""

    operator_key: str
    point_key: str
    direction: str = "entry"


@dataclass(frozen=True)
class Terminal:
    slug: str
    name: str
    country: str
    tier: str  # "core" | "secondary"
    entsog: tuple[PointDirection, ...]
    alsi_name_patterns: tuple[str, ...]  # case-insensitive substrings of ALSI facility names
    approx_lat: float
    approx_lon: float
    notes: str = ""
    extra: dict = field(default_factory=dict)


TERMINALS: tuple[Terminal, ...] = (
    Terminal(
        slug="gate",
        name="Gate Terminal (Rotterdam Maasvlakte)",
        country="NL",
        tier="core",
        entsog=(PointDirection("NL-TSO-0001", "LNG-00027"),),
        alsi_name_patterns=("gate",),
        approx_lat=51.955,
        approx_lon=4.062,
        notes="GTS reports next morning ~09:20 CET; hourly flows ~2h behind real time.",
    ),
    Terminal(
        slug="eems",
        name="EemsEnergyTerminal (Eemshaven FSRUs)",
        country="NL",
        tier="core",
        entsog=(PointDirection("NL-TSO-0001", "LNG-00068"),),
        alsi_name_patterns=("eems",),
        approx_lat=53.452,
        approx_lon=6.875,
    ),
    Terminal(
        slug="zeebrugge",
        name="Zeebrugge LNG (Fluxys)",
        country="BE",
        tier="core",
        entsog=(PointDirection("BE-TSO-0001", "LNG-00017"),),
        alsi_name_patterns=("zeebrugge",),
        approx_lat=51.348,
        approx_lon=3.183,
        notes="Also has an exit direction (reload/transshipment); entry = regas send-out.",
    ),
    Terminal(
        slug="dunkerque",
        name="Dunkerque LNG",
        country="FR",
        tier="core",
        entsog=(PointDirection("FR-TSO-0003", "LNG-00003"),),
        alsi_name_patterns=("dunkerque",),
        approx_lat=51.016,
        approx_lon=2.155,
    ),
    Terminal(
        slug="grain",
        name="Isle of Grain (National Gas)",
        country="UK",
        tier="core",
        entsog=(PointDirection("UK-TSO-0001", "LNG-00008"),),
        alsi_name_patterns=("grain",),
        approx_lat=51.443,
        approx_lon=0.713,
        notes=(
            "UK ENTSOG rows are backfilled with ~6-day lag (verified 2026-09-01); "
            "near-real-time UK flows need National Gas's own data portal (TODO W3+). "
            "Zero send-out for days at a time is genuine (summer idling)."
        ),
    ),
    Terminal(
        slug="milford_haven",
        name="Milford Haven (South Hook + Dragon, aggregated)",
        country="UK",
        tier="core",
        entsog=(PointDirection("UK-TSO-0001", "LNG-00049"),),
        alsi_name_patterns=("south hook", "dragon"),
        approx_lat=51.705,
        approx_lon=-5.105,
        notes=(
            "One grid point covering TWO ALSI facilities — the mapping is 2 ALSI : 1 ENTSOG. "
            "Same ~6-day UK reporting lag as Grain."
        ),
    ),
    Terminal(
        slug="montoir",
        name="Montoir-de-Bretagne (Elengy)",
        country="FR",
        tier="secondary",
        entsog=(PointDirection("FR-TSO-0003", "LNG-00024"),),
        alsi_name_patterns=("montoir",),
        approx_lat=47.315,
        approx_lon=-2.145,
        notes="Atlantic coast — outside the NW-core price story but same machinery.",
    ),
    Terminal(
        slug="lehavre",
        name="Le Havre FSRU (Cape Ann)",
        country="FR",
        tier="secondary",
        entsog=(PointDirection("FR-TSO-0003", "LNG-00071"),),
        alsi_name_patterns=("havre",),
        approx_lat=49.470,
        approx_lon=0.140,
    ),
    Terminal(
        slug="wilhelmshaven",
        name="Wilhelmshaven FSRUs (DET)",
        country="DE",
        tier="secondary",
        entsog=(PointDirection("DE-TSO-0009", "LNG-00083"),),
        alsi_name_patterns=("wilhelmshaven",),
        approx_lat=53.600,
        approx_lon=8.110,
        notes="Data under this OGE pairing starts ~Apr 2025; earlier flows may sit under another point key.",
    ),
    Terminal(
        slug="brunsbuettel",
        name="Brunsbüttel FSRU (DET)",
        country="DE",
        tier="secondary",
        entsog=(PointDirection("DE-TSO-0005", "LNG-00060"),),
        alsi_name_patterns=("brunsb",),
        approx_lat=53.890,
        approx_lon=9.120,
    ),
    Terminal(
        slug="stade",
        name="Stade FSRU",
        country="DE",
        tier="secondary",
        entsog=(PointDirection("DE-TSO-0005", "LNG-00078"),),
        alsi_name_patterns=("stade",),
        approx_lat=53.630,
        approx_lon=9.530,
        notes="Zero rows in 2024-2026 backfill (FSRU never properly commissioned) — kept for completeness.",
    ),
    Terminal(
        slug="mukran",
        name="Mukran / Baltic Energy Gate (Deutsche ReGas)",
        country="DE",
        tier="secondary",
        entsog=(
            PointDirection("DE-TSO-0001", "LNG-00079"),
            PointDirection("DE-TSO-0018", "LNG-00079"),
        ),
        alsi_name_patterns=("mukran", "regas", "baltic"),
        approx_lat=54.480,
        approx_lon=13.590,
        notes="Two TSO rows (GASCADE + Fluxys Deutschland) for one point — sum them.",
    ),
)


def by_slug(slug: str) -> Terminal:
    for t in TERMINALS:
        if t.slug == slug:
            return t
    raise KeyError(f"unknown terminal slug: {slug}")


def tier(name: str) -> tuple[Terminal, ...]:
    if name == "all":
        return TERMINALS
    return tuple(t for t in TERMINALS if t.tier == name)
