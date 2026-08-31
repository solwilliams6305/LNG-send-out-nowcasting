# lng-nowcast — notes for Claude sessions

Nowcasting NW-European LNG terminal send-out from free public data (ENTSOG,
GIE ALSI, AIS). Research/portfolio project for quant-research applications —
deliverables are a signal, a write-up, and this public repo. Not a trading
system; keep the honest-caveats section of README.md intact.

## Commands

```bash
.venv/bin/python scripts/snapshot_daily.py       # twice-daily revision snapshot
.venv/bin/python scripts/backfill_entsog.py      # merge-safe historical pull
.venv/bin/python scripts/bootstrap_alsi.py       # resolve ALSI EICs (needs key)
.venv/bin/python scripts/eda_overview.py         # reports/sendout_overview.png
```

## Hard-won facts (verified 2026-09-01, don't rediscover)

- ENTSOG needs no key; filter `operationalData` by operatorKey/pointKey/
  directionKey/indicator/periodType. Values kWh. Gas day = 06:00–06:00 CET.
- Timing: ENTSOG daily ~09:20 CET next morning; hourly ~2h behind real time;
  nominations for D published ~16:00 CET on D−1; ALSI publishes 19:30 CET
  + 23:00 CET catch-up. UK ENTSOG rows backfilled ~6 days late (zeros there
  can be genuine idling).
- ALSI (API manual V13): `x-key` header on data endpoints; the key must have
  ALSI or all-platforms scope (AGSI-only keys get "Invalid or missing API key");
  `/about?show=listing` is public. `inventory`/`dtmi` are dual-unit objects
  {"lng": 10³ m³, "gwh": GWh} — GIE's own conversion, free GCV calibration.
  Rows carry `updatedAt` and exact facility lat/lon. Status E/C/N. EICs
  resolved at runtime (bootstrap script), never hardcoded. 15 live-resolvable
  facilities across the 12 registered terminals.
- Revision landscape (first snapshot, 2026-09-01): EU facilities confirm within
  ~1 gas day — the E row exists intraday for the *running* day and updates
  during it, so the E→C action is intraday (hence the 11:00 UTC cron run).
  Post-C corrections still possible per GIE docs; snapshots catch both.
- ALSI coverage gaps: UK LSOs (Grain, South Hook — both GB and GB* datasets)
  are pure status-N since at least 2024; Dragon absent from ALSI entirely; Le
  Havre stopped reporting 2026-01-21; Stade never commissioned. UK ground truth
  → National Gas data portal (W2 priority); ENTSOG UK lags ~6 days.
- Mukran: two ENTSOG TSO rows for one point — sum them.
- His GIE key also unlocks AGSI (storage) — covariate for the W7 event study.

## Conventions

- Snapshots are append-only timestamped CSVs under data/snapshots/ (committed);
  bulk backfills live in data/raw/ (gitignored). Never rewrite old snapshots.
- data/snapshots and data/reference are committed by the Actions cron; don't
  add data/raw to git.
- Registry of terminals/points: lng_nowcast/terminals.py. Coordinates there are
  approximate — verify against ship tracks before using for AIS polygons.
