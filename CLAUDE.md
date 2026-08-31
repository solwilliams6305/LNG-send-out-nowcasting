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
- ALSI: `x-key` header; inventory in 10³ m³ LNG (volume), sendOut in GWh/d;
  status E/C/N. EICs resolved at runtime (bootstrap script), never hardcoded.
- Milford Haven: one ENTSOG point = two ALSI facilities (South Hook + Dragon).
  Mukran: two TSO rows for one point — sum them.

## Conventions

- Snapshots are append-only timestamped CSVs under data/snapshots/ (committed);
  bulk backfills live in data/raw/ (gitignored). Never rewrite old snapshots.
- data/snapshots and data/reference are committed by the Actions cron; don't
  add data/raw to git.
- Registry of terminals/points: lng_nowcast/terminals.py. Coordinates there are
  approximate — verify against ship tracks before using for AIS polygons.
