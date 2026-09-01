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
- National Gas portal (UK, keyless): POST data.nationalgas.com/api/find-gas-data
  with {latestFlag, applicableFor, dateFrom, dateTo, dateType:"GASDAY",
  ids:"PUBOB..,.."}; item registry in lng_nowcast/nationalgas.py (send-out at
  D+1/D+2/M+15 maturities, stocks, in/outflows, nominations, CV, Grain
  boil-off). latestFlag=N → all historical published versions (retroactive
  revision history). Values kWh (unit string "kw/h"). D+1 send-out generated
  ~12:01 next day; stock ~15:56 same day. Grain = NTS1+NTS2 summed; ENTSOG
  milford_haven = south_hook + dragon. UK gas day 05:00 local = EU 06:00 CET
  at the same UTC instant year-round, so gas_day labels align with ENTSOG.
  Intraday (cracked 2026-09-01 from the JS bundle): POST {"request": name} to
  /api/gas-system-status-table (flowRatesTable: 2-minutely per sub-terminal
  incl. GRAIN NTS 1/2, MILFORD HAVEN - SOUTH HOOK/- DRAGON, latest ~6 reads)
  and /api/gas-system-status-graph (flowRatesGraphs: full day-so-far 2-min
  trajectory per area). mcm/day rates; NOT archived publicly — our
  nationalgas_live snapshots are the archive. UK filter horizons become
  possible forward-only as these accumulate.
- AIS (lng_nowcast/ais.py): aisstream.io websocket, key AISSTREAM_KEY,
  subscribe within 3 s, [lat, lon] corner order, max 3 connections; the server
  drops connections periodically — the listener reconnects and returns partial
  data. Berth boxes are centred on ALSI facility coordinates (verified live
  2026-09-02; registry coords were 2-9 km off before); Dragon's box alone is
  provisional. Coverage is patchy by area/time (Milford Haven, Dunkerque,
  Mukran silent at first listen; Rotterdam/Elbe/Medway rich) — AIS silence
  means "no receiver", not "no ship". FSRUs (e.g. Energos Igloo at eems) sit
  permanently in berth boxes — infrastructure, not arrivals. Cargo inversion
  in physics.py: capacity-class primary, draught-delta secondary (ballast
  compensation ~60%). Offline test: scripts/selftest_ais.py. Snapshot runs
  listen ~480 s when the key exists (--ais-window to change).
- UK revision structure (quantified 2026-09-02, scripts/revision_report_uk.py):
  D+2 items republished ~30d later usually unchanged (median Δ=0); "M+15"
  items publish at ~M+1 and NEVER differ from final D+2 — drop them from
  models; D+1 physical ≈ final on 99%+ of days (p99 0.02 GWh); heavy tail up
  to 662.7 GWh. Canonical anomaly: Grain 2025-04-21 (physical+ENTSOG ~662,
  commercial finalized 0). Stocks occasionally corrected next day (up to
  340 GWh). Publication lags have tails (that D+1 came 3 days late).

- ENTSOG data traps (found 2026-09-01): dunkerque hourly rows are ALL-NaN
  (NaTran publishes empty values — use min_count so NaN never reads as 0);
  Nomination queries hang randomly with zero bytes forever (client retries
  timeouts once then skips; re-run passes fill gaps, merge is idempotent);
  hourly gas-day labeling needs the −6h shift (post-midnight hours belong to
  the previous gas day); LNG-point nominations are empirically worse than
  persistence as forecasts (~70% archived post-renomination) — the filter's
  φ-fit rejects them, don't hand-force them in.
- Filters: nowcast.py (bootstrap reference) and hsmm_rbpf.py (explicit-
  duration HSMM + Rao-Blackwellized Kalman; 500 particles ≈ BPF's 4000
  intraday, 15× faster, exact idle state, p_idle output; run via
  scripts/run_nowcast_rb.py on the same panels/metrics). pgas.py = particle
  Gibbs parameter learning (scripts/fit_pgas.py demo). RBPF gotcha learned:
  posterior views must Kalman-update per-particle means, not just weights.
  Evaluation headline: intraday 3–7× better than persistence, conformal
  cov 0.48–0.62/0.87–0.93. Open: Zeebrugge profile kernel, RBPF H0 gap
  (~3 GWh, PGAS-tunable), PGAS on full history.
- Time-gated analysis scripts (rerun as the archive fattens):
  revision_intraday.py (ALSI E→C trajectories + UK live-trajectory
  integrals; PRELIMINARY finding: intraday E-row = carry-forward of
  yesterday's C — verify over more days), berth_events.py (AIS occupancy
  spells; wilhelmshaven berth box too loose, catches port traffic).

## Conventions

- Snapshots are append-only timestamped CSVs under data/snapshots/ (committed);
  bulk backfills live in data/raw/ (gitignored). Never rewrite old snapshots.
- data/snapshots and data/reference are committed by the Actions cron; don't
  add data/raw to git.
- Registry of terminals/points: lng_nowcast/terminals.py. Coordinates there are
  approximate — verify against ship tracks before using for AIS polygons.
