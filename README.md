# lng-nowcast

Reconstructing the daily gas send-out of NW-European LNG terminals — a direct
short-term driver of TTF/NBP prices — from **free public physical data**
(ship-tracking, terminal inventories, pipeline flows), ahead of and independently
of the officially confirmed figures.

The commercial vendors (Kpler, Vortexa, ICIS) sell this signal from proprietary
AIS networks. The point here is not to out-scale them but to do two things they
skip, in the open: **model the revision process of the official data itself**
(ALSI's Estimated→Confirmed corrections), and publish **calibrated uncertainty**
rather than point estimates — checked against later-published ground truth.

## Findings log

*2026-09-01 (day one)* — verified live against the APIs:

- **ENTSOG beats ALSI by ~10 hours.** Daily physical flow for gas day *D*
  appears on ENTSOG ~09:20 CET on *D+1* for NW-EU TSOs; ALSI's official
  publication for the same day is 19:30 CET (with a 23:00 late-reporter pass).
- **Hourly flows trail real time by ~2 hours** at GTS points (Gate, Eems), so an
  intraday estimate of the *current* gas day is feasible — the nowcast horizon
  can move inside the gas day, not just ahead of the evening print.
- **Day-ahead nominations are public.** The nomination for gas day *D* appears
  ~16:00 CET on *D−1* (e.g. Gate: 406.7 GWh nominated for 2026-09-01, published
  2026-08-31 16:04). Free forward-looking input for the filter.
- **UK is the laggard**: National Gas rows on ENTSOG are backfilled ~6 days
  late, in batches. Near-real-time UK needs ALSI (19:30) or National Gas's own
  portal (TODO). Grain at exactly 0 for days at a stretch is *genuine* summer
  idling, not missing data.
- **ALSI inventory is a volume** (10³ m³ LNG), send-out an energy (GWh/d) — the
  volume→energy conversion (GCV × density, cargo-composition-dependent) is part
  of the inference problem, not bookkeeping. Since API V13 each row also carries
  GIE's own GWh conversion — a free per-facility GCV calibration to exploit.

*2026-09-01 (evening, first ALSI snapshot)*:

- **EU facilities confirm within ~1 gas day**, and the Estimated row exists
  *intraday for the running gas day*, updating as the day progresses. The
  revision study therefore reframes: the target is the intraday E-trajectory vs
  the final Confirmed value (plus rarer post-C corrections), not a multi-day
  E→C lag. Snapshot cadence raised to 3×/day accordingly.
- **UK terminals do not report to ALSI at all** (pure status-N since ≥ 2024;
  Dragon absent entirely) — the official 19:30 CET print simply has no UK leg
  anymore. UK ground truth must come from National Gas's data portal; ENTSOG's
  UK rows lag ~6 days. Le Havre stopped reporting 2026-01-21.

*2026-09-02 (National Gas portal cracked)*:

- The portal's SPA API (`POST /api/find-gas-data`, keyless) serves per-terminal
  UK data **richer than ALSI ever was**: send-out at three maturities — D+1
  physical (generated ~12:01 next day), D+2 commercial, M+15 reconciliation —
  a *built-in revision ladder*; opening tank stocks (generated ~15:56 same
  day); storage in/outflows (cargo discharges appear as inflows); prevailing
  nominations; per-terminal calorific values; and a Grain boil-off nomination.
- `latestFlag=N` returns **every historical published version** of every value
  — the UK revision history is reconstructable retroactively, no live logging
  required (unlike ALSI, where the archive only accumulates forward).
- Gas-day alignment is exact: UK 05:00 local ≡ EU 06:00 CET at the same UTC
  instant year-round, so UK rows line up with ENTSOG's with no shift.

*2026-09-02 (UK revision process quantified, from the retroactive
all-versions pull — 17,628 published versions, 2024→today)*:

- **27.8% of UK values get republished, but the median republication changes
  nothing**: D+2 commercial values are systematically re-issued ~30 days later,
  almost always bit-identical. The heavy tail is real — |Δ| up to 662.7 GWh
  (send-out) and 340 GWh (tank stock, corrected next day).
- **The "M+15" items actually publish at ~M+1 and never differ from final D+2**
  (max |Δ| = 0.00 across 2,829 matched days): the ladder's last rung carries no
  information, and UK values reach finality ~1 month after the gas day.
- **D+1 physical ≈ final commercial on 99%+ of days** (p99 |Δ| = 0.02 GWh): the
  next-morning physical print is an excellent proxy for final truth — except
  for rare large breaks, which is the whole game.
- Showcase anomaly, **Grain 2025-04-21**: D+1 physical and ENTSOG both record
  ~662 GWh (near max send-out); the commercial series initially agreed, then
  was retracted to 0.0 a month later and finalized at zero. The official record
  permanently disagrees with itself by a full day of near-max flow — and that
  day's D+1 also published 3 days late. This is why the filter carries
  heavy-tailed observation errors and three-way triangulation.

*2026-09-02 (AIS layer design)*:

- Berth calls last many hours and AIS draught persists after departure, so the
  3×-daily snapshot cadence already supports daily arrival detection: each run
  opens a bounded aisstream listening window (~8 min) and archives per-vessel
  state + berth occupancy. A continuous local logger adds density when running.
- Cargo size uses two estimators with propagated uncertainty: capacity-class
  from AIS dimensions (primary — a standard 295×46 m carrier ⇒ 1096 ± 107 GWh
  per full discharge) and draught-delta × TPC (secondary — 2.2 m ⇒
  1012 ± 386 GWh). The draught channel is honestly ~3.6× wider because ships
  ballast up as cargo comes off (compensation ~60% of cargo mass) and draught
  is hand-entered. UK arrivals get free ground truth from National Gas inflow
  data; EU arrivals from ALSI inventory jumps — the calibration anchors.

## Layout

```
lng_nowcast/           the package
  terminals.py         registry: ENTSOG point IDs (verified live), ALSI name
                       patterns, approx coordinates for AIS polygons
  entsog.py            ENTSOG client — daily/hourly flows + nominations, no key
  alsi.py              GIE ALSI client — inventory/send-out/status, free key
  physics.py           conversion constants with uncertainty ranges
  snapshot.py          the revision logger (see below)
scripts/
  snapshot_daily.py    twice-daily snapshot entrypoint (cron / Actions)
  backfill_entsog.py   historical daily flows → data/raw/
  bootstrap_alsi.py    resolve ALSI facility EICs from the live listing (once)
  eda_overview.py      small-multiples overview chart → reports/
data/
  snapshots/           committed — the accumulating revision archive
  reference/           committed — resolved facility registry
  raw/                 gitignored — bulk backfills (reproducible)
```

## The revision logger (start this before anything else)

ALSI rows carry `status` E/C (estimated/confirmed) and operators retroactively
correct values at any time; neither GIE nor ENTSOG archives its own past states.
So the Estimated→Confirmed process — the gap this project trades on — is only
observable **live**. `scripts/snapshot_daily.py` therefore re-fetches a trailing
45-day window twice a day and archives what each source *currently* claims, so
every revision is captured with timestamps on both sides.

Run it via GitHub Actions (survives laptops being closed): push this repo to
GitHub, add repo secret `ALSI_KEY`, and `.github/workflows/snapshot.yml` commits
snapshots back to the repo at 18:45 & 22:15 UTC daily — after ALSI's 19:30 CET
and 23:00 CET publications in both winter and summer time. The ENTSOG side logs
even while the ALSI key is missing.

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -e '.[viz]'
cp .env.example .env       # then fill in keys as you get them
```

Keys (all free): **ALSI** — register at <https://alsi.gie.eu/account>, key is on
the API page; **aisstream.io** — needed from week 3; **ENTSO-E** token — needed
for the price event study.

```bash
.venv/bin/python scripts/backfill_entsog.py --start 2024-01-01   # history
.venv/bin/python scripts/bootstrap_alsi.py                       # once, with key
.venv/bin/python scripts/snapshot_daily.py                       # one snapshot
.venv/bin/python scripts/eda_overview.py                         # chart
```

## Model (target state)

Per terminal, a slow–fast state-space model on gas-day resolution:

- slow state: tank inventory `I_t` (energy units after conversion)
- fast flow: send-out `S_t`; jump input: cargo arrivals `A_t` from AIS berth +
  draught-delta events
- dynamics: `I_{t+1} = I_t + A_t − S_t − boil-off`
- observations: ALSI inventory & send-out (noisy, *revised*), ENTSOG flows
  (noisy, partial), AIS arrival energy (draught error, heel, GCV)
- inference: particle / ensemble Kalman filter → daily posterior over `S_t`
  ahead of confirmation; non-Gaussianity from jump arrivals and heavy-tailed
  reporting errors

Validation: nowcast vs later-**Confirmed** ALSI; triangulation across the three
near-independent measurements; calibration (do 90% intervals cover 90%?).

## Roadmap

- [x] W1: ENTSOG ingestion verified live; registry of 12 terminals; backfill
- [x] W1: revision logger written; Actions workflow live in the cloud (3×/day)
- [x] W1: ALSI key armed; first full snapshots committed
- [x] W2: National Gas UK client (send-out ladder, stocks, flows, nominations)
- [x] W2: UK revision process quantified from the retroactive all-versions pull
- [x] W3: AIS layer built (aisstream client, berth boxes, cargo inversion
      physics, snapshot integration + continuous logger) — key pending
- [ ] W2/W3: EU intraday E→C EDA as snapshots accumulate; verify berth boxes
      against first observed traffic
- [ ] W3–4: AIS layer (aisstream.io): berth polygons, draught deltas → arrivals
- [ ] W5–6: state-space filter; intraday nowcast at the 19:30 CET horizon
- [ ] W7–8: validation, calibration, TTF event study; write-up

## Honest caveats

- Kpler/Vortexa/ICIS sell commercial versions of the level signal from
  proprietary AIS networks. The open contribution is the reconstruction from
  free data plus the revision/uncertainty layers they don't publish.
- Free AIS coverage has gaps; draught is hand-entered and laggy. If it proves
  too sparse, the project degrades gracefully to the ALSI+ENTSOG fusion problem.
- **Not retail-tradable**: ICE TTF futures granularity (~720 MWh/month) is far
  beyond a student account; this is a research signal, not a trading system.
- If ALSI revisions turn out tiny/instant, the headline pivots to
  revision-prediction and the ENTSOG-lead-time result — measured either way.

## Attribution

LNG inventory/send-out data: **GIE ALSI** (<https://alsi.gie.eu>) — GIE requires
naming them as source when data is used or repackaged. Flow and nomination data:
**ENTSOG Transparency Platform** (<https://transparency.entsog.eu>). This
project redistributes snapshots solely for research reproducibility.
