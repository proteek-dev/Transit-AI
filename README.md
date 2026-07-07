# SEQ Transit AI

> *"Your bus is 200m away, running 10 minutes late. Leave by 8:47 and you'll arrive at Broadbeach by 9:23."*

A transport **confidence layer** for the Gold Coast ↔ Brisbane corridor — built on a self-collected archive of TransLink's GTFS-Realtime feeds that doesn't exist anywhere else publicly.

**Status:** Phase 1 complete · Phase 2 (feature engineering) in progress · Phase 3 (app) not started

---

## What This Is

TransLink's own app is data-heavy and assumes prior transit knowledge. Google Maps is easier to use but not confidence-qualified. Neither tells you what to actually *do* with the information.

This project is **not** a journey planner and **not** a real-time tracker — both already exist. It's a weather-app-style layer on top: conditions at a glance, a leave-by time, and a plain-English reason, specifically for SEQ TransLink (bus, rail, tram, ferry).

The differentiator isn't delay prediction itself — Google Maps has shipped that since 2019. It's the confidence-layer UX and leave-by-time output, applied specifically to this corridor, backed by an archive TransLink doesn't publish.

### Why the archive matters

TransLink publishes system-wide monthly on-time averages, but no stop-level or per-line historical delay data. That granularity only exists if someone archives the live GTFS-Realtime feed as it happens — every day not captured is gone forever. This repo's daemon has been doing exactly that, continuously, since **28 June 2026**.

---

## Architecture

```
                    ┌─────────────────────────┐
                    │   TransLink public APIs  │
                    │  GTFS-RT · static GTFS   │
                    │  · monthly perf. CSVs    │
                    └────────────┬─────────────┘
                                 │  every 5 min (RT) / daily / 24h
                                 ▼
                    ┌─────────────────────────┐
                    │  scripts/archive_gtfsrt.py │
                    │  macOS launchd daemon      │
                    └────────────┬─────────────┘
                                 │  boto3 (no local writes)
                                 ▼
              ┌───────────────────────────────────────┐
              │   S3 · seq-transit-ai-data-ps           │
              │   (ap-southeast-2)                      │
              │  gtfs_realtime/ · gtfs_static/           │
              │  performance/ · ml_features/             │
              └───────┬─────────────────────┬───────────┘
                      │                     │
                      ▼                     ▼
      notebooks 02–04 (Phase 1 EDA)   notebook 05 (Phase 2 pipeline)
      static GTFS + performance CSVs  join + feature engineering
      → charts, LinkedIn insights     → versioned parquet + _latest.json
                                              │
                                              ▼
                                     notebook 06 (Phase 2 EDA)
                                     reads _latest.json only — fast,
                                     no re-ingest / re-join
```

**Key principle:** raw archive data is immutable. All enrichment (joins, features) happens on copies, written to separate, versioned S3 prefixes.

---

## Data Sources

| Source | What | Auth | Notes |
|---|---|---|---|
| GTFS-Realtime (SEQ) | Live trip updates, vehicle positions, service alerts | None (public) | Polled every 5 min by the daemon — this is the archive that doesn't exist elsewhere |
| GTFS Static (SEQ) | Routes, stops, timetables | None (public) | Refreshed daily |
| TransLink Monthly Performance | System-wide on-time rates by mode | None (public portal) | Published on a lag (government cadence); aggregate only, no per-line breakdown |
| Transitland historical archive | 100+ historical GTFS versions | Free (Hobbyist/Academic, email required) | Deferred — not currently used; revisit only if the GTFS-RT archive alone proves insufficient |

---

## Repo Structure

```
Transit-AI/
├── notebooks/
│   ├── 02_load_static_gtfs.ipynb        # static GTFS → S3 parquet
│   ├── 03_load_performance_data.ipynb   # performance CSVs → S3
│   ├── 04_eda.ipynb                     # Phase 1 EDA (on performance CSVs)
│   ├── 05_phase2_feature_pipeline.ipynb # GTFS-RT archive + static join → ML features (S3, versioned)
│   └── 06_phase2_eda.ipynb              # Phase 2 EDA — reads latest feature snapshot only, no re-ingest
├── scripts/
│   └── archive_gtfsrt.py                # continuous daemon — GTFS-RT, static GTFS, performance → S3
├── config/
│   └── feeds.yaml                       # all feed URLs (GTFS-RT, static, performance CSVs)
├── .env.example                         # AWS_S3_BUCKET, AWS_REGION, etc.
├── requirements.txt
└── README.md
```

> Notebook `01` (a pre-daemon manual test fetch) has been removed — the daemon fully replaced its purpose.
> All data lives in S3 (`seq-transit-ai-data-ps`, `ap-southeast-2`). The local machine holds only logs and code — there is no local or in-repo data storage.

---

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in AWS_S3_BUCKET, AWS_REGION, and credentials/profile
```

**Start the archiver** (or install it as a background service — see below):

```bash
python scripts/archive_gtfsrt.py
```

**Run the pipeline notebooks in order:**

```bash
jupyter lab
# 02 → static GTFS parquet
# 03 → performance CSVs
# 04 → Phase 1 EDA
# 05 → Phase 2 feature pipeline (long-running — see note below)
# 06 → Phase 2 EDA (fast — reads notebook 05's latest output only)
```

Notebook 05 processes days of 5-minute GTFS-RT snapshots and can take a while on a full date range. Run it in the background rather than blocking a terminal:

```bash
nohup jupyter nbconvert --to notebook --execute --inplace notebooks/05_phase2_feature_pipeline.ipynb > 05_run.log 2>&1 &
tail -f 05_run.log
```

A `DATE_LIMIT` variable inside the notebook restricts a run to a single day for fast validation — set it to `None` for a full-range run.

### Running the archiver continuously (macOS)

The daemon is designed to run indefinitely via `launchd` so it survives reboots and restarts on crash. See `scripts/archive_gtfsrt.py` and the plist template for the exact configuration.

---

## Progress

### Now — Done

**Phase 1 — Data & EDA**
Historical on-time performance for Citytrain, pulled from TransLink's public monthly CSVs (Dec 2023 – Mar 2026, 28 months):

| Mode | Average on-time rate |
|---|---|
| Tram | 95.8% |
| Train (Citytrain) | 93.2% (low of ~88% in the worst months; +0.5%/yr trend) |
| Bus | 89.7% |

TransLink's own data is system-wide and aggregate only — no per-route or per-time-of-day breakdown is published anywhere. That gap is exactly what the self-collected GTFS-RT archive is built to close.

**Phase 2 v0 — Feature pipeline infrastructure**
- GTFS-RT trip-update archive joined against a static GTFS snapshot, engineered into ML-ready features (route, stop, hour of day, day of week, `is_weekend`, `is_peak`, mode, delay)
- Feature snapshots versioned by `run_date` in S3, with a `_latest.json` manifest written only after a verified, complete write
- Verified end-to-end on a single-day test run with no regression after the versioning changes

### Next — In progress

- Re-run the full multi-day feature range under the versioned pipeline
- Validate the `is_peak` feature path against a weekday (only tested on a weekend so far)
- Decide the minimum data volume needed before model training starts

### Later — Directional, not yet scoped

Deliberately kept light here — these depend on decisions above and are likely to change shape once made:

- Weather data as a model feature
- Per-date-matched static GTFS snapshot joins (v0 uses a single snapshot for the whole archive)
- Phase 3 (the app itself) — not started, scope to be defined once Phase 2 has a validated model

---

## License & Data Attribution

TransLink open data is published under CC-BY. This project archives and derives features from that public data; it does not redistribute TransLink's raw feeds.
