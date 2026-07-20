# SEQ Transit AI

> *"Your bus is 200m away, running 10 minutes late. Leave by 8:47 and you'll arrive at Broadbeach by 9:23."*

A transport **confidence layer** for the Gold Coast ↔ Brisbane corridor — built on a self-collected archive of TransLink's GTFS-Realtime feeds that doesn't exist anywhere else publicly.

**Status:** Phase 1 complete · Phase 2 v0 model trained (enrichment deferred pending more archive data) · Phase 3 Streamlit POC live

---

## What This Is

TransLink's own app is data-heavy and assumes prior transit knowledge. Google Maps is easier to use but not confidence-qualified. Neither tells you what to actually *do* with the information.

This project is **not** a journey planner and **not** a real-time tracker — both already exist. It's a weather-app-style layer on top: conditions at a glance, a leave-by time, and a plain-English reason, specifically for SEQ TransLink (bus, rail, tram, ferry).

The differentiator isn't delay prediction itself — Google Maps has shipped that since 2019. It's the confidence-layer UX and leave-by-time output, applied specifically to this corridor, backed by an archive TransLink doesn't publish.

### Why the archive matters

TransLink publishes system-wide monthly on-time averages, but no stop-level or per-line historical delay data. That granularity only exists if someone archives the live GTFS-Realtime feed as it happens — every day not captured is gone forever. This repo's daemon has been doing exactly that, continuously, since **28 June 2026**, including dedicated tram endpoints (TransLink's combined feed excludes tram data — see Progress below).

---

## Architecture

```
                    ┌─────────────────────────┐
                    │   TransLink public APIs  │
                    │  GTFS-RT (combined +     │
                    │  dedicated tram) ·       │
                    │  static GTFS · perf CSVs │
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
              │  alerts/ (service_alerts archive)        │
              │  phase3/model/ (trained model artifacts) │
              └───┬─────────────┬─────────────┬──────────┘
                  │             │             │
                  ▼             ▼             ▼
      notebooks 02–04   notebook 05      notebook 05b
      (Phase 1 EDA)      (Phase 2         (alert features)
      static GTFS +      pipeline)        service_alerts →
      performance CSVs   per-date         route-hourly +
      → charts,          static-snapshot  stop-hourly parquet
      LinkedIn insights  join + feature   → S3 alerts/features/
                         eng. → versioned          │
                         parquet +                 │
                         _latest.json               │
                              │                      │
              ┌───────────────┼──────────┐           │
              ▼               ▼          │           │
      notebook 06       notebook 07      │           │
      (Phase 2 EDA)      (baseline       │           │
      reads              model)          │           │
      _latest.json       loads           │           │
      only — fast,       _latest.json,   │           │
      no re-ingest       trains XGBoost  │           │
      / re-join          v0 baseline     │           │
                                         ▼           ▼
                              notebook 08 (enrichment A/B)
                              loads _latest.json + 05b + 05c
                              output via enrich.py (shared
                              helper); trains 4 XGBoost variants:
                              baseline / +weather / +alerts / +both
                                                   ▲
                                                   │
                    ┌──────────────────┐          │
                    │ Open-Meteo ERA5   │          │
                    │ API (external,    │          │
                    │ no auth)          │          │
                    └────────┬──────────┘          │
                             ▼                      │
                       notebook 05c ─────────────────┘
                       (weather dimension)
                       → S3 weather/era5/hourly.parquet
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │  phase3/ Streamlit POC    │
                    │  loads model from S3,     │
                    │  blends v0 predictions     │
                    │  with the live GTFS-RT     │
                    │  feed at request time      │
                    └─────────────────────────┘
```

**Key principle:** raw archive data is immutable. All enrichment (joins, features) happens on copies, written to separate, versioned S3 prefixes.

---

## Data Sources

| Source | What | Auth | Notes |
|---|---|---|---|
| GTFS-Realtime (SEQ) | Live trip updates, vehicle positions, service alerts | None (public) | Polled every 5 min by the daemon — combined feed plus dedicated tram-only endpoints (the combined feed excludes tram) |
| GTFS Static (SEQ) | Routes, stops, timetables | None (public) | Refreshed daily |
| TransLink Monthly Performance | System-wide on-time rates by mode | None (public portal) | Published on a lag (government cadence); aggregate only, no per-line breakdown |
| Transitland historical archive | 100+ historical GTFS versions | Free (Hobbyist/Academic, email required) | Deferred — not currently used; revisit only if the GTFS-RT archive alone proves insufficient |

---

## Repo Structure

```
Transit-AI/
├── notebooks/
│   ├── 02_load_static_gtfs.ipynb          # Parse static GTFS → S3 parquet
│   ├── 03_load_performance_data.ipynb     # Download + validate performance CSVs → S3
│   ├── 04_eda.ipynb                       # Phase 1 EDA on performance CSVs (charts → S3)
│   ├── 05_phase2_feature_pipeline.ipynb   # GTFS-RT archive + per-date static GTFS join → versioned ML features (S3)
│   ├── 05b_alert_features.ipynb           # Service alerts archive → route-hourly + stop-hourly tables (S3)
│   ├── 05c_weather_dimension.ipynb        # Open-Meteo ERA5 historical weather → hourly parquet (S3)
│   ├── 06_phase2_eda.ipynb                # Phase 2 EDA — reads _latest.json only, no re-ingest
│   ├── 07_phase2_baseline_model.ipynb     # XGBoost v0 baseline — leakage-filtered, temporal split
│   ├── 08_enrichment_ab.ipynb             # Four-way A/B: baseline vs +weather vs +alerts vs +both
│   └── enrich.py                          # Shared load/join/split helper for notebook 08
├── scripts/
│   └── archive_gtfsrt.py                # continuous daemon — GTFS-RT (combined + tram), static GTFS, performance → S3
├── config/
│   └── feeds.yaml                       # all feed URLs (GTFS-RT combined + per-mode, static, performance CSVs)
├── phase3/                              # Streamlit POC front end — see "Phase 3 POC" below
│   ├── app.py                           # Streamlit UI: stop search, time picker, results
│   ├── gtfs_data.py                     # Static GTFS data layer: stop search, direct + multi-leg trip finder
│   ├── live_gtfs.py                     # Live GTFS-RT fetch (TripUpdates/VehiclePositions), 60s cache
│   ├── prediction.py                    # Model load/train + inference-time feature building + delay blending
│   └── config.py                        # Shared credential loading (st.secrets → env vars → .env)
├── .env.example                         # AWS_S3_BUCKET, AWS_REGION, etc.
├── requirements.txt
└── README.md
```

> Notebook `01` (a pre-daemon manual test fetch) has been removed — the daemon fully replaced its purpose.
> All data lives in S3 (`seq-transit-ai-data-ps`, `ap-southeast-2`). The local machine holds only logs and code — there is no local or in-repo data storage. `phase3/model/` is a local/S3 cache of trained model artifacts, gitignored — S3 is the source of truth.

---

## Notebook Run Order

The notebooks have a strict dependency chain. Run them in this order:

### Phase 1 — Data & EDA (one-time setup)

| Step | Notebook | Input | Output | Notes |
|------|----------|-------|--------|-------|
| 1 | `02_load_static_gtfs` | S3 static GTFS CSVs | S3 parquet files | Run once; re-run if static GTFS changes |
| 2 | `03_load_performance_data` | S3 performance CSVs | Validated DataFrames | Run once to confirm CSVs load cleanly |
| 3 | `04_eda` | Output from 02 + 03 | Charts → S3 `eda_charts/` | Phase 1 analysis — Citytrain/Bus/Tram on-time trends |

### Phase 2 — Feature Engineering + Model

| Step | Notebook | Input | Output | Notes |
|------|----------|-------|--------|-------|
| 4 | `05_phase2_feature_pipeline` | S3 GTFS-RT archive (combined + tram) + all static GTFS snapshots | Versioned parquet → S3 `ml_features/` + `_latest.json` | **Long-running** — use `nohup` (see Setup). Matches each realtime date to the static snapshot in effect at the time (nearest snapshot ≤ date), not one global snapshot, and streams the load+join per date to keep peak memory bounded on a full historical reprocess. |
| 5a | `05b_alert_features` | S3 service_alerts archive | `alerts/features/route_hourly.parquet` + `stop_hourly.parquet` | Independent of 05. Can run in any order relative to 05c. |
| 5b | `05c_weather_dimension` | Open-Meteo API (no auth) | `weather/era5/hourly.parquet` | Independent of 05. Cached; skips fetch unless `REFRESH=True`. |
| 6 | `06_phase2_eda` | `_latest.json` from step 4 | Charts + summary stats | Fast — reads finished parquet only, no re-ingest |
| 7 | `07_phase2_baseline_model` | `_latest.json` from step 4 | Trained XGBoost model + residual analysis | v0 baseline — no weather or alert features |
| 8 | `08_enrichment_ab` | `_latest.json` + 05b output + 05c output | Four-way A/B comparison JSON → S3 | Uses `enrich.py` for shared load/split logic. Currently inconclusive — needs more archive data. |

Steps 5a and 5b are independent of each other and of step 4 — they can
run in parallel. Steps 6, 7, and 8 all depend on step 4's output
(`_latest.json`) but are independent of each other.

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

## Phase 3 POC (Streamlit)

A trip planner front end wired directly on top of the Phase 2 pipeline: pick a "From" and "To" stop and a departure time, and it finds direct GTFS trips (falling back to multi-leg transfer routing if none exist), then estimates arrival delay by blending the v0 XGBoost model with TransLink's live GTFS-RT feed.

**Features:**
- Fuzzy stop search with rapidfuzz-powered typeahead (`streamlit-searchbox`)
- Live GTFS-RT fetching from TransLink's public feeds at request time
- v0 XGBoost delay predictions blended with live delay data, with confidence scoring (High/Medium/Low)
- **Leave-by time** as the hero output on every result card
- Collapsible trip cards — collapsed shows the leave-by time + a plain-English summary; expanded reveals full route detail (mode, headsign, delay badge, from/to stops, live tracking status, confidence, departure/arrival times)
- Now / Later / Custom departure time quick-select (AEST, refreshed at search time — not frozen at page load)
- Multi-leg transfer routing via BFS over the static GTFS route-intersection graph, at arbitrary transfer depth
- Human-readable route display: mode emoji + route name + headsign (e.g. "🚊 Tram GCL3 towards Helensvale")
- DD/MM/YYYY dates, AM/PM times, AEST timezone throughout
- Fallback to a projected timetable on thin-calendar days (e.g. Sundays) when the exact date's schedule can't be resolved
- Model loading with a fallback chain: S3 (source of truth) → local cache → retrain from the S3 feature snapshot if neither is available
- Credential resolution: `st.secrets` → environment variables → `.env` file, so the same code runs unmodified locally and on Streamlit Community Cloud

**Modules:**

| File | Responsibility |
|---|---|
| `app.py` | Streamlit UI only — stop search, date/time input, result rendering. No business logic. |
| `gtfs_data.py` | Static GTFS data layer: loads the latest snapshot from S3, caches it, exposes stop search and direct/multi-leg trip finding. |
| `live_gtfs.py` | Fetches TripUpdates/VehiclePositions directly from TransLink's public GTFS-RT feeds at request time (60s in-memory cache). |
| `prediction.py` | Loads or trains the v0 model, builds inference-time features matching the training schema exactly, and blends model output with live delay data. |
| `config.py` | Centralizes AWS credential loading across local and Streamlit Cloud environments. |

**Run locally:**

```bash
cd phase3
streamlit run app.py
```

The app opens at `http://localhost:8501`. Prerequisites: a `.env` file at the repo root with AWS credentials (same as the pipeline setup above), and `pip install -r requirements.txt` from the repo root. On first run, if no saved model exists yet at `phase3/model/xgb_v0.json`, `prediction.py` trains it from the S3 feature snapshot (a few minutes); subsequent runs load the saved model instantly.

**Phone access (same WiFi):**

```bash
streamlit run app.py --server.address 0.0.0.0
```

Then find the machine's local IP (e.g. on macOS: `ipconfig getifaddr en0`) and browse to `http://<local-ip>:8501` from the phone.

**Deployment:** the app is deployed to Streamlit Community Cloud, reading credentials from `st.secrets` and the trained model from S3 — no local model files or `.env` needed on Cloud.

**Known limitations:**
- Tram training data is still thin (~4k rows, one route) — predictions for tram trips carry lower confidence than bus/rail
- No natural-language query parser yet — stop selection is fuzzy search, not free-text trip queries
- Enrichment features (weather, service alerts) are built but deferred — see Progress below

---

## Progress

### Now — Done

**Phase 1 — Data & EDA**
Historical on-time performance from TransLink's public monthly CSVs (Dec 2023 – Mar 2026, 28 months):

| Mode | Average on-time rate |
|---|---|
| Tram | 95.8% |
| Train (Citytrain) | 93.2% (low of ~88% in the worst months; +0.5%/yr trend) |
| Bus | 89.7% |

TransLink's own data is system-wide and aggregate only — no per-route or per-time-of-day breakdown is published anywhere. That gap is exactly what the self-collected GTFS-RT archive is built to close.

**Phase 2 v0 — Feature pipeline + baseline model**
- GTFS-RT trip-update archive joined per-date against the static GTFS snapshot actually in effect on that date (nearest snapshot ≤ date), engineered into ML-ready features (route, stop, hour, day, peak flag, mode, delay). Earlier versions joined every date against a single global snapshot, which silently dropped realtime data whose trip_ids belonged to a different static release.
- Archiver now also polls dedicated tram-only GTFS-RT endpoints (TripUpdates, VehiclePositions) alongside the combined feed, since TransLink's combined feed excludes tram entities.
- Feature snapshots versioned by `run_date` in S3, with `_latest.json` manifest written only after verified write
- XGBoost v0 baseline retrained on the corrected, per-date-matched snapshot: MAE 2.084 minutes, RMSE 4.385 minutes (previous baseline: MAE 2.26 min) — 40.5M training rows across 23 archive dates (previously 20.4M rows across ~19-20 dates), now including tram (4,036 rows, GCL3 route)
- Enrichment A/B (weather + service alerts) built and tested — results inconclusive with 21 days of archive; deferred until more data accumulates (daemon runs passively)

**Phase 3 — Streamlit POC**
- Full trip search UX shipped: fuzzy stop search, direct + multi-leg transfer routing, leave-by hero output, collapsible result cards, Now/Later/Custom departure time selection
- Model serving wired to the S3-trained v0 model with a local/S3/retrain fallback chain
- See "Phase 3 POC" section above for the full feature list

### Next

- Confidence API refinements as more live traffic is observed against the app
- NL query parser: "Varsity Lakes to Broadbeach" → route resolution (LLM-assisted)
- Broader deployment beyond the current POC

### Later — Revisit with more data

- Re-run enrichment A/B once archive covers 3+ months (ideally including Nov–Mar wet season)
- Weather as a model feature (currently inconclusive — one Brisbane CBD point across ~21 dry winter days)
- More tram archive data — current training set has one tram route at low volume

---

## License & Data Attribution

TransLink open data is published under CC-BY. This project archives and derives features from that public data; it does not redistribute TransLink's raw feeds.
