# SEQ Transit AI

> *"Your bus is 200m away, running 10 minutes late. Leave by 8:47 and you'll arrive at Broadbeach by 9:23."*

A transport **confidence layer** for the Gold Coast ↔ Brisbane corridor — built on a self-collected archive of TransLink's GTFS-Realtime feeds that doesn't exist anywhere else publicly.

TransLink's own app assumes prior transit knowledge. Google Maps is easier to use but not confidence-qualified. Neither tells you what to actually *do* with the information. This project isn't a journey planner and isn't a real-time tracker — both already exist. It's a layer on top of them: a leave-by time and a plain-English confidence read, specifically for SEQ (bus, rail, tram).

The differentiator isn't delay prediction itself — it's the confidence-layer UX and leave-by-time output, backed by a historical archive TransLink doesn't publish anywhere.

### Why the archive matters

TransLink publishes system-wide monthly on-time averages, but no stop-level or per-line historical delay data. That granularity only exists by archiving the live GTFS-Realtime feed as it happens — every interval not captured is gone forever. This project has been doing exactly that, continuously, since late June 2026, including dedicated tram endpoints (TransLink's combined feed excludes tram).

---

## Architecture

```
TransLink public APIs (GTFS-Realtime, static GTFS, monthly performance CSVs)
              │  polled every ~5 min
              ▼
   EC2 archiver daemon  ──────────────▶  S3 (seq-transit-ai-data-ps)
   (continuous, archiving-only)          raw realtime + static snapshots
                                                    │
                                                    ▼
                                   Feature pipeline (notebooks, run locally)
                                   per-date static/realtime join → ML features
                                                    │
                                                    ▼
                              Model training (local) → S3, date/time-versioned
                              XGBoost delay model + auto-recorded MAE baseline
                                                    │
                                                    ▼
                                     Live app (phase3/) — Streamlit
                                     map-based stop search, live GTFS-RT,
                                     leave-by predictions, route path on map
```

**Key principle:** raw archive data is immutable. All feature engineering and enrichment happen on derived, versioned copies — the archive itself is never rewritten.

---

## Data Sources

| Source | What | Auth | Notes |
|---|---|---|---|
| GTFS-Realtime (SEQ) | Live trip updates, vehicle positions, service alerts | None (public) | Polled continuously — combined feed plus dedicated tram-only endpoints (the combined feed excludes tram) |
| GTFS Static (SEQ) | Routes, stops, timetables, shapes | None (public) | Refreshed daily |
| TransLink Monthly Performance | System-wide on-time rates by mode | None (public portal) | Aggregate only, no per-line breakdown — the gap this project's archive fills |

---

## Repo Structure

```
Transit-AI/
├── notebooks/              # Feature pipeline, EDA, and baseline model training
├── scripts/
│   └── archive_gtfsrt.py   # Continuous archiver daemon (EC2) — GTFS-RT, static GTFS, performance → S3
├── pipeline/                # Standalone rebuild + retrain scripts
├── config/
│   └── feeds.yaml           # Feed URLs (GTFS-RT combined + per-mode, static, performance)
├── phase3/                  # The live app
│   ├── app.py                 # Streamlit UI — stop search, results, shared route map
│   ├── gtfs_data.py            # Facade re-exporting gtfs/ below — external imports unchanged
│   ├── gtfs/
│   │   ├── loader.py             # GTFS static snapshot load, ferry exclusion, dtype/memory optimization
│   │   ├── search.py             # Typed stop search (fuzzy + proximity-biased) + nearest-stop ranking
│   │   ├── routing.py            # Direct + multi-leg transfer trip finding (BFS)
│   │   └── shapes.py             # Trip shape (road/rail path) lookup
│   ├── prediction.py           # Facade re-exporting ml/ below — external imports unchanged
│   ├── ml/
│   │   ├── training.py           # Model training (chunked read, leakage filter, fit, MAE baseline)
│   │   ├── inference.py          # Feature building + delay prediction/blending at request time
│   │   └── model_io.py           # Model/metadata load, caching, S3 upload
│   ├── ui/
│   │   ├── pickers.py            # From/To stop pickers (map tap + typed search)
│   │   ├── cards.py              # Result card rendering
│   │   └── formatting.py         # Pure display-formatting helpers
│   ├── map_picker.py           # Reusable map — stop picker + route path display
│   ├── live_gtfs.py            # Live GTFS-RT fetch, short-lived cache
│   ├── config.py               # Credential resolution (Streamlit Cloud / local / .env)
│   ├── route_types.py          # Shared GTFS route_type ↔ mode-name mapping
│   └── tests/                  # Diagnostic/smoke scripts (run directly, not pytest)
├── .env.example
├── requirements.txt
└── README.md
```

---

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in AWS_S3_BUCKET, AWS_REGION, and credentials/profile
nbstripout --install --attributes .gitattributes   # once per fresh clone
```

**Archiving** runs continuously on a dedicated EC2 instance as a `systemd` service — it's not something you need to run locally to use the app; the app reads pre-built features and a pre-trained model from S3.

**Rebuilding features and retraining the model** (only needed if you're changing the pipeline itself):

```bash
bash pipeline/01_rebuild_features.sh
bash pipeline/02_train_model.sh
```

**Running the app locally:**

```bash
cd phase3
streamlit run app.py
```

Opens at `http://localhost:8501`. Needs a `.env` at the repo root with AWS credentials. On Streamlit Community Cloud, the same code reads credentials from `st.secrets` instead — no `.env` needed there.

**Phone access on the same WiFi:**

```bash
streamlit run app.py --server.address 0.0.0.0
```

Then browse to `http://<local-ip>:8501` from your phone.

---

## The App

Pick a "From" and "To" stop and a departure time; it finds direct or multi-leg-transfer trips and predicts arrival delay by blending a trained model with TransLink's live GTFS-RT feed.

- Map-based stop selection for both origin and destination — type or use your location, and every nearby matching stop shows as a tappable pin, not a single auto-resolved match
- Typed destination search is proximity-biased toward your already-picked origin, so results near the wrong end of the corridor don't surface just because they share a name
- Distance shown ("180m away", "2.3km away") wherever the app actually knows your position relative to a stop
- Multi-leg transfer routing across bus, rail, and tram
- A single shared route map (not one per result) showing the actual GTFS road/rail path, defaulting to the top-ranked result and updating when you pick a different one
- Leave-by time as the headline output, with a plain-English summary and a confidence rating (High/Medium/Low)
- Live delay blended with the trained model at request time

**Known limitations:**
- Tram training data is still comparatively thin — tram predictions carry lower confidence than bus/rail
- Ferry is intentionally excluded from search, routing, and training (kept in the raw archive only)
- No natural-language query parsing — stop selection is search-based, not free-text

---

## Progress

**Data & model**
- Continuous GTFS-Realtime archive since late June 2026, now via a dedicated EC2 daemon (migrated off a laptop-based process for reliability)
- Feature pipeline joins realtime data against the static GTFS snapshot actually in effect on each date, ferry-excluded
- XGBoost delay model, retrained periodically, with an automatically recorded MAE baseline on every run and date/time-versioned model storage for rollback
- Current baseline, trained on the full ~156M-row ferry-free archive across 46 source-date partitions: train MAE 1.999 min, test MAE 1.957 min vs. a naive median baseline of 2.486 min — a 21.3% improvement over naive

**App**
- Full trip search: map-based stop pickers, direct + multi-leg transfer routing, leave-by output, live route map with real path shapes
- Deployed on Streamlit Community Cloud

**Next**
- Broader confidence-model refinement as more live traffic is observed
- Continued tram data accumulation

---

## License & Data Attribution

TransLink open data is published under CC-BY. This project archives and derives features from that public data; it does not redistribute TransLink's raw feeds.
