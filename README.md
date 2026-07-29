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
│   ├── app.py                # Streamlit UI — stop search, results, route map
│   ├── gtfs_data.py           # Static GTFS data layer — stop search, trip finding, shapes
│   ├── map_picker.py          # Reusable map-based stop picker + route path display
│   ├── live_gtfs.py           # Live GTFS-RT fetch, short-lived cache
│   ├── prediction.py          # Model load/train, feature building, delay blending
│   └── config.py              # Credential resolution (Streamlit Cloud / local / .env)
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
- Multi-leg transfer routing across bus, rail, and tram
- A route map showing the actual GTFS road/rail path for whichever result you select, not just a straight line
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

**App**
- Full trip search: map-based stop pickers, direct + multi-leg transfer routing, leave-by output, live route map with real path shapes
- Deployed on Streamlit Community Cloud

**Next**
- Broader confidence-model refinement as more live traffic is observed
- Continued tram data accumulation

---

## License & Data Attribution

TransLink open data is published under CC-BY. This project archives and derives features from that public data; it does not redistribute TransLink's raw feeds.
