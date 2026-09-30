# SEQ Transit AI

> *"Your bus is 200m away, running 10 minutes late. Leave by 8:47 and you'll arrive at Broadbeach by 9:23."*

**Live:** [https://d17e9wvem3lbaq.cloudfront.net](https://d17e9wvem3lbaq.cloudfront.net)

A transport **confidence layer** for the Gold Coast ↔ Brisbane corridor — built on a self-collected archive of TransLink's GTFS-Realtime feeds that doesn't exist anywhere else publicly.

TransLink's own app assumes prior transit knowledge. Google Maps is easier to use but not confidence-qualified. Neither tells you what to actually *do* with the information. This project isn't a journey planner and isn't a real-time tracker — both already exist. It's a layer on top of them: a leave-by time and a plain-English confidence read, specifically for SEQ (bus, rail, tram).

The differentiator isn't delay prediction itself — it's the confidence-layer UX and leave-by-time output, backed by a historical archive TransLink doesn't publish anywhere.

### Why the archive matters

TransLink publishes system-wide monthly on-time averages, but no stop-level or per-line historical delay data. That granularity only exists by archiving the live GTFS-Realtime feed as it happens — every interval not captured is gone forever. This project has been doing exactly that, continuously, since late June 2026, including dedicated tram endpoints (TransLink's combined feed excludes tram).

---

## Model baseline (v0)

| Metric | Value |
|---|---|
| Algorithm | XGBoost regression |
| Training rows | 117,157,755 |
| Test rows | 26,415,212 |
| Test MAE | **1.957 min** |
| Naive median MAE | 2.486 min |
| Improvement over naive | **21.3%** |
| Archive days used | 46 |
| Trained on | 2026-08-14 |

Confidence assignment (High / Medium / Low) is designed to blend live GTFS-RT delay with the model output when live data is available and fall back to model-only when it isn't. Out-of-vocabulary stops force Low confidence rather than silently degrading. See the "Known limitations" note below on the current wiring.

---

## Architecture

```text
┌─────────────────────────────────────────────────┐
│              TransLink public APIs              │
├─────────────────────────────────────────────────┤
│  GTFS-Realtime · static GTFS                    │
│  Monthly performance CSVs                       │
└─────────────────────────────────────────────────┘
                         │   polled every ~5 min
                         ▼
┌─────────────────────────────────────────────────┐
│               EC2 archiver daemon               │
├─────────────────────────────────────────────────┤
│  Continuous, archiving-only                     │
└─────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────┐
│          S3  ·  seq-transit-ai-data-ps          │
├─────────────────────────────────────────────────┤
│  Raw realtime + static snapshots                │
└─────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────┐
│   Feature pipeline  ·  notebooks, run locally   │
├─────────────────────────────────────────────────┤
│  Per-date static/realtime join → ML features    │
└─────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────┐
│             Model training  ·  local            │
├─────────────────────────────────────────────────┤
│  XGBoost delay model                            │
│  + auto-recorded MAE baseline                   │
│  → S3, date/time-versioned                      │
└─────────────────────────────────────────────────┘
                         │
                         ▼
╔═════════════════════════════════════════════════╗
║               Live app  ·  webui/               ║
╟─────────────────────────────────────────────────╢
║  FastAPI + Jinja2 SSR                           ║
║  Vanilla JS + Leaflet, containerised            ║
╚═════════════════════════════════════════════════╝
                         │
                         ▼
┌─────────────────────────────────────────────────┐
│                  EC2 t4g.medium                 │
├─────────────────────────────────────────────────┤
│  Docker, always-on, IAM instance role           │
└─────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────┐
│                    CloudFront                   │
├─────────────────────────────────────────────────┤
│  HTTPS, edge-cached /static/*                   │
└─────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────┐
│                    End users                    │
└─────────────────────────────────────────────────┘
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

## Tech Stack

- **Modelling:** Python 3.12, XGBoost, pandas, scikit-learn, pyarrow
- **Serving:** FastAPI, Jinja2, vanilla JS, Leaflet
- **Infrastructure:** AWS EC2 (Graviton arm64), S3, CloudFront, IAM instance roles
- **Dev:** Docker, systemd timers, GitHub

---

## Repo Structure

```text
Transit-AI/
├── notebooks/             # Feature pipeline, EDA, and baseline model training
├── scripts/
│   └── archive_gtfsrt.py  # Continuous archiver daemon (EC2) — GTFS-RT, static GTFS, performance → S3
├── pipeline/              # Standalone rebuild + retrain scripts
├── config/
│   └── feeds.yaml         # Feed URLs (GTFS-RT combined + per-mode, static, performance)
├── webui/                 # Production live app — FastAPI + Jinja2 SSR + vanilla JS + Leaflet
│   ├── app.py             # FastAPI endpoints: /, /results, /routes,
│   │                      # /stops/search, /stops/nearby, /stops/bbox,
│   │                      # /model/stats, /health
│   ├── templates/         # Jinja2 templates (base, index, results)
│   ├── static/            # CSS, JS, icons — edge-cached via CloudFront
│   └── requirements.txt   # Pinned production dependencies
├── phase3/                # Shared ML modules + retired Streamlit POC (retained for reference)
│   ├── app.py             # Streamlit UI — retired, no longer deployed
│   ├── gtfs_data.py       # Facade re-exporting gtfs/ below — external imports unchanged
│   ├── gtfs/
│   │   ├── loader.py      # GTFS static snapshot load, ferry exclusion, dtype/memory optimization
│   │   ├── search.py      # Typed stop search (fuzzy + proximity-biased) + nearest-stop ranking
│   │   ├── routing.py     # Direct + multi-leg transfer trip finding (BFS)
│   │   └── shapes.py      # Trip shape (road/rail path) lookup
│   ├── prediction.py      # Facade re-exporting ml/ below — external imports unchanged
│   ├── ml/
│   │   ├── training.py    # Model training (chunked read, leakage filter, fit, MAE baseline)
│   │   ├── inference.py   # Feature building + delay prediction/blending at request time
│   │   └── model_io.py    # Model/metadata load, caching, S3 upload
│   ├── ui/                # Streamlit-specific UI helpers (unused since POC retirement)
│   ├── map_picker.py      # Reusable map — stop picker + route path display
│   ├── live_gtfs.py       # Live GTFS-RT fetch, short-lived cache
│   ├── config.py          # Credential resolution (env / .env / Streamlit secrets)
│   ├── route_types.py     # Shared GTFS route_type ↔ mode-name mapping
│   └── tests/             # Diagnostic/smoke scripts (run directly, not pytest)
├── archive/               # Retired earlier iterations (Next.js frontend, previous FastAPI split)
├── Dockerfile             # Root Dockerfile builds the webui/ container for EC2 deployment
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
# Production-parity path (matches what runs on EC2)
docker build -t transit-ai-backend:local .
docker run --rm -p 8080:8080 \
  -e AWS_S3_BUCKET=seq-transit-ai-data-ps \
  -e AWS_DEFAULT_REGION=ap-southeast-2 \
  transit-ai-backend:local

# Faster iteration loop (Python directly, with hot reload)
pip install -r webui/requirements.txt
uvicorn webui.app:app --reload --port 8080
```

Open [http://localhost:8080](http://localhost:8080). Needs AWS credentials with read access to `s3://seq-transit-ai-data-ps/` — via `.env` at the repo root, an AWS profile, or an IAM role. Production uses an IAM instance role, so no static credentials are baked into the container.

---

## The App

Pick a "From" and "To" stop and a departure time; it finds direct or multi-leg-transfer trips and predicts arrival delay from a trained model.

- Map-based stop selection for both origin and destination — type or use your location, and every nearby matching stop shows as a tappable pin, not a single auto-resolved match
- Typed destination search is proximity-biased toward your already-picked origin, so results near the wrong end of the corridor don't surface just because they share a name
- Distance shown ("180m away", "2.3km away") wherever the app actually knows your position relative to a stop
- Multi-leg transfer routing across bus, rail, and tram
- A single shared route map (not one per result) showing the actual GTFS road/rail path, defaulting to the top-ranked result and updating when you pick a different one
- Station-by-station timeline showing every stop on the leg with transfer points and scheduled times
- Leave-by time as the headline output, with a plain-English "leave now / grab your keys / finish your coffee" contextual pill and a confidence rating (High / Medium / Low)

**Known limitations:**
- Tram training data is still comparatively thin — tram predictions carry lower confidence than bus/rail
- Ferry is intentionally excluded from search, routing, and training (kept in the raw archive only)
- Live GTFS-RT delay is not currently blended into the production prediction — the webui backend serves a model-only estimate. Live-vehicle animation on the timeline and RT-blended predictions are the next UI and modelling iterations
- No natural-language query parsing — stop selection is search-based, not free-text

---

## Progress

**Data & model**
- Continuous GTFS-Realtime archive since late June 2026, now via a dedicated EC2 daemon (migrated off a laptop-based process for reliability)
- Feature pipeline joins realtime data against the static GTFS snapshot actually in effect on each date, ferry-excluded
- XGBoost delay model, retrained periodically, with an automatically recorded MAE baseline on every run and date/time-versioned model storage for rollback
- Current baseline, trained on the full ~156M-row ferry-free archive across 46 source-date partitions: train MAE 1.999 min, test MAE 1.957 min vs. a naive median baseline of 2.486 min — a 21.3% improvement over naive

**App**
- Full trip search: map-based stop pickers, direct + multi-leg transfer routing, leave-by output, live route map with real path shapes, station timeline
- Deployed to production on AWS EC2 + CloudFront (FastAPI + Jinja2 SSR + vanilla JS)

**Next**
- Live GTFS-RT vehicle-position animation on the results timeline
- v1 model retrain with live-delay wired in as an input feature
- Broader confidence-model refinement as more live traffic is observed
- Continued tram data accumulation

---

## License & Data Attribution

TransLink open data is published under CC-BY. This project archives and derives features from that public data; it does not redistribute TransLink's raw feeds.

---

## Author

**Proteek Kumar Sanyal** — ML Engineer & Data Scientist, Gold Coast, Australia. Data Analyst at the City of Gold Coast.
