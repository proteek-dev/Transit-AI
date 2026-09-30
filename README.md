# SEQ Transit AI

**Weather-app-style delay confidence predictions for South East Queensland public transport.**

Live at **https://d17e9wvem3lbaq.cloudfront.net** — Gold Coast ↔ Brisbane corridor, all modes except ferry.

---

## What this is

A production ML app that answers a question TransLink and Google Maps don't:
**"Should I trust this scheduled trip, or will it run late?"**

Instead of just showing timetables, SEQ Transit AI serves a **per-journey delay
prediction** with a **confidence label** — High, Medium, or Low — trained on a
self-collected archive of GTFS-RT observations from the SEQ network.

The differentiator is the confidence layer. A journey planner tells you when
the bus is *scheduled*. A live-tracking app tells you where it *is right now*.
This app tells you what the model believes about how the trip will actually run,
and how much to trust that belief.

## Model baseline (v0)

| Metric | Value |
|---|---|
| Algorithm | XGBoost regression |
| Training rows | 117,157,755 |
| Test rows | 26,415,212 |
| Test MAE | **1.957 min** |
| Naive median MAE | 2.486 min |
| Improvement over naive | **21.3%** |
| Archive days | 46 |
| Trained on | 2026-08-14 |

Confidence assignment blends live GTFS-RT delay with the model output when live
data is available; falls back to model-only when it isn't. Out-of-vocabulary
stops force Low confidence rather than silently degrading.

## Architecture
                 TransLink GTFS + GTFS-RT
                          │
                          ▼
 ┌─────────────────────────────────────────────┐
 │   Archiver — EC2 t4g.small + systemd        │
 │   Polls combined + tram-dedicated feeds     │
 │   every 5 min, writes raw JSON to S3        │
 └─────────────────────────────────────────────┘
                          │
                          ▼
                S3 (ap-southeast-2)
                          │
                          ▼
 ┌─────────────────────────────────────────────┐
 │   Feature pipeline (notebook 05)            │
 │   Per-source-date snapshot matching,        │
 │   chunked read, categorical-dtype parquet   │
 └─────────────────────────────────────────────┘
                          │
                          ▼
 ┌─────────────────────────────────────────────┐
 │   XGBoost training (notebook 07)            │
 │   Temporal split, leakage filter,           │
 │   RSS-tracked chunked training              │
 └─────────────────────────────────────────────┘
                          │
                          ▼
 ┌─────────────────────────────────────────────┐
 │   FastAPI backend (webui/)                  │
 │   Jinja2 SSR + vanilla JS + Leaflet         │
 │   EC2 t4g.medium, Docker, always-on         │
 └─────────────────────────────────────────────┘
                          │
                          ▼
            CloudFront (HTTPS, edge cache /static/*)
                          │
                          ▼
                   End users

## Tech stack

- **Modelling:** Python 3.12, XGBoost, pandas, scikit-learn, pyarrow
- **Serving:** FastAPI, Jinja2, vanilla JS, Leaflet
- **Infrastructure:** AWS EC2 (Graviton arm64), S3, CloudFront, IAM instance roles
- **Data:** TransLink GTFS + GTFS-RT (combined feed + dedicated tram endpoints),
  Open-Meteo Historical Weather API (ERA5 reanalysis)
- **Dev:** Docker, systemd timers, GitHub

## Repository layout

webui/ Production FastAPI app + templates + static
phase3/ ML modules (predict, routing, GTFS loader, live GTFS-RT)
Retired Streamlit POC lives here — retained for reference
scripts/ Archiver daemon + precompute
pipeline/ Rebuild + train shell wrappers
notebooks/ Phase 1–2 feature engineering and model training
config/ Feed URLs and archive settings
archive/ Retired earlier iterations (kept for history)


## Running locally

Requires Python 3.12, Docker, and AWS credentials scoped to
`s3://seq-transit-ai-data-ps/`.

```bash
# Container path (matches production)
docker build -t transit-ai-backend:local .
docker run --rm -p 8080:8080 \
  -e AWS_S3_BUCKET=seq-transit-ai-data-ps \
  -e AWS_DEFAULT_REGION=ap-southeast-2 \
  transit-ai-backend:local

# Local Python path
pip install -r webui/requirements.txt
uvicorn webui.app:app --reload --port 8080
```

Open http://localhost:8080.

## Data provenance

All GTFS and GTFS-RT data is sourced from TransLink's public feeds under the
Queensland Government's open-data licence. The self-collected archive powering
the model is a rolling record of feed observations — it is *not* redistributed.

## Status

Active development on `work_ai`. `main` reflects the most recent
production-tagged release. Session-by-session decisions and learnings are kept
in the maintainer's Claude project rather than the repo tree.

## Author

**Proteek Kumar Sanyal** — ML Engineer & Data Scientist, Gold Coast, Australia.
Data Analyst at the City of Gold Coast.

- LinkedIn: https://www.linkedin.com/in/proteeksanyal/
- GitHub: [@proteek-dev](https://github.com/proteek-dev)
