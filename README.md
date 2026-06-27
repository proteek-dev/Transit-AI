# SEQ Transit AI — Phase 1: Data Collection & EDA

**Archiving TransLink SEQ GTFS-Realtime feeds to build the historical delay dataset that doesn't exist anywhere else.**

## Why This Matters

TransLink has no public stop-level historical delay data. The GTFS-Realtime API gives you a snapshot of live delays right now — but the moment you close the connection, that data is gone forever. This archiver captures every snapshot every 5 minutes and commits it to this repo, creating the first public stop-level delay history for South East Queensland.

## Data Sources

| Source | What | Auth | URL |
|---|---|---|---|
| GTFS-Realtime (SEQ) | Live delays, positions, alerts | None (public) | gtfsrt.api.translink.com.au |
| GTFS Static (SEQ) | Routes, stops, timetables | None (public) | gtfsrt.api.translink.com.au/GTFS/SEQ_GTFS.zip |
| Monthly Performance | Historical on-time rates by route | None (public portal) | data.qld.gov.au |

All data is CC-BY licensed. No API key required.

## Setup

```bash
pip install -r requirements.txt
```

**Start archiving immediately:**
```bash
python scripts/archive_gtfsrt.py
```

**Monthly performance data** — download CSVs manually from the portal and drop them into `source_files/performance/`:
```
https://www.data.qld.gov.au/dataset/translink-monthly-performance-data
```

## Folder Structure

```
Transit-AI/
├── source_files/
│   ├── gtfs_realtime/
│   │   ├── trip_updates/        ← GTFS-RT protobuf → JSON, every 5 min
│   │   ├── vehicle_positions/   ← GTFS-RT protobuf → JSON, every 5 min
│   │   └── service_alerts/      ← GTFS-RT protobuf → JSON, every 5 min
│   ├── gtfs_static/             ← SEQ_GTFS.zip extracted CSVs + Parquet, refreshed daily
│   └── performance/             ← TransLink monthly on-time running CSVs (manual download)
├── notebooks/
│   ├── 01_archive_gtfsrt.ipynb        ← fetch + archive GTFS-RT feeds (one-off/test)
│   ├── 02_load_static_gtfs.ipynb      ← parse static GTFS into DataFrames → Parquet
│   ├── 03_load_performance_data.ipynb ← load + clean monthly on-time running data
│   └── 04_eda.ipynb                   ← EDA: delay patterns by route, time, mode
├── scripts/
│   └── archive_gtfsrt.py              ← continuous archiver (runs indefinitely)
├── config/
│   └── feeds.yaml                     ← all feed URLs in one place
├── requirements.txt
└── README.md
```

## Notebook Sequence

Run in order:

1. **`01_archive_gtfsrt.ipynb`** — test fetch and one-off archive (or just run the script)
2. **`02_load_static_gtfs.ipynb`** — download static GTFS, save as Parquet
3. **`03_load_performance_data.ipynb`** — load monthly CSVs, standardise, save as Parquet
4. **`04_eda.ipynb`** — full EDA: worst routes, mode comparison, trends, GTFS-RT delay distribution

## Auto-Commit

`scripts/archive_gtfsrt.py` runs a `git commit + push` every 30 minutes, keeping this repo as the live archive. Each commit message follows the format:

```
archive: 2024-01-15 14:30 | trip_updates + vehicle_positions + alerts
```

This means the repo accumulates ~48 commits per day of archiving — one for each 30-minute window.
