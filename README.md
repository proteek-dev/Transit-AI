# SEQ Transit AI — Phase 1: Data Collection & EDA

Archiving TransLink SEQ GTFS-Realtime feeds and performance data to S3
to build the historical delay dataset that does not exist anywhere else.

## Why This Exists

TransLink has no public stop-level historical delay data. The GTFS-Realtime
API gives a live snapshot of delays — but the moment you close the connection,
that data is gone forever. This archiver captures every snapshot every 5
minutes and uploads it to Amazon S3, building the first stop-level delay
history for South East Queensland.

## Architecture

All data is stored in Amazon S3 (ap-southeast-2). Nothing is written locally
except logs. Notebooks read and write directly from S3 using s3fs.

S3 bucket structure:
  gtfs_realtime/
    trip_updates/        ← JSON per fetch, YYYY-MM-DD/HH-MM-SS.json
    vehicle_positions/   ← JSON per fetch, YYYY-MM-DD/HH-MM-SS.json
    service_alerts/      ← JSON per fetch, YYYY-MM-DD/HH-MM-SS.json
  gtfs_static/
    YYYY-MM-DD/          ← Extracted CSVs from SEQ_GTFS.zip (daily)
    parquet/             ← Parquet files from notebook 02
  performance/           ← TransLink monthly on-time CSVs
  eda_charts/            ← Chart outputs from notebook 04

## Data Sources

| Source | What | Auth |
|---|---|---|
| GTFS-Realtime (SEQ) | Live delays, positions, alerts | None — fully public |
| GTFS Static (SEQ) | Routes, stops, timetables | None — fully public |
| Monthly Performance CSVs | Historical on-time rates | None — fully public |

## Setup

pip install -r requirements.txt

Copy .env.example to .env and fill in your values:
  AWS_ACCESS_KEY_ID=
  AWS_SECRET_ACCESS_KEY=
  AWS_REGION=ap-southeast-2
  AWS_S3_BUCKET=

## Archiver Daemon (macOS launchd)

Runs every 5 minutes, survives reboots, auto-restarts on crash.

  launchctl list | grep transitai          # check status
  launchctl unload ~/Library/LaunchAgents/com.proteek.transitai.plist
  launchctl load ~/Library/LaunchAgents/com.proteek.transitai.plist
  tail -f ~/transit-ai-data/logs/archiver.log

## Notebook Sequence

All notebooks read from and write to S3 directly.

  02_load_static_gtfs.ipynb      — parse static GTFS, save Parquet to S3
  03_load_performance_data.ipynb — load and validate performance CSVs from S3
  04_eda.ipynb                   — EDA and charts, saved to S3

Note: notebook 01 is a legacy test file. Do not run it.

## One-Time Backfill

To migrate existing local data to S3:
  python scripts/backfill_s3.py

## Phase Roadmap

Phase 1 — Data collection and EDA (current)
Phase 2 — ML delay prediction model + live GTFS-RT integration
Phase 3 — React/Vite/Tailwind app with Python confidence API
