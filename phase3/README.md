# Phase 3 — SEQ Transit AI (Streamlit POC)

A local proof-of-concept trip planner for South East Queensland public transport.
Pick a "From" and "To" stop and a departure time, and it finds direct GTFS
trips, then estimates arrival delay by blending a baseline XGBoost model
(trained on ~21 days of historical GTFS-RT data) with TransLink's live
GTFS-RT feed when available.

## How to run

```
cd phase3
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## Prerequisites

- A `.env` file at the repo root with `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`,
  `AWS_REGION`, `AWS_S3_BUCKET` (same as Steps 1/2).
- Python deps installed from the repo root: `pip install -r requirements.txt`.
- On first run, if no saved model exists yet at `phase3/model/xgb_v0.json`,
  `prediction.py` will train it from the S3 feature snapshot — this can take
  a few minutes. Subsequent runs load the saved model instantly.

## Phone access

To open the app on your phone, connect it to the **same WiFi network** as
the machine running Streamlit, then find that machine's local IP (e.g. on
macOS: `ipconfig getifaddr en0`) and browse to `http://<local-ip>:8501` from
the phone.
