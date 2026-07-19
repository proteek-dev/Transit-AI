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

## Deployment

**Local:**

```
cd phase3
streamlit run app.py
```

**Phone (same WiFi):**

```
streamlit run app.py --server.address 0.0.0.0
```

Then open `http://<local-ip>:8501` from the phone (see "Phone access" above
for finding `<local-ip>`).

**Streamlit Community Cloud:**

1. Push this repo to GitHub (already done for `work_ai`).
2. On [share.streamlit.io](https://share.streamlit.io), create a new app
   connected to this GitHub repo, with main file path `phase3/app.py`.
3. In the app's Settings → Secrets, paste in the contents of
   `phase3/.streamlit/secrets.toml.example` with real AWS credentials filled
   in (never commit the real values — see that file's comments).
4. Deploy. The model is loaded from S3
   (`s3://seq-transit-ai-data-ps/phase3/model/`) — no local model files or
   `.env` are needed on Cloud since `phase3/config.py` reads credentials
   from `st.secrets` there.
