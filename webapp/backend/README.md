# Transit-AI backend

Step 1 of the FastAPI/Next.js replacement for the Streamlit app (`phase3/app.py`) — additive only, `phase3/` untouched.

## Run locally

`/model/stats` fetches `training_metadata.json` from S3 at startup (see below), so these env vars must be set before starting the server — e.g. via `export`, or `uvicorn --env-file`:

- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_DEFAULT_REGION`
- `AWS_S3_BUCKET` (same bucket phase3/ uses)

```bash
cd webapp/backend
pip install -r requirements.txt
uvicorn main:app --reload
```

Then `GET http://127.0.0.1:8000/model/stats`.

## Deployment

Deployed to [Render](https://render.com) — the service itself is set up manually in the Render dashboard, not automated here. `render.yaml` documents the build/start command and plan Render uses, so the config is committed and reproducible instead of living only in the dashboard.

- `GET /health` exists solely as a target for an external keep-alive cron (pings every ~10 min). It does no file I/O and returns no real data — just `{"status": "ok"}`.
- `GET /model/stats` is the actual functional endpoint the frontend will eventually call.

### Model stats and S3

`training_metadata.json` lives under `phase3/model/`, which is `.gitignore`'d and never deployed — Render has no local copy. Instead, `/model/stats` fetches the object from S3 **once, at process startup**, and caches it in memory; the endpoint always serves that cached copy and never makes an S3 call (or reads a local file) per-request. If the startup fetch fails, `/model/stats` returns `503` with a message explaining the fetch failed, rather than the app crashing.

One consequence: a new model retrain (which writes a fresh `training_metadata.json` to S3) is **not** picked up automatically — the cached copy only refreshes on the next process start. After a retrain, trigger a Render redeploy (or restart the service) to pick up the new stats.

Credentials use boto3's standard credential chain — set these as environment variables in the Render dashboard (**Settings → Environment**), never in `render.yaml`:

- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_DEFAULT_REGION`
- `AWS_S3_BUCKET`
