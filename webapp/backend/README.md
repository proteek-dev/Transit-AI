# Transit-AI backend

Step 1 of the FastAPI/Next.js replacement for the Streamlit app (`phase3/app.py`) — additive only, `phase3/` untouched.

## Run locally

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
