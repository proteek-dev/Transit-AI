# Build from the REPO ROOT, not webapp/backend/ -- this Dockerfile copies
# phase3/ files that live outside webapp/backend/, so the build context must
# include both:
#   docker build -f webapp/backend/Dockerfile -t transit-ai-backend .
#
# pyarrow==25.0.1 (pinned in webapp/backend/requirements.txt) has confirmed
# prebuilt manylinux_2_28 wheels for cp312 (checked against PyPI directly),
# and python:3.12-slim's Debian bookworm base satisfies that glibc floor --
# no source build / compiler needed for pyarrow or any other pinned dep.
FROM python:3.12-slim

WORKDIR /app

# Install only webapp/backend's own requirements -- not phase3/requirements.txt,
# which pulls in streamlit/folium/geopandas/jupyter/etc. that this service's
# import chain never touches (verified by tracing every import reachable from
# main.py: phase3/config.py, route_types.py, gtfs_data.py, live_gtfs.py,
# prediction.py, gtfs/*.py, ml/*.py -- the only third-party packages that
# chain imports are pandas, pyarrow (via gtfs/loader.py's optimized path),
# s3fs, requests, gtfs-realtime-bindings, rapidfuzz, joblib, xgboost, psutil,
# all already listed below).
COPY webapp/backend/requirements.txt ./webapp/backend/requirements.txt
RUN pip install --no-cache-dir -r webapp/backend/requirements.txt

# phase3/ is not a package (no __init__.py -- its modules assume phase3/
# itself is on sys.path; see main.py's sys.path.insert(0, PHASE3_DIR), where
# PHASE3_DIR resolves two directories up from main.py). Copy only the files
# main.py's import chain actually reaches, not the whole phase3/ tree:
#   gtfs_data    -> gtfs/loader, gtfs/routing, gtfs/search, gtfs/shapes
#   prediction   -> live_gtfs, ml/inference, ml/model_io
#   ml/model_io  -> config (+ lazily ml/training, only on the no-saved-model
#                   fallback path -- never hit in practice since S3 always
#                   has a saved model, but the module must exist so that
#                   lazy import doesn't ModuleNotFoundError if it ever runs)
#   gtfs/loader  -> config, route_types
# Deliberately excluded: app.py, map_picker.py, ui/, tests/, model/ (that
# last one is .gitignore'd and fetched from S3 at runtime, never baked in).
COPY phase3/config.py phase3/route_types.py phase3/gtfs_data.py phase3/live_gtfs.py phase3/prediction.py ./phase3/
COPY phase3/gtfs/ ./phase3/gtfs/
COPY phase3/ml/ ./phase3/ml/

COPY webapp/backend/ ./webapp/backend/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# uvicorn's Config inserts this directory onto sys.path before importing
# "main:app", the same way it would if you ran `uvicorn main:app` by hand
# from webapp/backend/ -- no PYTHONPATH juggling needed.
WORKDIR /app/webapp/backend

# $PORT is injected by Cloud Run at runtime -- shell form (no brackets) so
# /bin/sh expands it, instead of exec form passing the literal argv token
# "$PORT" through unexpanded. No --reload: this is the production entrypoint.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT"]
