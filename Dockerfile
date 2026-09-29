# Build from the REPO ROOT:
#   docker build -t transit-ai-webui .
#
# webui/app.py serves the SEQ Transit AI web frontend + JSON API on
# ${PORT:-8080} (behind CloudFront distribution E24LUW13GKO74P). It imports
# phase3/ modules via sys.path at startup (webui/app.py line 86), which is
# why phase3/ files have to be copied even though webui/ doesn't
# `from phase3 import` at module scope.
#
# pyarrow==25.0.1 (pinned in webui/requirements.txt for parity with the
# precompute venv on the EC2 box at /opt/transit-ai/.venv) has confirmed
# prebuilt manylinux_2_28 wheels for cp312, and python:3.12-slim's Debian
# bookworm base satisfies that glibc floor -- no source build or compiler
# needed for pyarrow or any other pinned dep.
FROM python:3.12-slim

WORKDIR /app

# Install webui's own requirements only -- NOT the repo-root requirements.txt,
# which pulls in streamlit/jupyterlab/geopandas/matplotlib/etc. that this
# service's import chain never touches. (Root requirements.txt is
# .dockerignored to enforce this.)
COPY webui/requirements.txt ./webui/requirements.txt
RUN pip install --no-cache-dir -r webui/requirements.txt

# phase3/ is not a package (no __init__.py -- its modules assume phase3/
# itself is on sys.path; see webui/app.py's sys.path.insert). Copy only the
# files webui/app.py's import chain reaches, not the whole phase3/ tree:
#   gtfs_data    -> gtfs/loader, gtfs/routing, gtfs/search, gtfs/shapes
#   prediction   -> live_gtfs, ml/inference, ml/model_io
#   ml/model_io  -> config (+ lazily ml/training, only on the no-saved-model
#                   fallback path -- never hit in practice since S3 always
#                   has a saved model, but the module must exist so that
#                   lazy import doesn't ModuleNotFoundError if it ever runs)
#   gtfs/loader  -> config, route_types
# Deliberately excluded: phase3/app.py (Streamlit POC), phase3/map_picker.py,
# phase3/tests/, phase3/model/ (.gitignore'd; fetched from S3 at runtime).
COPY phase3/config.py phase3/route_types.py phase3/gtfs_data.py phase3/live_gtfs.py phase3/prediction.py ./phase3/
COPY phase3/gtfs/ ./phase3/gtfs/
COPY phase3/ml/ ./phase3/ml/

# webui/ package: app.py, static/ (app.css, app.js, icons.js),
# templates/ (base.html, index.html, results.html). Works as an implicit
# namespace package (PEP 420) -- no __init__.py needed.
COPY webui/ ./webui/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# CMD matches Proteek's local dev command exactly:
#   PYTHONUNBUFFERED=1 uvicorn webui.app:app --port 8080 --log-level info
# ${PORT:-8080} gives a default so `docker run` works standalone. Fixes the
# branch-audit gotcha where the archived Dockerfile's $PORT had no default
# (Session 31's paste-vanished-$PORT bug -- container silently served on
# 8000, blocked the deploy).
CMD ["sh", "-c", "uvicorn webui.app:app --host 0.0.0.0 --port ${PORT:-8080}"]
