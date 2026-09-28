"""webui — Transit AI single-process web UI.

Run from the repo root:  uvicorn webui.app:app

Serves the HTML page at / plus the JSON API ported from
archive/webapp-v1-fastapi-nextjs-split/backend/main.py (/health, /model/stats,
/stops/search, /routes), with the response shapes of that archive's
frontend/lib/types.ts.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import boto3
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

logger = logging.getLogger('uvicorn.error')

REPO_ROOT = Path(__file__).resolve().parents[1]
# phase3/ isn't a package: its modules import each other as `import config`,
# `from gtfs.loader import ...`, so phase3/ itself has to be on sys.path.
PHASE3_DIR = REPO_ROOT / 'phase3'

# Same bucket env var and keys as archive/webapp-v1-fastapi-nextjs-split/backend/main.py.
S3_BUCKET_ENV_VAR = 'AWS_S3_BUCKET'
MODEL_SUBDIR = os.environ.get('MODEL_SUBDIR_OVERRIDE', 'latest')
S3_METADATA_KEY = f'phase3/model/{MODEL_SUBDIR}/training_metadata.json'
S3_FEATURE_MANIFEST_KEY = 'ml_features/v0_feature_snapshot/_latest.json'

UNKNOWN = 'unknown'

BRISBANE_TZ = ZoneInfo('Australia/Brisbane')

# Same search -> rank -> truncate constants as phase3/app.py and the archived
# backend: predict a wider pool, rank by predicted arrival, return the top 5.
CANDIDATE_POOL_SIZE = 10
MAX_RESULTS = 5
WINDOW_MINUTES = 60

# /model/stats graph_status threshold, from the archived backend.
HEALTHY_STOP_TIMES_MIN_ROWS = 1_000_000


def _fetch_s3_json(key: str) -> dict:
    bucket = os.environ.get(S3_BUCKET_ENV_VAR)
    if not bucket:
        raise RuntimeError(f'{S3_BUCKET_ENV_VAR} environment variable is not set')
    # boto3's standard credential chain (AWS_ACCESS_KEY_ID /
    # AWS_SECRET_ACCESS_KEY / AWS_DEFAULT_REGION env vars) -- nothing hardcoded.
    s3 = boto3.client('s3')
    response = s3.get_object(Bucket=bucket, Key=key)
    return json.loads(response['Body'].read())


def _fetch_s3_json_or_none(key: str, label: str) -> dict | None:
    try:
        return _fetch_s3_json(key)
    except Exception:
        logger.exception('[webui] failed to fetch %s (s3 key %s) -- continuing with None', label, key)
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Local dev: load the repo-root .env regardless of CWD. override=False so
    # real env vars always win; no-op when the file is absent.
    load_dotenv(REPO_ROOT / '.env', override=False)

    if str(PHASE3_DIR) not in sys.path:
        sys.path.insert(0, str(PHASE3_DIR))

    # phase3 imports live here, not at module top, so an import failure lands
    # in uvicorn's startup log. Imported by the same names phase3 uses
    # internally -- importing as `phase3.ml.model_io` too would create a second
    # module object with its own cache.
    import gtfs_data  # noqa: F401
    import live_gtfs  # noqa: F401
    import prediction  # noqa: F401
    from gtfs.loader import load_gtfs_data_optimized
    from ml import model_io

    logger.info('[webui] loading gtfs…')
    # GTFSData().load_optimized(), stored in the loader's shared cache so
    # gtfs_data's routing helpers reuse it instead of re-parsing the CSVs.
    app.state.gtfs = load_gtfs_data_optimized()
    logger.info(
        '[webui] gtfs loaded snapshot=%s rows=%d',
        app.state.gtfs.snapshot_date, len(app.state.gtfs.stop_times),
    )

    logger.info('[webui] loading model…')
    app.state.model = model_io.load_model()
    logger.info('[webui] model loaded id=%d', id(app.state.model))
    logger.info('[webui] load_model callable id=%d', id(model_io.load_model))

    # model_io.load_model is st.cache_resource(_load_model_impl) whenever
    # streamlit is importable. Outside a Streamlit runtime that cache has been
    # seen re-downloading the model from S3 minutes after startup, so pin every
    # reference to the instance loaded above. load_model takes no arguments,
    # so a constant-returning replacement is safe. inference.py and
    # prediction.py each bind load_model by name at import, so patching only
    # model_io would leave predict_delay() on the original.
    import prediction as _prediction
    from ml import inference as _inference

    _cached_model = app.state.model

    def _load_cached_model():
        return _cached_model

    model_io.load_model = _load_cached_model
    _inference.load_model = _load_cached_model
    _prediction.load_model = _load_cached_model
    logger.info('[webui] load_model patched to return cached instance')

    app.state.training_metadata = _fetch_s3_json_or_none(S3_METADATA_KEY, 'training_metadata.json')
    app.state.features_manifest = _fetch_s3_json_or_none(S3_FEATURE_MANIFEST_KEY, 'features _latest.json')

    logger.info('[webui] ready')
    yield


app = FastAPI(lifespan=lifespan, title='Transit AI')
app.mount('/static', StaticFiles(directory='webui/static'), name='static')
templates = Jinja2Templates(directory='webui/templates')


def _schedule_snapshot(state) -> str:
    gtfs = getattr(state, 'gtfs', None)
    return getattr(gtfs, 'snapshot_date', None) or UNKNOWN


def _model_trained(state) -> str:
    metadata = getattr(state, 'training_metadata', None) or {}
    return metadata.get('trained_at') or UNKNOWN


def _features_through(state) -> str:
    # Falls back to the model's training-window end when the features manifest
    # couldn't be fetched (expected in prod: EC2 role lacks GetObject on it).
    manifest = getattr(state, 'features_manifest', None)
    if manifest is not None:
        date_range = manifest.get('source_date_range') or []
        return max(date_range) if date_range else UNKNOWN
    metadata = getattr(state, 'training_metadata', None) or {}
    return metadata.get('data_window_end') or UNKNOWN


@app.get('/')
def index(request: Request):
    state = request.app.state
    return templates.TemplateResponse(request, 'index.html', {
        'schedule_snapshot': _schedule_snapshot(state),
        'model_trained': _model_trained(state),
        'features_through': _features_through(state),
    })


# ── JSON API (ported from archive/webapp-v1-fastapi-nextjs-split/backend/main.py) ──
#
# Handlers are plain `def`, so FastAPI runs them in its threadpool -- the
# blocking pandas/xgboost work never ties up the event loop (the archived
# backend got the same effect with asyncio.to_thread). phase3 modules are
# imported inside functions: lifespan has already put phase3/ on sys.path and
# loaded them, so these imports are dict lookups, not loads. GTFS and the model
# come from the caches lifespan warmed; nothing here reloads either.
#
# Optional[...] rather than `X | None` in route signatures: FastAPI evaluates
# them at runtime, and this runs on Python 3.9.


@app.get('/health')
def health() -> dict:
    """Cheap keep-alive target -- no app.state, no I/O."""
    return {'status': 'ok'}


def _graph_status(gtfs) -> str:
    """'healthy' at >= HEALTHY_STOP_TIMES_MIN_ROWS stop_times rows, 'empty' at 0
    rows or no GTFS loaded, 'unknown' for a non-zero count below the threshold
    -- same thresholds as the archived backend.
    """
    try:
        if gtfs is None or gtfs.stop_times is None:
            return 'empty'
        rows = len(gtfs.stop_times)
    except Exception:
        return 'unknown'
    if rows == 0:
        return 'empty'
    return 'healthy' if rows >= HEALTHY_STOP_TIMES_MIN_ROWS else 'unknown'


@app.get('/model/stats')
def model_stats(request: Request) -> dict:
    state = request.app.state
    metadata = getattr(state, 'training_metadata', None)
    if metadata is None:
        raise HTTPException(
            status_code=503,
            detail='Training metadata unavailable: S3 fetch failed at startup (see server log)',
        )

    # features_through keeps the archived semantics -- null when the features
    # manifest couldn't be fetched. (The page footer's _features_through()
    # falls back to data_window_end instead; the API deliberately doesn't.)
    manifest = getattr(state, 'features_manifest', None) or {}
    date_range = manifest.get('source_date_range') or []
    features_through = max(date_range) if date_range else None

    # The archived backend read this from gtfs_static_optimized/latest/_manifest.json;
    # webui reports the snapshot date of the GTFS actually loaded in memory.
    gtfs = getattr(state, 'gtfs', None)
    gtfs_static_snapshot = getattr(gtfs, 'snapshot_date', None)

    return {
        'training_rows': sum(metadata['coverage_counts'].values()),
        'days_archived': metadata['data_window_days'],
        'test_mae': metadata['test_mae'],
        'naive_mae': metadata['naive_mae'],
        'pct_improvement_over_naive': metadata['pct_improvement_over_naive'],
        'model_type': 'XGBoost (v0)',
        'training_window': {
            'start': metadata['data_window_start'],
            'end': metadata['data_window_end'],
        },
        'data_snapshot': {
            'features_through': features_through,
            'model_trained_through': metadata['data_window_end'],
            'gtfs_static_snapshot': gtfs_static_snapshot,
            'graph_status': _graph_status(gtfs),
        },
    }


def _to_stop_search_result(row: dict) -> dict:
    """Coerce one search_stops() dict to types.ts's StopSearchResult. The index
    carries numpy float32 coordinates, which FastAPI's JSON encoder can't
    serialize. distance_km stays absent (not null) when search_stops() omits it.
    """
    out = {
        'stop_id': str(row['stop_id']),
        'stop_ids': [str(s) for s in row['stop_ids']],
        'stop_name': str(row['stop_name']),
        'stop_lat': float(row['stop_lat']),
        'stop_lon': float(row['stop_lon']),
    }
    if 'distance_km' in row:
        out['distance_km'] = float(row['distance_km'])
    return out


@app.get('/stops/search')
def stops_search(
    q: str = Query(..., min_length=1, max_length=100),
    # ge=1 is load-bearing: search_stops() slices results[:limit], so a
    # negative limit would silently drop results from the end.
    limit: int = Query(8, ge=1, le=20),
    lat: Optional[float] = Query(None, ge=-90, le=90),
    lon: Optional[float] = Query(None, ge=-180, le=180),
) -> list:
    """Typeahead stop search -- wraps phase3/gtfs/search.py's search_stops()."""
    if (lat is None) != (lon is None):
        raise HTTPException(status_code=422, detail='lat and lon must be provided together or both omitted')
    from gtfs.search import search_stops

    try:
        rows = search_stops(q, limit=limit, ref_lat=lat, ref_lon=lon)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f'Stop search unavailable: {e}')
    return [_to_stop_search_result(r) for r in rows]


def _parse_departure(departure: str | None) -> datetime:
    """None -> naive Brisbane-local "now" (routing.py compares naive datetimes
    throughout). An ISO string with an offset is converted to Brisbane time
    then stripped; a naive one is taken as Brisbane wall-clock time.
    """
    if departure is None:
        now = datetime.now(BRISBANE_TZ)
        return datetime.combine(now.date(), now.time())
    try:
        parsed = datetime.fromisoformat(departure)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f'Invalid departure datetime: {departure!r} (expected ISO 8601)',
        )
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(BRISBANE_TZ).replace(tzinfo=None)
    return parsed


def _predict_leg(trip: dict, dest_stop_ids: list[str], search_departure_after: datetime) -> tuple[dict, dict] | None:
    """Enrich + predict_delay() one leg. live_delay is always None, as in the
    archived backend, so real predictions only come back Medium/Low
    confidence. Returns None on failure so one bad leg doesn't fail the request.
    """
    import prediction

    try:
        enriched = prediction.enrich_trip_with_dest_stop(trip, dest_stop_ids)
        pred = prediction.predict_delay(enriched, search_departure_after, live_delay=None)
    except Exception as e:
        logger.warning('[webui] /routes could not predict leg (trip_id=%r): %s', trip.get('trip_id'), e)
        return None
    return enriched, pred


def _predict_journey_legs(journey: dict) -> list[tuple[dict, dict]] | None:
    """All legs of a journey, enriched + predicted. None if any leg failed --
    a JSON leg can't represent "prediction failed", so the journey is dropped.
    """
    legs = []
    for leg in journey['legs']:
        result = _predict_leg(leg['trip'], leg['dest_stop_ids'], leg['search_departure_after'])
        if result is None:
            return None
        legs.append(result)
    return legs


def _serialize_journey(journey: dict, leg_predictions: list[tuple[dict, dict]], predicted_arrival: datetime) -> dict:
    from route_types import MODE_BY_ROUTE_TYPE

    legs_out = []
    for trip, pred in leg_predictions:
        mode = trip.get('mode') or MODE_BY_ROUTE_TYPE.get(trip.get('route_type'), 'unknown')
        leg_predicted_arrival = trip['dest_arrival_time'] + timedelta(minutes=pred['blended_delay_minutes'])
        legs_out.append({
            'stop_id': trip['dest_stop_id'],
            'route_short_name': trip.get('route_short_name') or trip['route_id'],
            'mode': mode,
            'departure_time': trip['origin_departure_time'].isoformat(),
            'predicted_arrival': leg_predicted_arrival.isoformat(),
            'confidence': pred['confidence'],
        })
    first_departure = leg_predictions[0][0]['origin_departure_time']
    total_predicted_duration_minutes = int(round((predicted_arrival - first_departure).total_seconds() / 60))
    return {
        'legs': legs_out,
        'total_predicted_duration_minutes': total_predicted_duration_minutes,
        'transfer_count': journey['num_transfers'],
    }


def _expand_stop_group(stop_id: str) -> list[str]:
    """Mirrors phase3's station-group expansion (stop_to_cluster -> cluster_stop_ids, as in gtfs/routing.py)."""
    from gtfs import loader as gtfs_loader  # added for webui: phase3 imports are function-local here

    data = gtfs_loader.load_gtfs_data()
    cluster = data.stop_to_cluster.get(stop_id)
    group = data.cluster_stop_ids.get(cluster, [stop_id]) if cluster else [stop_id]
    if group != [stop_id]:
        logger.debug('expanded stop_id %s -> %s', stop_id, group)
    return group


def _find_ranked_routes(from_stop_id: str, to_stop_id: str, departure_after: datetime) -> list[dict]:
    """Direct + transfer journeys (gtfs_data's BFS), every leg predicted,
    ranked by predicted (not scheduled) arrival -- the same pipeline
    phase3/app.py runs for its cards. Returns [] (not an error) when BFS finds
    nothing, e.g. the known HOTA -> Surfers Paradise gap.

    Each stop_id is expanded to its full station group first, the same
    stop_ids list phase3/app.py's pickers hand find_trips(): the UI sends one
    member (search_stops()'s stop_ids[0]), which alone can be a bus bay or a
    single-direction platform that no route from the other side reaches.
    """
    import gtfs_data

    origin_stop_ids = _expand_stop_group(from_stop_id)
    dest_stop_ids = _expand_stop_group(to_stop_id)

    trips = gtfs_data.find_trips(origin_stop_ids, dest_stop_ids, departure_after, window_minutes=WINDOW_MINUTES)
    transfer_journeys = gtfs_data.find_multi_leg_trips(
        origin_stop_ids, dest_stop_ids, departure_after, window_minutes=WINDOW_MINUTES,
        max_results=CANDIDATE_POOL_SIZE,
    )
    direct_journeys = [
        gtfs_data._direct_journey(trip, dest_stop_ids, departure_after) for trip in trips[:CANDIDATE_POOL_SIZE]
    ]
    # find_multi_leg_trips() runs its own depth-0 direct check too -- keep only
    # num_transfers >= 1 so those aren't double-counted (same filter as app.py).
    transfer_only_journeys = [j for j in transfer_journeys if j['num_transfers'] >= 1]
    candidate_journeys = direct_journeys + transfer_only_journeys

    ranked = []
    for journey in candidate_journeys:
        leg_predictions = _predict_journey_legs(journey)
        if leg_predictions is None:
            continue
        last_trip, last_pred = leg_predictions[-1]
        predicted_arrival = last_trip['dest_arrival_time'] + timedelta(minutes=last_pred['blended_delay_minutes'])
        ranked.append((predicted_arrival, journey, leg_predictions))

    ranked.sort(key=lambda r: r[0])
    return [
        _serialize_journey(journey, leg_predictions, predicted_arrival)
        for predicted_arrival, journey, leg_predictions in ranked[:MAX_RESULTS]
    ]


@app.get('/routes')
def routes(
    request: Request,
    from_stop_id: str = Query(..., min_length=1),
    to_stop_id: str = Query(..., min_length=1),
    departure: Optional[str] = Query(None),
) -> list:
    departure_after = _parse_departure(departure)
    # 404 only for stop_ids absent from the GTFS snapshot entirely (Session 32
    # decision) -- a known stop with no services in the window returns [].
    stop_to_cluster = request.app.state.gtfs.stop_to_cluster
    unknown = [sid for sid in (from_stop_id, to_stop_id) if sid not in stop_to_cluster]
    if unknown:
        raise HTTPException(status_code=404, detail=f'Unknown stop_id: {", ".join(unknown)}')

    # Model-cache check on every /routes call (see the load_model patch in
    # lifespan). inference_patched covers the reference predict_delay() calls.
    from ml import inference, model_io

    state_model = request.app.state.model
    logger.info(
        '[webui] /routes: model.state id=%d load_model()==app.state.model=%s inference_patched=%s',
        id(state_model), model_io.load_model() is state_model, inference.load_model is model_io.load_model,
    )
    try:
        return _find_ranked_routes(from_stop_id, to_stop_id, departure_after)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=f'Routing/prediction unavailable: {e}')
