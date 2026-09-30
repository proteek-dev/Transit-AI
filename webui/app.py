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
import re
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import boto3
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import RedirectResponse
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

# Journey-level confidence is the weakest leg's: Low < Medium < High.
CONFIDENCE_RANK = {'Low': 0, 'Medium': 1, 'High': 2}
# Same rule as phase3's predict_delay(): leave by = first departure - 3 min.
LEAVE_BY_BUFFER = timedelta(minutes=3)


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
    # stop_id -> (stop_name, lat, lon) off the stop table just loaded, for the
    # /routes legs' from_stop / to_stop. Built once here, not per request.
    stops = app.state.gtfs.stops
    app.state.stop_lookup = {
        str(sid): (str(name), round(float(lat), 6), round(float(lon), 6))
        for sid, name, lat, lon in zip(stops['stop_id'], stops['stop_name'], stops['stop_lat'], stops['stop_lon'])
    }

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


def _format_rows(n: int | None) -> str:
    """Row count for the /results stats card: 156_000_000 -> "156.0M"."""
    if n is None:
        return '—'
    if n >= 1_000_000:
        return f'{n / 1_000_000:.1f}M'
    if n >= 1_000:
        return f'{n / 1_000:.1f}k'
    return str(n)


templates.env.filters['format_rows'] = _format_rows


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


@app.get('/results')
def results_page(
    request: Request,
    # `from` is a Python keyword, hence the aliases (to/departure aliased too
    # for consistency). All optional here so a missing one redirects rather
    # than 422s.
    from_stop_id: Optional[str] = Query(None, alias='from'),
    to_stop_id: Optional[str] = Query(None, alias='to'),
    departure: Optional[str] = Query(None, alias='departure'),
    # The user's location fix from "Use my location", for the hero's walk
    # annotation. Strings, not floats: a bad value drops the pair rather
    # than 422ing the page.
    origin_lat: Optional[str] = Query(None),
    origin_lon: Optional[str] = Query(None),
):
    """Results page shell. The page fetches /routes itself from the echoed
    params, so a pasted /results URL works with no search-page state.
    """
    origin = _parse_origin(origin_lat, origin_lon)
    from_stop_id = (from_stop_id or '').strip()
    to_stop_id = (to_stop_id or '').strip()
    if not from_stop_id or not to_stop_id:
        return RedirectResponse('/', status_code=302)
    state = request.app.state
    return templates.TemplateResponse(request, 'results.html', {
        'schedule_snapshot': _schedule_snapshot(state),
        'model_trained': _model_trained(state),
        'features_through': _features_through(state),
        # Same shape as /model/stats; None (not a 503) when training metadata
        # failed to load at startup.
        'model_stats': _model_stats(state),
        'from_stop_id': from_stop_id,
        'to_stop_id': to_stop_id,
        'departure': (departure or '').strip() or None,
        'origin_lat': origin[0] if origin else None,
        'origin_lon': origin[1] if origin else None,
    })


_DECIMAL_RE = re.compile(r'-?\d+(?:\.\d+)?')


def _parse_origin(lat_raw: str | None, lon_raw: str | None) -> tuple[float, float] | None:
    """(lat, lon) when both are plain decimals in range, else None -- one
    bad or missing value drops both. The regex keeps float()'s extras
    ("nan", "inf", "1_0", "1e3") out.
    """
    if lat_raw is None or lon_raw is None:
        return None
    lat_raw, lon_raw = lat_raw.strip(), lon_raw.strip()
    if not (_DECIMAL_RE.fullmatch(lat_raw) and _DECIMAL_RE.fullmatch(lon_raw)):
        return None
    lat, lon = float(lat_raw), float(lon_raw)
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    return lat, lon


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


def _model_stats(state) -> dict | None:
    """The /model/stats payload, shared with the /results page context.
    None when the training metadata failed to load at startup -- /model/stats
    turns that into a 503, /results passes it through as model_stats=None.
    """
    metadata = getattr(state, 'training_metadata', None)
    if metadata is None:
        return None

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

    coverage_counts = metadata.get('coverage_counts')
    return {
        'training_rows': sum(coverage_counts.values()) if coverage_counts else None,
        'days_archived': metadata.get('data_window_days'),
        'test_mae': metadata.get('test_mae'),
        'naive_mae': metadata.get('naive_mae'),
        'pct_improvement_over_naive': metadata.get('pct_improvement_over_naive'),
        'model_type': 'XGBoost (v0)',
        'training_window': {
            'start': metadata.get('data_window_start'),
            'end': metadata.get('data_window_end'),
        },
        'data_snapshot': {
            'features_through': features_through,
            'model_trained_through': metadata.get('data_window_end'),
            'gtfs_static_snapshot': gtfs_static_snapshot,
            'graph_status': _graph_status(gtfs),
        },
    }


@app.get('/model/stats')
def model_stats(request: Request) -> dict:
    stats = _model_stats(request.app.state)
    if stats is None:
        raise HTTPException(
            status_code=503,
            detail='Training metadata unavailable: S3 fetch failed at startup (see server log)',
        )
    return stats


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


def _to_stop_row(row: dict, *, include_distance: bool) -> dict:
    """Serialize a phase3 stop dict to the API shape.
    include_distance=True  → adds distance_km (from /stops/nearby)
    include_distance=False → omits distance_km entirely (from /stops/bbox)

    nearest_stops() and stops_in_bbox() both split multi-mode stations into
    one row per mode, so route_types always has exactly one entry; mode is
    derived from it with the same mapping /routes legs use, so the map can
    colour the pin.
    """
    from route_types import MODE_BY_ROUTE_TYPE

    out = _to_stop_search_result(row)
    if include_distance:
        out['distance_km'] = float(row['distance_km'])
    else:
        out.pop('distance_km', None)
    route_types = row.get('route_types') or []
    out['mode'] = MODE_BY_ROUTE_TYPE.get(int(route_types[0]), 'unknown') if route_types else 'unknown'
    return out


@app.get('/stops/nearby')
def stops_nearby(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    limit: int = Query(15, ge=1, le=50),
) -> list:
    """Nearest useful stops to (lat, lon) -- wraps phase3/gtfs/search.py's
    nearest_stops(), which ranks by distance blended with service frequency
    (not pure distance) and reads the GTFS cache lifespan already warmed.
    """
    from gtfs.search import nearest_stops

    try:
        rows = nearest_stops(lat, lon, limit=limit)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f'Nearby stops unavailable: {e}')
    return [_to_stop_row(r, include_distance=True) for r in rows]


# Max bbox side, in degrees (~110 km at SEQ latitudes): covers the Gold
# Coast <-> Brisbane corridor while keeping one request's pin count sane.
BBOX_MAX_SIDE_DEGREES = 1.0


@app.get('/stops/bbox')
def stops_bbox(
    west: Optional[float] = Query(None, ge=-180, le=180),
    south: Optional[float] = Query(None, ge=-90, le=90),
    east: Optional[float] = Query(None, ge=-180, le=180),
    north: Optional[float] = Query(None, ge=-90, le=90),
    limit: int = Query(30, ge=1, le=100),
) -> list:
    """Stops inside a map viewport -- wraps phase3/gtfs/search.py's
    stops_in_bbox(), ranked by trip_count (busiest first), one row per mode.
    Rows are StopSearchResult + mode + trip_count, with no distance_km.
    """
    if west is None or south is None or east is None or north is None:
        raise HTTPException(status_code=422, detail='west, south, east, north are all required')
    if west >= east:
        raise HTTPException(status_code=422, detail='west must be less than east')
    if south >= north:
        raise HTTPException(status_code=422, detail='south must be less than north')
    if east - west > BBOX_MAX_SIDE_DEGREES or north - south > BBOX_MAX_SIDE_DEGREES:
        raise HTTPException(status_code=422, detail='bbox too large — max 1 degree per side')
    from gtfs.search import stops_in_bbox

    try:
        rows = stops_in_bbox(west, south, east, north, limit=limit)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f'Bbox stops unavailable: {e}')
    return [
        {**_to_stop_row(r, include_distance=False), 'trip_count': int(r['trip_count'])}
        for r in rows
    ]


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


def _stop_ref(stop_id: str, fallback_name: str | None, stop_lookup: dict) -> dict:
    """A leg endpoint for the map: {stop_id, stop_name, lat, lon}. lat/lon are
    null only if the stop_id is somehow missing from the loaded stop table.
    """
    name, lat, lon = stop_lookup.get(str(stop_id), (fallback_name or str(stop_id), None, None))
    return {'stop_id': str(stop_id), 'stop_name': name, 'lat': lat, 'lon': lon}


def _leg_shape(trip_id: str, origin_stop_id: str, dest_stop_id: str) -> list[list[float]] | None:
    """The leg's ridden segment of its trip shape as [[lat, lon], ...], or None
    when the trip has no shape. No straight-line stand-in here -- the client
    draws that fallback. A lookup failure degrades to None rather than
    dropping the route.
    """
    from gtfs.shapes import get_trip_shape_points

    try:
        points = get_trip_shape_points(trip_id, origin_stop_id=origin_stop_id, dest_stop_id=dest_stop_id)
    except Exception as e:
        logger.warning('[webui] /routes could not load shape (trip_id=%r): %s', trip_id, e)
        return None
    if not points:
        return None
    # Shape points are float32; round so JSON doesn't carry float32 noise digits.
    return [[round(float(lat), 6), round(float(lon), 6)] for lat, lon in points]


def _hhmm(seconds: int) -> str:
    """GTFS seconds-since-service-day-start -> "HH:MM", hours wrapped at 24
    ("24:15" -> "00:15"), the way TransLink displays after-midnight times.
    """
    return f'{(seconds // 3600) % 24:02d}:{(seconds % 3600) // 60:02d}'


def _leg_stops(
    trip: dict, origin_stop_id: str, dest_stop_id: str, stop_lookup: dict, is_first: bool, is_last: bool,
) -> list[dict]:
    """Every stop the leg's trip visits from origin to destination inclusive,
    in stop_sequence order, for the /results timeline.

    Read from GTFSData.route_trip_stops (route_id -> trip_id -> {stop_id:
    (stop_sequence, arrival_seconds)}), the in-memory index the loader built
    at startup -- no disk or stop_times scan per call. That index is keyed by
    stop_id, so a loop trip visiting the same stop twice keeps only its last
    visit; known and accepted (~none on the Gold Coast <-> Brisbane corridor).

    is_transfer_point marks the leg's own endpoints that sit between legs:
    the origin unless this is the first leg, the destination unless it's the
    last. Falls back to just the two endpoints if the index lookup misses.
    """
    from gtfs import loader as gtfs_loader

    def stop(stop_id: str, name: str | None, seconds: int | None, transfer: bool) -> dict:
        ref = _stop_ref(stop_id, name, stop_lookup)
        ref['scheduled_time'] = _hhmm(seconds) if seconds is not None else None
        ref['is_transfer_point'] = transfer
        return ref

    data = gtfs_loader.load_gtfs_data()
    visits = (data.route_trip_stops or {}).get(trip['route_id'], {}).get(trip['trip_id'])
    if visits and origin_stop_id in visits and dest_stop_id in visits:
        lo, hi = visits[origin_stop_id][0], visits[dest_stop_id][0]
        if lo < hi:
            ordered = sorted(
                (seq, str(sid), arr) for sid, (seq, arr) in visits.items() if lo <= seq <= hi
            )
            last = len(ordered) - 1
            return [
                stop(sid, None, int(arr),
                     (i == 0 and not is_first) or (i == last and not is_last))
                for i, (seq, sid, arr) in enumerate(ordered)
            ]

    logger.warning('[webui] /routes: no stop sequence for trip_id=%r; endpoints only', trip.get('trip_id'))
    dep, arr = trip['origin_departure_time'], trip['dest_arrival_time']
    return [
        stop(origin_stop_id, trip.get('origin_stop_name'), dep.hour * 3600 + dep.minute * 60, not is_first),
        stop(dest_stop_id, trip.get('dest_stop_name'), arr.hour * 3600 + arr.minute * 60, not is_last),
    ]


def _serialize_journey(
    journey: dict, leg_predictions: list[tuple[dict, dict]], predicted_arrival: datetime, stop_lookup: dict,
) -> dict:
    from route_types import MODE_BY_ROUTE_TYPE

    legs_out = []
    last_leg = len(leg_predictions) - 1
    for leg_index, (trip, pred) in enumerate(leg_predictions):
        mode = trip.get('mode') or MODE_BY_ROUTE_TYPE.get(trip.get('route_type'), 'unknown')
        leg_predicted_arrival = trip['dest_arrival_time'] + timedelta(minutes=pred['blended_delay_minutes'])
        trip_id = str(trip['trip_id'])
        origin_stop_id = str(trip['origin_stop_id'])
        dest_stop_id = str(trip['dest_stop_id'])
        legs_out.append({
            'stop_id': trip['dest_stop_id'],
            'route_short_name': trip.get('route_short_name') or trip['route_id'],
            'mode': mode,
            'departure_time': trip['origin_departure_time'].isoformat(),
            'predicted_arrival': leg_predicted_arrival.isoformat(),
            'confidence': pred['confidence'],
            # Additive map fields (Pass 5); the fields above are unchanged.
            'trip_id': trip_id,
            'from_stop': _stop_ref(origin_stop_id, trip.get('origin_stop_name'), stop_lookup),
            'to_stop': _stop_ref(dest_stop_id, trip.get('dest_stop_name'), stop_lookup),
            'shape': _leg_shape(trip_id, origin_stop_id, dest_stop_id),
            # Pass 9: ordered stops for the /results timeline.
            'stops': _leg_stops(
                trip, origin_stop_id, dest_stop_id, stop_lookup,
                is_first=leg_index == 0, is_last=leg_index == last_leg,
            ),
        })
    first_departure = leg_predictions[0][0]['origin_departure_time']
    total_predicted_duration_minutes = int(round((predicted_arrival - first_departure).total_seconds() / 60))
    # _predict_journey_legs() drops any journey with a failed leg, so the last
    # leg's prediction is always present here; the None guard is defensive.
    last_delay = leg_predictions[-1][1].get('blended_delay_minutes')
    weakest = min((pred['confidence'] for _, pred in leg_predictions), key=lambda c: CONFIDENCE_RANK.get(c, 0))
    return {
        'legs': legs_out,
        'total_predicted_duration_minutes': total_predicted_duration_minutes,
        'transfer_count': journey['num_transfers'],
        # Additive journey-level fields (Session 40); the fields above are unchanged.
        'leave_by': (first_departure - LEAVE_BY_BUFFER).strftime('%H:%M'),
        'predicted_delay_minutes': int(round(last_delay)) if last_delay is not None else None,
        'confidence': weakest.lower(),
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


def _find_ranked_routes(
    from_stop_id: str, to_stop_id: str, departure_after: datetime, stop_lookup: dict,
) -> list[dict]:
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
        _serialize_journey(journey, leg_predictions, predicted_arrival, stop_lookup)
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
        return _find_ranked_routes(from_stop_id, to_stop_id, departure_after, request.app.state.stop_lookup)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=f'Routing/prediction unavailable: {e}')
