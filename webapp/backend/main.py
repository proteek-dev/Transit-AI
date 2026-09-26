"""Minimal FastAPI backend — step 1 of the FastAPI/Next.js replacement for
the Streamlit app in phase3/app.py. Additive only; phase3/ is untouched.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import boto3
import psutil
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

# Memory visibility for diagnosing Render's free-tier 512MB OOM crash --
# purely external process-level observability (psutil reads this process's
# own RSS), not something that touches or wraps phase3/ code.
_process = psutil.Process()


def _rss_mb() -> float:
    return _process.memory_info().rss / (1024 * 1024)


print(f'[startup][mem] baseline RSS: {_rss_mb():.1f}MB')

app = FastAPI(title='Transit-AI API')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],  # open for now; restrict once the frontend domain exists
    allow_methods=['*'],
    allow_headers=['*'],
)


@app.get('/health')
def health() -> dict:
    """No file I/O, no S3 -- just a cheap target for the external keep-alive
    cron so pinging it doesn't cost anything on every hit.
    """
    return {'status': 'ok'}


# training_metadata.json lives under phase3/model/, which is .gitignore'd
# and doesn't exist on Render -- so instead of reading a local file, this
# fetches the object from S3 once at process startup and caches it in
# memory. /model/stats then serves the cache; it never touches S3 or disk
# per-request.
#
# Bucket env var and object key mirror phase3/ml/model_io.py's
# _s3_model_paths()['metadata'] (bucket from phase3/config.py's
# get_s3_bucket(), which reads AWS_S3_BUCKET; key is
# phase3/model/<MODEL_SUBDIR_OVERRIDE or 'latest'>/training_metadata.json)
# without importing that module directly -- model_io.py unconditionally
# imports xgboost/joblib/config-with-s3fs at load time and its only public
# loader (get_training_metadata) triggers a full model load, which this
# read-only endpoint doesn't need.
S3_BUCKET_ENV_VAR = 'AWS_S3_BUCKET'
MODEL_SUBDIR = os.environ.get('MODEL_SUBDIR_OVERRIDE', 'latest')
S3_METADATA_KEY = f'phase3/model/{MODEL_SUBDIR}/training_metadata.json'

_training_metadata: dict | None = None
_training_metadata_fetch_error: str | None = None


def _fetch_s3_json(key: str) -> dict:
    bucket = os.environ.get(S3_BUCKET_ENV_VAR)
    if not bucket:
        raise RuntimeError(f'{S3_BUCKET_ENV_VAR} environment variable is not set')
    # boto3.client('s3') with no explicit credentials uses boto3's standard
    # credential chain (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY /
    # AWS_DEFAULT_REGION env vars here) -- nothing hardcoded.
    s3 = boto3.client('s3')
    response = s3.get_object(Bucket=bucket, Key=key)
    return json.loads(response['Body'].read())


def _fetch_training_metadata_from_s3() -> dict:
    return _fetch_s3_json(S3_METADATA_KEY)


try:
    _training_metadata = _fetch_training_metadata_from_s3()
except Exception as e:
    # Never let a startup S3/network failure crash the whole app -- surface
    # it as a 503 from /model/stats instead (see below).
    _training_metadata_fetch_error = str(e)
    print(f'Startup: failed to fetch training_metadata.json from S3 ({e}) -- /model/stats will return 503.')


# data_snapshot freshness inputs for /model/stats, cached the same way as
# training_metadata.json above. Unlike that fetch, a failure here doesn't
# 503 the endpoint -- the affected data_snapshot field is just null.
#
# Feature manifest: same _latest.json phase3/ml/training.py reads;
# source_date_range is [first, last] feature date, so its max is the last
# day of features. GTFS manifest: the _manifest.json sidecar
# scripts/precompute_gtfs_static.py writes next to the 'latest' parquet the
# loader reads -- re-fetched after each warm-up GTFS load (see _warmup())
# so it tracks what's actually in memory.
S3_FEATURE_MANIFEST_KEY = 'ml_features/v0_feature_snapshot/_latest.json'
S3_GTFS_MANIFEST_KEY = 'phase3/gtfs_static_optimized/latest/_manifest.json'
HEALTHY_STOP_TIMES_MIN_ROWS = 1_000_000

_features_through: str | None = None
_gtfs_static_snapshot: str | None = None


def _refresh_features_through() -> None:
    global _features_through
    try:
        date_range = _fetch_s3_json(S3_FEATURE_MANIFEST_KEY).get('source_date_range') or []
        _features_through = max(date_range) if date_range else None
    except Exception as e:
        _features_through = None
        print(f'Startup: failed to fetch feature _latest.json from S3 ({e}) -- features_through will be null.')


def _refresh_gtfs_static_snapshot() -> None:
    global _gtfs_static_snapshot
    try:
        _gtfs_static_snapshot = _fetch_s3_json(S3_GTFS_MANIFEST_KEY).get('source_snapshot_date')
    except Exception as e:
        _gtfs_static_snapshot = None
        print(f'Failed to fetch GTFS _manifest.json from S3 ({e}) -- gtfs_static_snapshot will be null.')


_refresh_features_through()
_refresh_gtfs_static_snapshot()


def _graph_status() -> str:
    """'healthy' at >= HEALTHY_STOP_TIMES_MIN_ROWS stop_times rows, 'empty'
    at 0 rows or before warm-up has populated the GTFS cache, 'unknown'
    otherwise (including a non-zero count below the threshold, e.g. the
    ~714K-row stale-parquet symptom diagnostics/snapshot_integrity_check.py
    investigates). Reads the loader's shared cache directly rather than via
    load_gtfs_data(), which would trigger a full load if it's still cold.
    """
    try:
        data = gtfs_loader._gtfs_cache.get('data')
        if data is None or data.stop_times is None:
            return 'empty'
        rows = len(data.stop_times)
    except Exception:
        return 'unknown'
    if rows == 0:
        return 'empty'
    return 'healthy' if rows >= HEALTHY_STOP_TIMES_MIN_ROWS else 'unknown'


@app.get('/model/stats')
def model_stats() -> dict:
    if _training_metadata is None:
        raise HTTPException(
            status_code=503,
            detail=f'Training metadata unavailable: S3 fetch failed at startup ({_training_metadata_fetch_error})',
        )
    metadata = _training_metadata
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
            'features_through': _features_through,
            'model_trained_through': metadata['data_window_end'],
            'gtfs_static_snapshot': _gtfs_static_snapshot,
            'graph_status': _graph_status(),
        },
    }


# GET /routes wraps phase3/'s own routing + prediction stack rather than
# reimplementing any of it. phase3/ isn't a package (its modules do plain
# `import gtfs_data`, `from ml.model_io import ...`, etc, assuming phase3/
# itself is on sys.path) -- same fix phase3/tests/smoke_test.py already
# uses for the same reason.
#
# Imported here: gtfs_data.find_trips() / find_multi_leg_trips() /
# _direct_journey() (gtfs/routing.py's BFS engine, re-exported by
# gtfs_data.py's facade -- app.py itself calls _direct_journey the same
# underscore-prefixed way), prediction.enrich_trip_with_dest_stop() /
# predict_delay() (ml/inference.py's hot path, re-exported by
# prediction.py's facade), route_types.MODE_BY_ROUTE_TYPE (the GTFS
# route_type -> mode-name mapping predict_delay() itself uses internally),
# and gtfs.loader.load_gtfs_data_optimized() -- imported directly from
# gtfs.loader rather than through gtfs_data.py's facade, since that facade
# is out of scope for this change; populates the same shared GTFSData cache
# gtfs_data.find_trips() etc. read from (see its docstring), just via S3's
# precomputed stop_times parquet instead of parsing stop_times.txt.
#
# Streamlit-runtime check (see task item 4): nothing in this chain actually
# requires a running Streamlit app. The two places that touch `streamlit`
# -- ml/model_io.py's `load_model = st.cache_resource(...)(_load_model_impl)`
# and config.py's get_aws_credentials()/get_s3_bucket() trying st.secrets --
# both do `import streamlit as st` inside a try/except ImportError with a
# working non-Streamlit fallback already built in (an in-process dict cache
# for load_model, env-vars/.env for credentials). Since streamlit isn't in
# this service's requirements.txt, both fall back automatically. app.py's
# own @st.cache_data(ttl=60) around live_gtfs.fetch_trip_updates() is also
# just a UI-layer cache on top of live_gtfs.py's own manual 60s in-process
# cache -- moot here anyway since live GTFS-RT blending isn't used below
# (see _predict_leg).
PHASE3_DIR = Path(__file__).resolve().parents[2] / 'phase3'
if str(PHASE3_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE3_DIR))

import gtfs_data  # noqa: E402
import prediction  # noqa: E402
from gtfs import loader as gtfs_loader  # noqa: E402
from gtfs.loader import load_gtfs_data_optimized  # noqa: E402
from gtfs.search import search_stops  # noqa: E402
from route_types import MODE_BY_ROUTE_TYPE  # noqa: E402

BRISBANE_TZ = ZoneInfo('Australia/Brisbane')

# Mirrors phase3/app.py's own constants for this same search -> rank ->
# truncate pipeline (CANDIDATE_POOL_SIZE, MAX_RESULTS, the 60-minute
# window): a wider pool gets predicted so ranking-by-predicted-arrival can
# surface a transfer journey ahead of a direct trip if it actually arrives
# sooner, then only the top 5 are returned.
CANDIDATE_POOL_SIZE = 10
MAX_RESULTS = 5
WINDOW_MINUTES = 60

# gtfs_data.load_gtfs_data() (~46s: parses 900+ routes / 115K+ trips / 3M+
# stop_times from S3) and prediction.load_model() (~10s) each cache
# themselves in an in-process dict on first call -- lazily, by default, on
# whichever request happens to trigger them first. Without this, the first
# real /routes request after every container boot would eat that whole
# ~55s cost. Instead, the startup hook below fires this off as a background
# task (never awaited there) so uvicorn still binds its port and serves
# /health immediately; the actual blocking pandas/xgboost calls run via
# asyncio.to_thread so they never tie up the event loop while /health or
# any other request needs to be served concurrently. (Keep-alive strategy
# for Cloud Run -- min-instances vs. scale-to-zero -- is still being
# decided; /health stays useful either way.)
#
# GTFS and model loads run one after another, not concurrently: an earlier
# asyncio.gather() version ran both at once, which stacks both loads' peak
# memory footprints simultaneously -- that overshot Render free tier's
# 512MB cap and crash-looped the container on OOM. Sequential loading
# matches the pre-warm-up lazy-load behavior (which never OOM'd) at the
# cost of a longer total warm-up time (~sum of both stages instead of
# ~max of the two) -- an explicit memory-over-speed tradeoff on the free
# tier, not an oversight.
_warmup_ready = asyncio.Event()
# Set to True if _warmup()'s work crashes — checked by _wait_for_warmup() in /routes and /stops/search below.
_warmup_failed: bool = False


async def _warmup() -> None:
    print('[warmup] starting GTFS static load, then model load, in the background...')
    started_at = time.monotonic()
    try:
        gtfs_started_at = time.monotonic()
        # load_gtfs_data_optimized(): this process's own opt-in to
        # gtfs/loader.py's parquet-backed stop_times path (reads the
        # precomputed S3 parquet from scripts/precompute_gtfs_static.py
        # instead of parsing stop_times.txt) -- cuts out stop_times'
        # dominant share of peak RSS (the OOM driver on Render's 512MB free
        # tier), the same rows the earlier use_categorical_dtypes=True path
        # optimized the dtype of but still had to parse+chunk from CSV.
        # Populates the same shared cache gtfs_data.py's routing helpers
        # (find_trips() etc.) read from internally via load_gtfs_data() --
        # see load_gtfs_data_optimized()'s docstring. phase3/app.py's
        # Streamlit process never calls this and is unaffected.
        await asyncio.to_thread(load_gtfs_data_optimized)
        print(f'[warmup] GTFS static loaded in {time.monotonic() - gtfs_started_at:.1f}s')
        await asyncio.to_thread(_refresh_gtfs_static_snapshot)

        model_started_at = time.monotonic()
        await asyncio.to_thread(prediction.load_model)
        print(f'[warmup] model loaded in {time.monotonic() - model_started_at:.1f}s')
    except Exception as e:
        # _warmup_failed is checked by _wait_for_warmup() so requests fail
        # fast with 503 instead of retrying the same load on the request
        # path (the pre-guard behavior — see learnings.md's OOM section for
        # why that path is dangerous).
        global _warmup_failed
        _warmup_failed = True
        print(f'[warmup] failed after {time.monotonic() - started_at:.1f}s ({e}) -- /routes requests will return 503 until restart.')
    else:
        print(f'[warmup] complete in {time.monotonic() - started_at:.1f}s -- /routes is now warm.')
    finally:
        _warmup_ready.set()


async def _sample_memory_during_warmup() -> None:
    """Logs this process's RSS every ~1s for as long as _warmup() is still
    running, so an OOM that happens before the "GTFS static loaded" log
    line (i.e. during the load itself, not after it) is visible in Render's
    logs. Runs alongside _warmup() as its own task -- doesn't call into or
    wrap any phase3/ code, just observes the same process externally.
    """
    started_at = time.monotonic()
    while not _warmup_ready.is_set():
        elapsed = time.monotonic() - started_at
        print(f'[warmup][mem] {elapsed:.0f}s: {_rss_mb():.1f}MB')
        # Sleep up to 1s, but wake immediately (without an extra sample) if
        # warm-up finishes mid-sleep -- avoids sampling forever after warm-up.
        try:
            await asyncio.wait_for(_warmup_ready.wait(), timeout=1.0)
        except asyncio.TimeoutError:
            pass


@app.on_event('startup')
async def _on_startup() -> None:
    asyncio.create_task(_warmup())
    asyncio.create_task(_sample_memory_during_warmup())


def _parse_departure(departure: str | None) -> datetime:
    """None -> naive Brisbane-local "now" (matches app.py's Departure=Now
    handling: datetime.now(BRISBANE_TZ) then strip tzinfo, since routing.py's
    BFS compares against naive datetimes throughout). A given ISO string
    with an explicit offset/zone is converted to Brisbane time then
    stripped; a naive one is treated as already being Brisbane wall-clock
    time, same as app.py's Custom departure_mode.
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
    """Enrich + predict_delay() one leg's trip -- same as app.py's
    _predict_leg(), minus live GTFS-RT blending (live_delay always None
    here; predict_delay() natively supports this, falling back to its
    coverage-based Medium/Low confidence -- not a rebuild of that logic,
    just not exercising the live-blended branch). Returns None on failure
    rather than raising, so one bad leg doesn't take down the whole request.
    """
    try:
        enriched = prediction.enrich_trip_with_dest_stop(trip, dest_stop_ids)
        pred = prediction.predict_delay(enriched, search_departure_after, live_delay=None)
    except Exception as e:
        print(f'[/routes] could not predict leg (trip_id={trip.get("trip_id")!r}): {e}')
        return None
    return enriched, pred


def _predict_journey_legs(journey: dict) -> list[tuple[dict, dict]] | None:
    """All legs of a journey, enriched + predicted. None if any leg's
    prediction failed -- unlike app.py's UI (which renders a gap and keeps
    the journey), a JSON leg can't represent "prediction failed" inside the
    declared response shape, so the whole journey is dropped instead.
    """
    legs = []
    for leg in journey['legs']:
        result = _predict_leg(leg['trip'], leg['dest_stop_ids'], leg['search_departure_after'])
        if result is None:
            return None
        legs.append(result)
    return legs


def _serialize_journey(journey: dict, leg_predictions: list[tuple[dict, dict]], predicted_arrival: datetime) -> dict:
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


def _find_ranked_routes(from_stop_id: str, to_stop_id: str, departure_after: datetime) -> list[dict]:
    """find direct + transfer journeys (gtfs_data's BFS), predict every leg
    of every candidate, then rank by predicted (not scheduled) arrival --
    the exact pipeline app.py runs for its trip cards. Returns [] (not an
    error) when BFS finds nothing at all, e.g. the known Surfers Paradise ->
    HOTA gap.
    """
    origin_stop_ids = [from_stop_id]
    dest_stop_ids = [to_stop_id]

    trips = gtfs_data.find_trips(origin_stop_ids, dest_stop_ids, departure_after, window_minutes=WINDOW_MINUTES)
    transfer_journeys = gtfs_data.find_multi_leg_trips(
        origin_stop_ids, dest_stop_ids, departure_after, window_minutes=WINDOW_MINUTES,
        max_results=CANDIDATE_POOL_SIZE,
    )
    direct_journeys = [
        gtfs_data._direct_journey(trip, dest_stop_ids, departure_after) for trip in trips[:CANDIDATE_POOL_SIZE]
    ]
    # find_multi_leg_trips() also runs its own depth-0 direct check
    # internally -- filtered to num_transfers >= 1 here so those aren't
    # double-counted against direct_journeys above (same filter app.py uses).
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


async def _wait_for_warmup() -> None:
    """Await warmup completion. Raises HTTPException 503 if warmup crashed.

    Called by every request handler that reads GTFS data. Consolidates
    the wait-then-check pattern so /routes and /stops/search can't drift
    apart on how they handle a failed startup.
    """
    if not _warmup_ready.is_set():
        await _warmup_ready.wait()
    if _warmup_failed:
        raise HTTPException(
            status_code=503,
            detail='Backend warm-up failed at startup; check server logs.',
        )


@app.get('/routes')
async def routes(
    from_stop_id: str = Query(..., min_length=1),
    to_stop_id: str = Query(..., min_length=1),
    # Optional[str], not `str | None` -- FastAPI evaluates a route's
    # parameter annotations at runtime via get_typed_signature() even under
    # `from __future__ import annotations`, and Python 3.9 (still in play
    # here -- no runtime.txt/.python-version pin) can't eval PEP 604 `|`
    # union syntax at runtime, only PEP 585 generics like list[dict].
    # Confirmed by reproducing the crash locally.
    departure: Optional[str] = Query(None),
) -> list[dict]:
    departure_after = _parse_departure(departure)
    await _wait_for_warmup()
    try:
        # to_thread here for the same reason _warmup() uses it: this is a
        # sync function doing blocking pandas/xgboost work, and running it
        # inline on an `async def` handler would tie up the event loop for
        # its ~0.25-2s instead of the threadpool isolation a plain `def`
        # FastAPI route would have gotten automatically.
        return await asyncio.to_thread(_find_ranked_routes, from_stop_id, to_stop_id, departure_after)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=f'Routing/prediction unavailable: {e}')


@app.get('/stops/search')
async def stops_search(
    q: str = Query(..., min_length=1, max_length=100),
    # ge=1 is load-bearing: search_stops() slices results[:limit], so a
    # negative limit would silently drop results from the end.
    limit: int = Query(10, ge=1, le=50),
    # Optional[float], not `float | None` -- same Python 3.9 runtime-eval
    # constraint as /routes' departure param above.
    lat: Optional[float] = Query(None, ge=-90, le=90),
    lon: Optional[float] = Query(None, ge=-180, le=180),
) -> List[Dict[str, Any]]:
    """Typeahead stop search -- thin wrapper over phase3/gtfs/search.py's
    search_stops(). Returns its dicts as-is (distance_km only present on
    geo-biased fuzzy matches), so no response_model: forcing that key
    Optional would add nulls to the wire shape.
    """
    if (lat is None) != (lon is None):
        raise HTTPException(status_code=422, detail='lat and lon must be provided together or both omitted')
    await _wait_for_warmup()
    try:
        return await asyncio.to_thread(search_stops, q, limit=limit, ref_lat=lat, ref_lon=lon)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=f'Stop search unavailable: {e}')
