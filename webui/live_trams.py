"""Live tram VehiclePositions for /live_vehicles.

TransLink's combined SEQ VehiclePositions feed (what phase3/live_gtfs.py
reads) carries no tram entities -- tram is only published on a dedicated
per-mode endpoint, which scripts/archive_gtfsrt.py polls as TRAM_FEEDS.
This is the sibling fetcher for that endpoint, returning the same
trip_id -> vehicle shape as live_gtfs.fetch_vehicle_positions() so the two
can be merged. Same 60s in-memory cache as phase3.
"""
from __future__ import annotations

import logging
import time

import requests
from google.transit import gtfs_realtime_pb2

logger = logging.getLogger('uvicorn.error')

# config/feeds.yaml gtfs_realtime_by_mode.vehicle_positions_tram -- the URL
# archive_gtfsrt.py's TRAM_FEEDS['vehicle_positions'] resolves to. Hardcoded
# like phase3/live_gtfs.py's URLs: config/ isn't copied into the container.
TRAM_VEHICLE_POSITIONS_URL = 'https://gtfsrt.api.translink.com.au/api/realtime/SEQ/VehiclePositions/Tram'

CACHE_TTL_SECONDS = 60

_cache = {'ts': 0.0, 'data': {}}


def fetch_tram_vehicle_positions() -> dict[str, dict]:
    """trip_id -> {lat, lon, speed, timestamp, current_stop_sequence}.

    {} on any fetch/parse failure (logged). No stale fallback, matching phase3.
    """
    now = time.time()
    if _cache['ts'] and (now - _cache['ts']) < CACHE_TTL_SECONDS:
        return _cache['data']

    try:
        resp = requests.get(TRAM_VEHICLE_POSITIONS_URL, timeout=30)
        resp.raise_for_status()
        feed = gtfs_realtime_pb2.FeedMessage()
        feed.ParseFromString(resp.content)
    except Exception:
        logger.exception('[webui] tram VehiclePositions fetch failed -- returning no tram vehicles')
        return {}

    result = {}
    for entity in feed.entity:
        if not entity.HasField('vehicle'):
            continue
        vp = entity.vehicle
        trip_id = vp.trip.trip_id
        if not trip_id:
            continue

        result[trip_id] = {
            'lat': vp.position.latitude,
            'lon': vp.position.longitude,
            'speed': vp.position.speed if vp.position.HasField('speed') else None,
            'timestamp': int(vp.timestamp) if vp.HasField('timestamp') else int(now),
            'current_stop_sequence': vp.current_stop_sequence if vp.HasField('current_stop_sequence') else None,
        }

    _cache['ts'] = now
    _cache['data'] = result
    return result
