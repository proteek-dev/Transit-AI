"""Live tram VehiclePositions for /live_vehicles, and the fetch guard both of
its feeds share.

TransLink's combined SEQ VehiclePositions feed (what phase3/live_gtfs.py
reads) carries no tram entities -- tram is only published on a dedicated
per-mode endpoint, which scripts/archive_gtfsrt.py polls as TRAM_FEEDS.
`tram_feed` is the sibling fetcher for that endpoint, returning the same
trip_id -> vehicle shape as live_gtfs.fetch_vehicle_positions() so the two
can be merged. Same 60s in-memory cache as phase3.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable, NamedTuple, TypedDict

import requests
from google.transit import gtfs_realtime_pb2

logger = logging.getLogger('uvicorn.error')

# config/feeds.yaml gtfs_realtime_by_mode.vehicle_positions_tram -- the URL
# archive_gtfsrt.py's TRAM_FEEDS['vehicle_positions'] resolves to. Hardcoded
# like phase3/live_gtfs.py's URLs: config/ isn't copied into the container.
TRAM_VEHICLE_POSITIONS_URL = 'https://gtfsrt.api.translink.com.au/api/realtime/SEQ/VehiclePositions/Tram'

CACHE_TTL_SECONDS = 60
# After a failed fetch, the next retry waits this long, so an outage costs
# one request per window rather than one per poll.
FAILURE_TTL_SECONDS = 20
# Short: this runs on a request thread (/live_vehicles is a sync endpoint).
FETCH_TIMEOUT_SECONDS = 5


class VehiclePosition(TypedDict):
    """One trip's latest fix -- the shape live_gtfs.fetch_vehicle_positions() returns too."""
    lat: float
    lon: float
    speed: float | None
    timestamp: int  # epoch seconds of the fix
    current_stop_sequence: int | None


class FeedResult(NamedTuple):
    vehicles: dict[str, VehiclePosition]
    # False while the feed's latest fetch attempt failed; `vehicles` is then
    # the last good data (or {}), so the caller can report the outage rather
    # than pass it off as "no vehicle reporting".
    ok: bool


class GuardedFeed:
    """Single-flight, failure-cached wrapper around a blocking feed fetch.

    Only one thread runs `fetch` at a time -- the others get the current data
    rather than queueing behind a slow feed -- and a failure isn't retried for
    `failure_ttl_seconds`. A successful result is reused for
    `cache_ttl_seconds` (0: `fetch` does its own caching).
    """

    def __init__(self, fetch: Callable[[float], dict[str, VehiclePosition]], *, name: str,
                 cache_ttl_seconds: float, failure_ttl_seconds: float):
        self._fetch = fetch
        self.name = name
        self.cache_ttl_seconds = cache_ttl_seconds
        self.failure_ttl_seconds = failure_ttl_seconds
        self.lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        self.data: dict[str, VehiclePosition] = {}
        self.fetched_at = 0.0
        self.failed_at = 0.0
        self.last_ok = True

    def get(self) -> FeedResult:
        now = time.time()
        if self.fetched_at and (now - self.fetched_at) < self.cache_ttl_seconds:
            return FeedResult(self.data, True)
        if (now - self.failed_at) < self.failure_ttl_seconds:
            return FeedResult(self.data, False)
        if not self.lock.acquire(blocking=False):
            return FeedResult(self.data, self.last_ok)
        try:
            try:
                data = self._fetch(now)
            except Exception:
                logger.exception('[webui] %s fetch failed -- serving its last good positions', self.name)
                self.failed_at, self.last_ok = now, False
                return FeedResult(self.data, False)
            self.data, self.fetched_at, self.last_ok = data, now, True
            return FeedResult(data, True)
        finally:
            self.lock.release()


def _fetch_tram_positions(now: float) -> dict[str, VehiclePosition]:
    """trip_id -> VehiclePosition from the tram feed. Raises on fetch/parse
    failure. Entities without a trip_id, a position or a timestamp are
    dropped: an undated fix can't be aged, so it would pass /live_vehicles'
    staleness cut as brand new.
    """
    resp = requests.get(TRAM_VEHICLE_POSITIONS_URL, timeout=FETCH_TIMEOUT_SECONDS)
    resp.raise_for_status()
    feed = gtfs_realtime_pb2.FeedMessage()
    feed.ParseFromString(resp.content)

    result: dict[str, VehiclePosition] = {}
    for entity in feed.entity:
        if not entity.HasField('vehicle'):
            continue
        vp = entity.vehicle
        trip_id = vp.trip.trip_id
        if not trip_id or not vp.HasField('position') or not vp.HasField('timestamp'):
            continue
        result[trip_id] = {
            'lat': vp.position.latitude,
            'lon': vp.position.longitude,
            'speed': vp.position.speed if vp.position.HasField('speed') else None,
            'timestamp': int(vp.timestamp),
            'current_stop_sequence': vp.current_stop_sequence if vp.HasField('current_stop_sequence') else None,
        }
    return result


tram_feed = GuardedFeed(_fetch_tram_positions, name='tram VehiclePositions',
                        cache_ttl_seconds=CACHE_TTL_SECONDS, failure_ttl_seconds=FAILURE_TTL_SECONDS)
