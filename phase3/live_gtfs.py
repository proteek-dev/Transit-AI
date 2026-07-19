"""Live GTFS-RT fetcher for the Phase 3 Streamlit POC.

Fetches TripUpdates and VehiclePositions directly from TransLink's public
GTFS-RT feeds at request time (same endpoints as scripts/archive_gtfsrt.py).
No auth needed. Responses are cached in memory for 60 seconds to avoid
hammering TransLink on rapid repeated queries.
"""
from __future__ import annotations

import time

import requests
from google.transit import gtfs_realtime_pb2

TRIP_UPDATES_URL = 'https://gtfsrt.api.translink.com.au/api/realtime/SEQ/TripUpdates'
VEHICLE_POSITIONS_URL = 'https://gtfsrt.api.translink.com.au/api/realtime/SEQ/VehiclePositions'

CACHE_TTL_SECONDS = 60

_cache: dict = {}


def _fetch_feed(url: str) -> gtfs_realtime_pb2.FeedMessage:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    feed = gtfs_realtime_pb2.FeedMessage()
    feed.ParseFromString(resp.content)
    return feed


def fetch_trip_updates(force_refresh: bool = False) -> dict:
    """trip_id -> {delay_seconds, stop_id, timestamp, status}, from the latest
    stop_time_update that carries an arrival/departure delay for each trip.
    """
    cached = _cache.get('trip_updates')
    now = time.time()
    if not force_refresh and cached and (now - cached[0]) < CACHE_TTL_SECONDS:
        return cached[1]

    feed = _fetch_feed(TRIP_UPDATES_URL)
    result = {}
    for entity in feed.entity:
        if not entity.HasField('trip_update'):
            continue
        tu = entity.trip_update
        trip_id = tu.trip.trip_id
        if not trip_id:
            continue

        latest = None
        for stu in tu.stop_time_update:
            if stu.HasField('arrival'):
                delay_seconds = stu.arrival.delay
            elif stu.HasField('departure'):
                delay_seconds = stu.departure.delay
            else:
                continue
            latest = (delay_seconds, stu.stop_id, stu.schedule_relationship)

        if latest is None:
            continue

        delay_seconds, stop_id, schedule_relationship = latest
        result[trip_id] = {
            'delay_seconds': delay_seconds,
            'stop_id': stop_id,
            'timestamp': int(tu.timestamp) if tu.HasField('timestamp') else int(now),
            'status': gtfs_realtime_pb2.TripUpdate.StopTimeUpdate.ScheduleRelationship.Name(
                schedule_relationship
            ),
        }

    _cache['trip_updates'] = (now, result)
    return result


def fetch_vehicle_positions(force_refresh: bool = False) -> dict:
    """trip_id -> {lat, lon, speed, timestamp, current_stop_sequence}."""
    cached = _cache.get('vehicle_positions')
    now = time.time()
    if not force_refresh and cached and (now - cached[0]) < CACHE_TTL_SECONDS:
        return cached[1]

    feed = _fetch_feed(VEHICLE_POSITIONS_URL)
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

    _cache['vehicle_positions'] = (now, result)
    return result


def get_live_delay(trip_id: str) -> dict | None:
    """Current known delay for a trip: {delay_minutes, timestamp, stop_id}, or None."""
    updates = fetch_trip_updates()
    info = updates.get(trip_id)
    if info is None:
        return None

    return {
        'delay_minutes': info['delay_seconds'] / 60.0,
        'timestamp': info['timestamp'],
        'stop_id': info['stop_id'],
    }


if __name__ == '__main__':
    print('=== Fetching TripUpdates ===')
    updates = fetch_trip_updates()
    print(f'{len(updates):,} trips have live delay data right now')
    print()

    print('=== Fetching VehiclePositions ===')
    positions = fetch_vehicle_positions()
    print(f'{len(positions):,} vehicles are currently tracked')
    print()

    if updates:
        sample_trip_id = next(iter(updates))
        print(f'--- Sample live delay for trip_id={sample_trip_id!r} ---')
        print(f'  raw trip update: {updates[sample_trip_id]}')
        print(f'  get_live_delay(): {get_live_delay(sample_trip_id)}')
    else:
        print('No live trip updates available right now — nothing to sample.')
