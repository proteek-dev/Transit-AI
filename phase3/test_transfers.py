"""Diagnostic script for find_multi_leg_trips() — not a test suite, just prints.

Run: python phase3/test_transfers.py

Do NOT use Playwright/Selenium/any browser here — this only exercises the
gtfs_data data layer directly.
"""
from __future__ import annotations

from datetime import datetime

from gtfs_data import find_multi_leg_trips, find_trips, search_stops


def _pick_stop_group(query: str) -> dict | None:
    """Prefer a 'station' entry over a generic street stop sharing the name."""
    results = search_stops(query, limit=50)
    if not results:
        return None
    exact_station = f'{query.strip().lower()} station'
    for r in results:
        if r['stop_name'].lower() == exact_station:
            return r
    station_matches = [r for r in results if 'station' in r['stop_name'].lower()]
    return station_matches[0] if station_matches else results[0]


def run_case(label: str, origin_query: str, dest_query: str, departure_after: datetime) -> None:
    print(f'=== {label}: {origin_query} -> {dest_query} (from {departure_after}) ===')

    origin = _pick_stop_group(origin_query)
    dest = _pick_stop_group(dest_query)
    if not origin or not dest:
        print(f'  no stop match for {"origin" if not origin else "destination"} query {origin_query!r}/{dest_query!r}')
        print()
        return

    print(f'  origin group: {origin["stop_name"]} ({len(origin["stop_ids"])} stop_ids)')
    print(f'  dest group:   {dest["stop_name"]} ({len(dest["stop_ids"])} stop_ids)')

    direct = find_trips(origin['stop_ids'], dest['stop_ids'], departure_after, window_minutes=60)
    print(f'  find_trips (direct): {len(direct)} result(s)')

    journeys = find_multi_leg_trips(
        origin['stop_ids'], dest['stop_ids'], departure_after,
        window_minutes=60, max_transfers=3, max_results=5,
    )
    print(f'  find_multi_leg_trips: {len(journeys)} journey(s)')

    for i, journey in enumerate(journeys, start=1):
        print(f'  --- Journey {i}: {journey["num_transfers"]} transfer(s), '
              f'total ~{journey["total_minutes"]} min ---')
        for leg_idx, leg in enumerate(journey['legs'], start=1):
            trip = leg['trip']
            route_label = trip['route_short_name'] or trip['route_id']
            print(f'    Leg {leg_idx}: {route_label:<8} '
                  f'{leg["board_stop_name"]} {trip["origin_departure_time"]:%H:%M:%S} -> '
                  f'{leg["alight_stop_name"]} {trip["dest_arrival_time"]:%H:%M:%S}'
                  f'{"  (fallback schedule)" if trip.get("fallback_schedule") else ""}')
        for tp in journey['transfer_points']:
            print(f'    Transfer at {tp["stop_name"]}: {tp["connection_minutes"]} min connection')
    print()


if __name__ == '__main__':
    now = datetime.now()

    run_case('Case 1', 'Bundall Rd', 'Cavill Avenue', now)
    run_case('Case 2', 'Helensvale', 'Roma Street', now)
    run_case('Case 3', 'Broadbeach South', 'Sunnybank', now)
