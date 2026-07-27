"""Diagnostic script for nearest_stops() — not a test suite, just prints.

Run: python phase3/test_nearest_stops.py

Do NOT use Playwright/Selenium/any browser here — this only exercises the
gtfs_data data layer directly. Live map/geolocation interaction is tested
manually in the running Streamlit app.
"""
from __future__ import annotations

import importlib.metadata

from gtfs_data import nearest_stops

# GTFS route_type -> label, matching app.py's ROUTE_TYPE_MODE (kept as a
# local literal here since this script deliberately has no Streamlit/app.py
# dependency).
ROUTE_TYPE_LABEL = {0: 'Tram', 1: 'Metro', 2: 'Train', 3: 'Bus', 4: 'Ferry'}

KNOWN_POINTS = [
    ('Surfers Paradise', -27.9975, 153.4295),
    ('Southport', -27.9678, 153.4142),
]


def run_case(label: str, lat: float, lon: float) -> None:
    print(f'=== nearest_stops({label} @ {lat}, {lon}) ===')
    results = nearest_stops(lat, lon, limit=15)
    print(f'  {len(results)} candidate(s):')
    for r in results:
        modes = '/'.join(ROUTE_TYPE_LABEL.get(rt, f'type{rt}') for rt in r['route_types'])
        print(
            f"    {r['stop_name']:<40} modes={modes:<12} "
            f"dist={r['distance_km']:.3f}km  trips={r['trip_count']:<5} "
            f"stop_id={r['stop_id']}  ({len(r['stop_ids'])} stop_ids)"
        )
    print()


if __name__ == '__main__':
    for pkg in ('streamlit-geolocation', 'streamlit-folium', 'folium'):
        try:
            print(f'{pkg} version: {importlib.metadata.version(pkg)}')
        except importlib.metadata.PackageNotFoundError:
            print(f'{pkg} version: NOT INSTALLED')
    print()

    for label, lat, lon in KNOWN_POINTS:
        run_case(label, lat, lon)
