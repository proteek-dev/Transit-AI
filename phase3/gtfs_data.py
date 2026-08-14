"""GTFS data layer for the Phase 3 Streamlit POC.

Thin facade over the split-out modules: loader.py (S3/parquet loading, the
GTFSData class, shape_id -> [(lat,lon)] lookup), search.py (typeahead +
nearest-stop search), routing.py (the BFS multi-leg trip engine), and
shapes.py (trip shape-point trimming). Re-exports every name the rest of the
codebase imports from here (app.py, prediction.py, the test/stress scripts)
so `import gtfs_data` / `from gtfs_data import X` keep working unchanged.
No Streamlit or model code here — data layer only.
"""
from __future__ import annotations

from datetime import datetime

from gtfs.loader import DATE_PATTERN, load_gtfs_data
from gtfs.routing import FALLBACK_THIN_COVERAGE_DAYS, _direct_journey, find_multi_leg_trips, find_trips
from gtfs.search import nearest_stops, search_stops
from gtfs.shapes import get_trip_shape_points


def _pick_stop_group(query: str) -> dict | None:
    """Demo/diagnostic helper only — not used by app.py or map_picker.py.

    Prefers an exact stop_name match first. Only when there's no exact match
    does it fall back to preferring a 'station'-named result, and even then
    only among results that are actual PREFIX matches of the query (e.g.
    'Broadbeach South' -> 'Broadbeach South station' — a real multi-platform
    ambiguity). Earlier this station-preference applied to ANY result
    containing 'station', including unrelated fuzzy matches far down the
    list (e.g. querying 'Robina Town Centre' — which has its own exact
    match — incorrectly returned 'Indooroopilly Shopping Centre station', a
    same-word fuzzy match with no relation to the query).
    """
    results = search_stops(query, limit=50)
    if not results:
        return None
    query_lower = query.strip().lower()
    for r in results:
        if r['stop_name'].lower() == query_lower:
            return r
    prefix_matches = [r for r in results if r['stop_name'].lower().startswith(query_lower)]
    station_matches = [r for r in prefix_matches if 'station' in r['stop_name'].lower()]
    if station_matches:
        return station_matches[0]
    return prefix_matches[0] if prefix_matches else results[0]


def _print_trip_search(label: str, origin_query: str, dest_query: str, departure_after: datetime):
    print(f'--- find_trips: {label} (from {departure_after}, 60 min window) ---')
    origin = _pick_stop_group(origin_query)
    dest = _pick_stop_group(dest_query)
    if not origin or not dest:
        print(f'  no stop match for {"origin" if not origin else "destination"} query')
        print()
        return

    print(f'  origin group: {origin["stop_name"]} ({len(origin["stop_ids"])} stop_ids)')
    print(f'  dest group:   {dest["stop_name"]} ({len(dest["stop_ids"])} stop_ids)')
    trips = find_trips(origin['stop_ids'], dest['stop_ids'], departure_after, window_minutes=60)
    print(f'  {len(trips)} trip(s) found, showing first 5:')
    for t in trips[:5]:
        print(f"    {t['route_short_name'] or t['route_id']:<8} "
              f"dep {t['origin_departure_time']:%H:%M:%S} -> arr {t['dest_arrival_time']:%H:%M:%S}  "
              f"({t['n_stops_between']} stops between)  trip_id={t['trip_id']}")
    print()


if __name__ == '__main__':
    data = load_gtfs_data()
    print(
        f'\n=== Loaded snapshot {data.snapshot_date}: '
        f'{len(data.stops):,} stops, {len(data.routes):,} routes, {len(data.trips):,} trips ===\n'
    )

    for q in ['Broadbeach', 'Bun', 'Sunny']:
        print(f'--- search_stops({q!r}) ---')
        for r in search_stops(q):
            print(f"  {r['stop_name']:<45} lat={r['stop_lat']:.5f} lon={r['stop_lon']:.5f} "
                  f"stop_ids={r['stop_ids'][:3]}{'...' if len(r['stop_ids']) > 3 else ''}")
        print()

    now = datetime.now()
    _print_trip_search('Broadbeach South -> Surfers Paradise', 'Broadbeach South', 'Surfers Paradise', now)
    _print_trip_search('Helensvale -> Brisbane', 'Helensvale', 'Brisbane', now)
