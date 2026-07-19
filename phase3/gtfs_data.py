"""GTFS data layer for the Phase 3 Streamlit POC.

Loads the latest static GTFS snapshot from S3, caches it in memory, and
exposes stop search (typeahead) and a trip finder (origin -> destination)
on top of it. No Streamlit or model code here — data layer only.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta

import pandas as pd
from rapidfuzz import fuzz, process

import config

DATE_PATTERN = re.compile(r'^\d{4}-\d{2}-\d{2}$')
DAY_NAMES = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']

STATIC_FILES = ['stops.txt', 'stop_times.txt', 'trips.txt', 'routes.txt',
                'calendar.txt', 'calendar_dates.txt']

# Sunday/thin-calendar fallback: a query date within this many days of the
# static snapshot's capture date is where TransLink's calendar_dates.txt
# additions (e.g. special Sunday services) are least likely to be published
# yet — see find_trips() fallback below.
FALLBACK_THIN_COVERAGE_DAYS = 3


def _get_env():
    return config.get_s3_bucket(), config.get_s3_filesystem()


def _parse_gtfs_time(series: pd.Series) -> pd.Series:
    """Parse 'HH:MM:SS' GTFS time strings (hours can exceed 24) into a Timedelta."""
    parts = series.str.split(':', expand=True).astype(int)
    return (
        pd.to_timedelta(parts[0], unit='h')
        + pd.to_timedelta(parts[1], unit='m')
        + pd.to_timedelta(parts[2], unit='s')
    )


class GTFSData:
    """Holds one parsed static GTFS snapshot in memory."""

    def __init__(self):
        self.snapshot_date = None
        self.stops = None
        self.stop_times = None
        self.trips = None
        self.routes = None
        self.calendar = None
        self.calendar_dates = None
        self._stop_index = None  # one row per unique stop_name, built at load time

    def load(self):
        bucket, fs = _get_env()
        static_prefix = f'{bucket}/gtfs_static'

        entries = fs.ls(static_prefix)
        snapshot_dates = sorted([
            e.rstrip('/').split('/')[-1] for e in entries
            if DATE_PATTERN.match(e.rstrip('/').split('/')[-1])
        ])
        if not snapshot_dates:
            raise FileNotFoundError(f'No YYYY-MM-DD snapshots found under s3://{static_prefix}')

        self.snapshot_date = max(snapshot_dates)
        print(f'Loading static GTFS snapshot: {self.snapshot_date}')

        frames = {}
        for name in STATIC_FILES:
            path = f's3://{static_prefix}/{self.snapshot_date}/{name}'
            frames[name] = pd.read_csv(path, dtype=str)

        self.stops = frames['stops.txt']
        self.stops['stop_lat'] = self.stops['stop_lat'].astype(float)
        self.stops['stop_lon'] = self.stops['stop_lon'].astype(float)

        self.stop_times = frames['stop_times.txt']
        self.stop_times['stop_sequence'] = self.stop_times['stop_sequence'].astype(int)

        self.trips = frames['trips.txt']

        self.routes = frames['routes.txt']
        self.routes['route_type'] = self.routes['route_type'].astype(int)

        self.calendar = frames['calendar.txt']
        for day in DAY_NAMES:
            self.calendar[day] = self.calendar[day].astype(int)
        self.calendar['start_date'] = self.calendar['start_date'].astype(int)
        self.calendar['end_date'] = self.calendar['end_date'].astype(int)

        self.calendar_dates = frames['calendar_dates.txt']

        self._build_stop_index()

        print(
            f'Loaded snapshot {self.snapshot_date}: '
            f'{len(self.stops):,} stops, {len(self.routes):,} routes, '
            f'{len(self.trips):,} trips, {len(self.stop_times):,} stop_times'
        )

    def _build_stop_index(self):
        # Platforms reference a parent_station (e.g. tram/train platforms all
        # sharing one hub) — collapse those onto the parent's name first, so a
        # hub isn't split into several same-place, differently-named groups
        # (some hubs fold platforms under the bare station name, others don't).
        stops = self.stops
        name_by_id = stops.set_index('stop_id')['stop_name']
        canonical_id = stops['parent_station'].fillna(stops['stop_id'])
        canonical_name = canonical_id.map(name_by_id).fillna(stops['stop_name'])

        grouped = stops.assign(canonical_name=canonical_name).groupby(
            'canonical_name', sort=False
        ).agg(
            stop_ids=('stop_id', lambda s: sorted(set(s))),
            stop_lat=('stop_lat', 'mean'),
            stop_lon=('stop_lon', 'mean'),
        ).reset_index().rename(columns={'canonical_name': 'stop_name'})
        self._stop_index = grouped

    def active_service_ids(self, query_date) -> set:
        date_int = int(query_date.strftime('%Y%m%d'))
        day_col = DAY_NAMES[query_date.weekday()]

        cal = self.calendar
        mask = (
            (cal[day_col] == 1)
            & (cal['start_date'] <= date_int)
            & (cal['end_date'] >= date_int)
        )
        active = set(cal.loc[mask, 'service_id'])

        cd = self.calendar_dates
        exceptions = cd[cd['date'] == str(date_int)]
        added = set(exceptions.loc[exceptions['exception_type'] == '1', 'service_id'])
        removed = set(exceptions.loc[exceptions['exception_type'] == '2', 'service_id'])

        return (active | added) - removed


_gtfs_cache = {}


def load_gtfs_data(force_reload: bool = False) -> GTFSData:
    """Load (or return the cached) parsed static GTFS snapshot."""
    if force_reload or 'data' not in _gtfs_cache:
        data = GTFSData()
        data.load()
        _gtfs_cache['data'] = data
    return _gtfs_cache['data']


def search_stops(query: str, limit: int = 10) -> list[dict]:
    """Typeahead stop search: prefix matches first, then fuzzy matches, deduped by stop_name."""
    data = load_gtfs_data()
    index = data._stop_index
    query_lower = query.lower().strip()
    if not query_lower:
        return []

    names_lower = index['stop_name'].str.lower()
    prefix_mask = names_lower.str.startswith(query_lower)
    prefix_rows = index[prefix_mask].sort_values('stop_name')

    results = list(prefix_rows.itertuples(index=False))
    remaining = limit - len(results)

    if remaining > 0:
        non_prefix = index[~prefix_mask]
        choices = non_prefix['stop_name'].tolist()
        matches = process.extract(
            query, choices, scorer=fuzz.WRatio, limit=remaining, processor=str.lower
        )
        matched_names = [m[0] for m in matches if m[1] > 0]
        fuzzy_rows = non_prefix[non_prefix['stop_name'].isin(matched_names)]
        # preserve rapidfuzz's ranked order
        fuzzy_rows = fuzzy_rows.set_index('stop_name').loc[matched_names].reset_index()
        results += list(fuzzy_rows.itertuples(index=False))

    results = results[:limit]

    return [
        {
            'stop_id': row.stop_ids[0],
            'stop_ids': row.stop_ids,
            'stop_name': row.stop_name,
            'stop_lat': row.stop_lat,
            'stop_lon': row.stop_lon,
        }
        for row in results
    ]


def find_trips(
    origin_stop_ids: list[str],
    dest_stop_ids: list[str],
    departure_after: datetime,
    window_minutes: int = 60,
) -> list[dict]:
    """Find trips visiting origin then destination, departing origin within the
    time window. Each returned dict carries `fallback_schedule` (bool).

    If the direct query comes back empty and the query date falls within
    FALLBACK_THIN_COVERAGE_DAYS of the static snapshot's capture date — the
    part of the calendar where day-specific exceptions are least likely to be
    published yet — retry one week ahead (same day-of-week, same time), then
    shift the results' times back onto the original date and flag them.
    """
    trips = _find_trips_core(origin_stop_ids, dest_stop_ids, departure_after, window_minutes)
    for trip in trips:
        trip['fallback_schedule'] = False
    if trips:
        return trips

    data = load_gtfs_data()
    snapshot_date = datetime.strptime(data.snapshot_date, '%Y-%m-%d').date()
    query_date = departure_after.date()
    if abs((query_date - snapshot_date).days) > FALLBACK_THIN_COVERAGE_DAYS:
        return trips

    fallback_departure_after = departure_after + timedelta(days=7)
    fallback_trips = _find_trips_core(
        origin_stop_ids, dest_stop_ids, fallback_departure_after, window_minutes
    )
    for trip in fallback_trips:
        trip['origin_departure_time'] -= timedelta(days=7)
        trip['dest_arrival_time'] -= timedelta(days=7)
        trip['fallback_schedule'] = True
    return fallback_trips


def _find_trips_core(
    origin_stop_ids: list[str],
    dest_stop_ids: list[str],
    departure_after: datetime,
    window_minutes: int = 60,
) -> list[dict]:
    """Same-day trip search — no fallback. See find_trips() for the public entry point."""
    data = load_gtfs_data()
    query_date = departure_after.date()
    day_midnight = datetime(query_date.year, query_date.month, query_date.day)
    window_end = departure_after + timedelta(minutes=window_minutes)

    active_service_ids = data.active_service_ids(query_date)
    trips_today = data.trips[data.trips['service_id'].isin(active_service_ids)]
    if trips_today.empty:
        return []

    st = data.stop_times[data.stop_times['trip_id'].isin(trips_today['trip_id'])]

    origin_st = st[st['stop_id'].isin(origin_stop_ids)][
        ['trip_id', 'stop_id', 'stop_sequence', 'departure_time']
    ].rename(columns={
        'stop_id': 'origin_stop_id',
        'stop_sequence': 'origin_stop_sequence',
        'departure_time': 'origin_departure_time_raw',
    })
    dest_st = st[st['stop_id'].isin(dest_stop_ids)][
        ['trip_id', 'stop_id', 'stop_sequence', 'arrival_time']
    ].rename(columns={
        'stop_id': 'dest_stop_id',
        'stop_sequence': 'dest_stop_sequence',
        'arrival_time': 'dest_arrival_time_raw',
    })

    merged = origin_st.merge(dest_st, on='trip_id', how='inner')
    merged = merged[merged['origin_stop_sequence'] < merged['dest_stop_sequence']]
    if merged.empty:
        return []

    merged['origin_departure_dt'] = day_midnight + _parse_gtfs_time(merged['origin_departure_time_raw'])
    merged['dest_arrival_dt'] = day_midnight + _parse_gtfs_time(merged['dest_arrival_time_raw'])

    in_window = (
        (merged['origin_departure_dt'] >= departure_after)
        & (merged['origin_departure_dt'] <= window_end)
    )
    merged = merged[in_window]
    if merged.empty:
        return []

    merged = merged.merge(
        trips_today[['trip_id', 'route_id']], on='trip_id', how='left'
    )
    merged = merged.merge(
        data.routes[['route_id', 'route_short_name', 'route_long_name', 'route_type']],
        on='route_id', how='left',
    )

    stop_names = data.stops.set_index('stop_id')['stop_name']
    merged['origin_stop_name'] = merged['origin_stop_id'].map(stop_names)
    merged['dest_stop_name'] = merged['dest_stop_id'].map(stop_names)
    merged['n_stops_between'] = (
        merged['dest_stop_sequence'] - merged['origin_stop_sequence'] - 1
    ).clip(lower=0)

    merged = merged.sort_values('origin_departure_dt')

    return [
        {
            'trip_id': row.trip_id,
            'route_id': row.route_id,
            'route_short_name': row.route_short_name,
            'route_long_name': row.route_long_name,
            'route_type': row.route_type,
            'origin_stop_name': row.origin_stop_name,
            'origin_departure_time': row.origin_departure_dt.to_pydatetime(),
            'dest_stop_name': row.dest_stop_name,
            'dest_arrival_time': row.dest_arrival_dt.to_pydatetime(),
            'n_stops_between': int(row.n_stops_between),
        }
        for row in merged.itertuples(index=False)
    ]


def _pick_stop_group(query: str) -> dict | None:
    """Demo helper: from search_stops results, prefer a 'station' entry (a real
    transit hub) over a generic street stop that happens to share the name.
    """
    results = search_stops(query, limit=50)
    if not results:
        return None
    exact_station = f'{query.strip().lower()} station'
    for r in results:
        if r['stop_name'].lower() == exact_station:
            return r
    station_matches = [r for r in results if 'station' in r['stop_name'].lower()]
    return station_matches[0] if station_matches else results[0]


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
