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
        self.stop_to_routes = None   # stop_id -> set of route_id, built at load time
        self.route_to_stops = None   # route_id -> ordered list of stop_id, built at load time
        self.route_trip_counts = None  # route_id -> trip count, used for fan-out ranking
        self.stop_to_cluster = None    # stop_id -> canonical station name
        self.cluster_to_routes = None  # canonical station name -> set of route_id
        self.cluster_stop_ids = None   # canonical station name -> list of stop_id

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
        self._build_route_indexes()

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
        # stop_id -> canonical station name, reused by _build_route_indexes()
        # so transfer detection recognizes same-station platforms that carry
        # different stop_ids per mode (e.g. a tram platform and a train
        # platform at the same interchange).
        self._stop_id_to_canonical_name = dict(zip(stops['stop_id'], canonical_name))

    def _build_route_indexes(self):
        """stop_to_routes / route_to_stops / route_trip_counts — the route
        graph find_multi_leg_trips() BFSes over to find transfer chains.

        Also builds cluster_to_routes / cluster_stop_ids / stop_to_cluster:
        real interchanges often split platforms across stop_ids with no stop_id
        in common (a tram platform and a train platform at the same station),
        so two routes are also considered "connected" if they serve the same
        canonical station name, not just the same literal stop_id.
        """
        merged = self.stop_times[['trip_id', 'stop_id', 'stop_sequence']].merge(
            self.trips[['trip_id', 'route_id']], on='trip_id', how='left'
        ).dropna(subset=['route_id'])

        self.route_trip_counts = self.trips.groupby('route_id')['trip_id'].nunique().to_dict()

        stop_to_routes = {}
        pairs = merged[['stop_id', 'route_id']].drop_duplicates()
        for stop_id, route_id in pairs.itertuples(index=False):
            stop_to_routes.setdefault(stop_id, set()).add(route_id)
        self.stop_to_routes = stop_to_routes

        route_to_stops = {}
        ordered = merged.sort_values('stop_sequence')[['route_id', 'stop_id']].drop_duplicates()
        for route_id, group in ordered.groupby('route_id', sort=False):
            route_to_stops[route_id] = list(dict.fromkeys(group['stop_id']))
        self.route_to_stops = route_to_stops

        stop_to_cluster = self._stop_id_to_canonical_name
        self.stop_to_cluster = stop_to_cluster
        self.cluster_stop_ids = dict(zip(self._stop_index['stop_name'], self._stop_index['stop_ids']))

        cluster_to_routes = {}
        for stop_id, routes in stop_to_routes.items():
            cluster = stop_to_cluster.get(stop_id, stop_id)
            cluster_to_routes.setdefault(cluster, set()).update(routes)
        self.cluster_to_routes = cluster_to_routes

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


# Performance guards for find_multi_leg_trips()'s route-graph BFS — see
# _expand_route_frontier(). Without these, a hub like Roma Street (served by
# dozens of routes) makes the frontier explode combinatorially by depth 2.
_ROUTE_FAN_OUT_CAP = 15
_STOP_CLUSTER_CAP = 50


def _expand_route_frontier(
    data: GTFSData, frontier: list[dict], dest_routes: set, visited_routes: set,
) -> tuple[list[dict], list[dict]]:
    """One BFS hop over the route-intersection graph.

    `frontier` entries are {'routes': [route_id, ...], 'transfer_stops': [[stop_id, ...], ...]}
    chains that haven't reached a route serving the destination yet (each
    transfer_stops entry is the full list of stop_ids at that hop's station,
    not a single stop_id — see cluster note below). Returns (completed_chains,
    next_frontier) where completed_chains are chains whose newly-added route
    is in dest_routes.

    Two routes are considered "connected" at a *station cluster* — same
    canonical station name — not just a literal shared stop_id, because real
    interchanges often split platforms across stop_ids with nothing in common
    (e.g. a tram platform and a train platform at the same station). Without
    this, transfers at exactly those interchanges are invisible to the BFS.

    `visited_routes` is a global set of every route_id already reached by an
    earlier or the current hop, mutated in place. Without it, the same route
    gets re-discovered via every path that leads to it and the frontier grows
    combinatorially (this previously made a 3-hop BFS at a busy hub run for
    30+ minutes); with it, each route is expanded from at most once, so total
    work is bounded by the size of the route network, not the number of paths
    through it.
    """
    route_to_stops = data.route_to_stops
    route_trip_counts = data.route_trip_counts
    stop_to_cluster = data.stop_to_cluster
    cluster_to_routes = data.cluster_to_routes
    cluster_stop_ids = data.cluster_stop_ids

    # Cap 1: pool every station cluster the frontier's routes visit, keep the
    # 50 clusters served by the most routes (the interchanges most likely to
    # yield a transfer).
    pooled_clusters = set()
    for chain in frontier:
        for stop in route_to_stops.get(chain['routes'][-1], []):
            pooled_clusters.add(stop_to_cluster.get(stop, stop))
    candidate_clusters = sorted(
        pooled_clusters, key=lambda c: -len(cluster_to_routes.get(c, ()))
    )[:_STOP_CLUSTER_CAP]
    candidate_cluster_set = set(candidate_clusters)

    completed, next_frontier = [], []

    for chain in frontier:
        last_route = chain['routes'][-1]
        clusters_here = {
            stop_to_cluster.get(s, s) for s in route_to_stops.get(last_route, [])
        } & candidate_cluster_set

        for cluster in clusters_here:
            candidate_routes = cluster_to_routes.get(cluster, set()) - visited_routes
            if not candidate_routes:
                continue

            def _chain_to(next_route):
                return {
                    'routes': chain['routes'] + [next_route],
                    'transfer_stops': chain['transfer_stops'] + [cluster_stop_ids.get(cluster, [cluster])],
                }

            # A route that already reaches the destination is a free win —
            # recognize it regardless of the fan-out cap below. GTFS feeds
            # that version route_ids per direction/pattern (e.g. dozens of
            # near-duplicate rail route_ids at one interchange) can otherwise
            # rank the exact direction needed well outside the top 15 by
            # trip count, hiding a real connection.
            for next_route in candidate_routes & dest_routes:
                if next_route in visited_routes:
                    continue
                visited_routes.add(next_route)
                completed.append(_chain_to(next_route))

            # Cap 2: only the exploratory (non-destination) routes are capped
            # to the 15 busiest — this is what actually bounds how much
            # deeper BFS work a hub like Roma Street can generate.
            exploratory = candidate_routes - dest_routes - visited_routes
            if len(exploratory) > _ROUTE_FAN_OUT_CAP:
                exploratory = set(
                    sorted(exploratory, key=lambda r: -route_trip_counts.get(r, 0))[:_ROUTE_FAN_OUT_CAP]
                )

            for next_route in exploratory:
                if next_route in visited_routes:
                    continue
                visited_routes.add(next_route)
                next_frontier.append(_chain_to(next_route))

    return completed, next_frontier


def _resolve_chain(
    chain: dict,
    origin_stop_ids: list[str],
    dest_stop_ids: list[str],
    departure_after: datetime,
    window_minutes: int,
    min_connection: int,
    max_connection: int,
) -> dict | None:
    """Resolve a route-graph chain into a timed journey by calling find_trips()
    leg-by-leg. Returns None if any leg has no timed trip within its window, or
    any connection falls outside [min_connection, max_connection].
    """
    routes = chain['routes']
    leg_stop_bounds = [origin_stop_ids] + list(chain['transfer_stops']) + [dest_stop_ids]
    connection_window = max(max_connection - min_connection, 1)

    legs = []
    transfer_points = []
    next_departure_after = departure_after
    next_window = window_minutes

    for i, route_id in enumerate(routes):
        from_stops = leg_stop_bounds[i]
        to_stops = leg_stop_bounds[i + 1]
        candidates = find_trips(from_stops, to_stops, next_departure_after, next_window)
        candidates = [t for t in candidates if t['route_id'] == route_id]
        if not candidates:
            return None
        trip = candidates[0]

        if i > 0:
            prev_arrival = legs[i - 1]['trip']['dest_arrival_time']
            connection_minutes = int((trip['origin_departure_time'] - prev_arrival).total_seconds() // 60)
            if not (min_connection <= connection_minutes <= max_connection):
                return None
            transfer_points.append({
                'stop_name': legs[i - 1]['trip']['dest_stop_name'],
                'connection_minutes': connection_minutes,
            })

        legs.append({
            'trip': trip,
            'board_stop_name': trip['origin_stop_name'],
            'alight_stop_name': trip['dest_stop_name'],
            'dest_stop_ids': to_stops,
            'search_departure_after': next_departure_after,
        })

        next_departure_after = trip['dest_arrival_time'] + timedelta(minutes=min_connection)
        next_window = connection_window

    total_minutes = int(
        (legs[-1]['trip']['dest_arrival_time'] - legs[0]['trip']['origin_departure_time']).total_seconds() // 60
    )
    return {
        'legs': legs,
        'transfer_points': transfer_points,
        'total_minutes': total_minutes,
        'num_transfers': len(legs) - 1,
    }


def _direct_journey(trip: dict, dest_stop_ids: list[str], departure_after: datetime) -> dict:
    """Wrap a single find_trips() result in the same journey shape find_multi_leg_trips() returns."""
    total_minutes = int((trip['dest_arrival_time'] - trip['origin_departure_time']).total_seconds() // 60)
    return {
        'legs': [{
            'trip': trip,
            'board_stop_name': trip['origin_stop_name'],
            'alight_stop_name': trip['dest_stop_name'],
            'dest_stop_ids': dest_stop_ids,
            'search_departure_after': departure_after,
        }],
        'transfer_points': [],
        'total_minutes': total_minutes,
        'num_transfers': 0,
    }


def find_multi_leg_trips(
    origin_stop_ids: list[str],
    dest_stop_ids: list[str],
    departure_after: datetime,
    window_minutes: int = 60,
    min_connection: int = 5,
    max_connection: int = 45,
    max_transfers: int = 3,
    max_results: int = 5,
) -> list[dict]:
    """Find journeys (direct or with transfers) from origin to destination via
    route-intersection BFS over the static GTFS network.

    Depth 0 checks for a route serving both origin and destination directly
    (delegating to find_trips()). Depth 1+ BFSes the route graph — two routes
    are "connected" if they share a stop — expanding one hop per depth and
    resolving newly-completed chains into timed journeys via find_trips() per
    leg. BFS stops deepening as soon as `max_results` timed journeys have been
    found, so shorter-transfer-count journeys are always preferred.
    """
    data = load_gtfs_data()
    stop_to_routes = data.stop_to_routes

    origin_routes = set()
    for sid in origin_stop_ids:
        origin_routes |= stop_to_routes.get(sid, set())
    dest_routes = set()
    for sid in dest_stop_ids:
        dest_routes |= stop_to_routes.get(sid, set())

    journeys = []

    # Depth 0: direct trips on a route serving both origin and destination.
    if origin_routes & dest_routes:
        for trip in find_trips(origin_stop_ids, dest_stop_ids, departure_after, window_minutes):
            journeys.append(_direct_journey(trip, dest_stop_ids, departure_after))

    if len(journeys) >= max_results:
        return sorted(journeys, key=lambda j: j['total_minutes'])[:max_results]

    # Depth 1+: transfer chains via route-intersection BFS. `visited_routes`
    # is global across the whole search (see _expand_route_frontier) so each
    # route is only ever expanded from once, keeping the BFS tractable at
    # busy hubs like Roma Street.
    visited_routes = set(origin_routes)
    frontier = [{'routes': [r], 'transfer_stops': []} for r in origin_routes]

    for _depth in range(1, max_transfers + 1):
        if not frontier:
            break
        completed, frontier = _expand_route_frontier(data, frontier, dest_routes, visited_routes)

        for chain in completed:
            if len(journeys) >= max_results:
                break
            journey = _resolve_chain(
                chain, origin_stop_ids, dest_stop_ids, departure_after, window_minutes,
                min_connection, max_connection,
            )
            if journey is not None:
                journeys.append(journey)

        if len(journeys) >= max_results:
            break

    return sorted(journeys, key=lambda j: j['total_minutes'])[:max_results]


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
