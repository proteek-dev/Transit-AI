"""Multi-leg trip routing (BFS engine) for the Phase 3 Streamlit POC.

find_trips() answers a single-route origin -> destination query;
find_multi_leg_trips() BFSes the route-intersection graph on top of it to
find transfer journeys. Ferry service never reaches this module -- it's
excluded upstream in loader.GTFSData.load() (see route_types.FERRY_ROUTE_TYPE),
so every route/stop/trip id seen here is already ferry-free.
"""
from __future__ import annotations

import bisect
from datetime import datetime, timedelta

from gtfs.loader import GTFSData, _parse_gtfs_time, load_gtfs_data

# Sunday/thin-calendar fallback: a query date within this many days of the
# static snapshot's capture date is where TransLink's calendar_dates.txt
# additions (e.g. special Sunday services) are least likely to be published
# yet -- see find_trips() fallback below.
FALLBACK_THIN_COVERAGE_DAYS = 3


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

    trip_cols = ['trip_id', 'route_id']
    if 'trip_headsign' in trips_today.columns:
        trip_cols.append('trip_headsign')
    merged = merged.merge(trips_today[trip_cols], on='trip_id', how='left')
    if 'trip_headsign' not in merged.columns:
        merged['trip_headsign'] = None

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
            'trip_headsign': row.trip_headsign if isinstance(row.trip_headsign, str) and row.trip_headsign.strip() else None,
            'origin_stop_id': row.origin_stop_id,
            'origin_stop_name': row.origin_stop_name,
            'origin_departure_time': row.origin_departure_dt.to_pydatetime(),
            'dest_stop_id': row.dest_stop_id,
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

# Cap 3: at most this many alternative connecting chains are kept PER
# destination-serving route (see _expand_route_frontier's docstring on why
# more than one alternative must be tried at all). Without this cap, a hub
# whose destination is served by dozens of near-duplicate route_id variants
# (e.g. one GTFS route_id per direction/pattern on the same physical rail
# line) combined with several independent connecting routes at the origin
# makes the number of candidate chains grow combinatorially with BFS depth
# (observed: 67 -> 2,014 -> 15,195 across 3 depths for one real origin/dest
# pair) — each candidate requires a full _resolve_chain() call, so this
# alone can make a query take minutes. Capping to the busiest few
# alternatives per destination route keeps the fix for the premature-
# visited-routes bug from reintroducing that blowup.
_DEST_ROUTE_ALTERNATIVES_CAP = 3


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
    earlier or the current hop, mutated in place *for exploratory routes
    only* (see below). Without it, the same route gets re-discovered via
    every path that leads to it and the frontier grows combinatorially (this
    previously made a 3-hop BFS at a busy hub run for 30+ minutes); with it,
    each route is expanded from at most once, so total work is bounded by the
    size of the route network, not the number of paths through it.

    Completing routes (routes already in `dest_routes`) are handled
    differently: `visited_routes` is deliberately NOT updated for them here.
    Reaching a destination-serving route topologically doesn't guarantee
    find_trips() can actually build a timed trip through it (wrong direction,
    no schedule overlap, connection window too tight) — and several distinct
    connecting routes/stations can independently reach the same destination
    route. If the first one discovered claimed it here, an unlucky (hash-
    randomized set iteration) processing order could permanently discard a
    reachable destination in favor of one that fails to resolve. So every
    chain that reaches a `dest_routes` member is returned in `completed`
    (ranked, not deduped) and it is the *caller's* job to try them in order
    and only mark the route settled once one resolves or all have failed.
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

    next_frontier = []
    # next_route -> list of candidate chains reaching it, capped per-route at
    # the end (Cap 3) rather than during collection, so the cap keeps the
    # busiest few candidates regardless of which cluster/chain happens to be
    # processed first (still not hash-order-dependent).
    completed_by_route: dict[str, list[dict]] = {}

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
            #
            # Every alternative chain reaching a dest route is kept (not just
            # the first found) — see the visited_routes note in the
            # docstring above for why first-reach-wins is wrong here. Cap 3
            # (applied after this loop, once all candidates are collected)
            # bounds how many alternatives per route actually get resolved.
            for next_route in candidate_routes & dest_routes:
                completed_by_route.setdefault(next_route, []).append(_chain_to(next_route))

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

    # Cap 3 + deterministic trial order: per destination route, keep only the
    # _DEST_ROUTE_ALTERNATIVES_CAP candidates with the busiest connecting
    # route, ranked busiest-first. Bounds how many _resolve_chain() calls the
    # caller makes per route (see the constant's comment for why this is
    # needed) while still trying multiple alternatives, and makes trial order
    # reproducible run to run rather than dependent on Python's hash-
    # randomized set iteration order.
    completed = []
    for candidates in completed_by_route.values():
        candidates.sort(key=lambda c: -route_trip_counts.get(c['routes'][-2], 0))
        completed.extend(candidates[:_DEST_ROUTE_ALTERNATIVES_CAP])
    completed.sort(key=lambda c: -route_trip_counts.get(c['routes'][-2], 0))

    return completed, next_frontier


_INDEX_UNRESOLVED = object()  # sentinel: the fast index has no data for this route/stop -- caller must fall back to the pandas path


def _any_route_reaches_dest(data, from_stops, to_stops, query_date, search_departure_after, window_minutes):
    """Mirrors find_trips()'s own non-empty gate (any route, any trip, from
    any of from_stops to any of to_stops, sequence-ordered, within window, on
    an active service) -- NOT scoped to one route_id.

    find_trips()'s FALLBACK_THIN_COVERAGE_DAYS day-shift only triggers when
    _find_trips_core() is empty across ALL routes for this stop pair -- if
    some other route already serves it, find_trips() returns those rows
    immediately and never reaches its own fallback, even though the ONE
    route _lookup_leg_trip cares about has nothing that day. Without this
    check, _lookup_leg_trip's day-shift retry (below) would fire whenever
    just its own route is empty, misapplying a 7-day-shifted schedule to a
    route that's correctly just not running that day (e.g. a calendar_dates
    exception for that one date) while a near-duplicate route/service
    happens to run a week later -- confirmed by differential testing against
    the old code on a real case (Southport -> Beenleigh, route VLBD-4999).
    """
    active = data.active_service_ids(query_date)
    day_midnight = datetime(query_date.year, query_date.month, query_date.day)
    window_start = (search_departure_after - day_midnight).total_seconds()
    window_end = window_start + window_minutes * 60
    to_stop_set = set(to_stops)

    routes_here = set()
    for sid in from_stops:
        routes_here |= data.stop_to_routes.get(sid, set())

    for route_id in routes_here:
        stop_departures = data.route_stop_departures.get(route_id)
        trip_stops = data.route_trip_stops.get(route_id)
        if not stop_departures or not trip_stops:
            continue
        for sid in from_stops:
            entries = stop_departures.get(sid)
            if not entries:
                continue
            idx = bisect.bisect_left(entries, (window_start,))
            for dep_sec, trip_id, seq, service_id in entries[idx:]:
                if dep_sec > window_end:
                    break
                if service_id not in active:
                    continue
                trip_map = trip_stops.get(trip_id)
                if not trip_map:
                    continue
                for dsid in to_stop_set:
                    dest_entry = trip_map.get(dsid)
                    if dest_entry and dest_entry[0] > seq:
                        return True
    return False


def _lookup_leg_trip(
    data: GTFSData,
    route_id: str,
    from_stops: list[str],
    to_stops: list[str],
    departure_after: datetime,
    window_minutes: int,
):
    """Fast index-based replacement for _resolve_chain()'s old
    find_trips()-then-filter-by-route_id existence check. Mirrors
    find_trips()'s semantics exactly -- earliest valid departure on
    `route_id` from any of `from_stops` to any of `to_stops` within the
    window, including its FALLBACK_THIN_COVERAGE_DAYS 7-day-ahead retry --
    using GTFSData.route_stop_departures/route_trip_stops instead of a
    pandas merge over the whole day's stop_times.

    Returns a trip dict shaped like find_trips()'s rows, None if genuinely no
    connecting trip exists on this route (a real negative answer -- the old
    code would have reached the same conclusion, just slower), or the
    _INDEX_UNRESOLVED sentinel if the index has no data to answer this query
    at all (route_id or all of from_stops missing from the index -- should
    not happen given route_to_stops is built from the same data, but this is
    flagged explicitly by the caller rather than silently guessing).
    """
    stop_departures = data.route_stop_departures.get(route_id)
    trip_stops = data.route_trip_stops.get(route_id)
    if not stop_departures or not trip_stops:
        return _INDEX_UNRESOLVED
    if not any(sid in stop_departures for sid in from_stops):
        return _INDEX_UNRESOLVED

    to_stop_set = set(to_stops)

    def _search(query_date, search_departure_after):
        active = data.active_service_ids(query_date)
        day_midnight = datetime(query_date.year, query_date.month, query_date.day)
        window_start = (search_departure_after - day_midnight).total_seconds()
        window_end = window_start + window_minutes * 60

        best = None  # (dep_sec, trip_id, origin_stop_id, origin_seq, dest_stop_id, dest_seq, arr_sec)
        for sid in from_stops:
            entries = stop_departures.get(sid)
            if not entries:
                continue
            idx = bisect.bisect_left(entries, (window_start,))
            for dep_sec, trip_id, seq, service_id in entries[idx:]:
                if dep_sec > window_end:
                    break
                if best is not None and dep_sec >= best[0]:
                    break  # sorted ascending -- nothing further here can beat the current best
                if service_id not in active:
                    continue
                trip_map = trip_stops.get(trip_id)
                if not trip_map:
                    continue
                for dsid in to_stop_set:
                    dest_entry = trip_map.get(dsid)
                    if dest_entry and dest_entry[0] > seq:
                        best = (dep_sec, trip_id, sid, seq, dsid, dest_entry[0], dest_entry[1])
                        break
        return best, day_midnight

    query_date = departure_after.date()
    best, day_midnight = _search(query_date, departure_after)

    fallback_schedule = False
    if best is None:
        snapshot_date = datetime.strptime(data.snapshot_date, '%Y-%m-%d').date()
        if abs((query_date - snapshot_date).days) <= FALLBACK_THIN_COVERAGE_DAYS:
            # Only attempt the day-shift if NO route at all serves this stop
            # pair in this window on this date -- matching find_trips()'s
            # real trigger condition exactly (see _any_route_reaches_dest).
            if not _any_route_reaches_dest(data, from_stops, to_stops, query_date, departure_after, window_minutes):
                fb_departure_after = departure_after + timedelta(days=7)
                best, day_midnight = _search(fb_departure_after.date(), fb_departure_after)
                fallback_schedule = best is not None

    if best is None:
        return None

    dep_sec, trip_id, origin_stop_id, origin_seq, dest_stop_id, dest_seq, arr_sec = best
    origin_departure_time = day_midnight + timedelta(seconds=dep_sec)
    dest_arrival_time = day_midnight + timedelta(seconds=arr_sec)
    if fallback_schedule:
        origin_departure_time -= timedelta(days=7)
        dest_arrival_time -= timedelta(days=7)

    meta = data.route_meta.get(route_id, {})
    return {
        'trip_id': trip_id,
        'route_id': route_id,
        'route_short_name': meta.get('route_short_name'),
        'route_long_name': meta.get('route_long_name'),
        'route_type': meta.get('route_type'),
        'trip_headsign': data.trip_headsign_by_id.get(trip_id),
        'origin_stop_id': origin_stop_id,
        'origin_stop_name': data.stop_name_by_id.get(origin_stop_id),
        'origin_departure_time': origin_departure_time,
        'dest_stop_id': dest_stop_id,
        'dest_stop_name': data.stop_name_by_id.get(dest_stop_id),
        'dest_arrival_time': dest_arrival_time,
        'n_stops_between': max(dest_seq - origin_seq - 1, 0),
        'fallback_schedule': fallback_schedule,
    }


def _resolve_chain(
    data: GTFSData,
    chain: dict,
    origin_stop_ids: list[str],
    dest_stop_ids: list[str],
    departure_after: datetime,
    window_minutes: int,
    min_connection: int,
    max_connection: int,
) -> dict | None:
    """Resolve a route-graph chain into a timed journey, leg-by-leg. Each leg
    first tries _lookup_leg_trip()'s fast per-route index; only when that
    index can't answer at all (_INDEX_UNRESOLVED) does it fall back to the
    original find_trips()-then-filter-by-route_id pandas path, logging the
    fallthrough. Returns None if any leg has no timed trip within its window,
    or any connection falls outside [min_connection, max_connection].
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

        trip = _lookup_leg_trip(data, route_id, from_stops, to_stops, next_departure_after, next_window)
        if trip is _INDEX_UNRESOLVED:
            print(
                f'[_resolve_chain] index fallthrough: route_id={route_id!r} not covered by '
                f'route_stop_departures/route_trip_stops -- falling back to full find_trips() scan'
            )
            candidates = find_trips(from_stops, to_stops, next_departure_after, next_window)
            candidates = [t for t in candidates if t['route_id'] == route_id]
            trip = candidates[0] if candidates else None
        if trip is None:
            return None

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

        # `completed` may hold several competing chains that all end on the
        # same destination-serving route (different connecting routes or
        # transfer stations reaching it — see _expand_route_frontier's
        # docstring). Try each, in the ranked order _expand_route_frontier
        # already sorted them into, until one resolves; only then treat that
        # destination route as settled for this depth, so an early candidate
        # that fails to resolve doesn't shadow a later one that would have
        # worked.
        settled_routes = set()
        for chain in completed:
            if len(journeys) >= max_results:
                break
            terminal_route = chain['routes'][-1]
            if terminal_route in settled_routes:
                continue
            journey = _resolve_chain(
                data, chain, origin_stop_ids, dest_stop_ids, departure_after, window_minutes,
                min_connection, max_connection,
            )
            if journey is not None:
                journeys.append(journey)
                settled_routes.add(terminal_route)

        # Whether or not it resolved, every destination route seen at this
        # depth has now had every currently-known alternative tried — mark
        # it visited so deeper depths don't keep re-attempting it.
        visited_routes.update(chain['routes'][-1] for chain in completed)

        if len(journeys) >= max_results:
            break

    return sorted(journeys, key=lambda j: j['total_minutes'])[:max_results]
