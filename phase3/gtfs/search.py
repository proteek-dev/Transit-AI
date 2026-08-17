"""Stop search for the Phase 3 Streamlit POC.

Typeahead search (search_stops) and map-picker nearest-stop ranking
(nearest_stops) on top of the stop index loader.GTFSData._build_stop_index()
builds -- including its turnback-stop exclusion (Session 22).
"""
from __future__ import annotations

from math import atan2, cos, log1p, radians, sin, sqrt

from rapidfuzz import fuzz, process

from gtfs.loader import load_gtfs_data


def search_stops(
    query: str,
    limit: int = 10,
    ref_lat: float | None = None,
    ref_lon: float | None = None,
) -> list[dict]:
    """Typeahead stop search: prefix matches first, then fuzzy matches, deduped by stop_name.

    When ref_lat/ref_lon are both given, the fuzzy-match branch pulls a
    wider candidate pool from process.extract() and re-sorts it by haversine
    distance from (ref_lat, ref_lon) before truncating to `remaining` --
    narrows fuzzy typeahead results toward a known reference point (e.g. a
    confirmed origin) instead of a query fragment matching stops scattered
    across the whole service area. Prefix matches are untouched either way
    -- already query-specific enough that geo-biasing isn't needed there, so
    no distance_km is computed or attached for them. When ref_lat/ref_lon
    are omitted (the default), this function is byte-for-byte identical to
    the un-biased version: no widened pool, no re-sort, no distance_km field
    on any result.
    """
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
        has_ref = ref_lat is not None and ref_lon is not None
        # Widen the pool pulled from process.extract() when there's a
        # reference point to re-rank against, so a closer-but-slightly-
        # lower-scoring match isn't excluded before re-ranking gets a
        # chance. Capped so a large `remaining` can't request an unbounded
        # extract() pool.
        extract_limit = min(remaining * 4, 40) if has_ref else remaining
        matches = process.extract(
            query, choices, scorer=fuzz.WRatio, limit=extract_limit, processor=str.lower
        )
        matched_names = [m[0] for m in matches if m[1] > 0]
        fuzzy_rows = non_prefix[non_prefix['stop_name'].isin(matched_names)]
        # preserve rapidfuzz's ranked order
        fuzzy_rows = fuzzy_rows.set_index('stop_name').loc[matched_names].reset_index()

        if has_ref:
            fuzzy_rows = fuzzy_rows.copy()
            fuzzy_rows['distance_km'] = [
                _haversine_km(ref_lat, ref_lon, lat, lon)
                for lat, lon in zip(fuzzy_rows['stop_lat'], fuzzy_rows['stop_lon'])
            ]
            fuzzy_rows = fuzzy_rows.sort_values('distance_km').head(remaining)

        results += list(fuzzy_rows.itertuples(index=False))

    results = results[:limit]

    return [
        {
            'stop_id': row.stop_ids[0],
            'stop_ids': row.stop_ids,
            'stop_name': row.stop_name,
            'stop_lat': row.stop_lat,
            'stop_lon': row.stop_lon,
            # Only present on fuzzy-matched rows when a reference point was
            # given (see has_ref above) -- reuses the distance already
            # computed for sorting rather than recomputing it here, and is
            # omitted (not set to None) everywhere else, so downstream code
            # can distinguish "we don't know" from "we know and it's far."
            **({'distance_km': row.distance_km} if hasattr(row, 'distance_km') else {}),
        }
        for row in results
    ]


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(dlambda / 2) ** 2
    return 2 * r * atan2(sqrt(a), sqrt(1 - a))


def nearest_stops(lat: float, lon: float, limit: int = 15) -> list[dict]:
    """Nearest *useful* stops to (lat, lon) for a map-pin origin picker.

    Ranks candidates by a blend of distance and service frequency (via
    route_trip_counts) rather than pure distance, so a stop with frequent
    service a bit farther away can outrank a barely-served stop that's
    slightly closer.

    Reuses the parent_station clustering built by _build_stop_index() (same
    one search_stops() uses) so platform variants of one station don't spam
    the candidate list. But when a cluster's member stops serve genuinely
    different modes (e.g. a tram platform and a train platform folded under
    the same station name), it's split back into one pin per mode — that's
    exactly the ambiguity a map should resolve visually, not hide.
    """
    data = load_gtfs_data()
    stop_to_routes = data.stop_to_routes
    route_trip_counts = data.route_trip_counts
    route_type_by_route = data.routes.set_index('route_id')['route_type'].to_dict()
    stop_lat_lon = data.stops.set_index('stop_id')[['stop_lat', 'stop_lon']]

    candidates = []
    for row in data._stop_index.itertuples(index=False):
        by_mode: dict[int, list[str]] = {}
        for stop_id in row.stop_ids:
            route_types = {route_type_by_route.get(r) for r in stop_to_routes.get(stop_id, set())}
            route_types.discard(None)
            for rt in route_types:
                by_mode.setdefault(rt, []).append(stop_id)

        if not by_mode:
            continue

        if len(by_mode) == 1:
            # Common case: every member stop serves the same single mode —
            # one merged pin, reusing the cluster's already-averaged coords.
            groups = [(set(by_mode), row.stop_ids, row.stop_lat, row.stop_lon)]
        else:
            groups = []
            for rt, sids in by_mode.items():
                coords = stop_lat_lon.loc[stop_lat_lon.index.intersection(sids)]
                if coords.empty:
                    continue
                groups.append(({rt}, sids, coords['stop_lat'].mean(), coords['stop_lon'].mean()))

        for route_types, sids, clat, clon in groups:
            routes_here = set()
            for sid in sids:
                routes_here |= stop_to_routes.get(sid, set())
            trip_count = sum(route_trip_counts.get(r, 0) for r in routes_here)

            distance_km = _haversine_km(lat, lon, clat, clon)
            # Usefulness bonus is log-scaled so a handful of extra trips
            # doesn't dominate, but a well-served stop meaningfully outranks
            # a barely-served one at similar distance.
            score = distance_km / (1.0 + log1p(trip_count))

            candidates.append({
                'stop_id': sids[0],
                'stop_ids': sorted(set(sids)),
                'stop_name': row.stop_name,
                'stop_lat': clat,
                'stop_lon': clon,
                'route_types': sorted(route_types),
                'distance_km': distance_km,
                'trip_count': trip_count,
                '_score': score,
            })

    candidates.sort(key=lambda c: c['_score'])
    for c in candidates:
        del c['_score']
    return candidates[:limit]
