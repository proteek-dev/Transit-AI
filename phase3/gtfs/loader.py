"""GTFS static-snapshot loader for the Phase 3 Streamlit POC.

Loads the latest static GTFS snapshot from S3, applies ferry exclusion, and
builds the stop/route indexes that search.py and routing.py read off
GTFSData, plus the shape_id -> [(lat, lon)] lookup shapes.py reads.
"""
from __future__ import annotations

import re

import pandas as pd

import config
from route_types import FERRY_ROUTE_TYPE

DATE_PATTERN = re.compile(r'^\d{4}-\d{2}-\d{2}$')
DAY_NAMES = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']

# Column set load_stop_times_optimized() validates the S3 parquet against --
# matches load()'s own stop_time_cols and scripts/precompute_gtfs_static.py's
# STOP_TIME_COLS (the artifact producer), so a schema drift between the two
# is caught explicitly rather than surfacing as a downstream KeyError.
STOP_TIME_COLS = ['trip_id', 'stop_id', 'stop_sequence', 'arrival_time', 'departure_time']

# Ferry is out of scope everywhere except the raw S3 archiver
# (scripts/archive_gtfsrt.py, untouched) -- routes, trips, stop_times, and
# stops are all filtered against this in load() so no ferry service can ever
# reach search, BFS transfer routing, or prediction downstream.
# (FERRY_ROUTE_TYPE itself now lives in route_types.py, imported above.)


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
        self.shape_points = None       # shape_id -> ordered list of (lat, lon), built at load time
        # Per-route departure/arrival index used by _resolve_chain()'s fast
        # path (see _build_route_departure_index) -- a static structural
        # artifact of the snapshot, same rationale as route_to_stops above.
        self.route_stop_departures = None  # route_id -> stop_id -> sorted [(departure_seconds, trip_id, stop_sequence, service_id)]
        self.route_trip_stops = None       # route_id -> trip_id -> {stop_id: (stop_sequence, arrival_seconds)}
        self.route_meta = None             # route_id -> {route_short_name, route_long_name, route_type}
        self.stop_name_by_id = None        # stop_id -> stop_name
        self.trip_headsign_by_id = None    # trip_id -> trip_headsign or None

    def load(self, use_categorical_dtypes: bool = False):
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

        snapshot_root = f's3://{static_prefix}/{self.snapshot_date}'

        # TODO: consider category dtype for repeated ID columns (trip_id,
        # stop_id, route_id, service_id) read below, needs its own isolated
        # change + full test pass -- left as plain str/object for now since
        # they're used as raw dict keys, merge keys, and itertuples fields
        # throughout _build_route_indexes() and _build_route_departure_index(),
        # and category dtype risks silent equality/hash surprises there.

        # --- Ferry exclusion, in dependency order (routes -> trips ->
        # stop_times -> stops) -- see FERRY_ROUTE_TYPE above. Filtering here,
        # before _build_stop_index()/_build_route_indexes() run, means every
        # downstream structure (search index, stop_to_routes, route_to_stops,
        # cluster_to_routes) is ferry-free by construction, with no separate
        # filtering needed at the search/BFS call sites themselves. Each read
        # below is also restricted to the columns this module actually uses
        # (usecols) and stop_times.txt -- the ~2.9M-row dominant contributor
        # to peak memory -- is read+filtered in chunks so the full unfiltered
        # frame is never held in memory at once. ---
        routes_raw = pd.read_csv(
            f'{snapshot_root}/routes.txt',
            usecols=['route_id', 'route_short_name', 'route_long_name', 'route_type'],
            dtype=str,
        )
        routes_raw = routes_raw.assign(route_type=routes_raw['route_type'].astype('int32'))
        ferry_route_ids = set(routes_raw.loc[routes_raw['route_type'] == FERRY_ROUTE_TYPE, 'route_id'])
        self.routes = routes_raw[routes_raw['route_type'] != FERRY_ROUTE_TYPE].reset_index(drop=True)

        # trips.txt is small (no chunking needed) -- filtered against
        # ferry_route_ids up front so the stop_times chunk loop below can
        # filter against non_ferry_trip_ids directly.
        trips_raw = pd.read_csv(
            f'{snapshot_root}/trips.txt',
            usecols=['trip_id', 'route_id', 'service_id', 'shape_id', 'trip_headsign'],
            dtype=str,
        )
        self.trips = trips_raw[~trips_raw['route_id'].isin(ferry_route_ids)].reset_index(drop=True)
        non_ferry_trip_ids = set(self.trips['trip_id'])

        stop_time_cols = ['trip_id', 'stop_id', 'stop_sequence', 'arrival_time', 'departure_time']
        stop_times_chunks = []
        stop_time_rows_loaded = 0
        for chunk in pd.read_csv(
            f'{snapshot_root}/stop_times.txt',
            usecols=stop_time_cols,
            dtype=str,
            chunksize=500_000,
        ):
            stop_time_rows_loaded += len(chunk)
            chunk = chunk.assign(stop_sequence=chunk['stop_sequence'].astype('int32'))
            chunk = chunk[chunk['trip_id'].isin(non_ferry_trip_ids)]
            if use_categorical_dtypes:
                # Convert per-chunk (~500k rows), before concat -- NOT on the
                # full concatenated frame after the fact. Converting the
                # already-concatenated ~3M-row object-dtype frame requires
                # holding both the object and categorical representations of
                # the WHOLE frame in memory simultaneously during that one
                # .astype('category') call, and glibc's allocator doesn't
                # return that transient peak to the OS afterward -- measured
                # RSS actually went UP (901.9MB vs 655.7MB baseline) despite
                # stop_times' own reported memory_usage() going down.
                # Converting each chunk right after its ferry-filter slice
                # bounds that transient peak to one chunk's size instead.
                chunk['trip_id'] = chunk['trip_id'].astype('category')
                chunk['stop_id'] = chunk['stop_id'].astype('category')
            stop_times_chunks.append(chunk)
        self.stop_times = (
            pd.concat(stop_times_chunks, ignore_index=True) if stop_times_chunks
            else pd.DataFrame(columns=stop_time_cols)
        )
        del stop_times_chunks
        non_ferry_stop_ids = set(self.stop_times['stop_id'])

        stops_raw = pd.read_csv(
            f'{snapshot_root}/stops.txt',
            usecols=['stop_id', 'stop_name', 'stop_lat', 'stop_lon', 'parent_station'],
            dtype=str,
        )
        # Keep any parent_station referenced by a surviving stop too, so
        # _build_stop_index()'s canonical-name lookup (parent id -> parent
        # name) still resolves for stops that share a hub with a kept mode
        # (e.g. a train platform's parent station record).
        kept_parent_ids = set(
            stops_raw.loc[stops_raw['stop_id'].isin(non_ferry_stop_ids), 'parent_station'].dropna()
        )
        keep_stop_ids = non_ferry_stop_ids | kept_parent_ids
        self.stops = stops_raw[stops_raw['stop_id'].isin(keep_stop_ids)].reset_index(drop=True)
        self.stops['stop_lat'] = self.stops['stop_lat'].astype('float32')
        self.stops['stop_lon'] = self.stops['stop_lon'].astype('float32')

        # Optional, additive path -- only taken when a caller explicitly asks
        # for it (see load_gtfs_data(); default False, so phase3/app.py's
        # Streamlit call is completely unaffected). stop_times' trip_id/
        # stop_id are already categorical at this point (converted per-chunk
        # above, before concat). trips and stops are read as single
        # non-chunked frames (small enough that a post-hoc convert here isn't
        # implicated in the peak-memory regression the chunk-level change
        # above exists to fix), so trip_id/stop_id are converted here to
        # match stop_times' categorical dtype -- every merge/join against
        # stop_times' trip_id/stop_id (gtfs/routing.py, _build_route_indexes(),
        # _build_route_departure_index() below) then sees consistent
        # categorical columns on both sides rather than mixing categorical
        # against plain object dtype. arrival_time/departure_time are
        # deliberately left alone -- lower repetition, and this pass isn't
        # touching downstream time-formatting code.
        if use_categorical_dtypes:
            self.trips['trip_id'] = self.trips['trip_id'].astype('category')
            self.stops['stop_id'] = self.stops['stop_id'].astype('category')

        print(
            f'Excluded ferry (route_type={FERRY_ROUTE_TYPE}): '
            f'{len(ferry_route_ids):,} route(s), '
            f'{len(trips_raw) - len(self.trips):,} trip(s), '
            f'{stop_time_rows_loaded - len(self.stop_times):,} stop_time row(s), '
            f'{len(stops_raw) - len(self.stops):,} ferry-only stop(s)'
        )

        # --- DIAGNOSTIC (print-only) -- investigate 'turnback' stops. Runs
        # against the post-ferry-filter frames (self.stops / self.stop_times),
        # NOT the raw unfiltered frames the previous (pre-chunking) version
        # of this diagnostic used: now that stop_times.txt is read+filtered
        # in chunks (above), the full unfiltered frame is never materialized
        # as a single object to diagnose against, and reconstructing it here
        # would mean concatenating every chunk unfiltered first -- reintroducing
        # the exact peak-memory problem this change exists to fix. Deliberate
        # call: turnback stops are not a ferry concern (see _build_stop_index()
        # below) and this diagnostic's purpose -- auditing turnback stop_times
        # occurrences -- doesn't need ferry rows present, so running it
        # post-filter is equivalent for its purpose. Flagging this explicitly
        # since it is a behavior change from the prior "raw frame" version. ---
        turnback_mask = self.stops['stop_name'].str.contains('turnback', case=False, na=False)
        turnback_stops = self.stops[turnback_mask]
        print(f"[DIAGNOSTIC] 'turnback' stop_name matches: {len(turnback_stops)}")
        for name in sorted(turnback_stops['stop_name'].unique()):
            print(f'  - {name!r}')

        trip_min_max = self.stop_times.groupby('trip_id')['stop_sequence'].agg(['min', 'max'])

        for row in turnback_stops.itertuples(index=False):
            stop_id = row.stop_id
            parent_station = getattr(row, 'parent_station', None)
            has_parent = isinstance(parent_station, str) and parent_station.strip() != ''

            matches = self.stop_times[self.stop_times['stop_id'] == stop_id]
            n_trips = matches['trip_id'].nunique()
            print(f'\n[DIAGNOSTIC] stop_id={stop_id!r} stop_name={row.stop_name!r}')
            print(f"  in stop_times.txt: {'yes' if not matches.empty else 'no'} ({n_trips} distinct trip_id(s))")
            print(f'  parent_station: {parent_station!r} (has_parent={has_parent})')

            if not matches.empty:
                first_count = last_count = middle_count = 0
                for m in matches.itertuples(index=False):
                    tmin, tmax = trip_min_max.loc[m.trip_id, 'min'], trip_min_max.loc[m.trip_id, 'max']
                    if m.stop_sequence == tmin:
                        first_count += 1
                    elif m.stop_sequence == tmax:
                        last_count += 1
                    else:
                        middle_count += 1
                print(f'  occurrences: {len(matches)} across {n_trips} distinct trip(s) -- '
                      f'position: first={first_count} last={last_count} middle={middle_count}')

        self.calendar = pd.read_csv(
            f'{snapshot_root}/calendar.txt',
            usecols=['service_id', *DAY_NAMES, 'start_date', 'end_date'],
            dtype=str,
        )
        for day in DAY_NAMES:
            self.calendar[day] = self.calendar[day].astype('int32')
        self.calendar['start_date'] = self.calendar['start_date'].astype('int32')
        self.calendar['end_date'] = self.calendar['end_date'].astype('int32')

        self.calendar_dates = pd.read_csv(
            f'{snapshot_root}/calendar_dates.txt',
            usecols=['service_id', 'date', 'exception_type'],
            dtype=str,
        )

        shapes_raw = pd.read_csv(
            f'{snapshot_root}/shapes.txt',
            usecols=['shape_id', 'shape_pt_lat', 'shape_pt_lon', 'shape_pt_sequence'],
            dtype=str,
        )
        shapes_raw = shapes_raw.assign(
            shape_pt_lat=shapes_raw['shape_pt_lat'].astype('float32'),
            shape_pt_lon=shapes_raw['shape_pt_lon'].astype('float32'),
            shape_pt_sequence=shapes_raw['shape_pt_sequence'].astype('int32'),
        ).sort_values(['shape_id', 'shape_pt_sequence'])
        self.shape_points = {
            shape_id: list(zip(group['shape_pt_lat'], group['shape_pt_lon']))
            for shape_id, group in shapes_raw.groupby('shape_id', sort=False)
        }

        self._build_stop_index()
        self._build_route_indexes()
        self._build_route_departure_index()

        print(
            f'Loaded snapshot {self.snapshot_date}: '
            f'{len(self.stops):,} stops, {len(self.routes):,} routes, '
            f'{len(self.trips):,} trips, {len(self.stop_times):,} stop_times'
        )
        print(
            f'Excluded {len(self._turnback_excluded_names)} turnback stop(s) from the search index '
            f'(still present in routing/prediction data): {self._turnback_excluded_names}'
        )

        total_trips = len(self.trips)
        trips_with_shape = self.trips['shape_id'].notna().sum()
        pct_with_shape = (trips_with_shape / total_trips * 100) if total_trips else 0.0
        print(
            f'Loaded {len(self.shape_points):,} shapes; '
            f'{trips_with_shape:,}/{total_trips:,} trips ({pct_with_shape:.1f}%) have a shape_id'
        )
        shape_id_by_trip = self.trips.set_index('trip_id')['shape_id']
        for trip_id in self.trips['trip_id'].head(3):
            shape_id = shape_id_by_trip.get(trip_id)
            points = self.shape_points.get(shape_id) if pd.notna(shape_id) else None
            n_points = len(points) if points is not None else 0
            print(f'  spot-check trip_id={trip_id!r} shape_id={shape_id!r}: {n_points} shape points')

    def load_optimized(self):
        """Opt-in, memory-lean sibling of load() for webapp/backend's
        Render process -- not called by phase3/app.py's Streamlit process,
        which keeps using load() unchanged.

        Reads routes/trips/stops/calendar/calendar_dates/shapes from S3 CSV
        exactly like load() does (they're small; not the OOM driver), but
        sources stop_times from the precomputed S3 parquet
        (scripts/precompute_gtfs_static.py) via load_stop_times_optimized()
        instead of load()'s chunked stop_times.txt CSV parse -- the ~3M-row
        table that dominates peak memory on Render's 512MB free tier.

        Builds the exact same derived structures load() does
        (_build_stop_index / _build_route_indexes / _build_route_departure_index)
        off the resulting frames, so a GTFSData populated via this method is
        indistinguishable, shape-wise, to anything reading it (routing.py,
        search.py, shapes.py) from one populated via load().
        """
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
        print(f'Loading static GTFS snapshot (optimized stop_times): {self.snapshot_date}')

        snapshot_root = f's3://{static_prefix}/{self.snapshot_date}'

        # --- Same ferry-exclusion pipeline as load() (routes -> trips ->
        # stop_times -> stops), same usecols/dtype choices for the small
        # tables. Only stop_times' source differs (parquet, not chunked CSV). ---
        routes_raw = pd.read_csv(
            f'{snapshot_root}/routes.txt',
            usecols=['route_id', 'route_short_name', 'route_long_name', 'route_type'],
            dtype=str,
        )
        routes_raw = routes_raw.assign(route_type=routes_raw['route_type'].astype('int32'))
        ferry_route_ids = set(routes_raw.loc[routes_raw['route_type'] == FERRY_ROUTE_TYPE, 'route_id'])
        self.routes = routes_raw[routes_raw['route_type'] != FERRY_ROUTE_TYPE].reset_index(drop=True)

        trips_raw = pd.read_csv(
            f'{snapshot_root}/trips.txt',
            usecols=['trip_id', 'route_id', 'service_id', 'shape_id', 'trip_headsign'],
            dtype=str,
        )
        self.trips = trips_raw[~trips_raw['route_id'].isin(ferry_route_ids)].reset_index(drop=True)
        non_ferry_trip_ids = set(self.trips['trip_id'])

        # The parquet artifact is unfiltered (see scripts/precompute_gtfs_static.py's
        # docstring) -- ferry exclusion against non_ferry_trip_ids happens here,
        # same cross-reference load()'s chunk loop does against stop_times.txt.
        self.stop_times = load_stop_times_optimized(non_ferry_trip_ids=non_ferry_trip_ids)
        non_ferry_stop_ids = set(self.stop_times['stop_id'])

        stops_raw = pd.read_csv(
            f'{snapshot_root}/stops.txt',
            usecols=['stop_id', 'stop_name', 'stop_lat', 'stop_lon', 'parent_station'],
            dtype=str,
        )
        kept_parent_ids = set(
            stops_raw.loc[stops_raw['stop_id'].isin(non_ferry_stop_ids), 'parent_station'].dropna()
        )
        keep_stop_ids = non_ferry_stop_ids | kept_parent_ids
        self.stops = stops_raw[stops_raw['stop_id'].isin(keep_stop_ids)].reset_index(drop=True)
        self.stops['stop_lat'] = self.stops['stop_lat'].astype('float32')
        self.stops['stop_lon'] = self.stops['stop_lon'].astype('float32')

        # stop_times' trip_id/stop_id are already categorical (parquet-sourced,
        # see load_stop_times_optimized()) -- match that on trips/stops here,
        # same as load()'s use_categorical_dtypes=True path, so every merge
        # against stop_times downstream (_build_route_indexes(),
        # _build_route_departure_index()) sees consistent dtypes on both sides.
        self.trips['trip_id'] = self.trips['trip_id'].astype('category')
        self.stops['stop_id'] = self.stops['stop_id'].astype('category')

        self.calendar = pd.read_csv(
            f'{snapshot_root}/calendar.txt',
            usecols=['service_id', *DAY_NAMES, 'start_date', 'end_date'],
            dtype=str,
        )
        for day in DAY_NAMES:
            self.calendar[day] = self.calendar[day].astype('int32')
        self.calendar['start_date'] = self.calendar['start_date'].astype('int32')
        self.calendar['end_date'] = self.calendar['end_date'].astype('int32')

        self.calendar_dates = pd.read_csv(
            f'{snapshot_root}/calendar_dates.txt',
            usecols=['service_id', 'date', 'exception_type'],
            dtype=str,
        )

        shapes_raw = pd.read_csv(
            f'{snapshot_root}/shapes.txt',
            usecols=['shape_id', 'shape_pt_lat', 'shape_pt_lon', 'shape_pt_sequence'],
            dtype=str,
        )
        shapes_raw = shapes_raw.assign(
            shape_pt_lat=shapes_raw['shape_pt_lat'].astype('float32'),
            shape_pt_lon=shapes_raw['shape_pt_lon'].astype('float32'),
            shape_pt_sequence=shapes_raw['shape_pt_sequence'].astype('int32'),
        ).sort_values(['shape_id', 'shape_pt_sequence'])
        self.shape_points = {
            shape_id: list(zip(group['shape_pt_lat'], group['shape_pt_lon']))
            for shape_id, group in shapes_raw.groupby('shape_id', sort=False)
        }

        self._build_stop_index()
        self._build_route_indexes()
        self._build_route_departure_index()

        print(
            f'Loaded snapshot {self.snapshot_date} (optimized stop_times): '
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

        # Tram turnback points (e.g. 'Cavill Avenue turnback') are operational
        # waypoints a trip passes through mid-journey while reversing
        # direction, not passenger-facing boarding stops -- confirmed via
        # diagnostic: every occurrence in stop_times.txt sits at a middle
        # stop_sequence position (never first/last), and none carry a
        # parent_station. Excluded here from the search/nearest-stop index
        # ONLY (name-based, not a hardcoded stop_id list, since stop_ids are
        # snapshot-specific). routing (route_to_stops, stop_to_routes,
        # cluster_to_routes, _stop_id_to_canonical_name below) still needs
        # the full stop set -- real trips physically pass through them.
        turnback_mask = stops['stop_name'].str.contains('turnback', case=False, na=False)
        self._turnback_excluded_names = sorted(stops.loc[turnback_mask, 'stop_name'].unique())
        searchable_stops = stops.loc[~turnback_mask]

        grouped = searchable_stops.assign(canonical_name=canonical_name).groupby(
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
        # platform at the same interchange). Built from the FULL stops frame
        # (not searchable_stops) -- must not lose turnback stop_ids, which
        # _build_route_indexes() needs for correct transfer clustering.
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

    def _build_route_departure_index(self):
        """Per-route stop-level departure/arrival index -- a static structural
        artifact of the snapshot (same rationale as route_to_stops), built
        once here so _resolve_chain()'s per-leg existence check doesn't need
        a fresh pandas merge over the whole day's stop_times for every
        candidate chain. Profiling on real hub-to-hub queries showed this
        merge (via find_trips() -> _find_trips_core()) costing ~127ms/call,
        with hundreds of candidate chains per query -- that's the >50s cost
        this index replaces with an O(log n) lookup.

        route_stop_departures[route_id][stop_id] = sorted list of
        (departure_seconds, trip_id, stop_sequence, service_id) -- the
        origin-side lookup, sorted so a window search is a bisect.

        route_trip_stops[route_id][trip_id] = {stop_id: (stop_sequence,
        arrival_seconds)} -- every stop a given trip visits on this route,
        for the destination-side reachability + ordering check.

        Also caches route_meta / stop_name_by_id / trip_headsign_by_id so a
        fast-path hit can build a full trip dict without touching pandas.
        """
        merged = self.stop_times[['trip_id', 'stop_id', 'stop_sequence', 'arrival_time', 'departure_time']].merge(
            self.trips[['trip_id', 'route_id', 'service_id']], on='trip_id', how='left'
        ).dropna(subset=['route_id'])
        merged = merged.assign(
            departure_seconds=_parse_gtfs_time(merged['departure_time']).dt.total_seconds().astype(int),
            arrival_seconds=_parse_gtfs_time(merged['arrival_time']).dt.total_seconds().astype(int),
        )

        route_stop_departures: dict[str, dict[str, list]] = {}
        route_trip_stops: dict[str, dict[str, dict]] = {}

        for route_id, group in merged.groupby('route_id', sort=False):
            by_stop: dict[str, list] = {}
            by_trip: dict[str, dict] = {}
            for row in group.itertuples(index=False):
                by_stop.setdefault(row.stop_id, []).append(
                    (row.departure_seconds, row.trip_id, row.stop_sequence, row.service_id)
                )
                by_trip.setdefault(row.trip_id, {})[row.stop_id] = (row.stop_sequence, row.arrival_seconds)
            for entries in by_stop.values():
                entries.sort(key=lambda t: t[0])
            route_stop_departures[route_id] = by_stop
            route_trip_stops[route_id] = by_trip

        self.route_stop_departures = route_stop_departures
        self.route_trip_stops = route_trip_stops

        self.route_meta = {
            row.route_id: {
                'route_short_name': row.route_short_name,
                'route_long_name': row.route_long_name,
                'route_type': row.route_type,
            }
            for row in self.routes.itertuples(index=False)
        }
        self.stop_name_by_id = self.stops.set_index('stop_id')['stop_name'].to_dict()
        if 'trip_headsign' in self.trips.columns:
            self.trip_headsign_by_id = {
                tid: (h if isinstance(h, str) and h.strip() else None)
                for tid, h in zip(self.trips['trip_id'], self.trips['trip_headsign'])
            }
        else:
            self.trip_headsign_by_id = {}

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


def load_stop_times_optimized(non_ferry_trip_ids: set | None = None) -> pd.DataFrame:
    """Opt-in stop_times loader for memory-constrained callers (webapp/backend's
    Render process) -- reads the precomputed, already-categorical-dtyped
    parquet artifact from S3 (scripts/precompute_gtfs_static.py) via a single
    pd.read_parquet() call, same S3-URI-read pattern as
    diagnostics/verify_optimized_read.py, instead of GTFSData.load()'s
    chunked stop_times.txt CSV parse. Does not touch, wrap, or replace that
    CSV path -- phase3/app.py's Streamlit process is completely unaffected.

    Returns the same columns GTFSData.load() populates self.stop_times with
    (trip_id, stop_id, stop_sequence, arrival_time, departure_time) -- a
    drop-in column-for-column, usable by anything currently reading that
    frame. dtypes are a strict superset of load(use_categorical_dtypes=True)'s:
    trip_id/stop_id/arrival_time/departure_time all come back as category
    (not just trip_id/stop_id) and stop_sequence as int32 -- confirmed
    compatible with downstream code (_parse_gtfs_time()'s .str.split() and
    the trip_id/stop_id merges in _build_route_indexes() /
    _build_route_departure_index() both work unchanged against categorical
    input).

    non_ferry_trip_ids, when given, ferry-filters the frame the same way
    GTFSData.load() does -- the parquet artifact itself is unfiltered (see
    scripts/precompute_gtfs_static.py's docstring: ferry exclusion was
    deliberately left to this follow-up wiring step).

    Raises rather than silently falling back to the CSV path on any S3 read
    failure (missing artifact, bad credentials) or schema mismatch --
    callers that opted into this for its memory savings need to know
    immediately if the optimized artifact isn't there, not silently eat the
    CSV parse's memory cost behind their back.
    """
    bucket, _fs = _get_env()
    parquet_uri = f's3://{bucket}/phase3/gtfs_static_optimized/latest/stop_times.parquet'

    try:
        stop_times = pd.read_parquet(parquet_uri)
    except Exception as e:
        raise RuntimeError(
            f'Failed to read optimized stop_times parquet from {parquet_uri}: {e}. '
            f'Has scripts/precompute_gtfs_static.py been run for the current snapshot?'
        ) from e

    missing_cols = [c for c in STOP_TIME_COLS if c not in stop_times.columns]
    if missing_cols:
        raise RuntimeError(
            f'Optimized stop_times parquet at {parquet_uri} is missing expected column(s) '
            f'{missing_cols} (got {list(stop_times.columns)}) -- schema drift from '
            f'scripts/precompute_gtfs_static.py?'
        )

    if non_ferry_trip_ids is not None:
        stop_times = stop_times[stop_times['trip_id'].isin(non_ferry_trip_ids)].reset_index(drop=True)

    return stop_times


_gtfs_cache = {}


def load_gtfs_data(force_reload: bool = False, use_categorical_dtypes: bool = False) -> GTFSData:
    """Load (or return the cached) parsed static GTFS snapshot.

    use_categorical_dtypes only affects a fresh load -- once cached, later
    calls (with any combination of these args) return the same cached
    object as-is. Not an issue in practice: phase3/app.py's Streamlit
    process and webapp/backend/'s process are separate processes with their
    own cache, and each only ever calls this with one fixed set of args for
    its process's lifetime (see webapp/backend/main.py's warm-up).
    """
    if force_reload or 'data' not in _gtfs_cache:
        data = GTFSData()
        data.load(use_categorical_dtypes=use_categorical_dtypes)
        _gtfs_cache['data'] = data
    return _gtfs_cache['data']


def load_gtfs_data_optimized(force_reload: bool = False) -> GTFSData:
    """Opt-in sibling of load_gtfs_data() for webapp/backend's Render process.

    Populates the SAME shared _gtfs_cache this module's load_gtfs_data()
    reads from, via GTFSData.load_optimized() instead of GTFSData.load().
    This matters, not just for symmetry: phase3/gtfs/routing.py's
    find_trips() / find_multi_leg_trips() / _direct_journey() all call the
    plain load_gtfs_data() (no args) internally to fetch the cached
    instance -- populating a separate cache here would leave those calls
    re-triggering their own full stop_times.txt CSV parse on first use,
    silently defeating the point of warming up with the lean parquet path.
    """
    if force_reload or 'data' not in _gtfs_cache:
        data = GTFSData()
        data.load_optimized()
        _gtfs_cache['data'] = data
    return _gtfs_cache['data']
