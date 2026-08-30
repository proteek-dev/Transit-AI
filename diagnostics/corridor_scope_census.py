#!/usr/bin/env python3
"""Diagnostic-only corridor-scoping census over the current GTFS static
snapshot.

Answers two questions for the corridor-scoping decision:

  1. What is the TRUE physical route count, independent of route_id
     inflation? (route_id is re-issued in a new format across static
     snapshot versions; (route_short_name, route_type) is the stable
     identifier -- see phase3/route_types.py and the loader's route_meta
     convention.)
  2. How much of the static snapshot -- stops, routes, and stop_times ROWS
     (the memory driver) -- falls inside the Gold Coast + Logan + Brisbane
     LGA corridor vs. outside it?

Read-only against S3 (static GTFS only -- never touches gtfs_realtime/ or
any prefix the RT archive census script reads). No pipeline/phase3 changes.

LGA boundary source (real, cited, not guessed):
  ABS ASGS2023 Local Government Areas -- served live via the ABS
  Geospatial Solutions ArcGIS FeatureServer:
    https://geo.abs.gov.au/arcgis/rest/services/ASGS2023/LGA/FeatureServer/0
  Documented at:
    https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs-edition-3/jul2021-jun2026/access-and-downloads/digital-boundary-files
  Queried for exactly the 3 target LGAs (confirmed by direct query against
  the live service): LGA_CODE_2023 31000=Brisbane, 33430=Gold Coast,
  34590=Logan. Response is cached locally under diagnostics/cache/ so
  repeat runs don't re-download. If the download fails or returns anything
  other than exactly 3 polygons, this script stops with an error -- it does
  NOT fall back to a guessed bounding box.

Usage:
    python3 diagnostics/corridor_scope_census.py
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests

# Reuse the existing credential-loading pattern (st.secrets -> env -> .env)
# instead of duplicating or hardcoding credentials -- see phase3/config.py.
_PHASE3_DIR = Path(__file__).resolve().parent.parent / 'phase3'
if str(_PHASE3_DIR) not in sys.path:
    sys.path.insert(0, str(_PHASE3_DIR))
import config  # noqa: E402

DATE_PATTERN = re.compile(r'^\d{4}-\d{2}-\d{2}$')

ABS_LGA_QUERY_URL = 'https://geo.abs.gov.au/arcgis/rest/services/ASGS2023/LGA/FeatureServer/0/query'
CORRIDOR_LGA_CODES = {'31000': 'Brisbane', '33430': 'Gold Coast', '34590': 'Logan'}
CORRIDOR_LGA_NAMES = set(CORRIDOR_LGA_CODES.values())
CACHE_PATH = Path(__file__).resolve().parent / 'cache' / 'abs_lga_gold_coast_logan_brisbane.geojson'


def list_date_partitions(fs, prefix: str) -> list[str]:
    """Same YYYY-MM-DD enumeration logic phase3/gtfs/loader.py's
    GTFSData.load() and notebook 05 both use for gtfs_static/.
    """
    entries = fs.ls(prefix)
    return sorted(
        e.rstrip('/').split('/')[-1] for e in entries
        if DATE_PATTERN.match(e.rstrip('/').split('/')[-1])
    )


def fetch_lga_boundaries() -> gpd.GeoDataFrame:
    """The 3 target LGA polygons, from cache if present, else downloaded
    from the live ABS FeatureServer (see module docstring for the exact
    URL). Stops with a clear error on any failure -- never falls back to a
    guessed bounding box.
    """
    if CACHE_PATH.exists():
        print(f'Using cached LGA boundaries: {CACHE_PATH}')
        with open(CACHE_PATH) as f:
            geojson = json.load(f)
    else:
        print(f'Fetching LGA boundaries from ABS ASGS2023 FeatureServer:\n  {ABS_LGA_QUERY_URL}')
        params = {
            'where': "LGA_CODE_2023 IN ('" + "','".join(sorted(CORRIDOR_LGA_CODES)) + "')",
            'outFields': 'LGA_CODE_2023,LGA_NAME_2023',
            'outSR': '4326',
            'f': 'geojson',
        }
        try:
            resp = requests.get(ABS_LGA_QUERY_URL, params=params, timeout=60)
            resp.raise_for_status()
            geojson = resp.json()
        except Exception as e:
            raise RuntimeError(
                f'Failed to download LGA boundaries from {ABS_LGA_QUERY_URL}: {e}\n'
                'STOPPING -- not falling back to a guessed bounding box. '
                'Check connectivity and retry.'
            ) from e

        features = geojson.get('features', [])
        got_codes = {f['properties'].get('lga_code_2023') for f in features}
        if got_codes != set(CORRIDOR_LGA_CODES):
            raise RuntimeError(
                f'Expected exactly LGA codes {sorted(CORRIDOR_LGA_CODES)} '
                f'({sorted(CORRIDOR_LGA_CODES.values())}) from {ABS_LGA_QUERY_URL}, '
                f'got {sorted(c for c in got_codes if c)}.\n'
                'STOPPING -- refusing to proceed with incomplete/unexpected boundary data.'
            )

        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_PATH, 'w') as f:
            json.dump(geojson, f)
        print(f'Cached LGA boundaries -> {CACHE_PATH}')

    gdf = gpd.GeoDataFrame.from_features(geojson['features'], crs='EPSG:4326')
    gdf['lga_name'] = gdf['lga_name_2023']
    return gdf


def classify_stops(stops_df: pd.DataFrame, lga_gdf: gpd.GeoDataFrame) -> tuple[pd.DataFrame, int]:
    """Point-in-polygon classification of every stop into one of the 3
    corridor LGAs, 'other' (a real stop outside all 3), or flagged
    separately if lat/lon is missing (never silently dropped).
    """
    missing_mask = stops_df['stop_lat'].isna() | stops_df['stop_lon'].isna()
    n_missing = int(missing_mask.sum())

    valid = stops_df.loc[~missing_mask]
    points_gdf = gpd.GeoDataFrame(
        valid[['stop_id']],
        geometry=gpd.points_from_xy(valid['stop_lon'], valid['stop_lat']),
        crs='EPSG:4326',
    )
    joined = gpd.sjoin(points_gdf, lga_gdf[['lga_name', 'geometry']], how='left', predicate='within')
    # ABS LGA polygons are non-overlapping by construction -- a stop can
    # only legitimately match one, but sjoin can emit >1 row for a point
    # exactly on a shared boundary; dedupe defensively rather than double-count.
    joined = joined.drop_duplicates(subset='stop_id', keep='first')
    lga_by_stop = dict(zip(joined['stop_id'], joined['lga_name']))

    result = stops_df.copy()
    result['lga'] = result['stop_id'].map(lga_by_stop)
    result.loc[missing_mask, 'lga'] = 'MISSING_COORDS'
    result['lga'] = result['lga'].fillna('other')
    return result, n_missing


def main() -> None:
    t0 = time.time()
    bucket = config.get_s3_bucket()
    fs = config.get_s3_filesystem()
    print(f'Bucket: s3://{bucket}\n')

    static_prefix = f'{bucket}/gtfs_static'
    static_dates = list_date_partitions(fs, static_prefix)
    if not static_dates:
        raise FileNotFoundError(f'No YYYY-MM-DD subfolders found under s3://{static_prefix}')
    snapshot_date = max(static_dates)
    root = f's3://{static_prefix}/{snapshot_date}'
    print(f'Static snapshot: {snapshot_date} (latest of {len(static_dates)} available -- same selection '
          f'phase3/gtfs/loader.py uses)\n')

    # ------------------------------------------------------------------
    # PART 1 -- True physical route count
    # ------------------------------------------------------------------
    print('=' * 100)
    print('PART 1 -- True physical route count (fixes route_id inflation)')
    print('=' * 100)
    routes_df = pd.read_csv(f'{root}/routes.txt', dtype=str,
                             usecols=['route_id', 'route_short_name', 'route_type'])

    raw_route_id_count = routes_df['route_id'].nunique()
    true_route_keys = routes_df[['route_short_name', 'route_type']].drop_duplicates()
    true_route_count = len(true_route_keys)
    inflation_ratio = raw_route_id_count / true_route_count if true_route_count else float('nan')

    print(f'Raw distinct route_id count                              : {raw_route_id_count:,}')
    print(f'True route count (distinct route_short_name+route_type)  : {true_route_count:,}')
    print(f'Inflation ratio (route_id / true route)                  : {inflation_ratio:.2f}x')

    # ------------------------------------------------------------------
    # PART 2 -- Geographic corridor classification of stops
    # ------------------------------------------------------------------
    print('\n' + '=' * 100)
    print('PART 2 -- Geographic corridor classification (Gold Coast / Logan / Brisbane LGA)')
    print('=' * 100)
    lga_gdf = fetch_lga_boundaries()

    stops_df = pd.read_csv(f'{root}/stops.txt', dtype=str, usecols=['stop_id', 'stop_lat', 'stop_lon'])
    stops_df['stop_lat'] = pd.to_numeric(stops_df['stop_lat'], errors='coerce')
    stops_df['stop_lon'] = pd.to_numeric(stops_df['stop_lon'], errors='coerce')

    classified_stops, n_missing_coords = classify_stops(stops_df, lga_gdf)
    total_stops = len(classified_stops)
    lga_counts = classified_stops['lga'].value_counts()

    print(f'Total stops in static snapshot: {total_stops:,}')
    for name in ['Brisbane', 'Gold Coast', 'Logan', 'other', 'MISSING_COORDS']:
        label = 'Missing lat/lon (unclassifiable)' if name == 'MISSING_COORDS' else name
        print(f'  {label:<34}: {int(lga_counts.get(name, 0)):,}')
    if n_missing_coords:
        print(f'\n  FLAGGED: {n_missing_coords:,} stop(s) have null/missing lat or lon and could not be '
              f'classified -- counted above, not dropped.')

    # ------------------------------------------------------------------
    # PART 3 -- Route-level and row-level corridor membership
    # ------------------------------------------------------------------
    print('\n' + '=' * 100)
    print('PART 3 -- Route-level and row-level corridor membership')
    print('=' * 100)

    trips_df = pd.read_csv(f'{root}/trips.txt', dtype=str, usecols=['trip_id', 'route_id'])
    stop_times_df = pd.read_csv(f'{root}/stop_times.txt', dtype=str, usecols=['trip_id', 'stop_id'])
    total_stop_times_rows = len(stop_times_df)

    stop_lga_by_id = dict(zip(classified_stops['stop_id'], classified_stops['lga']))
    stop_times_df['in_corridor_stop'] = stop_times_df['stop_id'].map(stop_lga_by_id).isin(CORRIDOR_LGA_NAMES)

    # "In corridor" at the TRIP level: does this trip touch >= 1 corridor
    # stop anywhere along its journey? (Logan is full first-class scope,
    # not pass-through-only, so any of the 3 LGAs qualifies.)
    trip_touches_corridor = stop_times_df.groupby('trip_id')['in_corridor_stop'].any()

    # Row-level: ALL stop_times rows belonging to an in-corridor-touching
    # trip count as in-corridor, even rows at an out-of-corridor stop on
    # that same trip -- the whole trip's schedule is what a corridor-scoped
    # rebuild would need to keep in memory, not just its corridor-side stops.
    stop_times_df['trip_in_corridor'] = stop_times_df['trip_id'].map(trip_touches_corridor).fillna(False)
    in_corridor_rows = int(stop_times_df['trip_in_corridor'].sum())
    out_of_corridor_rows = total_stop_times_rows - in_corridor_rows

    # Route-level, on raw route_id (the natural join key from trips.txt).
    all_route_ids = set(routes_df['route_id'])
    route_ids_with_trips = set(trips_df['route_id'])
    n_routes_no_trips = len(all_route_ids - route_ids_with_trips)

    trip_route_touch = trips_df.assign(
        touches=trips_df['trip_id'].map(trip_touches_corridor).fillna(False)
    )
    route_in_corridor = trip_route_touch.groupby('route_id')['touches'].any()
    route_in_corridor_full = pd.Series(False, index=sorted(all_route_ids))
    route_in_corridor_full.update(route_in_corridor)

    raw_in_count = int(route_in_corridor_full.sum())
    raw_out_count = len(all_route_ids) - raw_in_count

    # Same membership rolled up to the TRUE route (route_short_name,
    # route_type) key from Part 1 -- a true route counts as in-corridor if
    # ANY of its constituent route_id variants touches the corridor.
    routes_with_membership = routes_df.copy()
    routes_with_membership['in_corridor'] = routes_with_membership['route_id'].map(route_in_corridor_full)
    true_route_membership = routes_with_membership.groupby(
        ['route_short_name', 'route_type']
    )['in_corridor'].any()
    true_in_count = int(true_route_membership.sum())
    true_out_count = len(true_route_membership) - true_in_count

    print('Route counts (raw route_id -- natural join key from trips.txt):')
    print(f'  In-corridor      : {raw_in_count:,}')
    print(f'  Out-of-corridor  : {raw_out_count:,}')
    print(f'  Total            : {len(all_route_ids):,}')
    if n_routes_no_trips:
        print(f'  (of which {n_routes_no_trips:,} route_id(s) have zero scheduled trips in this snapshot -- '
              f'counted as out-of-corridor above by default, flagged here rather than silently folded in)')

    print('\nRoute counts (TRUE route = route_short_name + route_type, from Part 1):')
    print(f'  In-corridor      : {true_in_count:,}')
    print(f'  Out-of-corridor  : {true_out_count:,}')
    print(f'  Total            : {len(true_route_membership):,}  (matches Part 1\'s true route count)')

    print(f'\nstop_times row counts (static snapshot, full file, {total_stop_times_rows:,} rows -- '
          f'this is a ROW-level figure, not a route count, and drives memory/RSS directly):')
    in_pct = 100 * in_corridor_rows / total_stop_times_rows if total_stop_times_rows else 0.0
    out_pct = 100 * out_of_corridor_rows / total_stop_times_rows if total_stop_times_rows else 0.0
    print(f'  In-corridor (trip touches >=1 corridor stop) : {in_corridor_rows:>10,} rows  ({in_pct:5.1f}%)')
    print(f'  Out-of-corridor                              : {out_of_corridor_rows:>10,} rows  ({out_pct:5.1f}%)')
    print(f'  Total                                        : {total_stop_times_rows:>10,} rows')

    print(f'\nDone in {time.time() - t0:.1f}s')


if __name__ == '__main__':
    main()
