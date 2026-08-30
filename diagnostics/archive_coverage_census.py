#!/usr/bin/env python3
"""Diagnostic-only coverage census over the S3 GTFS-RT archive.

Answers "what routes/modes/agencies exist in the archive and static
snapshot" -- nothing more. No geographic/LGA classification, no writes to
S3, no changes to pipeline/ or phase3/ files. Samples 5 source_date
partitions spread across the archive's date range (not a full rebuild).

Usage:
    python3 diagnostics/archive_coverage_census.py
"""
from __future__ import annotations

import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

# Reuse the existing credential-loading pattern (st.secrets -> env -> .env)
# instead of duplicating or hardcoding credentials -- see phase3/config.py.
_PHASE3_DIR = Path(__file__).resolve().parent.parent / 'phase3'
if str(_PHASE3_DIR) not in sys.path:
    sys.path.insert(0, str(_PHASE3_DIR))
import config  # noqa: E402
from route_types import MODE_BY_ROUTE_TYPE_STR  # noqa: E402

DATE_PATTERN = re.compile(r'^\d{4}-\d{2}-\d{2}$')
N_SAMPLES = 5

# A single day's trip_updates folder can hold 500+ 5-min-poll files, each
# carrying every downstream StopTimeUpdate for every active trip -- reading
# a full day for all 5 sampled partitions took ~5.7 minutes in practice, well
# over the "couple minutes" budget for a coverage check (vs. a full rebuild).
# Files are thinned to an evenly-spaced subset per partition (still spanning
# the whole day, so early-morning/late-night-only routes aren't systematically
# missed) -- distinct route/mode coverage and row counts below are therefore
# from the SAMPLED files only, not the full day, and are reported as such.
MAX_FILES_PER_PARTITION = 60


def list_date_partitions(fs, prefix: str) -> list[str]:
    """YYYY-MM-DD subfolders directly under `prefix` -- same enumeration
    logic notebooks/05_phase2_feature_pipeline.ipynb uses to list both the
    gtfs_realtime/trip_updates and gtfs_static partitions (Cells 1 and 2).
    """
    entries = fs.ls(prefix)
    return sorted(
        e.rstrip('/').split('/')[-1] for e in entries
        if DATE_PATTERN.match(e.rstrip('/').split('/')[-1])
    )


def sample_evenly(items: list, n: int) -> list:
    """n evenly-spaced items spanning the full list (first ... last),
    deduped, order preserved. Used both for picking sample dates (earliest/
    ~25%/~50%/~75%/latest) and for thinning one partition's file list.
    """
    if len(items) <= n:
        return items
    idxs = [round(i * (len(items) - 1) / (n - 1)) for i in range(n)]
    picked = []
    for i in idxs:
        if items[i] not in picked:
            picked.append(items[i])
    return picked


def match_static_snapshot(source_date: str, snapshot_dates: list[str]) -> str:
    """Nearest static snapshot <= source_date, else the earliest available
    -- mirrors notebook 05's match_static_snapshot() (same schedule-in-effect
    logic the real pipeline uses to join a realtime date).
    """
    eligible = [d for d in snapshot_dates if d <= source_date]
    return max(eligible) if eligible else min(snapshot_dates)


def load_routes_for_snapshot(bucket: str, snapshot_date: str) -> pd.DataFrame | None:
    path = f's3://{bucket}/gtfs_static/{snapshot_date}/routes.txt'
    try:
        return pd.read_csv(path, dtype=str)
    except FileNotFoundError:
        return None


def _load_json_file(fs, path: str) -> list[dict]:
    with fs.open(path) as f:
        return json.load(f)


def census_one_partition(fs, bucket: str, rt_prefix: str, source_date: str,
                          static_dates: list[str], routes_cache: dict) -> dict:
    all_files = sorted(f for f in fs.ls(f'{rt_prefix}/{source_date}') if f.endswith('.json'))
    files = sample_evenly(all_files, MAX_FILES_PER_PARTITION)

    records: list[dict] = []
    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = [executor.submit(_load_json_file, fs, p) for p in files]
        for fut in as_completed(futures):
            records.extend(fut.result())

    df = pd.DataFrame(records, columns=['trip_id', 'route_id', 'stop_id', 'delay_seconds', 'timestamp'])
    row_count = len(df)
    distinct_route_ids = sorted(df['route_id'].dropna().unique()) if row_count else []

    snap_date = match_static_snapshot(source_date, static_dates)
    if snap_date not in routes_cache:
        routes_cache[snap_date] = load_routes_for_snapshot(bucket, snap_date)
    routes_df = routes_cache[snap_date]

    route_short_names: list[str] = []
    route_types: list[str] = []
    if routes_df is not None and row_count:
        merged = df[['route_id']].drop_duplicates().merge(routes_df, on='route_id', how='left')
        route_short_names = sorted(merged['route_short_name'].dropna().unique())
        route_types = sorted(merged['route_type'].dropna().unique())

    agency_id_present = routes_df is not None and 'agency_id' in routes_df.columns
    agency_ids = sorted(routes_df['agency_id'].dropna().unique()) if agency_id_present else []

    return {
        'source_date': source_date,
        'n_files_total': len(all_files),
        'n_files_sampled': len(files),
        'row_count': row_count,
        'static_snapshot_used': snap_date,
        'distinct_route_id': distinct_route_ids,
        'distinct_route_short_name': route_short_names,
        'distinct_route_type': route_types,
        'agency_id_present': agency_id_present,
        'distinct_agency_id': agency_ids,
    }


def census_static_snapshot(fs, bucket: str) -> dict:
    """Same snapshot-selection logic as phase3/gtfs/loader.py's
    GTFSData.load(): the latest YYYY-MM-DD snapshot under gtfs_static/ --
    the dataset the Render OOM issue is about.
    """
    static_prefix = f'{bucket}/gtfs_static'
    static_dates = list_date_partitions(fs, static_prefix)
    latest = max(static_dates)
    root = f's3://{static_prefix}/{latest}'

    routes = pd.read_csv(f'{root}/routes.txt', dtype=str)
    agency_id_present = 'agency_id' in routes.columns

    stop_times_rows = 0
    for chunk in pd.read_csv(f'{root}/stop_times.txt', usecols=['trip_id'], dtype=str, chunksize=500_000):
        stop_times_rows += len(chunk)

    return {
        'snapshot_date': latest,
        'all_snapshot_dates': static_dates,
        'distinct_route_id': sorted(routes['route_id'].dropna().unique()),
        'distinct_route_short_name': sorted(routes['route_short_name'].dropna().unique()),
        'distinct_route_type': sorted(routes['route_type'].dropna().unique()),
        'agency_id_present': agency_id_present,
        'distinct_agency_id': sorted(routes['agency_id'].dropna().unique()) if agency_id_present else [],
        'stop_times_row_count': stop_times_rows,
    }


def main() -> None:
    t0 = time.time()
    bucket = config.get_s3_bucket()
    fs = config.get_s3_filesystem()
    print(f'Bucket: s3://{bucket}\n')

    rt_prefix = f'{bucket}/gtfs_realtime/trip_updates'
    all_rt_dates = list_date_partitions(fs, rt_prefix)
    if not all_rt_dates:
        raise FileNotFoundError(f'No YYYY-MM-DD subfolders found under s3://{rt_prefix}')

    sampled = sample_evenly(all_rt_dates, N_SAMPLES)
    print(f'Archive has {len(all_rt_dates)} source_date partition(s): {all_rt_dates[0]} -> {all_rt_dates[-1]}')
    print(f'Sampling {len(sampled)} (earliest/~25%/~50%/~75%/latest): {sampled}\n')

    static_dates = list_date_partitions(fs, f'{bucket}/gtfs_static')
    routes_cache: dict = {}

    print('=' * 100)
    print('Reading sampled archive partitions ...')
    print('=' * 100)
    partition_results = []
    for d in sampled:
        print(f'  {d} ...', end=' ', flush=True)
        result = census_one_partition(fs, bucket, rt_prefix, d, static_dates, routes_cache)
        partition_results.append(result)
        print(f'{result["row_count"]:,} rows from {result["n_files_sampled"]}/{result["n_files_total"]} file(s) sampled, '
              f'{len(result["distinct_route_id"])} route_id(s), '
              f'{len(result["distinct_route_type"])} route_type(s) '
              f'(static snapshot used: {result["static_snapshot_used"]})')

    print('\n' + '=' * 100)
    print('Reading static GTFS snapshot (the one phase3/gtfs/loader.py loads) ...')
    print('=' * 100)
    static_result = census_static_snapshot(fs, bucket)
    print(f'  snapshot_date={static_result["snapshot_date"]}  '
          f'routes={len(static_result["distinct_route_id"])}  '
          f'stop_times rows={static_result["stop_times_row_count"]:,}')

    # --- Summary table ---
    print('\n' + '=' * 100)
    print('SUMMARY -- coverage census')
    print(f'(per-partition rows/route counts are from the {MAX_FILES_PER_PARTITION}-file-cap sample below, '
          f'not the full day -- see "files" column)')
    print('=' * 100)
    header = f'{"source_date":<14} {"files":>11} {"rows (sampled)":>15} {"route_id":>9} {"short_name":>11} {"route_type":>11} {"agency_id":>10}'
    print(header)
    print('-' * len(header))
    for r in partition_results:
        agency_col = len(r['distinct_agency_id']) if r['agency_id_present'] else 'N/A'
        files_col = f'{r["n_files_sampled"]}/{r["n_files_total"]}'
        print(f'{r["source_date"]:<14} {files_col:>11} {r["row_count"]:>15,} {len(r["distinct_route_id"]):>9} '
              f'{len(r["distinct_route_short_name"]):>11} {len(r["distinct_route_type"]):>11} '
              f'{agency_col!s:>10}')
    print('-' * len(header))
    static_agency_col = len(static_result['distinct_agency_id']) if static_result['agency_id_present'] else 'N/A'
    print(f'{"STATIC " + static_result["snapshot_date"]:<14} {"n/a":>11} {"n/a":>15} '
          f'{len(static_result["distinct_route_id"]):>9} {len(static_result["distinct_route_short_name"]):>11} '
          f'{len(static_result["distinct_route_type"]):>11} {static_agency_col!s:>10}'
          f'   (stop_times rows: {static_result["stop_times_row_count"]:,}, full snapshot -- not sampled)')

    # --- Full distinct value lists (small enough to eyeball directly) ---
    all_route_types = sorted(set().union(
        *(set(r['distinct_route_type']) for r in partition_results),
        set(static_result['distinct_route_type']),
    ))
    print('\nAll distinct route_type values seen (archive samples + static snapshot):')
    for rt in all_route_types:
        mode = MODE_BY_ROUTE_TYPE_STR.get(rt, 'UNKNOWN/unexpected -- not in phase3/route_types.py')
        print(f'  route_type={rt!r} -> mode={mode}')

    any_agency_present = static_result['agency_id_present'] or any(r['agency_id_present'] for r in partition_results)
    if any_agency_present:
        all_agency_ids = sorted(set().union(
            *(set(r['distinct_agency_id']) for r in partition_results),
            set(static_result['distinct_agency_id']),
        ))
        print(f'\nAll distinct agency_id values seen: {all_agency_ids}')
    else:
        print('\nNo agency_id column found in routes.txt for any sampled/static snapshot -- '
              'this is a single-agency feed (agency.txt lists exactly one agency, "Translink", '
              'with no agency_id column either), so there is nothing to check for an unexpected '
              'agency_id value.')

    expected_modes = {'tram', 'bus', 'rail'}
    seen_modes = {MODE_BY_ROUTE_TYPE_STR.get(rt) for rt in all_route_types}
    missing_modes = expected_modes - seen_modes
    print(f'\nExpected modes (tram/bus/rail) all present: {"YES" if not missing_modes else f"NO -- missing {missing_modes}"}')
    print(f'ferry (route_type=\'4\') present in this census: {"4" in all_route_types}')

    print(f'\nDone in {time.time() - t0:.1f}s')


if __name__ == '__main__':
    main()
