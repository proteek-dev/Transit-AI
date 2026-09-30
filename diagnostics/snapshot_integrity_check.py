#!/usr/bin/env python3
"""Read-only S3 integrity check for the optimized GTFS stop_times snapshot
(webapp/backend's production data source) against the raw GTFS static
snapshot it's meant to be derived from.

WHY: EC2 production logs a startup snapshot summary drastically smaller than
a prior local run's --
  EC2  : 7,410 stops, 1,336 routes, 144,728 trips,   714,195 stop_times
  local: 13,035 stops, 1,516 routes, 144,008 trips, 1,673,780 stop_times
-- trips roughly unchanged, stops down ~43%, stop_times down ~57%, and
GET /routes?from_stop_id=600118&to_stop_id=600016 returns an empty list.

scripts/precompute_gtfs_static.py (read separately -- this script does not
import or modify it) applies NO row filter, subset, sample, or early-exit
when writing the optimized parquet: every chunk read from stop_times.txt is
kept, so it can't be the direct cause of a row-count drop by itself. The
filtering that COULD explain this symptom happens downstream, in
phase3/gtfs/loader.py:

  - GTFSData.load_optimized() (loader.py ~L330-342) re-derives
    non_ferry_trip_ids from whatever gtfs_static/<date>/ is CURRENTLY
    max() -- every call, live.
  - load_stop_times_optimized() (loader.py ~L649-650) always reads the SAME
    fixed S3 key -- phase3/gtfs_static_optimized/latest/stop_times.parquet
    -- regardless of which dated snapshot that parquet was actually built
    from, then ferry-filters it via
    `stop_times[stop_times['trip_id'].isin(non_ferry_trip_ids)]`
    (loader.py ~L669).

If scripts/precompute_gtfs_static.py hasn't been re-run since a newer
gtfs_static/<date>/ snapshot appeared, 'latest' silently holds an OLDER
snapshot's trip_id namespace. Two different dated GTFS publishes don't
generally share trip_id values, so that isin() filter would then reject
nearly every row -- not because they're ferries, but because the two
snapshots' trip_id spaces don't overlap -- which matches this symptom's
shape (trips ~stable per-snapshot, stop_times/stops collapse). The other
live candidate is simply a genuinely smaller raw feed on the date EC2
picked up, unrelated to any bug in this repo. This script gathers the
read-only evidence to tell those two apart.

Read-only: only lists/reads S3 (stop_times.txt / trips.txt / stops.txt CSVs
under gtfs_static/<date>/, and the optimized parquet's metadata + selected
columns under phase3/gtfs_static_optimized/). Never writes to S3 or to any
local file. Not invoked by anything else in this repo -- run manually.

Usage:
    python3 diagnostics/snapshot_integrity_check.py
    python3 diagnostics/snapshot_integrity_check.py --date 2026-09-23
    python3 diagnostics/snapshot_integrity_check.py --stop-ids 600118,600016
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
import psutil
import pyarrow.parquet as pq

# Reuse the existing credential-loading pattern (st.secrets -> env -> .env)
# instead of duplicating or hardcoding credentials -- see phase3/config.py.
_PHASE3_DIR = Path(__file__).resolve().parent.parent / 'phase3'
if str(_PHASE3_DIR) not in sys.path:
    sys.path.insert(0, str(_PHASE3_DIR))
import config  # noqa: E402

DATE_PATTERN = re.compile(r'^\d{4}-\d{2}-\d{2}$')
DEFAULT_STOP_IDS = ['600118', '600016']
CSV_CHUNKSIZE = 500_000


def _rss_mb() -> float:
    return psutil.Process().memory_info().rss / 1e6


def list_date_partitions(fs, prefix: str) -> list[str]:
    """Same YYYY-MM-DD enumeration logic as phase3/gtfs/loader.py's
    GTFSData.load() / scripts/precompute_gtfs_static.py use. Non-date
    entries (e.g. a 'latest/' pointer folder) are dropped by the regex.
    """
    entries = fs.ls(prefix)
    return sorted(
        e.rstrip('/').split('/')[-1] for e in entries
        if DATE_PATTERN.match(e.rstrip('/').split('/')[-1])
    )


def count_csv_rows(uri: str, usecols: list[str]) -> int:
    """Streaming row count via chunked read -- never materializes the full
    file at once, same CHUNKSIZE convention as
    scripts/precompute_gtfs_static.py's own stop_times.txt read.
    """
    total = 0
    for chunk in pd.read_csv(uri, usecols=usecols, dtype=str, chunksize=CSV_CHUNKSIZE):
        total += len(chunk)
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '--date', default=None,
        help="gtfs_static/ snapshot date to compare the optimized parquet against "
             "(YYYY-MM-DD). Defaults to max() of available dates -- the same "
             "selection logic loader.py and precompute_gtfs_static.py both use, "
             "and what EC2's '2026-09-23' startup log line reflects.",
    )
    parser.add_argument(
        '--stop-ids', default=','.join(DEFAULT_STOP_IDS),
        help=f'Comma-separated stop_ids to check (default: {",".join(DEFAULT_STOP_IDS)})',
    )
    args = parser.parse_args()
    check_stop_ids = [s.strip() for s in args.stop_ids.split(',') if s.strip()]

    bucket = config.get_s3_bucket()
    fs = config.get_s3_filesystem()
    print(f'Bucket: s3://{bucket}')
    print(f'[MEM] start: {_rss_mb():,.1f} MB RSS\n')

    # --- Snapshot date selection ------------------------------------------
    static_prefix = f'{bucket}/gtfs_static'
    static_dates = list_date_partitions(fs, static_prefix)
    if not static_dates:
        raise FileNotFoundError(f'No YYYY-MM-DD subfolders found under s3://{static_prefix}')
    snapshot_date = args.date or max(static_dates)
    if snapshot_date not in static_dates:
        raise FileNotFoundError(
            f'--date {snapshot_date!r} not found under s3://{static_prefix} '
            f'(available: {static_dates})'
        )
    print(f'Comparing against gtfs_static snapshot: {snapshot_date} '
          f'({"max() of " + str(len(static_dates)) + " available" if args.date is None else "explicit --date"})\n')
    snapshot_root = f's3://{static_prefix}/{snapshot_date}'

    output_prefix = f'{bucket}/phase3/gtfs_static_optimized'
    latest_uri = f's3://{output_prefix}/latest/stop_times.parquet'

    # --- (a) Optimized parquet: row count, size, LastModified -------------
    print('-' * 100)
    print('(a) Optimized stop_times.parquet (latest/ pointer)')
    print('-' * 100)
    latest_info = fs.info(latest_uri)
    latest_size_bytes = latest_info.get('size', 0)
    latest_modified = latest_info.get('LastModified') or latest_info.get('last_modified')
    latest_etag = latest_info.get('ETag') or latest_info.get('etag')

    with fs.open(latest_uri, 'rb') as f:
        optimized_row_count = pq.ParquetFile(f).metadata.num_rows

    print(f'URI          : {latest_uri}')
    print(f'Row count    : {optimized_row_count:,}')
    print(f'File size    : {latest_size_bytes / 1e6:,.1f} MB ({latest_size_bytes:,} bytes)')
    print(f'LastModified : {latest_modified}')
    print(f'ETag         : {latest_etag}')
    print(f'(raw fs.info(): {latest_info})')

    # --- (b) Raw stop_times.txt row count, same-dated prefix --------------
    print('\n' + '-' * 100)
    print('(b) Raw stop_times.txt (same-dated gtfs_static/<date>/ prefix)')
    print('-' * 100)
    raw_stop_times_uri = f'{snapshot_root}/stop_times.txt'
    raw_row_count = count_csv_rows(raw_stop_times_uri, usecols=['trip_id'])
    print(f'URI       : {raw_stop_times_uri}')
    print(f'Row count : {raw_row_count:,}')
    if raw_row_count:
        drop_pct = 100 * (1 - optimized_row_count / raw_row_count)
        print(f'Optimized vs raw: {optimized_row_count:,} / {raw_row_count:,} rows '
              f'({drop_pct:.1f}% fewer in the optimized parquet). The optimized parquet '
              f'itself is unfiltered (precompute_gtfs_static.py writes 1:1 from this same '
              f'raw file) -- any drop here reflects loader.py\'s downstream ferry filter '
              f'(normally small) plus whatever gap section (bonus) below finds between '
              f"'latest' and this date's own dated parquet.")

    # --- (c) stop_id presence + stop_times reference counts ---------------
    print('\n' + '-' * 100)
    print('(c) Target stop_ids: presence in stops.txt + stop_times reference counts')
    print('-' * 100)
    stops_uri = f'{snapshot_root}/stops.txt'
    stops_df = pd.read_csv(stops_uri, usecols=['stop_id', 'stop_name', 'parent_station'], dtype=str)
    stop_ids_present = set(stops_df['stop_id'])

    optimized_stop_id_col = pd.read_parquet(latest_uri, columns=['stop_id'])['stop_id'].astype(str)

    for sid in check_stop_ids:
        in_stops = sid in stop_ids_present
        stop_name = None
        if in_stops:
            match = stops_df.loc[stops_df['stop_id'] == sid, 'stop_name']
            stop_name = match.iloc[0] if not match.empty else None
        ref_count = int((optimized_stop_id_col == sid).sum())
        print(f'stop_id={sid!r}: in {snapshot_date}/stops.txt={in_stops} (stop_name={stop_name!r}), '
              f"stop_times rows referencing it in 'latest' parquet={ref_count:,}")

    # --- (d) Distinct trip_id count: optimized parquet vs raw trips.txt ---
    print('\n' + '-' * 100)
    print('(d) Distinct trip_id count: optimized parquet vs raw trips.txt')
    print('-' * 100)
    optimized_trip_id_col = pd.read_parquet(latest_uri, columns=['trip_id'])['trip_id']
    optimized_distinct_trips = set(optimized_trip_id_col.astype(str).unique())

    trips_uri = f'{snapshot_root}/trips.txt'
    trips_df = pd.read_csv(trips_uri, usecols=['trip_id'], dtype=str)
    raw_trip_row_count = len(trips_df)
    raw_distinct_trip_ids = set(trips_df['trip_id'].unique())

    overlap = optimized_distinct_trips & raw_distinct_trip_ids
    print(f'Optimized parquet distinct trip_id : {len(optimized_distinct_trips):,}')
    print(f'Raw trips.txt row count             : {raw_trip_row_count:,}')
    print(f'Raw trips.txt distinct trip_id      : {len(raw_distinct_trip_ids):,}')
    if raw_distinct_trip_ids:
        print(f'Overlap (optimized parquet trip_id ∩ raw trips.txt trip_id) : {len(overlap):,} '
              f'({100 * len(overlap) / len(raw_distinct_trip_ids):.1f}% of {snapshot_date}\'s '
              f"trips.txt trip_ids also appear in the 'latest' parquet -- a LOW percentage here "
              f"is the direct signature of the stale-pointer hypothesis in this script's module "
              f'docstring: loader.py\'s isin() ferry filter would reject everything outside '
              f'this overlap as a side effect, not because it\'s a ferry)')

    # --- (e) stop_sequence range + single-row-trip truncation signature ---
    print('\n' + '-' * 100)
    print('(e) stop_sequence range + trips with exactly one stop_time row')
    print('-' * 100)
    seq_df = pd.read_parquet(latest_uri, columns=['trip_id', 'stop_sequence'])
    seq_min = seq_df['stop_sequence'].min()
    seq_max = seq_df['stop_sequence'].max()
    rows_per_trip = seq_df.groupby('trip_id', observed=True).size()
    single_row_trips = rows_per_trip[rows_per_trip == 1]

    print(f'stop_sequence min                  : {seq_min}')
    print(f'stop_sequence max                  : {seq_max}')
    print(f'Trips with exactly 1 stop_time row : {len(single_row_trips):,} '
          f'(of {len(rows_per_trip):,} distinct trip_id in the optimized parquet)')
    if len(single_row_trips):
        sample = [str(x) for x in single_row_trips.index[:10]]
        print(f'Sample single-row trip_id(s) (up to 10): {sample}')

    # --- (bonus, not explicitly requested) does 'latest' match the dated
    # sibling for the snapshot_date being compared against? This is the
    # single check that most directly tests candidate #1 from this script's
    # module docstring: a stale 'latest' pointer left over from an older
    # precompute_gtfs_static.py run, predating a newer gtfs_static/<date>/
    # snapshot. ---
    print('\n' + '-' * 100)
    print(f"(bonus) Does 'latest' match the {snapshot_date} dated sibling parquet?")
    print('-' * 100)
    dated_uri = f's3://{output_prefix}/{snapshot_date}/stop_times.parquet'
    if fs.exists(dated_uri):
        dated_info = fs.info(dated_uri)
        dated_size = dated_info.get('size', 0)
        dated_etag = dated_info.get('ETag') or dated_info.get('etag')
        size_match = dated_size == latest_size_bytes
        etag_match = bool(dated_etag) and dated_etag == latest_etag
        print(f'Dated sibling : {dated_uri}')
        print(f'Dated size    : {dated_size / 1e6:,.1f} MB ({dated_size:,} bytes)')
        print(f'Dated ETag    : {dated_etag}')
        print(f"latest/ matches {snapshot_date}'s dated parquet -- by size: {size_match}"
              + (f', by ETag: {etag_match}' if dated_etag else ' (no ETag to compare)'))
        if not (size_match or etag_match):
            print(f"WARNING: 'latest' does NOT match the {snapshot_date} dated parquet -- "
                  f"'latest' most likely still points at an OLDER precompute run. "
                  f'scripts/precompute_gtfs_static.py probably needs to be re-run for '
                  f'{snapshot_date}.')
    else:
        print(f'No dated sibling found at {dated_uri} -- scripts/precompute_gtfs_static.py '
              f"has apparently never been run for {snapshot_date}. 'latest' (if it exists) "
              f'must be pointing at an older snapshot date.')
        optimized_dates = list_date_partitions(fs, output_prefix)
        print(f'Dated snapshots that DO exist under s3://{output_prefix}/: {optimized_dates}')

    print(f'\n[MEM] end: {_rss_mb():,.1f} MB RSS')
    print('\n' + '=' * 100)
    print('Done -- read-only, nothing was written.')
    print('=' * 100)


if __name__ == '__main__':
    main()
