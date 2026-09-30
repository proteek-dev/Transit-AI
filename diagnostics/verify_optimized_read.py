#!/usr/bin/env python3
"""Diagnostic-only round-trip verification for the optimized stop_times
parquet produced by scripts/precompute_gtfs_static.py.

Confirms:
  1. Reading the parquet back gives trip_id/stop_id/arrival_time/
     departure_time as categorical dtype -- not silently widened back to
     object on read (parquet round-trips pandas' category dtype natively
     via its dictionary encoding, but this verifies that actually holds
     rather than assuming it). arrival_time/departure_time were added here
     after precompute_gtfs_static.py was extended to convert them too --
     profile_stop_times_columns.py found they were 89.8% of the frame's
     memory, so this check would otherwise silently miss the two columns
     that matter most.
  2. Row count matches the known total (3,153,579).
  3. Peak RSS for this read-only operation, via the same psutil RSS pattern
     scripts/precompute_gtfs_static.py / phase3/ml/training.py use.

Read-only: only reads the already-optimized parquet at
s3://seq-transit-ai-data-ps/phase3/gtfs_static_optimized/latest/stop_times.parquet.
Never touches raw stop_times.txt, phase3/, or any other existing file.

Usage:
    python3 diagnostics/verify_optimized_read.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import psutil

# Reuse the existing credential-loading pattern (st.secrets -> env -> .env)
# instead of duplicating or hardcoding credentials -- see phase3/config.py.
_PHASE3_DIR = Path(__file__).resolve().parent.parent / 'phase3'
if str(_PHASE3_DIR) not in sys.path:
    sys.path.insert(0, str(_PHASE3_DIR))
import config  # noqa: E402

EXPECTED_ROW_COUNT = 3_153_579
EXPECTED_CATEGORY_COLS = ['trip_id', 'stop_id', 'arrival_time', 'departure_time']


def _rss_mb() -> float:
    return psutil.Process().memory_info().rss / 1e6


def main() -> None:
    # Same calling shape as scripts/precompute_gtfs_static.py's main():
    # config.get_s3_bucket() + config.get_s3_filesystem() (the latter's
    # .env-loading side effect is what lets the plain pd.read_*('s3://...')
    # URI below authenticate, exactly as precompute_gtfs_static.py's own
    # pd.read_csv(f'{snapshot_root}/stop_times.txt', ...) call relies on --
    # no explicit storage_options/fs.open(), same bare-URI read pattern.
    bucket = config.get_s3_bucket()
    config.get_s3_filesystem()
    parquet_uri = f's3://{bucket}/phase3/gtfs_static_optimized/latest/stop_times.parquet'

    rss_before = _rss_mb()
    print(f'[MEM] before read: {rss_before:,.1f} MB RSS')
    print(f'Reading {parquet_uri} ...')

    df = pd.read_parquet(parquet_uri)

    rss_after = _rss_mb()
    print(f'[MEM] after read : {rss_after:,.1f} MB RSS')
    peak_rss_mb = max(rss_before, rss_after)

    row_count = len(df)
    row_count_ok = row_count == EXPECTED_ROW_COUNT

    dtypes = {col: str(df[col].dtype) for col in EXPECTED_CATEGORY_COLS}
    all_categorical = all(dtypes[col] == 'category' for col in EXPECTED_CATEGORY_COLS)

    print('\n' + '=' * 100)
    print('SUMMARY -- optimized parquet round-trip verification')
    print('=' * 100)
    print(f'Peak RSS (this read-only operation) : {peak_rss_mb:,.1f} MB')
    print(f'Row count                           : {row_count:,} (expected {EXPECTED_ROW_COUNT:,}) '
          f'-- {"MATCH" if row_count_ok else "MISMATCH"}')
    print('Dtypes confirmed:')
    for col in EXPECTED_CATEGORY_COLS:
        status = 'OK (categorical)' if dtypes[col] == 'category' else f'FAILED (widened to {dtypes[col]!r})'
        print(f'  {col:<10}: {dtypes[col]:<10} -- {status}')

    if row_count_ok and all_categorical:
        print('\nRound-trip verification PASSED: categorical dtype survived parquet, row count matches.')
    else:
        raise RuntimeError('Round-trip verification FAILED -- see summary above.')


if __name__ == '__main__':
    main()
