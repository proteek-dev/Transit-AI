#!/usr/bin/env python3
"""Diagnostic-only per-column memory profile of the optimized stop_times
parquet.

Loads s3://seq-transit-ai-data-ps/phase3/gtfs_static_optimized/latest/stop_times.parquet
and reports, for EVERY column (not just trip_id/stop_id): dtype, distinct
value count, and memory_usage(deep=True) -- so which column is actually
driving the read-back RSS is visible directly, rather than assumed from the
two columns already suspected. deep=True is required here, not the shallow
default: for object/category columns, the shallow default only counts the
pointer array, not the actual string/category payload, and would
under-report exactly the columns most likely to be expensive.

Read-only: only reads the already-optimized parquet. Does not touch
scripts/precompute_gtfs_static.py, raw stop_times.txt, phase3/, or any
other existing file.

Usage:
    python3 diagnostics/profile_stop_times_columns.py
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


def _rss_mb() -> float:
    return psutil.Process().memory_info().rss / 1e6


def main() -> None:
    # Same calling shape as scripts/precompute_gtfs_static.py's main() /
    # diagnostics/verify_optimized_read.py: config.get_s3_bucket() +
    # config.get_s3_filesystem() (the latter's .env-loading side effect is
    # what lets the plain pd.read_parquet('s3://...') URI below
    # authenticate) -- no explicit storage_options/fs.open().
    bucket = config.get_s3_bucket()
    config.get_s3_filesystem()
    parquet_uri = f's3://{bucket}/phase3/gtfs_static_optimized/latest/stop_times.parquet'

    rss_before = _rss_mb()
    print(f'[MEM] before read: {rss_before:,.1f} MB RSS')
    print(f'Reading {parquet_uri} ...')

    df = pd.read_parquet(parquet_uri)

    rss_after = _rss_mb()
    print(f'[MEM] after read : {rss_after:,.1f} MB RSS')

    row_count = len(df)
    mem_by_col = df.memory_usage(deep=True, index=False)
    total_mem_bytes = int(mem_by_col.sum())

    rows = []
    for col in df.columns:
        rows.append({
            'column': col,
            'dtype': str(df[col].dtype),
            'distinct_values': df[col].nunique(dropna=False),
            'memory_bytes': int(mem_by_col[col]),
        })
    profile = pd.DataFrame(rows).sort_values('memory_bytes', ascending=False).reset_index(drop=True)

    print('\n' + '=' * 100)
    print(f'SUMMARY -- per-column memory profile (deep=True, {row_count:,} rows)')
    print('=' * 100)
    header = f'{"column":<20} {"dtype":<12} {"distinct":>10} {"memory (MB)":>14} {"% of total":>11}'
    print(header)
    print('-' * len(header))
    for r in profile.itertuples(index=False):
        mem_mb = r.memory_bytes / 1e6
        pct = 100 * r.memory_bytes / total_mem_bytes if total_mem_bytes else 0.0
        print(f'{r.column:<20} {r.dtype:<12} {r.distinct_values:>10,} {mem_mb:>14,.1f} {pct:>10.1f}%')
    print('-' * len(header))
    print(f'{"TOTAL":<20} {"":<12} {"":>10} {total_mem_bytes / 1e6:>14,.1f} {100.0:>10.1f}%')

    print(f'\n(For reference, df.memory_usage(deep=False, index=False).sum() = '
          f'{df.memory_usage(deep=False, index=False).sum() / 1e6:,.1f} MB -- the shallow default, '
          f'shown only to make the deep-vs-shallow under-reporting gap visible, not used above.)')


if __name__ == '__main__':
    main()
