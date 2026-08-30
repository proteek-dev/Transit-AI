#!/usr/bin/env python3
"""Precompute an already-optimized stop_times.parquet for the GTFS static
snapshot, upstream of Render's memory-constrained /routes backend.

WHY: webapp/backend's /routes endpoint OOMs (peaks ~851MB RSS) parsing
stop_times.txt (~3M rows) inside a 512MB-constrained Render process --
phase3/gtfs/loader.py's GTFSData.load() already reads it in per-chunk
category-dtype-converted batches (the `use_categorical_dtypes=True` path,
see loader.py's stop_time_cols chunk loop), but even that optimized parse
still costs real transient memory to run in the first place. This script
does that same parse ONCE, upstream, on a machine with real RAM headroom
(this Mac, or the archiver EC2 -- either has plenty for a 3M-row file), and
writes the finished artifact to S3 so Render can eventually just read a
lean parquet file directly instead of re-parsing the raw CSV every warm-up.

Wiring loader.py to actually READ this artifact is a separate follow-up
prompt -- this script only produces it. Standalone; does not import from or
modify phase3/gtfs/loader.py.

SOURCE (confirmed from phase3/gtfs/loader.py's GTFSData.load(), not
guessed): the latest s3://{bucket}/gtfs_static/{YYYY-MM-DD}/ snapshot
(max() of the YYYY-MM-DD subfolders) -- same snapshot-selection logic and
same usecols/dtype=str/chunksize=500_000 chunked read of stop_times.txt.

arrival_time/departure_time were originally left as object dtype here (and
still are in loader.py), on the assumption that they have "lower
repetition" than trip_id/stop_id. diagnostics/profile_stop_times_columns.py
measured this directly against the actual file and found that assumption
wrong: arrival_time and departure_time each have only ~14-15K distinct
HH:MM:SS values across all 3,153,579 rows (~99.5% repetition) -- the same
profile that made trip_id/stop_id worth converting -- and were in fact the
two largest columns in the frame, 205.0MB each, 89.8% of its 456.8MB total
memory_usage(deep=True). Both are now included in the same category-dtype
treatment below (this script only -- loader.py's own object-dtype choice
for these columns is untouched, per this prompt's scope).

CATEGORY DTYPE FIX (2026-08-30): the first version of this script applied
.astype('category') to trip_id/stop_id independently PER CHUNK -- mirroring
loader.py's own per-chunk conversion, whose comment documents that converting
the whole ~3M-row frame in one .astype('category') call measured 901.9MB RSS
vs. 655.7MB baseline (glibc's allocator doesn't return that transient peak to
the OS). That part was correct, but it left an undiscovered gap: each chunk's
independent .astype('category') builds its OWN local category set, and
pd.concat() silently falls back to object dtype for a column whenever the
chunks being concatenated don't all share identical categories -- which is
near-guaranteed across ~500k-row slices of a 3M-row file. Confirmed via
diagnostics/verify_optimized_read.py: the saved parquet's trip_id/stop_id
came back as plain object (pandas_type 'unicode' in the file's own embedded
schema metadata), and read-back RSS was 651.4MB -- still over Render's
512MB cap, because the persisted artifact was never actually categorical.

Fixed by building ONE shared pandas.CategoricalDtype per column (from a
lightweight first pass reading ONLY the CATEGORY_COLS columns across the
full file -- see build_shared_category_dtypes()) BEFORE the main chunk loop, then
.astype()-ing every chunk to that same shared dtype instead of a fresh
'category' inference each time. All chunks now share identical categories,
so concat preserves categorical dtype instead of falling back. A post-concat
assertion (see load_and_optimize_stop_times()) fails the script loudly if
this class of regression ever reappears, rather than shipping silently.

OUTPUT PATH (confirmed from phase3/ml/model_io.py / pipeline/02_train_model.sh's
model-artifact convention, not invented): that convention is
phase3/model/latest/ (live pointer, MODEL_SUBDIR='latest') plus a permanent
dated/timed sibling under phase3/model/runs/{date}/{time}/, promoted
S3-to-S3 via fs.copy() from one staged upload rather than re-uploaded twice.
This script follows the same family under phase3/, combined with
gtfs_static/{date}/'s own existing date-partition convention (there's
exactly one static snapshot per calendar date, unlike training which can
run multiple times a day, so no {time} subfolder is needed here):
  s3://{bucket}/phase3/gtfs_static_optimized/{snapshot_date}/stop_times.parquet   (dated, canonical upload)
  s3://{bucket}/phase3/gtfs_static_optimized/latest/stop_times.parquet           (S3-side copy of the above)

Usage:
    python3 scripts/precompute_gtfs_static.py
"""
from __future__ import annotations

import re
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd
import psutil

# Reuse the existing credential-loading pattern (st.secrets -> env -> .env)
# instead of duplicating or hardcoding credentials -- see phase3/config.py.
_PHASE3_DIR = Path(__file__).resolve().parent.parent / 'phase3'
if str(_PHASE3_DIR) not in sys.path:
    sys.path.insert(0, str(_PHASE3_DIR))
import config  # noqa: E402

DATE_PATTERN = re.compile(r'^\d{4}-\d{2}-\d{2}$')

# Exact usecols/dtype loader.py's GTFSData.load() reads for stop_times.txt --
# kept identical so this artifact is schema-compatible with what loader.py
# will eventually read instead of the raw CSV.
STOP_TIME_COLS = ['trip_id', 'stop_id', 'stop_sequence', 'arrival_time', 'departure_time']
# arrival_time/departure_time added here per diagnostics/profile_stop_times_columns.py's
# measurement -- see module docstring's SOURCE section for the ~99.5%-repetition finding.
CATEGORY_COLS = ['trip_id', 'stop_id', 'arrival_time', 'departure_time']
CHUNKSIZE = 500_000

# One (label, rss_mb) tuple per checkpoint -- same pattern as
# phase3/ml/training.py's _log_mem()/_mem_checkpoints (psutil RSS, printed
# and retained so a final peak can be reported).
_mem_checkpoints: list[tuple[str, float]] = []


def _log_mem(label: str) -> float:
    rss_mb = psutil.Process().memory_info().rss / 1e6
    print(f'[MEM] {label}: {rss_mb:,.1f} MB RSS')
    _mem_checkpoints.append((label, rss_mb))
    return rss_mb


def list_date_partitions(fs, prefix: str) -> list[str]:
    """Same YYYY-MM-DD enumeration logic as phase3/gtfs/loader.py's
    GTFSData.load() uses for gtfs_static/.
    """
    entries = fs.ls(prefix)
    return sorted(
        e.rstrip('/').split('/')[-1] for e in entries
        if DATE_PATTERN.match(e.rstrip('/').split('/')[-1])
    )


def build_shared_category_dtypes(snapshot_root: str) -> dict[str, pd.CategoricalDtype]:
    """First pass: read ONLY CATEGORY_COLS (usecols -- not the full
    5-column row width) across the WHOLE file, in ONE chunked pass, to
    collect the complete distinct-value set for each of the 4 columns
    (trip_id, stop_id, arrival_time, departure_time). Still chunked (not
    one bulk read) so this pass's own peak stays bounded to one chunk's
    four narrow columns at a time, same rationale as the main loop.

    The resulting CategoricalDtype per column is then shared by every chunk
    in the main loop below -- see module docstring's "CATEGORY DTYPE FIX"
    section for why a per-chunk-independent category set (the original
    approach) silently loses categorical dtype on concat.
    """
    _log_mem('before category first pass')
    distinct_values: dict[str, set[str]] = {col: set() for col in CATEGORY_COLS}

    for chunk in pd.read_csv(
        f'{snapshot_root}/stop_times.txt',
        usecols=CATEGORY_COLS,
        dtype=str,
        chunksize=CHUNKSIZE,
    ):
        for col in CATEGORY_COLS:
            distinct_values[col].update(chunk[col].unique())

    counts_str = ', '.join(f'{len(distinct_values[col]):,} distinct {col}' for col in CATEGORY_COLS)
    _log_mem(f'after category first pass ({counts_str})')

    return {
        col: pd.CategoricalDtype(categories=sorted(values))
        for col, values in distinct_values.items()
    }


def load_and_optimize_stop_times(
    snapshot_root: str, category_dtypes: dict[str, pd.CategoricalDtype]
) -> tuple[pd.DataFrame, int]:
    """Chunked read of stop_times.txt with per-chunk (pre-concat) category
    conversion -- same design as loader.py's use_categorical_dtypes=True
    path, just without the ferry-exclusion cross-reference against
    trips.txt/routes.txt (out of scope here: this script's job is the raw
    file's memory-layout transform, not loader.py's full filtering
    pipeline -- that stays loader.py's job in the follow-up wiring prompt).

    Every chunk is cast to the SAME `category_dtypes` (built once up front
    by build_shared_category_dtypes(), not re-inferred per chunk) so
    pd.concat() below sees identical categories across all chunks and
    preserves categorical dtype instead of silently falling back to object.
    """
    _log_mem('before read')
    chunks: list[pd.DataFrame] = []
    rows_processed = 0

    for i, chunk in enumerate(pd.read_csv(
        f'{snapshot_root}/stop_times.txt',
        usecols=STOP_TIME_COLS,
        dtype=str,
        chunksize=CHUNKSIZE,
    )):
        rows_processed += len(chunk)
        chunk = chunk.assign(stop_sequence=chunk['stop_sequence'].astype('int32'))
        # Per-chunk, before concat, using the shared dtype -- see module
        # docstring's "CATEGORY DTYPE FIX" section for why this has to be
        # the SAME CategoricalDtype object across every chunk, not a fresh
        # .astype('category') inference each time.
        for col in CATEGORY_COLS:
            chunk[col] = chunk[col].astype(category_dtypes[col])
        chunks.append(chunk)
        if (i + 1) % 3 == 0:
            _log_mem(f'after chunk {i + 1} ({rows_processed:,} rows read)')

    _log_mem(f'after all chunks read ({rows_processed:,} rows total)')

    stop_times = (
        pd.concat(chunks, ignore_index=True) if chunks
        else pd.DataFrame(columns=STOP_TIME_COLS)
    )
    del chunks
    _log_mem('after concat')

    # Fail loudly, not a printed warning -- this is exactly the class of
    # regression (category dtype silently lost on concat) that shipped
    # undetected before diagnostics/verify_optimized_read.py caught it.
    for col in CATEGORY_COLS:
        actual_dtype = stop_times[col].dtype
        assert isinstance(actual_dtype, pd.CategoricalDtype), (
            f'{col} lost its categorical dtype after concat (got {actual_dtype!r}, '
            f'expected CategoricalDtype) -- the shared-dtype fix has regressed. '
            f'Refusing to write a non-categorical parquet artifact.'
        )

    return stop_times, rows_processed


def main() -> None:
    _log_mem('start')
    bucket = config.get_s3_bucket()
    fs = config.get_s3_filesystem()
    print(f'Bucket: s3://{bucket}\n')

    static_prefix = f'{bucket}/gtfs_static'
    static_dates = list_date_partitions(fs, static_prefix)
    if not static_dates:
        raise FileNotFoundError(f'No YYYY-MM-DD subfolders found under s3://{static_prefix}')
    snapshot_date = max(static_dates)
    snapshot_root = f's3://{static_prefix}/{snapshot_date}'
    print(f'Static snapshot: {snapshot_date} (latest of {len(static_dates)} available -- same '
          f'selection phase3/gtfs/loader.py uses)\n')

    raw_csv_key = f'{static_prefix}/{snapshot_date}/stop_times.txt'
    raw_csv_size_bytes = fs.info(raw_csv_key)['size']

    category_dtypes = build_shared_category_dtypes(snapshot_root)
    stop_times, rows_processed = load_and_optimize_stop_times(snapshot_root, category_dtypes)

    tmp_dir = Path(tempfile.mkdtemp(prefix='gtfs_static_precompute_'))
    try:
        local_parquet_path = tmp_dir / 'stop_times.parquet'
        stop_times.to_parquet(local_parquet_path, index=False)
        _log_mem('after local parquet write')
        parquet_size_bytes = local_parquet_path.stat().st_size

        output_prefix = f'{bucket}/phase3/gtfs_static_optimized'
        dated_uri = f's3://{output_prefix}/{snapshot_date}/stop_times.parquet'
        latest_uri = f's3://{output_prefix}/latest/stop_times.parquet'

        print(f'\nUploading to S3 (dated, canonical) -> {dated_uri}')
        fs.put(str(local_parquet_path), dated_uri)
        # S3-side copy for the latest/ pointer -- same fs.copy() promotion
        # pattern pipeline/02_train_model.sh uses (one upload, then
        # server-side copy_object to each destination) rather than a second
        # local->S3 upload of the same bytes.
        print(f'Copying (S3-side) to latest pointer -> {latest_uri}')
        fs.copy(dated_uri, latest_uri)

        dated_ok = fs.exists(dated_uri)
        latest_ok = fs.exists(latest_uri)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    peak_rss_mb = max(rss for _, rss in _mem_checkpoints)

    print('\n' + '=' * 100)
    print('SUMMARY -- GTFS static stop_times precompute')
    print('=' * 100)
    print(f'Snapshot date              : {snapshot_date}')
    print(f'Rows processed             : {rows_processed:,}')
    print(f'Peak RSS during run        : {peak_rss_mb:,.1f} MB')
    print(f'Raw stop_times.txt size    : {raw_csv_size_bytes / 1e6:,.1f} MB ({raw_csv_size_bytes:,} bytes)')
    print(f'Optimized parquet size     : {parquet_size_bytes / 1e6:,.1f} MB ({parquet_size_bytes:,} bytes)')
    reduction_pct = 100 * (1 - parquet_size_bytes / raw_csv_size_bytes) if raw_csv_size_bytes else 0.0
    print(f'Size reduction             : {reduction_pct:.1f}%')
    print(f'S3 write (dated)           : {dated_uri} -- {"OK" if dated_ok else "FAILED (fs.exists() returned False)"}')
    print(f'S3 write (latest pointer)  : {latest_uri} -- {"OK" if latest_ok else "FAILED (fs.exists() returned False)"}')

    if not (dated_ok and latest_ok):
        raise RuntimeError(
            'S3 write verification failed for one or both destinations -- see S3 write lines above. '
            'Not reporting success.'
        )


if __name__ == '__main__':
    main()
