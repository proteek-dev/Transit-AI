"""Model training for the Phase 3 prediction service -- heavy, rare path.

Reproduces notebook 07 end to end (chunked per-date read across the
migration cutoff, leakage filter, time-feature derivation, dropna, temporal
train/test split, fit, MAE baseline) and saves + uploads the resulting
artifacts. Only reached via model_io._load_model_impl()'s lazy import, when
no saved model is found anywhere (S3 or local).
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import joblib
import pandas as pd
import psutil
import xgboost as xgb

import gtfs_data
from ml.model_io import (
    CATEGORICAL_COLS,
    CATEGORIES_PATH,
    FEATURE_COLS,
    MODEL_DIR,
    MODEL_PATH,
    TRAINING_METADATA_PATH,
    _get_env,
    _upload_model_to_s3,
)

# Leakage filtering moved upstream to notebook 05 (write time) on 2026-08-08.
# Max source_date present in any snapshot as of that migration (from the
# _latest.json manifest's source_date_range at the time) -- rows at/before
# this date may be S3 copy_object carry-forwards of pre-migration,
# unfiltered snapshot data (carry-forward bypasses notebook 05's Python
# write path entirely, so its new write-time filter never touches them).
# Rows after this date can only exist if written by the already-filtered
# post-migration code path. See _load_training_frames()'s leakage-filter
# safety net below, and notebook 05's "Step 5b — Leakage filter (write-time)"
# cell. Intentionally NOT auto-derived from the manifest -- it's a snapshot
# of a point in time (when this code shipped), not a live value.
MIGRATION_CUTOFF_SOURCE_DATE = '2026-07-30'

# Populated by _log_mem() -- one (label, rss_mb) tuple per checkpoint taken
# during a training run, so _build_training_metadata() can persist the RSS
# trend into training_metadata.json instead of it being print-only and lost
# after the run. Reset by _reset_mem_checkpoints() at the start of every
# _train_and_save_model() call.
_mem_checkpoints: list[tuple[str, float]] = []


def _log_mem(label: str) -> None:
    """Debug checkpoint: current process RSS, so a slow/OOM-prone training
    run can be narrowed down to a specific stage without a profiler."""
    rss_mb = psutil.Process().memory_info().rss / 1e6
    print(f'[MEM] {label}: {rss_mb:,.1f} MB RSS')
    _mem_checkpoints.append((label, rss_mb))


def _reset_mem_checkpoints() -> None:
    """Clears _mem_checkpoints. Called at the very start of
    _train_and_save_model(), before _load_training_frames() runs, so a stale
    checkpoint list from a prior run in the same process can never leak into
    this run's training_metadata.json -- this module is normally invoked as
    a single-run script, but don't rely on that.
    """
    _mem_checkpoints.clear()


def _mae(y_true, y_pred) -> float:
    """Mean absolute error (notebook 07 Cells 8/8b). `y_pred` may be a
    per-row array (model predictions) or a single scalar broadcast across
    every row (the naive median-baseline case) -- both work via plain
    ndarray broadcasting, no numpy import needed.
    """
    return float(abs(y_true - y_pred).mean())


def _load_training_frames():
    """Reproduces notebook 07 Cells 2-6 verbatim: load via _latest.json,
    leakage filter, re-derive time features, target/feature build, temporal
    split. Returns (X_train, y_train, X_test, y_test, categories) — both
    partitions, so _train_and_save_model() can compute a real held-out MAE
    (notebook 07 Cells 8/8b) as part of every training run instead of that
    being a separate manual step.
    """
    bucket, fs = _get_env()
    ml_features_prefix = f'{bucket}/ml_features/v0_feature_snapshot'

    manifest_path = f's3://{ml_features_prefix}/_latest.json'
    with fs.open(manifest_path) as f:
        manifest = json.load(f)
    run_date = manifest['latest_run']
    print(f'Loading training snapshot: run_date={run_date}')

    # --- Debug-only: sample down each partition as it's read, so the entire
    # downstream path (dtype optimization, DMatrix construction, fit, save)
    # runs against a small dataset in seconds instead of the full ~94M-row
    # snapshot. Unset (or 1.0) means full data -- the default, normal path.
    debug_sample_frac = float(os.environ.get('DEBUG_SAMPLE_FRAC', 1.0))

    def _optimize_dtypes(chunk: pd.DataFrame) -> pd.DataFrame:
        """Same float64->float32 downcast + category casts as before, just
        applied per-chunk now instead of once on the full bulk-loaded frame
        -- see the module-level MIGRATION_CUTOFF_SOURCE_DATE comment for why
        the read itself is chunked. Kept as its own step (not folded into
        the read) so pd.concat() below combines already-compact, already-
        categorical pieces rather than raw object-dtype ones -- concatenating
        33 raw chunks into one ~88M-row frame would recreate the exact
        transient-doubling memory cost this rewrite exists to avoid.
        """
        float64_cols = chunk.select_dtypes(include='float64').columns
        for c in float64_cols:
            chunk[c] = chunk[c].astype('float32')
        for c in ('route_id', 'stop_id', 'mode', 'day_of_week', 'source_date', 'trip_id'):
            if c in chunk.columns and chunk[c].dtype == object:
                chunk[c] = chunk[c].astype('category')
        return chunk

    def _compute_scheduled_arrival_dt(chunk: pd.DataFrame, source_date_str: str) -> pd.Series:
        """Reconstructs scheduled_arrival_dt for ONE source_date's rows -- a
        single scalar Timestamp broadcast across the chunk (not the many-
        dates .cat.codes trick this replaced, which only paid off when this
        ran once on the full multi-date combined frame). Shared by both the
        pre-cutoff leak check and the post-cutoff feature derivation below,
        so this is computed exactly once per chunk, never again later on
        the full concatenated df.
        """
        raw_time = chunk['scheduled_arrival_time']
        parsed = pd.to_datetime(raw_time, format='%H:%M:%S', errors='coerce')
        offset = parsed - parsed.dt.normalize()
        past_midnight = parsed.isna()
        if past_midnight.any():
            hh = raw_time.loc[past_midnight].str.slice(0, 2).astype(int)
            rest = raw_time.loc[past_midnight].str.slice(2)
            days, hh_mod = divmod(hh, 24)
            wrapped_time = hh_mod.astype(str).str.zfill(2) + rest
            fixed_parsed = pd.to_datetime(wrapped_time, format='%H:%M:%S')
            offset.loc[past_midnight] = (
                (fixed_parsed - fixed_parsed.dt.normalize()) + pd.to_timedelta(days, unit='D')
            )
        source_date_midnight = pd.to_datetime(source_date_str, format='%Y-%m-%d').tz_localize('Australia/Brisbane')
        return source_date_midnight + offset

    def _filter_leaky_chunk(chunk: pd.DataFrame, source_date_str: str) -> tuple[pd.DataFrame, pd.Series, int]:
        """Leak-check + drop for ONE source_date's rows only. Returns the
        clean chunk alongside its already-computed scheduled_arrival_dt
        (filtered to match) so the caller doesn't need to recompute it.
        """
        scheduled_arrival_dt = _compute_scheduled_arrival_dt(chunk, source_date_str)
        leak_mask = chunk['snapshot_timestamp'] >= scheduled_arrival_dt
        n_leaky = int(leak_mask.sum())
        clean_chunk = chunk.loc[~leak_mask].copy()
        clean_scheduled_arrival_dt = scheduled_arrival_dt.loc[~leak_mask]
        return clean_chunk, clean_scheduled_arrival_dt, n_leaky

    def _add_time_features_and_dropna(
        chunk: pd.DataFrame, scheduled_arrival_dt: pd.Series
    ) -> tuple[pd.DataFrame, int, int]:
        """Cell 4 (hour_of_day/day_of_week/is_weekend/is_peak) + Cell 5's
        dropna(delay_minutes), applied to ONE already-leak-filtered chunk
        instead of once on the full concatenated frame -- keeps peak memory
        bounded to chunk size instead of the ~147M-row combined frame.
        """
        chunk['hour_of_day'] = scheduled_arrival_dt.dt.hour.astype('int32')
        chunk['day_of_week'] = scheduled_arrival_dt.dt.day_name()
        chunk['is_weekend'] = chunk['day_of_week'].isin(['Saturday', 'Sunday'])
        chunk['is_peak'] = (~chunk['is_weekend']) & chunk['hour_of_day'].isin([7, 8, 16, 17])
        n_before_dropna = len(chunk)
        chunk = chunk.dropna(subset=['delay_minutes'])
        n_after_dropna = len(chunk)
        return chunk, n_before_dropna, n_after_dropna

    # Never load the full ~94M-row archive into one dataframe and then try
    # to remove leaky rows from it in-memory -- every variant of that
    # (.loc[mask].copy(), chunked df.drop(inplace=True)) OOM-killed on this
    # machine, because pandas has no way to remove rows from a DataFrame
    # without reallocating a full new set of column arrays for what remains,
    # so the old (pre-filter) and new (post-filter) versions of the ~94M-row
    # frame both exist in memory at once, transiently. Instead: read + filter
    # one source_date partition at a time (each ~1-7M rows depending on the
    # date, never the whole archive), and only concatenate the small,
    # already-clean, already-optimized results once at the end.
    partition_prefix = f'{ml_features_prefix}/run_date={run_date}'
    all_source_dates = sorted(
        e.rstrip('/').rsplit('source_date=', 1)[-1]
        for e in fs.ls(partition_prefix) if 'source_date=' in e
    )
    pre_cutoff_dates = [d for d in all_source_dates if d <= MIGRATION_CUTOFF_SOURCE_DATE]
    post_cutoff_dates = [d for d in all_source_dates if d > MIGRATION_CUTOFF_SOURCE_DATE]
    print(f'[CHUNKED READ] {len(all_source_dates)} source_date partition(s) total: '
          f'{len(pre_cutoff_dates)} pre-cutoff (chunked read + safety-net filter, one date at a '
          f'time), {len(post_cutoff_dates)} post-cutoff (chunked read, no leak filter -- already '
          f'filtered at write time by notebook 05, one date at a time).')

    # Debug-only: cap the number of dates read on each side of the cutoff,
    # so the whole pipeline (load, filter, dropna, split, fit, MAE) can be
    # validated end-to-end on a small slice without waiting on the full
    # archive -- proves logic correctness independent of memory/scale.
    # Unset (the default) leaves pre_cutoff_dates/post_cutoff_dates untouched.
    debug_max_dates = os.environ.get('DEBUG_MAX_DATES')
    if debug_max_dates is not None:
        debug_max_dates = int(debug_max_dates)
        pre_cutoff_dates = pre_cutoff_dates[:debug_max_dates]
        post_cutoff_dates = post_cutoff_dates[:debug_max_dates]
        print(f'=== DEBUG MODE: DEBUG_MAX_DATES={debug_max_dates} -- processing '
              f'{len(pre_cutoff_dates)} pre-cutoff + {len(post_cutoff_dates)} post-cutoff dates only. '
              f'MODEL WILL NOT BE SAVED OR PROMOTED. ===')

    _log_mem('before data load')

    all_chunks = []
    rows_loaded = 0
    rows_after_sample_total = 0
    n_dropped = 0
    rows_before_dropna_total = 0
    rows_after_dropna_total = 0
    for i, d in enumerate(pre_cutoff_dates, 1):
        chunk = pd.read_parquet(f's3://{partition_prefix}/source_date={d}/')
        chunk['source_date'] = d
        n_chunk_loaded = len(chunk)
        if debug_sample_frac < 1.0:
            chunk = chunk.sample(frac=debug_sample_frac, random_state=42)
        rows_loaded += n_chunk_loaded
        rows_after_sample_total += len(chunk)

        chunk = _optimize_dtypes(chunk)
        clean_chunk, chunk_scheduled_arrival_dt, n_chunk_dropped = _filter_leaky_chunk(chunk, d)
        n_dropped += n_chunk_dropped

        clean_chunk, n_chunk_before_dropna, n_chunk_after_dropna = _add_time_features_and_dropna(
            clean_chunk, chunk_scheduled_arrival_dt
        )
        rows_before_dropna_total += n_chunk_before_dropna
        rows_after_dropna_total += n_chunk_after_dropna
        all_chunks.append(clean_chunk)

        print(f'[LEAKAGE-BY-DATE] source_date={d}  before={n_chunk_loaded:,}  '
              f'after={n_chunk_loaded - n_chunk_dropped:,}  dropped={n_chunk_dropped:,}  '
              f'({(n_chunk_dropped / n_chunk_loaded * 100) if n_chunk_loaded else 0:.2f}%)')
        print(f'[DROPNA-BY-DATE] source_date={d}  before={n_chunk_before_dropna:,}  '
              f'after={n_chunk_after_dropna:,}  dropped={n_chunk_before_dropna - n_chunk_after_dropna:,}')
        _log_mem(f'chunked read: after source_date={d} ({i}/{len(pre_cutoff_dates)})')

    # Already filtered at write time (notebook 05 Step 5b) -- read one
    # source_date partition at a time just like the pre-cutoff loop above
    # (same memory-avoidance rationale), just without the leak check/drop,
    # since these partitions are already trusted clean. Appended into the
    # SAME all_chunks list as the pre-cutoff chunks above -- one shared list
    # feeding a single final concat, instead of three separate concats
    # (pre-cutoff-only, post-cutoff-only, then combining those two), which
    # is what actually OOM-killed a 46-date run: each intermediate concat
    # transiently held multiple large frames in memory at once.
    for i, d in enumerate(post_cutoff_dates, 1):
        chunk = pd.read_parquet(f's3://{partition_prefix}/source_date={d}/')
        chunk['source_date'] = d
        n_chunk_loaded = len(chunk)
        if debug_sample_frac < 1.0:
            chunk = chunk.sample(frac=debug_sample_frac, random_state=42)
        rows_loaded += n_chunk_loaded
        rows_after_sample_total += len(chunk)

        chunk = _optimize_dtypes(chunk)
        chunk_scheduled_arrival_dt = _compute_scheduled_arrival_dt(chunk, d)
        chunk, n_chunk_before_dropna, n_chunk_after_dropna = _add_time_features_and_dropna(
            chunk, chunk_scheduled_arrival_dt
        )
        rows_before_dropna_total += n_chunk_before_dropna
        rows_after_dropna_total += n_chunk_after_dropna
        all_chunks.append(chunk)

        print(f'[POST-CUTOFF-BY-DATE] source_date={d}  rows={len(chunk):,} (already leak-filtered at write time)')
        print(f'[DROPNA-BY-DATE] source_date={d}  before={n_chunk_before_dropna:,}  '
              f'after={n_chunk_after_dropna:,}  dropped={n_chunk_before_dropna - n_chunk_after_dropna:,}')
        _log_mem(f'chunked read: after source_date={d} ({i}/{len(post_cutoff_dates)})')

    print(f'Loaded {rows_loaded:,} rows total ({len(pre_cutoff_dates):,} pre-cutoff date(s) + '
          f'{len(post_cutoff_dates):,} post-cutoff date(s))')
    print(f'[ROW ACCOUNTING] 1. Loaded from ferry-filtered S3 snapshot: {rows_loaded:,} rows')
    if debug_sample_frac < 1.0:
        print(f'[ROW ACCOUNTING] 1b. Debug sample active (frac={debug_sample_frac}) -- applied per partition above')
    rows_before_leakage_filter = rows_loaded

    # Single concat of every per-date chunk (pre-cutoff + post-cutoff)
    # directly into df -- no separate pre_cutoff_df/post_cutoff_df
    # intermediates, so only one combined frame is ever materialized here,
    # not three.
    df = (
        pd.concat(all_chunks, ignore_index=True) if all_chunks
        else pd.DataFrame(columns=['source_date', 'scheduled_arrival_time', 'snapshot_timestamp'])
    )
    del all_chunks
    # pd.concat() of per-chunk categoricals whose category SETS differ (e.g.
    # source_date -- each chunk only ever has its own single date as a
    # category) silently coerces the combined column back to object dtype,
    # not a unioned category -- confirmed by direct test before this code
    # was run for real. Re-optimizing once here, on the already-reduced
    # final frame (same size class the old bulk approach's dtype step
    # always handled fine), restores category dtype with the correct
    # unioned vocabulary for the CATEGORICAL_COLS cast below.
    df = _optimize_dtypes(df)
    _log_mem(f'after concatenating all {len(pre_cutoff_dates) + len(post_cutoff_dates)} chunks (pre+post)')

    # rows_before_dropna_total/rows_after_dropna_total were accumulated per
    # chunk above (post-leak-filter, pre-dropna -> post-dropna) inside the
    # two read loops -- scheduled_arrival_dt reconstruction, Cell 4's
    # hour_of_day/day_of_week/is_weekend/is_peak derivation, and dropna all
    # now happen once per (small) chunk instead of once on the full
    # (~147M-row) combined frame, so `df` here is already post-dropna.
    rows_after_leakage_filter = rows_before_dropna_total
    print(f'Leakage filter: dropped {n_dropped:,} rows (post-arrival captures)')
    print(f'[ROW ACCOUNTING] 2. After leakage filter: {rows_before_leakage_filter:,} -> '
          f'{rows_after_leakage_filter:,} rows ({n_dropped:,} dropped)')

    rows_after_dropna = rows_after_dropna_total
    print(f'[ROW ACCOUNTING] 3. After dropna(delay_minutes): {rows_before_dropna_total:,} -> '
          f'{rows_after_dropna:,} rows ({rows_before_dropna_total - rows_after_dropna:,} dropped)')
    _log_mem('after dropna(delay_minutes) (already applied per-chunk above)')
    y = df['delay_minutes'].astype('float32')  # already float32 from the downcast above; explicit for clarity

    for c in CATEGORICAL_COLS:
        # astype('category') on an ALREADY-categorical column (df[c] may be
        # category dtype from the memory-optimization pass above) does not
        # drop categories no longer present after the leakage filter/dropna --
        # remove_unused_categories() keeps the persisted category vocabulary
        # identical to what it would be without that earlier optimization.
        # Applied directly on df (not a separate X copy) -- X_train/X_test
        # below inherit this exact vocabulary via plain slicing, since
        # slicing/copying a categorical column preserves its category list
        # unchanged. Doing it here, once, on the full train+test-combined
        # set (before the split) is what keeps X_test's codes aligned with
        # X_train's / categories.joblib's -- a second remove_unused_categories()
        # per split partition would desync them.
        df[c] = df[c].astype('category').cat.remove_unused_categories()
    df['stop_sequence'] = df['stop_sequence'].astype('float32')
    df['is_weekend'] = df['is_weekend'].astype('int8')
    df['is_peak'] = df['is_peak'].astype('int8')
    _log_mem('after feature dtype cast (pre-split)')

    # Small (single-column) string copy of source_date used for the date-span
    # summary and the split boundary below -- not a full-frame copy.
    source_date_str = df['source_date'].astype(str)

    # Full pre-split date span -- 'source_date' is already string-cast (ISO
    # 'YYYY-MM-DD'), so lexicographic min/max is chronological min/max.
    # Captured here (train+test combined) rather than after the split, since
    # this is what the model was actually trained+evaluated on as a whole.
    data_window_start = source_date_str.min()
    data_window_end = source_date_str.max()
    data_window_days = source_date_str.nunique()

    # --- Cell 6: temporal split (train = earliest dates, test = latest ~20%),
    # done BEFORE building the feature frame so X_train/X_test are each
    # sliced straight out of df -- df, X_train, and X_test are the only
    # full/partial-size frames ever alive together, never an extra
    # combined-X copy on top of both splits.
    rows_by_date = source_date_str.groupby(source_date_str).size().sort_index()
    total_rows = rows_by_date.sum()
    cum_from_end = rows_by_date[::-1].cumsum()[::-1]
    candidate_dates = cum_from_end[cum_from_end <= 0.20 * total_rows].index
    boundary_date = candidate_dates.min() if len(candidate_dates) else rows_by_date.index.max()
    train_mask = source_date_str < boundary_date
    del source_date_str

    X_train = df.loc[train_mask, FEATURE_COLS].copy()
    y_train = y.loc[train_mask]
    n_train = len(X_train)
    _log_mem('after building X_train')

    # Mirrors X_train/y_train exactly -- same already-cast df columns
    # (CATEGORICAL_COLS were cast + remove_unused_categories()'d on the full
    # train+test df above, before this split), so X_test's categorical codes
    # are guaranteed to align with X_train's / categories.joblib's, with no
    # separate remove_unused_categories() call needed (or wanted: that would
    # re-derive a test-only category list and desync its codes from what the
    # model was actually fit on).
    X_test = df.loc[~train_mask, FEATURE_COLS].copy()
    y_test = y.loc[~train_mask]
    n_test = int(total_rows) - n_train
    _log_mem('after building X_test')

    del df, y, train_mask
    print(f'Train: {n_train:,} rows (boundary date {boundary_date})')
    print(f'[ROW ACCOUNTING] 4. Temporal split: {int(total_rows):,} rows -> '
          f'train={n_train:,}, test={n_test:,} (boundary date {boundary_date})')

    print('=' * 70)
    print('Row accounting: loaded -> final train/test split')
    print('=' * 70)
    print(f'  1. Rows loaded from ferry-filtered snapshot : {rows_loaded:,}')
    if debug_sample_frac < 1.0:
        print(f'  1b. After debug sample (frac={debug_sample_frac})     : '
              f'{rows_after_sample_total:,} ({rows_loaded - rows_after_sample_total:,} dropped by sampling)')
    print(f'  2. After leakage filter                     : {rows_after_leakage_filter:,} '
          f'({rows_after_sample_total - rows_after_leakage_filter:,} dropped)')
    print(f'  3. After dropna(delay_minutes)               : {rows_after_dropna:,} '
          f'({rows_after_leakage_filter - rows_after_dropna:,} dropped)')
    print(f'  4. Final split: train={n_train:,} + test={n_test:,} = {n_train + n_test:,}')
    if (n_train + n_test) != rows_after_dropna:
        print(f'  WARNING: train+test ({n_train + n_test:,}) does not match post-dropna '
              f'count ({rows_after_dropna:,}) -- gap of '
              f'{rows_after_dropna - (n_train + n_test):,} row(s) is unexplained by any stage above.')
    else:
        print('  No gap: train+test exactly matches the post-dropna row count.')

    # Persisted alongside the model: the exact category->code mapping used at
    # fit time, so inference-time categorical columns can be reconstructed
    # identically (XGBoost's categorical splits are keyed on these codes).
    categories = {c: X_train[c].cat.categories.tolist() for c in CATEGORICAL_COLS}
    return X_train, y_train, X_test, y_test, categories, data_window_start, data_window_end, data_window_days


def _find_best_matching_static_snapshot(train_route_ids: set):
    """route_id carries a version suffix that changes between static GTFS
    snapshots (e.g. '100-4799' vs '100-4948'), so the snapshot whose
    routes.txt matches the *training* data's route_id vocabulary is not
    necessarily the latest one — it has to be located by checking each
    available snapshot's overlap with train_route_ids.
    """
    bucket, fs = _get_env()
    static_prefix = f'{bucket}/gtfs_static'
    entries = fs.ls(static_prefix)
    snapshot_dates = sorted(
        [e.rstrip('/').split('/')[-1] for e in entries if gtfs_data.DATE_PATTERN.match(e.rstrip('/').split('/')[-1])],
        reverse=True,
    )

    best_date, best_routes, best_match = None, None, -1
    for snap in snapshot_dates:
        routes = pd.read_csv(f's3://{static_prefix}/{snap}/routes.txt', dtype=str)
        match = len(train_route_ids & set(routes['route_id']))
        if match > best_match:
            best_date, best_routes, best_match = snap, routes, match
        if match == len(train_route_ids):
            break  # perfect match — no need to check older snapshots

    print(f'Best static snapshot match for training route_ids: {best_date} '
          f'({best_match}/{len(train_route_ids)} route_ids matched)')
    return best_date, best_routes


def _build_training_metadata(
    X_train: pd.DataFrame,
    train_mae: float,
    test_mae: float,
    naive_mae: float,
    pct_improvement_over_naive: float,
    data_window_start: str,
    data_window_end: str,
    data_window_days: int,
    mem_checkpoints: list[tuple[str, float]],
) -> dict:
    """Coverage counts per (route_short_name, mode), used by predict_delay()
    to judge whether a live route was well represented in training — keyed
    on route_short_name rather than the version-drifting route_id. Also
    records this run's train/test/naive MAE baseline (notebook 07 Cells
    8/8b), so it's permanently available in training_metadata.json rather
    than only printed to the training log.

    mem_checkpoints is the (label, rss_mb) list accumulated by _log_mem()
    over the run (see module-level _mem_checkpoints) -- persisted here so RSS
    trend is comparable across runs as the archive grows (~5-6M rows/day)
    instead of only existing in console output that's lost after each run.
    """
    train_route_ids = set(X_train['route_id'].astype(str).unique())
    snapshot_date, routes = _find_best_matching_static_snapshot(train_route_ids)
    short_name_by_id = routes.set_index('route_id')['route_short_name']

    lookup = X_train[['route_id', 'mode']].astype(str).copy()
    lookup['route_short_name'] = lookup['route_id'].map(short_name_by_id).fillna('UNKNOWN')

    counts = lookup.groupby(['route_short_name', 'mode'], observed=True).size()
    coverage_counts = {f'{name}|{mode}': int(n) for (name, mode), n in counts.items()}

    if not mem_checkpoints:
        raise ValueError(
            'No memory checkpoints captured (_mem_checkpoints is empty) -- '
            '_log_mem() should have run at least once during this training run. '
            'Refusing to write training_metadata.json with missing RSS data.'
        )
    peak_rss_mb = max(rss_mb for _, rss_mb in mem_checkpoints)

    def _checkpoint_rss(label: str) -> float:
        for checkpoint_label, rss_mb in mem_checkpoints:
            if checkpoint_label == label:
                return rss_mb
        raise ValueError(
            f'Expected memory checkpoint {label!r} not found in captured checkpoints '
            f'({[l for l, _ in mem_checkpoints]!r}) -- a _log_mem() call site may have '
            'been renamed or removed. Refusing to silently persist a missing RSS value.'
        )

    return {
        'trained_at': datetime.now(ZoneInfo('Australia/Brisbane')).isoformat(),
        'static_snapshot_used_for_route_names': snapshot_date,
        'coverage_counts': coverage_counts,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'naive_mae': naive_mae,
        'pct_improvement_over_naive': pct_improvement_over_naive,
        'data_window_start': data_window_start,
        'data_window_end': data_window_end,
        'data_window_days': data_window_days,
        'peak_rss_mb': peak_rss_mb,
        'rss_at_categorical_cast_mb': _checkpoint_rss('after feature dtype cast (pre-split)'),
        'rss_at_x_train_build_mb': _checkpoint_rss('after building X_train'),
        'rss_at_x_test_build_mb': _checkpoint_rss('after building X_test'),
        'rss_after_fit_mb': _checkpoint_rss('after fit completes'),
    }


def _train_and_save_model() -> None:
    _reset_mem_checkpoints()
    print('No saved model found — training v0 XGBoost model from the S3 feature snapshot...')
    X_train, y_train, X_test, y_test, categories, data_window_start, data_window_end, data_window_days = (
        _load_training_frames()
    )

    model = xgb.XGBRegressor(
        enable_categorical=True,
        tree_method='hist',
        random_state=42,
        n_jobs=-1,
    )
    _log_mem('before DMatrix construction')
    model.fit(X_train, y_train)
    _log_mem('after fit completes')
    print('Training complete.')

    # --- notebook 07 Cells 8/8b, ported so every retrain permanently records
    # a real MAE baseline instead of that being a separate manual step.
    # model.predict(X_train) reuses the already-fitted model above -- no
    # retraining, just one extra inference pass. ---
    test_pred = model.predict(X_test)
    test_mae = _mae(y_test.values, test_pred)

    train_pred = model.predict(X_train)
    train_mae = _mae(y_train.values, train_pred)

    train_median = float(y_train.median())
    naive_mae = _mae(y_test.values, train_median)
    pct_improvement_over_naive = (1 - test_mae / naive_mae) * 100

    print('=== Train/test MAE summary ===')
    print(f'Train: MAE {train_mae:.3f} min ({len(X_train):,} rows)')
    print(f'Test:  MAE {test_mae:.3f} min ({len(X_test):,} rows)')
    print(f'Naive baseline (predict train median = {train_median:.3f} min): MAE {naive_mae:.3f} min')
    print(f'Test MAE improvement over naive median baseline: {pct_improvement_over_naive:.1f}%')

    # DEBUG_MAX_DATES validation runs must prove the pipeline works end-to-end
    # (everything above this line just did) but must never let a partial-data
    # model reach save/promotion. Aborting with a non-zero exit here is
    # already handled correctly by pipeline/02_train_model.sh (runs/ and
    # latest/ are left untouched on any non-zero exit), so no separate
    # promotion-guard logic is needed in the bash script.
    if os.environ.get('DEBUG_MAX_DATES') is not None:
        raise RuntimeError(
            'DEBUG_MAX_DATES is set -- aborting before save to prevent a partial-data '
            'model from being promoted to latest/. Unset DEBUG_MAX_DATES for a real training run.'
        )

    training_metadata = _build_training_metadata(
        X_train, train_mae, test_mae, naive_mae, pct_improvement_over_naive,
        data_window_start, data_window_end, data_window_days,
        list(_mem_checkpoints),
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_PATH))
    joblib.dump(categories, CATEGORIES_PATH)
    with open(TRAINING_METADATA_PATH, 'w') as f:
        json.dump(training_metadata, f)
    print(f'Saved model to {MODEL_PATH.name}')
    print(f'Saved categorical mappings to {CATEGORIES_PATH.name}')
    print(f'Saved training coverage metadata to {TRAINING_METADATA_PATH.name}')

    _upload_model_to_s3()
