"""Prediction service for the Phase 3 Streamlit POC.

Loads (training + saving first if needed) the v0 XGBoost delay model from
notebook 07, builds inference-time feature rows matching that training
schema exactly, and blends model predictions with live GTFS-RT delay data
into a rider-facing summary.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import pandas as pd
import psutil
import xgboost as xgb

import config
import gtfs_data
import live_gtfs

# The live app always reads/writes phase3/model/latest/. pipeline/02_train_model.sh
# points a training run at an isolated phase3/model/_staging_{date}_{time}/
# subdir instead via MODEL_SUBDIR_OVERRIDE, so a live model.predict() call can
# never observe a partially-written retrain -- it promotes staging to latest/
# only after all three artifact files are confirmed present.
MODEL_SUBDIR = os.environ.get('MODEL_SUBDIR_OVERRIDE', 'latest')
MODEL_DIR = Path(__file__).parent / 'model' / MODEL_SUBDIR
MODEL_PATH = MODEL_DIR / 'xgb_v0.json'
CATEGORIES_PATH = MODEL_DIR / 'categories.joblib'
TRAINING_METADATA_PATH = MODEL_DIR / 'training_metadata.json'

# Must match notebook 07 Cell 5 exactly (feature_cols after EXCLUDE_COLS).
FEATURE_COLS = ['route_id', 'stop_id', 'mode', 'stop_sequence',
                'hour_of_day', 'day_of_week', 'is_weekend', 'is_peak']
CATEGORICAL_COLS = ['route_id', 'stop_id', 'mode', 'day_of_week']

# Same mapping notebook 05 uses to derive the training `mode` column from
# GTFS route_type — needed here to turn a find_trips() route_type back into
# the same string the model was trained on.
MODE_BY_ROUTE_TYPE = {0: 'tram', 2: 'rail', 3: 'bus', 4: 'ferry'}
MODE_NOUN = {'tram': 'tram', 'rail': 'train', 'bus': 'bus', 'ferry': 'ferry', 'unknown': 'service'}

# A (route_short_name, mode) combo with at least this many training rows is
# considered "well represented" for the coverage check in predict_delay().
WELL_REPRESENTED_MIN_ROWS = 1000

_cache: dict = {}


def format_time_ampm(dt: datetime) -> str:
    """Format a datetime as rider-facing 'HH:MM AM/PM', e.g. '09:20 PM'."""
    return dt.strftime('%I:%M %p')


def _log_mem(label: str) -> None:
    """Debug checkpoint: current process RSS, so a slow/OOM-prone training
    run can be narrowed down to a specific stage without a profiler."""
    rss_mb = psutil.Process().memory_info().rss / 1e6
    print(f'[MEM] {label}: {rss_mb:,.1f} MB RSS')


def _get_env():
    return config.get_s3_bucket(), config.get_s3_filesystem()


def _s3_model_prefix() -> str:
    return f'{config.get_s3_bucket()}/phase3/model/{MODEL_SUBDIR}'


def _s3_model_paths() -> dict:
    prefix = _s3_model_prefix()
    return {
        'model': f's3://{prefix}/xgb_v0.json',
        'categories': f's3://{prefix}/categories.joblib',
        'metadata': f's3://{prefix}/training_metadata.json',
    }


def _download_model_from_s3() -> bool:
    """Download the saved model files from S3 into MODEL_DIR. Returns True on success."""
    try:
        fs = config.get_s3_filesystem()
        s3_paths = _s3_model_paths()
        if not all(fs.exists(p) for p in s3_paths.values()):
            return False

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        fs.get(s3_paths['model'], str(MODEL_PATH))
        fs.get(s3_paths['categories'], str(CATEGORIES_PATH))
        fs.get(s3_paths['metadata'], str(TRAINING_METADATA_PATH))
        print(f"Downloaded model files from s3://{_s3_model_prefix()}/")
        return True
    except Exception as e:
        print(f'Could not load model from S3 ({e}) — will try local files instead.')
        return False


def _upload_model_to_s3() -> None:
    fs = config.get_s3_filesystem()
    s3_paths = _s3_model_paths()
    fs.put(str(MODEL_PATH), s3_paths['model'])
    fs.put(str(CATEGORIES_PATH), s3_paths['categories'])
    fs.put(str(TRAINING_METADATA_PATH), s3_paths['metadata'])
    print(f"Uploaded model files to s3://{_s3_model_prefix()}/")


def _load_training_frames():
    """Reproduces notebook 07 Cells 2-6 verbatim: load via _latest.json,
    leakage filter, re-derive time features, target/feature build, temporal
    split. Returns (X_train, y_train, categories) — only the train partition,
    matching what notebook 07 actually fits on.
    """
    bucket, fs = _get_env()
    ml_features_prefix = f'{bucket}/ml_features/v0_feature_snapshot'

    manifest_path = f's3://{ml_features_prefix}/_latest.json'
    with fs.open(manifest_path) as f:
        manifest = json.load(f)
    run_date = manifest['latest_run']
    load_path = f's3://{ml_features_prefix}/run_date={run_date}/'
    print(f'Loading training snapshot: run_date={run_date}')

    df = pd.read_parquet(load_path)
    print(f'Loaded {len(df):,} rows x {df.shape[1]} cols')
    _log_mem('after data load')

    # --- Debug-only: sample down before any dtype conversion so the entire
    # downstream path (dtype optimization, DMatrix construction, fit, save)
    # runs against a small dataset in seconds instead of the full ~62M-row
    # snapshot. Unset (or 1.0) means full data -- the default, normal path.
    debug_sample_frac = float(os.environ.get('DEBUG_SAMPLE_FRAC', 1.0))
    if debug_sample_frac < 1.0:
        n_before_sample = len(df)
        df = df.sample(frac=debug_sample_frac, random_state=42)
        print(f'[DEBUG_SAMPLE_FRAC={debug_sample_frac}] Sampling active -- '
              f'{n_before_sample:,} rows -> {len(df):,} rows')

    # --- Memory footprint reduction: downcast float64 -> float32, and
    # convert repeating low-cardinality string/object columns to category
    # dtype. Pure memory optimization on the raw ~62M-row snapshot -- no
    # modeling logic, feature selection, or hyperparameters change here.
    # Columns XGBoost needs as true numeric for splitting (delay_minutes,
    # stop_sequence) are downcast in place but never categorized;
    # scheduled_arrival_time stays a string (parsed via .str below, and
    # never reaches XGBoost -- dropped before X is built).
    mem_before = df.memory_usage(deep=True).sum()

    float64_cols = df.select_dtypes(include='float64').columns
    for c in float64_cols:
        df[c] = df[c].astype('float32')

    category_cols = ['route_id', 'stop_id', 'mode', 'day_of_week', 'source_date', 'trip_id']
    for c in category_cols:
        if c in df.columns and df[c].dtype == object:
            df[c] = df[c].astype('category')

    mem_after = df.memory_usage(deep=True).sum()
    print(f'Memory usage: {mem_before / 1e6:,.1f} MB -> {mem_after / 1e6:,.1f} MB '
          f'({(1 - mem_after / mem_before) * 100:.1f}% reduction)')
    _log_mem('after dtype optimization')

    # --- Cell 3: leakage filter ---
    _log_mem('before leakage filter')
    # pd.to_datetime with an explicit format parses in vectorized C code with
    # no per-row Python objects, unlike .str.split(), which allocates a
    # Python list + string per row at 50M+ row scale. GTFS times can have
    # hours >= 24 for past-midnight trips, which %H (00-23) rejects --
    # errors='coerce' turns those rows into NaT for free, which doubles as
    # the mask for a small divmod-based fallback on just that subset.
    raw_time = df['scheduled_arrival_time']
    parsed = pd.to_datetime(raw_time, format='%H:%M:%S', errors='coerce')
    offset = parsed - parsed.dt.normalize()

    past_midnight = parsed.isna()
    if past_midnight.any():
        hh = raw_time.loc[past_midnight].str.slice(0, 2).astype(int)
        rest = raw_time.loc[past_midnight].str.slice(2)
        days, hh_mod = divmod(hh, 24)
        wrapped_time = hh_mod.astype(str).str.zfill(2) + rest
        fixed_parsed = pd.to_datetime(wrapped_time, format='%H:%M:%S')
        # `days` is added back as a standalone Timedelta rather than folded
        # into fixed_parsed's date -- .dt.normalize() would otherwise
        # re-derive midnight from the shifted date and cancel it out.
        offset.loc[past_midnight] = (
            (fixed_parsed - fixed_parsed.dt.normalize()) + pd.to_timedelta(days, unit='D')
        )

    source_date_midnight = pd.to_datetime(df['source_date'].astype(str)).dt.tz_localize('Australia/Brisbane')
    scheduled_arrival_dt = source_date_midnight + offset
    _log_mem('after scheduled_arrival_dt construction')

    leak_mask = df['snapshot_timestamp'] >= scheduled_arrival_dt
    n_dropped = int(leak_mask.sum())
    df = df.loc[~leak_mask].copy()
    scheduled_arrival_dt = scheduled_arrival_dt.loc[~leak_mask]
    print(f'Leakage filter: dropped {n_dropped:,} rows (post-arrival captures)')
    _log_mem('after leakage filter')

    # --- Cell 4: re-derive hour_of_day / day_of_week / is_weekend / is_peak ---
    df['hour_of_day'] = scheduled_arrival_dt.dt.hour.astype('int32')
    df['day_of_week'] = scheduled_arrival_dt.dt.day_name()
    df['is_weekend'] = df['day_of_week'].isin(['Saturday', 'Sunday'])
    df['is_peak'] = (~df['is_weekend']) & df['hour_of_day'].isin([7, 8, 16, 17])

    # --- Cell 5: target + feature matrix ---
    df = df.dropna(subset=['delay_minutes']).copy()
    y = df['delay_minutes'].astype('float32')  # already float32 from the downcast above; explicit for clarity

    X = df[FEATURE_COLS + ['source_date']].copy()
    X['source_date'] = X['source_date'].astype(str)
    for c in CATEGORICAL_COLS:
        # astype('category') on an ALREADY-categorical column (df[c] may be
        # category dtype from the memory-optimization pass above) does not
        # drop categories no longer present after the leakage filter/split --
        # remove_unused_categories() keeps the persisted category vocabulary
        # identical to what it would be without that earlier optimization
        # (i.e. exactly X_train's own values, not a superset).
        X[c] = X[c].astype('category').cat.remove_unused_categories()
    X['stop_sequence'] = X['stop_sequence'].astype('float32')
    X['is_weekend'] = X['is_weekend'].astype('int8')
    X['is_peak'] = X['is_peak'].astype('int8')

    # --- Cell 6: temporal split (train = earliest dates, test = latest ~20%) ---
    rows_by_date = X.groupby('source_date', observed=True).size().sort_index()
    total_rows = rows_by_date.sum()
    cum_from_end = rows_by_date[::-1].cumsum()[::-1]
    candidate_dates = cum_from_end[cum_from_end <= 0.20 * total_rows].index
    boundary_date = candidate_dates.min() if len(candidate_dates) else rows_by_date.index.max()
    train_mask = X['source_date'] < boundary_date

    X_train = X.loc[train_mask, FEATURE_COLS].copy()
    y_train = y.loc[train_mask]
    print(f'Train: {len(X_train):,} rows (boundary date {boundary_date})')

    # Persisted alongside the model: the exact category->code mapping used at
    # fit time, so inference-time categorical columns can be reconstructed
    # identically (XGBoost's categorical splits are keyed on these codes).
    categories = {c: X_train[c].cat.categories.tolist() for c in CATEGORICAL_COLS}
    return X_train, y_train, categories


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


def _build_training_metadata(X_train: pd.DataFrame) -> dict:
    """Coverage counts per (route_short_name, mode), used by predict_delay()
    to judge whether a live route was well represented in training — keyed
    on route_short_name rather than the version-drifting route_id.
    """
    train_route_ids = set(X_train['route_id'].astype(str).unique())
    snapshot_date, routes = _find_best_matching_static_snapshot(train_route_ids)
    short_name_by_id = routes.set_index('route_id')['route_short_name']

    lookup = X_train[['route_id', 'mode']].astype(str).copy()
    lookup['route_short_name'] = lookup['route_id'].map(short_name_by_id).fillna('UNKNOWN')

    counts = lookup.groupby(['route_short_name', 'mode'], observed=True).size()
    coverage_counts = {f'{name}|{mode}': int(n) for (name, mode), n in counts.items()}

    return {
        'trained_at': datetime.now(ZoneInfo('Australia/Brisbane')).isoformat(),
        'static_snapshot_used_for_route_names': snapshot_date,
        'coverage_counts': coverage_counts,
    }


def _train_and_save_model() -> None:
    print('No saved model found — training v0 XGBoost model from the S3 feature snapshot...')
    X_train, y_train, categories = _load_training_frames()

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

    training_metadata = _build_training_metadata(X_train)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_PATH))
    joblib.dump(categories, CATEGORIES_PATH)
    with open(TRAINING_METADATA_PATH, 'w') as f:
        json.dump(training_metadata, f)
    print(f'Saved model to {MODEL_PATH.name}')
    print(f'Saved categorical mappings to {CATEGORIES_PATH.name}')
    print(f'Saved training coverage metadata to {TRAINING_METADATA_PATH.name}')

    _upload_model_to_s3()


def _load_model_impl() -> xgb.XGBRegressor:
    """S3 first (source of truth for Streamlit Cloud, which has no local model
    files), then local files (local dev fallback), training only if neither
    is available.
    """
    have_model = _download_model_from_s3()
    if not have_model:
        have_model = MODEL_PATH.exists() and CATEGORIES_PATH.exists() and TRAINING_METADATA_PATH.exists()
        if have_model:
            # Locally trained but not yet mirrored to S3 (e.g. first Cloud deploy prep) — upload now.
            _upload_model_to_s3()
    if not have_model:
        _train_and_save_model()

    model = xgb.XGBRegressor(enable_categorical=True, tree_method='hist')
    model.load_model(str(MODEL_PATH))
    categories = joblib.load(CATEGORIES_PATH)
    with open(TRAINING_METADATA_PATH) as f:
        training_metadata = json.load(f)

    _cache['categories'] = categories
    _cache['training_metadata'] = training_metadata
    return model


try:
    import streamlit as st
    load_model = st.cache_resource(
        show_spinner='Loading prediction model (training on first run can take a few minutes)...'
    )(_load_model_impl)
except ImportError:
    def load_model() -> xgb.XGBRegressor:
        """Load the trained v0 model (training + saving first if needed). Cached in memory."""
        if 'model' not in _cache:
            _cache['model'] = _load_model_impl()
        return _cache['model']


def _get_training_metadata() -> dict:
    if 'training_metadata' not in _cache:
        load_model()
    return _cache['training_metadata']


def _get_categories() -> dict:
    if 'categories' not in _cache:
        load_model()
    return _cache['categories']


def build_features(trip_info: dict, departure_time: datetime) -> pd.DataFrame:
    """Build one feature row matching notebook 07's training schema exactly.

    trip_info must supply route_id, stop_id, stop_sequence (the destination
    stop being predicted for), and either `mode` directly or `route_type`
    (GTFS int, mapped via MODE_BY_ROUTE_TYPE). `departure_time` is the
    scheduled clock time hour_of_day/day_of_week/is_weekend/is_peak are
    derived from — notebook 07 re-derives these from scheduled_arrival_time,
    not capture time, so this should be the trip's scheduled time, not
    wall-clock "now".
    """
    categories = _get_categories()

    mode = trip_info.get('mode')
    if mode is None:
        mode = MODE_BY_ROUTE_TYPE.get(trip_info.get('route_type'), 'unknown')

    hour_of_day = departure_time.hour
    day_of_week = departure_time.strftime('%A')
    is_weekend = day_of_week in ('Saturday', 'Sunday')
    is_peak = (not is_weekend) and hour_of_day in (7, 8, 16, 17)

    row = {
        'route_id': trip_info['route_id'],
        'stop_id': trip_info['stop_id'],
        'mode': mode,
        'stop_sequence': float(trip_info['stop_sequence']),
        'hour_of_day': hour_of_day,
        'day_of_week': day_of_week,
        'is_weekend': int(is_weekend),
        'is_peak': int(is_peak),
    }
    X = pd.DataFrame([row])

    for c in CATEGORICAL_COLS:
        X[c] = pd.Categorical(X[c], categories=categories[c])
    X['hour_of_day'] = X['hour_of_day'].astype('int32')
    X['is_weekend'] = X['is_weekend'].astype('int8')
    X['is_peak'] = X['is_peak'].astype('int8')
    # float32 to match the training-time downcast (_load_training_frames /
    # X_train['stop_sequence']) -- a model fit on float32 can behave
    # inconsistently at inference time if fed float64 columns.
    X['stop_sequence'] = X['stop_sequence'].astype('float32')

    return X[FEATURE_COLS]


def enrich_trip_with_dest_stop(trip: dict, dest_stop_ids: list[str]) -> dict:
    """Add `stop_id` / `stop_sequence` (the destination stop this specific
    trip actually visits) to a find_trips() result dict, by reading the same
    cached static GTFS data gtfs_data.py already loaded. gtfs_data.py itself
    is never modified — this only reads its already-loaded stop_times.
    """
    data = gtfs_data.load_gtfs_data()
    st = data.stop_times
    match = st[(st['trip_id'] == trip['trip_id']) & (st['stop_id'].isin(dest_stop_ids))]
    if match.empty:
        raise ValueError(f"No stop_times row for trip {trip['trip_id']!r} at stops {dest_stop_ids!r}")
    match = match.sort_values('stop_sequence').iloc[-1]

    enriched = dict(trip)
    enriched['stop_id'] = match['stop_id']
    enriched['stop_sequence'] = int(match['stop_sequence'])
    return enriched


def predict_delay(trip_info: dict, departure_time: datetime, live_delay: dict | None = None) -> dict:
    """Predict delay for a trip's arrival at its destination stop, blending
    the v0 model prediction with live GTFS-RT data when available.
    """
    model = load_model()
    training_metadata = _get_training_metadata()

    X = build_features(trip_info, departure_time)
    predicted_delay_minutes = float(model.predict(X)[0])

    mode = trip_info.get('mode') or MODE_BY_ROUTE_TYPE.get(trip_info.get('route_type'), 'unknown')
    route_short_name = trip_info.get('route_short_name') or trip_info['route_id']
    coverage_count = training_metadata['coverage_counts'].get(f'{route_short_name}|{mode}', 0)
    well_represented = coverage_count >= WELL_REPRESENTED_MIN_ROWS

    if live_delay is not None:
        live_delay_minutes = float(live_delay['delay_minutes'])
        agrees = abs(live_delay_minutes - predicted_delay_minutes) <= 2.0
        blended_delay_minutes = 0.7 * live_delay_minutes + 0.3 * predicted_delay_minutes
        confidence = 'High' if agrees else 'Medium'
    else:
        live_delay_minutes = None
        blended_delay_minutes = predicted_delay_minutes
        confidence = 'Medium' if well_represented else 'Low'

    scheduled_arrival = trip_info['dest_arrival_time']
    estimated_arrival = scheduled_arrival + timedelta(minutes=blended_delay_minutes)
    leave_by = trip_info['origin_departure_time'] - timedelta(minutes=3)

    mode = trip_info.get('mode') or MODE_BY_ROUTE_TYPE.get(trip_info.get('route_type'), 'unknown')
    mode_noun = MODE_NOUN.get(mode, 'service')
    route_label = trip_info.get('route_short_name') or trip_info['route_id']

    if blended_delay_minutes >= 1:
        delay_phrase = f'running ~{blended_delay_minutes:.0f} min late'
    elif blended_delay_minutes <= -1:
        delay_phrase = f'running ~{abs(blended_delay_minutes):.0f} min early'
    else:
        delay_phrase = 'on time'

    summary = (
        f"The {route_label} {mode_noun} is {delay_phrase}. "
        f"Leave by {format_time_ampm(leave_by)} to catch the "
        f"{format_time_ampm(trip_info['origin_departure_time'])} "
        f"from {trip_info.get('origin_stop_name', 'origin')}. "
        f"Expected arrival at {trip_info.get('dest_stop_name', 'destination')}: "
        f"{format_time_ampm(estimated_arrival)}. Confidence: {confidence}."
    )

    return {
        'predicted_delay_minutes': predicted_delay_minutes,
        'live_delay_minutes': live_delay_minutes,
        'blended_delay_minutes': blended_delay_minutes,
        'confidence': confidence,
        'scheduled_arrival': format_time_ampm(scheduled_arrival),
        'estimated_arrival': format_time_ampm(estimated_arrival),
        'leave_by': format_time_ampm(leave_by),
        'summary': summary,
    }


if __name__ == '__main__':
    print('=== Loading v0 model (training + saving first if needed) ===')
    model = load_model()
    print(f'Model loaded: {type(model).__name__}, n_estimators={model.n_estimators}')
    print()

    print('=== Finding a trip: Broadbeach South -> Surfers Paradise ===')
    origin_candidates = gtfs_data.search_stops('Broadbeach South', limit=50)
    dest_candidates = gtfs_data.search_stops('Surfers Paradise', limit=50)
    origin = next((r for r in origin_candidates if 'station' in r['stop_name'].lower()), origin_candidates[0])
    dest = next((r for r in dest_candidates if 'station' in r['stop_name'].lower()), dest_candidates[0])
    print(f'  origin: {origin["stop_name"]}  dest: {dest["stop_name"]}')

    now = datetime.now()
    trips = gtfs_data.find_trips(origin['stop_ids'], dest['stop_ids'], now, window_minutes=60)
    print(f'  {len(trips)} trip(s) found in the next 60 min')
    if not trips:
        raise SystemExit('No trips found in the next 60 minutes — try again later.')

    trip = enrich_trip_with_dest_stop(trips[0], dest['stop_ids'])
    print(f'  using trip: {trip}')
    print()

    print('=== predict_delay WITHOUT live data ===')
    result_no_live = predict_delay(trip, now)
    for k, v in result_no_live.items():
        print(f'  {k}: {v}')
    print()

    print('=== predict_delay WITH live data ===')
    live_delay = live_gtfs.get_live_delay(trip['trip_id'])
    print(f'  live_delay lookup for trip_id={trip["trip_id"]!r}: {live_delay}')
    if live_delay is None:
        print('  (no live update yet for this trip — it has not started running; '
              'using a synthetic live_delay below to demonstrate the blending path)')
        live_delay = {'delay_minutes': 3.0, 'timestamp': int(now.timestamp()), 'stop_id': trip['stop_id']}
    result_with_live = predict_delay(trip, now, live_delay=live_delay)
    for k, v in result_with_live.items():
        print(f'  {k}: {v}')
    print()

    print('=== Finding a trip: Helensvale -> Roma Street (Citytrain, route_short_name+mode coverage check) ===')
    next_monday = now + timedelta(days=(7 - now.weekday()) % 7 or 7)
    next_monday = next_monday.replace(hour=9, minute=0, second=0, microsecond=0)
    origin_candidates = gtfs_data.search_stops('Helensvale station', limit=50)
    dest_candidates = gtfs_data.search_stops('Roma Street station', limit=50)
    origin = next(r for r in origin_candidates if r['stop_name'] == 'Helensvale station')
    dest = next(r for r in dest_candidates if r['stop_name'] == 'Roma Street station')
    print(f'  origin: {origin["stop_name"]}  dest: {dest["stop_name"]}  departure_after: {next_monday}')

    trips = gtfs_data.find_trips(origin['stop_ids'], dest['stop_ids'], next_monday, window_minutes=120)
    print(f'  {len(trips)} trip(s) found in the next 120 min')
    if trips:
        trip = enrich_trip_with_dest_stop(trips[0], dest['stop_ids'])
        print(f'  using trip: {trip}')
        result = predict_delay(trip, next_monday)
        print(f"  confidence (no live data): {result['confidence']}")
        for k, v in result.items():
            print(f'  {k}: {v}')
