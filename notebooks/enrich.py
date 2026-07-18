"""Shared prep functions for the enrichment A/B (08_enrichment_ab.ipynb).

Reproduces notebook 07 Cells 1-6 verbatim (load via _latest.json, leakage
filter, time-feature rederivation, target/feature build, temporal split) so
the A/B is apples-to-apples against the pristine baseline. Notebook 07 is
never imported or modified — this module only replicates its logic.
"""
import gc
import json
import os
import subprocess
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv, find_dotenv
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'python-dotenv', '-q'], check=True)
    from dotenv import load_dotenv, find_dotenv

try:
    import s3fs
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', 's3fs', '-q'], check=True)
    import s3fs

import numpy as np
import pandas as pd


EXCLUDE_COLS = [
    'delay_seconds',          # same label, different unit
    'delay_minutes',          # this is y
    'snapshot_timestamp',     # capture metadata, not a real-world feature
    'scheduled_arrival_time',  # used for filtering/deriving only
    'trip_id',                # identifier, not predictive
]
CATEGORICAL_COLS = ['route_id', 'stop_id', 'mode', 'day_of_week']

WEATHER_COLS = ['temperature_2m', 'precipitation', 'wind_speed_10m', 'wind_gusts_10m',
                'is_raining', 'is_weather_missing']
ROUTE_ALERT_COLS = ['route_has_alert', 'route_n_alerts', 'route_alert_effect',
                     'is_construction_routelevel', 'is_maintenance_routelevel',
                     'is_rail_replacement_routelevel']
STOP_ALERT_COLS = ['stop_has_alert', 'stop_n_alerts', 'stop_alert_effect',
                    'is_construction_stoplevel', 'is_maintenance_stoplevel',
                    'is_rail_replacement_stoplevel']
ALERT_COLS = ROUTE_ALERT_COLS + STOP_ALERT_COLS


def get_env():
    """Load .env and return (S3_BUCKET, fs) — mirrors notebook 07 Cell 1."""
    _dotenv_path = find_dotenv(usecwd=True)
    if _dotenv_path:
        load_dotenv(_dotenv_path, override=False)
        print(f'Loaded .env from: {_dotenv_path}')
    else:
        print('No .env file found — using defaults.')

    S3_BUCKET = os.environ.get('AWS_S3_BUCKET', '')
    if not S3_BUCKET:
        raise EnvironmentError('AWS_S3_BUCKET is not set. Check your .env file.')

    fs = s3fs.S3FileSystem()
    print(f'S3_BUCKET: {S3_BUCKET}')
    return S3_BUCKET, fs


def load_prepared(S3_BUCKET, fs, run_date_override=None):
    """Reproduces notebook 07 Cells 2-5 verbatim.

    Returns (X, y, scheduled_arrival_dt, snapshot_timestamp, feature_cols, run_date):
      - X: feature matrix, still carries 'source_date' (str) for the split —
        caller must select feature_cols (+ any enrichment cols) before fit.
      - scheduled_arrival_dt / snapshot_timestamp: tz-aware Series aligned to
        X's index (post leakage-filter, post null-target-drop).
    """
    # --- Cell 2: load feature snapshot via _latest.json run pointer ---
    ML_FEATURES_PREFIX = f'{S3_BUCKET}/ml_features/v0_feature_snapshot'
    if run_date_override is not None:
        RUN_DATE = run_date_override
        load_path = f's3://{ML_FEATURES_PREFIX}/run_date={RUN_DATE}/'
        manifest = None
        print(f'RUN_DATE_OVERRIDE set — loading run_date={RUN_DATE} directly (bypassing _latest.json)')
    else:
        manifest_path = f's3://{ML_FEATURES_PREFIX}/_latest.json'
        with fs.open(manifest_path) as f:
            manifest = json.load(f)
        RUN_DATE = manifest['latest_run']
        load_path = f's3://{ML_FEATURES_PREFIX}/run_date={RUN_DATE}/'
        print(f'_latest.json found -> latest_run={RUN_DATE}')

    df = pd.read_parquet(load_path)
    print(f'Loaded {len(df):,} rows x {df.shape[1]} cols from {load_path}')

    if manifest is not None:
        expected = manifest['row_count']
        actual = len(df)
        match = 'MATCH' if actual == expected else 'MISMATCH'
        print(f'Manifest row_count: {expected:,} | Loaded: {actual:,} -> {match}')

    # --- Cell 3: parse scheduled_arrival_time, drop post-arrival captures (leakage filter) ---
    _time_parts = df['scheduled_arrival_time'].str.split(':', expand=True).astype(int)
    _offset = (
        pd.to_timedelta(_time_parts[0], unit='h')
        + pd.to_timedelta(_time_parts[1], unit='m')
        + pd.to_timedelta(_time_parts[2], unit='s')
    )
    _source_date_midnight = pd.to_datetime(df['source_date'].astype(str)).dt.tz_localize('Australia/Brisbane')
    scheduled_arrival_dt = _source_date_midnight + _offset

    n_before = len(df)
    leak_mask = df['snapshot_timestamp'] >= scheduled_arrival_dt
    n_dropped = int(leak_mask.sum())

    df = df.loc[~leak_mask].copy()
    scheduled_arrival_dt = scheduled_arrival_dt.loc[~leak_mask]

    print(f'Leakage filter: dropped {n_dropped:,} / {n_before:,} rows '
          f'({n_dropped / n_before * 100:.2f}%) — post-arrival captures')

    del _time_parts, _offset, _source_date_midnight, leak_mask
    gc.collect()

    # --- Cell 4: overwrite hour_of_day / day_of_week / is_weekend / is_peak ---
    df['hour_of_day'] = scheduled_arrival_dt.dt.hour.astype('int32')
    df['day_of_week'] = scheduled_arrival_dt.dt.day_name()
    df['is_weekend'] = df['day_of_week'].isin(['Saturday', 'Sunday'])
    df['is_peak'] = (~df['is_weekend']) & df['hour_of_day'].isin([7, 8, 16, 17])

    # --- Cell 5: build target + feature matrix ---
    n_before_target = len(df)
    df = df.dropna(subset=['delay_minutes']).copy()
    n_null_target = n_before_target - len(df)
    print(f'Dropped {n_null_target:,} rows with null delay_minutes (no observed label)')

    # scheduled_arrival_dt / snapshot_timestamp must track the same row drop —
    # notebook 07 never needed this (it discards scheduled_arrival_dt after
    # Cell 4), but this module needs both aligned to the final row set for
    # the weather/alert joins and the alert-join leakage check downstream.
    scheduled_arrival_dt = scheduled_arrival_dt.loc[df.index]
    snapshot_timestamp = df['snapshot_timestamp'].copy()

    y = df['delay_minutes'].astype('float64')

    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS and c != 'source_date']
    print(f'Feature columns ({len(feature_cols)}): {feature_cols}')

    X = df[feature_cols + ['source_date']].copy()
    X['source_date'] = X['source_date'].astype(str)

    for c in CATEGORICAL_COLS:
        X[c] = X[c].astype('category')

    X['stop_sequence'] = X['stop_sequence'].astype('float64')
    X['is_weekend'] = X['is_weekend'].astype('int8')
    X['is_peak'] = X['is_peak'].astype('int8')

    return X, y, scheduled_arrival_dt, snapshot_timestamp, feature_cols, RUN_DATE


def temporal_split(X):
    """Reproduces notebook 07 Cell 6's boundary-date logic verbatim.

    Split on sorted, unique source_date values — never a random shuffle.
    Boundary date chosen so the test partition (latest dates) is as close
    to 20% of rows as possible. Returns (train_mask, test_mask, boundary_date).
    """
    rows_by_date = X.groupby('source_date', observed=True).size().sort_index()
    total_rows = rows_by_date.sum()
    cum_from_end = rows_by_date[::-1].cumsum()[::-1]  # rows remaining from this date onward

    candidate_dates = cum_from_end[cum_from_end <= 0.20 * total_rows].index
    boundary_date = candidate_dates.min() if len(candidate_dates) else rows_by_date.index.max()

    train_mask = X['source_date'] < boundary_date
    test_mask = ~train_mask

    print(f'Boundary date (test = this date onward): {boundary_date}')
    print(f'Train: {train_mask.sum():,} rows ({train_mask.sum() / total_rows * 100:.1f}%), '
          f'dates {X.loc[train_mask, "source_date"].min()} .. {X.loc[train_mask, "source_date"].max()}')
    print(f'Test:  {test_mask.sum():,} rows ({test_mask.sum() / total_rows * 100:.1f}%), '
          f'dates {X.loc[test_mask, "source_date"].min()} .. {X.loc[test_mask, "source_date"].max()}')

    return train_mask, test_mask, boundary_date


def attach_weather(X, scheduled_arrival_dt, S3_BUCKET, fs):
    """Left-join weather/era5/hourly.parquet on date+hour derived from
    scheduled_arrival_dt (the time being predicted for, not snapshot_timestamp).
    Adds is_weather_missing; ERA5 lags a few days so the latest dates may be null.
    Returns (X_with_weather, match_rate).
    """
    weather_path = f's3://{S3_BUCKET}/weather/era5/hourly.parquet'
    weather_df = pd.read_parquet(weather_path, filesystem=fs)

    join_key = pd.DataFrame({
        'weather_date': scheduled_arrival_dt.dt.strftime('%Y-%m-%d'),
        'weather_hour': scheduled_arrival_dt.dt.hour.astype('int32'),
    })
    join_key.index = X.index

    merged = join_key.merge(weather_df, on=['weather_date', 'weather_hour'], how='left')
    merged.index = X.index  # merge on columns resets to RangeIndex; restore alignment

    X_out = X.copy()
    X_out['temperature_2m'] = merged['temperature_2m'].astype('float64')
    X_out['precipitation'] = merged['precipitation'].astype('float64')
    X_out['wind_speed_10m'] = merged['wind_speed_10m'].astype('float64')
    X_out['wind_gusts_10m'] = merged['wind_gusts_10m'].astype('float64')
    # nullable boolean -> float64 (1.0/0.0/NaN): keeps missing consistent with
    # the other weather columns for XGBoost's native missing-value handling,
    # rather than relying on pandas' nullable-boolean dtype.
    X_out['is_raining'] = merged['is_raining'].astype('float64')
    X_out['is_weather_missing'] = merged['temperature_2m'].isna().astype('int8')

    match_rate = 1.0 - X_out['is_weather_missing'].mean()
    print(f'Weather join match rate: {match_rate * 100:.2f}%')

    return X_out, float(match_rate)


def attach_alerts(X, scheduled_arrival_dt, S3_BUCKET, fs):
    """Left-join BOTH alert tables — route_hourly on route_id, stop_hourly on
    stop_id — each on date+hour from scheduled_arrival_dt. Fill misses with
    0 / 'NONE'. Route-level and stop-level columns are kept separate.
    Returns (X_with_alerts, route_match_rate, stop_match_rate).
    """
    route_path = f's3://{S3_BUCKET}/alerts/features/route_hourly.parquet'
    stop_path = f's3://{S3_BUCKET}/alerts/features/stop_hourly.parquet'
    route_alerts = pd.read_parquet(route_path, filesystem=fs)
    stop_alerts = pd.read_parquet(stop_path, filesystem=fs)

    alert_date = scheduled_arrival_dt.dt.strftime('%Y-%m-%d')
    alert_hour = scheduled_arrival_dt.dt.hour.astype('int64')

    route_key = pd.DataFrame({
        'route_id': X['route_id'].astype(str),
        'alert_date': alert_date,
        'alert_hour': alert_hour,
    })
    route_key.index = X.index
    route_merged = route_key.merge(route_alerts, on=['route_id', 'alert_date', 'alert_hour'], how='left')
    route_merged.index = X.index

    stop_key = pd.DataFrame({
        'stop_id': X['stop_id'].astype(str),
        'alert_date': alert_date,
        'alert_hour': alert_hour,
    })
    stop_key.index = X.index
    stop_merged = stop_key.merge(stop_alerts, on=['stop_id', 'alert_date', 'alert_hour'], how='left')
    stop_merged.index = X.index

    X_out = X.copy()

    X_out['route_has_alert'] = route_merged['route_has_alert'].fillna(0).astype('int8')
    X_out['route_n_alerts'] = route_merged['route_n_alerts'].fillna(0).astype('int32')
    X_out['route_alert_effect'] = route_merged['route_alert_effect'].fillna('NONE').astype('category')
    X_out['is_construction_routelevel'] = route_merged['is_construction_routelevel'].fillna(False).astype('int8')
    X_out['is_maintenance_routelevel'] = route_merged['is_maintenance_routelevel'].fillna(False).astype('int8')
    X_out['is_rail_replacement_routelevel'] = route_merged['is_rail_replacement_routelevel'].fillna(False).astype('int8')

    X_out['stop_has_alert'] = stop_merged['stop_has_alert'].fillna(0).astype('int8')
    X_out['stop_n_alerts'] = stop_merged['stop_n_alerts'].fillna(0).astype('int32')
    X_out['stop_alert_effect'] = stop_merged['stop_alert_effect'].fillna('NONE').astype('category')
    X_out['is_construction_stoplevel'] = stop_merged['is_construction_stoplevel'].fillna(False).astype('int8')
    X_out['is_maintenance_stoplevel'] = stop_merged['is_maintenance_stoplevel'].fillna(False).astype('int8')
    X_out['is_rail_replacement_stoplevel'] = stop_merged['is_rail_replacement_stoplevel'].fillna(False).astype('int8')

    route_match_rate = float(X_out['route_has_alert'].mean())
    stop_match_rate = float(X_out['stop_has_alert'].mean())
    print(f'Route-alert join match rate: {route_match_rate * 100:.4f}%')
    print(f'Stop-alert join match rate:  {stop_match_rate * 100:.4f}%')

    return X_out, route_match_rate, stop_match_rate
