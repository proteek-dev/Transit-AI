"""Inference for the Phase 3 prediction service -- hot path, runs per request.

build_features() builds one feature row matching notebook 07's training
schema exactly (reusing model_io.FEATURE_COLS/CATEGORICAL_COLS so it can
never drift from what training.py fit on) and flags out-of-vocabulary
stop_ids; predict_delay() runs the cached model and blends its prediction
with live GTFS-RT delay data into a rider-facing summary.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

import gtfs_data
from ml.model_io import CATEGORICAL_COLS, FEATURE_COLS, _get_categories, _get_training_metadata, load_model
from route_types import MODE_BY_ROUTE_TYPE

# MODE_BY_ROUTE_TYPE (route_types.py) is the same mapping notebook 05 uses to
# derive the training `mode` column from GTFS route_type — needed here to
# turn a find_trips() route_type back into the same string the model was
# trained on.
MODE_NOUN = {'tram': 'tram', 'rail': 'train', 'bus': 'bus', 'ferry': 'ferry', 'unknown': 'service'}

# A (route_short_name, mode) combo with at least this many training rows is
# considered "well represented" for the coverage check in predict_delay().
WELL_REPRESENTED_MIN_ROWS = 1000


def format_time_ampm(dt: datetime) -> str:
    """Format a datetime as rider-facing 'HH:MM AM/PM', e.g. '09:20 PM'."""
    return dt.strftime('%I:%M %p')


def build_features(trip_info: dict, departure_time: datetime) -> tuple[pd.DataFrame, bool]:
    """Build one feature row matching notebook 07's training schema exactly.

    trip_info must supply route_id, stop_id, stop_sequence (the destination
    stop being predicted for), and either `mode` directly or `route_type`
    (GTFS int, mapped via MODE_BY_ROUTE_TYPE). `departure_time` is the
    scheduled clock time hour_of_day/day_of_week/is_weekend/is_peak are
    derived from — notebook 07 re-derives these from scheduled_arrival_time,
    not capture time, so this should be the trip's scheduled time, not
    wall-clock "now".

    Returns (X, stop_id_is_oov). stop_id_is_oov is True when trip_info['stop_id']
    isn't in the trained stop_id vocabulary -- pd.Categorical() below silently
    encodes an out-of-vocabulary stop_id as NaN, which XGBoost still produces a
    prediction for (via its learned default split direction) rather than
    erroring, so callers need this signal to flag the result as lower-confidence
    instead of it looking like a normal, fully-informed prediction.
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

    stop_id_is_oov = trip_info['stop_id'] not in categories['stop_id']

    for c in CATEGORICAL_COLS:
        X[c] = pd.Categorical(X[c], categories=categories[c])
    X['hour_of_day'] = X['hour_of_day'].astype('int32')
    X['is_weekend'] = X['is_weekend'].astype('int8')
    X['is_peak'] = X['is_peak'].astype('int8')
    # float32 to match the training-time downcast (_load_training_frames /
    # X_train['stop_sequence']) -- a model fit on float32 can behave
    # inconsistently at inference time if fed float64 columns.
    X['stop_sequence'] = X['stop_sequence'].astype('float32')

    return X[FEATURE_COLS], stop_id_is_oov


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

    X, stop_id_is_oov = build_features(trip_info, departure_time)
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

    # Override, not a replacement: an out-of-vocabulary stop_id encodes as NaN
    # in build_features() and still produces a real prediction from XGBoost's
    # learned default split direction, but the route+mode coverage_counts
    # heuristic above has no way to know that happened -- without this, a
    # well-represented route with an OOV destination stop would report the
    # same confidence as a fully in-vocabulary prediction.
    if stop_id_is_oov:
        confidence = 'Low'

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
