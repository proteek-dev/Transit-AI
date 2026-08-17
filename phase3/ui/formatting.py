"""Pure data-to-display-string helpers for the Phase 3 Streamlit app.

No Streamlit calls here -- every function just takes data and returns a
string (or, for the departure-time slot list, plain time/int values feeding
a display). route_badge()/route_label_plain() and the ROUTE_TYPE_MODE map
are also the shared "how do we label this route_type" vocabulary ui/pickers.py
uses for its map-picker legend, so both live here rather than in cards.py to
avoid one importing the other for a single dict.
"""
from __future__ import annotations

from datetime import datetime, time

CONFIDENCE_COLOR = {'High': '#3AA65B', 'Medium': '#E0A526', 'Low': '#8A8A8A'}

# GTFS route_type -> (emoji, human-readable mode label), per the GTFS spec's
# extended route types 0-4 (SEQ doesn't run metro, but route_type 1 is handled).
ROUTE_TYPE_MODE = {
    0: ('🚊', 'Tram'),
    1: ('🚇', 'Metro'),
    2: ('🚆', 'Train'),
    3: ('🚌', 'Bus'),
    4: ('⛴', 'Ferry'),
}
DEFAULT_ROUTE_TYPE_MODE = ('🚍', 'Transit')


def route_badge(trip: dict) -> str:
    """'[emoji] [mode] [route_short_name] towards [trip_headsign]'.

    Falls back to route_long_name if trip_headsign is missing, and drops the
    "towards ..." suffix entirely if both are missing.
    """
    emoji, mode_label = ROUTE_TYPE_MODE.get(trip.get('route_type'), DEFAULT_ROUTE_TYPE_MODE)
    route_name = trip.get('route_short_name') or trip['route_id']
    direction = trip.get('trip_headsign') or trip.get('route_long_name')

    label = f'{emoji} {mode_label} {route_name}'
    if direction:
        label += f' towards {direction}'
    return label


def route_label_plain(trip: dict) -> str:
    """'[mode] [route_short_name] towards [trip_headsign]', no emoji — used
    in expander labels that already carry their own leading emoji.
    """
    _, mode_label = ROUTE_TYPE_MODE.get(trip.get('route_type'), DEFAULT_ROUTE_TYPE_MODE)
    route_name = trip.get('route_short_name') or trip['route_id']
    direction = trip.get('trip_headsign') or trip.get('route_long_name')

    label = f'{mode_label} {route_name}'
    if direction:
        label += f' towards {direction}'
    return label


def format_distance(distance_km: float) -> str:
    """'180m away' below 1km (rounded to the nearest 10m), else '2.3km away'
    (one decimal place). Callers are responsible for only calling this when
    a distance_km value actually exists on a candidate -- this never
    computes or estimates a distance itself.
    """
    if distance_km < 1.0:
        meters = round(distance_km * 1000 / 10) * 10
        return f'{meters:.0f}m away'
    return f'{distance_km:.1f}km away'


def delay_color(minutes: float) -> str:
    if minutes > 5:
        return '#D64545'  # red
    if minutes >= 2:
        return '#E0A526'  # amber
    return '#3AA65B'  # green


def badge_html(text: str, color: str) -> str:
    return (
        f'<span style="background:{color}22;color:{color};padding:2px 10px;'
        f'border-radius:12px;font-weight:600;font-size:0.85em;">{text}</span>'
    )


def _time_slot_options() -> list[time]:
    """15-minute time slots spanning a full day, as time objects."""
    return [time(hour=h, minute=m) for h in range(24) for m in (0, 15, 30, 45)]


def _format_time_ampm_short(t: time) -> str:
    """'h:MM AM/PM' without a leading zero on the hour, e.g. '10:15 PM'."""
    hour_12 = t.hour % 12 or 12
    period = 'AM' if t.hour < 12 else 'PM'
    return f'{hour_12}:{t.minute:02d} {period}'


def _closest_slot_index(slots: list[time], target: time) -> int:
    """Index of the slot with the smallest minutes-of-day distance to target."""
    target_minutes = target.hour * 60 + target.minute
    diffs = [abs((s.hour * 60 + s.minute) - target_minutes) for s in slots]
    return diffs.index(min(diffs))


def _rotate_slots(slots: list[time], start_idx: int) -> list[time]:
    """Rotate a chronological slot list so it starts at start_idx and wraps
    back around to itself -- every slot is still present, just reordered so
    the closest-to-now slot leads instead of midnight.
    """
    return slots[start_idx:] + slots[:start_idx]


def training_window_caption(training_metadata: dict) -> str:
    """Footer caption text describing the training data window, threaded
    from training_metadata.json's data_window_start/end/days (only present
    after a full retrain -- DEBUG_MAX_DATES runs never reach save, so this
    can't be populated by a partial-data run). Falls back to just the
    trained-at date for older training_metadata.json files predating these
    fields, never a hardcoded day count that could silently go stale.
    """
    trained_at = datetime.fromisoformat(training_metadata['trained_at']).strftime('%d %b %Y')

    data_window_start = training_metadata.get('data_window_start')
    data_window_end = training_metadata.get('data_window_end')
    data_window_days = training_metadata.get('data_window_days')

    if data_window_start and data_window_end and data_window_days:
        return (
            f'Predictions based on {data_window_days} days of historical data '
            f'({data_window_start} to {data_window_end}). Model last trained {trained_at}.'
        )
    return f'Model last trained {trained_at}.'
