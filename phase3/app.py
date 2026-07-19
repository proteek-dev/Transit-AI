"""SEQ Transit AI — Streamlit POC front end.

Wires the Phase 3 data layer (gtfs_data), live GTFS-RT feed (live_gtfs), and
the v0 delay-prediction model (prediction) into a single "From -> To -> when"
trip search. No business logic lives here — presentation + wiring only.
"""
from __future__ import annotations

from datetime import datetime, time
from zoneinfo import ZoneInfo

import streamlit as st
from streamlit_searchbox import st_searchbox

import gtfs_data
import live_gtfs
import prediction

st.set_page_config(page_title='SEQ Transit AI', page_icon='🚊', layout='centered')

BRISBANE_TZ = ZoneInfo('Australia/Brisbane')

MODE_ICON = {'tram': '🚊', 'rail': '🚆', 'bus': '🚌', 'ferry': '⛴️', 'unknown': '🚏'}
CONFIDENCE_COLOR = {'High': '#3AA65B', 'Medium': '#E0A526', 'Low': '#8A8A8A'}


# ── Cached loaders ──────────────────────────────────────────────────────────

@st.cache_resource(show_spinner='Loading GTFS schedule data from S3...')
def get_gtfs_data():
    return gtfs_data.load_gtfs_data()


@st.cache_resource(show_spinner='Loading prediction model (training on first run can take a few minutes)...')
def get_model():
    return prediction.load_model()


@st.cache_data(ttl=60, show_spinner=False)
def get_live_updates():
    """Fetch the live TripUpdates feed, cached 60s. Returns (updates, error)."""
    try:
        return live_gtfs.fetch_trip_updates(), None
    except Exception as e:
        return {}, str(e)


# ── Presentation helpers ─────────────────────────────────────────────────────

def route_badge(trip: dict) -> str:
    mode = prediction.MODE_BY_ROUTE_TYPE.get(trip.get('route_type'), 'unknown')
    icon = MODE_ICON.get(mode, '🚏')
    short_name = trip.get('route_short_name') or trip['route_id']
    if mode == 'bus':
        return f'{icon} Route {short_name} Bus'
    return f'{icon} {short_name} · {mode.title()}'


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


def _search_stops(query: str) -> list[tuple[str, dict]]:
    """search_function for st_searchbox: (display_label, stop_data) tuples."""
    if len(query.strip()) < 2:
        return []
    return [(m['stop_name'], m) for m in gtfs_data.search_stops(query, limit=10)]


def stop_picker(label: str, key_prefix: str) -> dict | None:
    """Live typeahead stop search. Returns the selected stop dict or None."""
    return st_searchbox(
        _search_stops,
        label=label,
        placeholder='Type a stop name...',
        key=f'{key_prefix}_searchbox',
    )


# ── Header ────────────────────────────────────────────────────────────────

st.title('SEQ Transit AI')
st.caption('Live delay predictions for South East Queensland')

# Fail gracefully up front if the data/model can't load at all.
try:
    data = get_gtfs_data()
except Exception as e:
    st.error(f'Could not load GTFS schedule data: {e}')
    st.stop()

try:
    model = get_model()
except Exception as e:
    st.error(f'Could not load the prediction model: {e}')
    st.stop()

# ── Input section ─────────────────────────────────────────────────────────

col1, col2 = st.columns(2)
with col1:
    origin = stop_picker('From', 'origin')
with col2:
    dest = stop_picker('To', 'dest')

now_brisbane = datetime.now(BRISBANE_TZ)

time_slots = _time_slot_options()
label_to_time = {_format_time_ampm_short(t): t for t in time_slots}
time_labels = list(label_to_time.keys())
default_time_index = _closest_slot_index(time_slots, now_brisbane.time())

col3, col4 = st.columns(2)
with col3:
    travel_date = st.date_input('Date', value=now_brisbane.date(), format='DD/MM/YYYY')
with col4:
    selected_time_label = st.selectbox('Departure time', time_labels, index=default_time_index)
    travel_time = label_to_time[selected_time_label]

search_clicked = st.button('Search', type='primary', use_container_width=True)

# ── Results ────────────────────────────────────────────────────────────────

if search_clicked:
    if origin is None or dest is None:
        st.warning('Pick both a "From" and a "To" stop first.')
    else:
        departure_after = datetime.combine(travel_date, travel_time)
        with st.spinner('Searching for trips...'):
            trips = gtfs_data.find_trips(origin['stop_ids'], dest['stop_ids'], departure_after, window_minutes=60)
        st.session_state['results'] = {
            'trips': trips,
            'dest_stop_ids': dest['stop_ids'],
            'departure_after': departure_after,
        }

results = st.session_state.get('results')
if results:
    trips = results['trips']
    st.divider()

    if not trips:
        st.info(
            'No direct services found between these stops in the next 60 minutes. '
            'Try a different time or check if a transfer is needed.'
        )
    else:
        updates, live_error = get_live_updates()
        if live_error:
            st.warning('Live GTFS-RT feed is currently unavailable — showing model predictions only.')

        if any(t.get('fallback_schedule') for t in trips):
            st.info('Schedule based on projected timetable — times may vary')

        stop_names = data.stops.set_index('stop_id')['stop_name']

        with st.spinner('Generating predictions...'):
            predicted_trips = []
            for trip in trips[:5]:
                try:
                    enriched = prediction.enrich_trip_with_dest_stop(trip, results['dest_stop_ids'])
                    raw_update = updates.get(enriched['trip_id'])
                    live_delay = None
                    if raw_update is not None:
                        live_delay = {
                            'delay_minutes': raw_update['delay_seconds'] / 60.0,
                            'timestamp': raw_update['timestamp'],
                            'stop_id': raw_update['stop_id'],
                        }
                    pred = prediction.predict_delay(enriched, results['departure_after'], live_delay=live_delay)
                except Exception as e:
                    st.warning(f'Could not predict this trip ({trip.get("trip_id")}): {e}')
                    continue
                predicted_trips.append((trip, pred, raw_update))

        for trip, pred, raw_update in predicted_trips:
            with st.container(border=True):
                st.markdown(
                    f'<div style="background:#2563eb18;border-radius:10px;padding:10px 16px;'
                    f'margin-bottom:10px;">'
                    f'<span style="font-size:1.4em;font-weight:700;color:#2563eb;">'
                    f'🕒 Leave by {pred["leave_by"]}</span></div>',
                    unsafe_allow_html=True,
                )

                head_col, delay_col = st.columns([3, 2])
                with head_col:
                    st.markdown(f'**{route_badge(trip)}**')
                with delay_col:
                    color = delay_color(pred['blended_delay_minutes'])
                    st.markdown(
                        badge_html(f'{pred["blended_delay_minutes"]:+.0f} min', color),
                        unsafe_allow_html=True,
                    )

                dep_col, arr_col = st.columns(2)
                with dep_col:
                    st.metric(
                        'Departure',
                        prediction.format_time_ampm(trip['origin_departure_time']),
                        help=trip['origin_stop_name'],
                    )
                with arr_col:
                    st.metric('Est. arrival', pred['estimated_arrival'], help=trip['dest_stop_name'])

                if raw_update is not None:
                    live_stop_name = stop_names.get(raw_update['stop_id'], raw_update['stop_id'])
                    st.caption(f'📡 Live: currently {raw_update["delay_seconds"] / 60:.0f} min late at {live_stop_name}')
                else:
                    st.caption('📡 No live tracking yet')

                conf = pred['confidence']
                st.markdown(
                    badge_html(f'Confidence: {conf}', CONFIDENCE_COLOR.get(conf, '#8A8A8A')),
                    unsafe_allow_html=True,
                )

                if trip.get('fallback_schedule'):
                    st.caption('📅 Schedule based on projected timetable — times may vary')

                st.write(pred['summary'])

# ── Footer ──────────────────────────────────────────────────────────────

st.divider()
with st.expander('About this app'):
    st.write(
        "SEQ Transit AI is a proof-of-concept that blends TransLink's scheduled GTFS "
        'timetable with live GTFS-RT vehicle delay data and a baseline XGBoost model to '
        'estimate arrival times across South East Queensland public transport. It is a '
        'research prototype, not an official TransLink product.'
    )

st.caption(
    'Predictions based on ~21 days of historical data. '
    'Model: XGBoost v0 baseline. Live data from TransLink GTFS-RT feeds.'
)
