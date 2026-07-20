"""SEQ Transit AI — Streamlit POC front end.

Wires the Phase 3 data layer (gtfs_data), live GTFS-RT feed (live_gtfs), and
the v0 delay-prediction model (prediction) into a single "From -> To -> when"
trip search. No business logic lives here — presentation + wiring only.
"""
from __future__ import annotations

from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

import streamlit as st
from streamlit_searchbox import st_searchbox

import gtfs_data
import live_gtfs
import prediction

st.set_page_config(page_title='SEQ Transit AI', page_icon='🚊', layout='centered')

BRISBANE_TZ = ZoneInfo('Australia/Brisbane')

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


def _predict_leg(trip: dict, dest_stop_ids: list[str], search_departure_after: datetime, updates: dict):
    """Enrich + predict_delay() one leg's trip. Returns (trip, pred, raw_update), or None on failure."""
    try:
        enriched = prediction.enrich_trip_with_dest_stop(trip, dest_stop_ids)
        raw_update = updates.get(enriched['trip_id'])
        live_delay = None
        if raw_update is not None:
            live_delay = {
                'delay_minutes': raw_update['delay_seconds'] / 60.0,
                'timestamp': raw_update['timestamp'],
                'stop_id': raw_update['stop_id'],
            }
        pred = prediction.predict_delay(enriched, search_departure_after, live_delay=live_delay)
    except Exception as e:
        st.warning(f'Could not predict this leg ({trip.get("trip_id")}): {e}')
        return None
    return trip, pred, raw_update


def render_trip_card(trip: dict, pred: dict, raw_update: dict | None, stop_names,
                      label_prefix: tuple[str, str] = ('From', 'To')) -> None:
    """Render one leg's prediction card: leave-by banner, route badge,
    from/to (or board/alight) stops, delay badge, departure/arrival metrics,
    live tracking caption, confidence, summary.

    `label_prefix` distinguishes a direct trip ('From'/'To') from a transfer
    journey leg ('Board'/'Alight') — same card layout either way.
    """
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

    st.write(f"{label_prefix[0]}: {trip['origin_stop_name']} → {label_prefix[1]}: {trip['dest_stop_name']}")

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


def render_direct_trips(trips: list[dict], dest_stop_ids: list[str], departure_after: datetime,
                         updates: dict, stop_names) -> None:
    if any(t.get('fallback_schedule') for t in trips):
        st.info('Schedule based on projected timetable — times may vary')

    with st.spinner('Generating predictions...'):
        predicted = []
        for trip in trips[:5]:
            result = _predict_leg(trip, dest_stop_ids, departure_after, updates)
            if result is not None:
                predicted.append(result)

    for trip, pred, raw_update in predicted:
        with st.container(border=True):
            render_trip_card(trip, pred, raw_update, stop_names)


def render_transfer_journeys(journeys: list[dict], updates: dict, stop_names) -> None:
    with st.spinner('Generating predictions...'):
        for idx, journey in enumerate(journeys, start=1):
            with st.container(border=True):
                n = journey['num_transfers']
                st.markdown(f"### Journey {idx} ({n} transfer{'' if n == 1 else 's'})")
                st.caption(f"Total: ~{journey['total_minutes']} min")

                for leg_idx, leg in enumerate(journey['legs']):
                    st.divider()
                    st.markdown(f'**Leg {leg_idx + 1}**')
                    trip = leg['trip']

                    result = _predict_leg(trip, leg['dest_stop_ids'], leg['search_departure_after'], updates)
                    if result is None:
                        continue
                    _, pred, raw_update = result
                    render_trip_card(trip, pred, raw_update, stop_names, label_prefix=('Board', 'Alight'))

                    if leg_idx < len(journey['transfer_points']):
                        tp = journey['transfer_points'][leg_idx]
                        st.divider()
                        st.markdown(f"🔄 **Transfer at {tp['stop_name']}** — {tp['connection_minutes']} min connection")


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
    departure_mode = st.segmented_control(
        'Departure', ['Now', 'Later', 'Custom'], default='Now',
    )
    # segmented_control returns None if the user clicks the selected pill
    # again (deselecting it) — fall back to the "Now" default rather than
    # leaving departure_mode unset.
    departure_mode = departure_mode or 'Now'

travel_time = None
if departure_mode == 'Custom':
    selected_time_label = st.selectbox('Departure time', time_labels, index=default_time_index)
    travel_time = label_to_time[selected_time_label]

search_clicked = st.button('Search', type='primary', use_container_width=True)

# ── Results ────────────────────────────────────────────────────────────────

if search_clicked:
    if origin is None or dest is None:
        st.warning('Pick both a "From" and a "To" stop first.')
    else:
        if departure_mode == 'Now':
            now_at_click = datetime.now(BRISBANE_TZ)
            departure_after = datetime.combine(now_at_click.date(), now_at_click.time())
        elif departure_mode == 'Later':
            later_at_click = datetime.now(BRISBANE_TZ) + timedelta(minutes=30)
            departure_after = datetime.combine(later_at_click.date(), later_at_click.time())
        else:
            departure_after = datetime.combine(travel_date, travel_time)
        window_minutes = 60
        with st.spinner('Searching for trips...'):
            trips = gtfs_data.find_trips(origin['stop_ids'], dest['stop_ids'], departure_after, window_minutes=window_minutes)
            transfer_journeys = []
            if not trips:
                transfer_journeys = gtfs_data.find_multi_leg_trips(
                    origin['stop_ids'], dest['stop_ids'], departure_after, window_minutes=window_minutes,
                )
        st.session_state['results'] = {
            'trips': trips,
            'transfer_journeys': transfer_journeys,
            'dest_stop_ids': dest['stop_ids'],
            'departure_after': departure_after,
            'window_minutes': window_minutes,
        }

results = st.session_state.get('results')
if results:
    trips = results['trips']
    transfer_journeys = results.get('transfer_journeys', [])
    st.divider()

    if not trips and not transfer_journeys:
        st.info(
            f'No services found between these stops within the next {results["window_minutes"]} minutes, '
            'even with transfers. Try a different time or check nearby stops.'
        )
    else:
        updates, live_error = get_live_updates()
        if live_error:
            st.warning('Live GTFS-RT feed is currently unavailable — showing model predictions only.')

        stop_names = data.stops.set_index('stop_id')['stop_name']

        if trips:
            if transfer_journeys:
                st.subheader('Direct services')
            render_direct_trips(trips, results['dest_stop_ids'], results['departure_after'], updates, stop_names)

        if transfer_journeys:
            if trips:
                st.subheader('Services with transfers')
            render_transfer_journeys(transfer_journeys, updates, stop_names)

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
