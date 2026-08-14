"""SEQ Transit AI — Streamlit POC front end.

Wires the Phase 3 data layer (gtfs_data), live GTFS-RT feed (live_gtfs), and
the v0 delay-prediction model (prediction) into a single "From -> To -> when"
trip search. No business logic lives here — presentation + wiring only.
Orchestrator only: the From/To pickers live in ui/pickers.py, result-card
rendering in ui/cards.py, and pure display-formatting helpers in
ui/formatting.py.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import streamlit as st

import gtfs_data
import live_gtfs
import prediction
from ui.cards import render_ranked_journey
from ui.formatting import (
    _closest_slot_index,
    _format_time_ampm_short,
    _rotate_slots,
    _time_slot_options,
    training_window_caption,
)
from ui.pickers import get_gtfs_data, render_from_picker, render_to_picker

st.set_page_config(page_title='SEQ Transit AI', page_icon='🚊', layout='centered')

BRISBANE_TZ = ZoneInfo('Australia/Brisbane')

# Final number of ranked results shown. CANDIDATE_POOL_SIZE widens each path
# (direct trips, transfer journeys) beyond MAX_RESULTS before they're merged
# and ranked by predicted arrival -- otherwise a candidate that wasn't in
# either path's own top-MAX_RESULTS-by-schedule could never surface in the
# combined ranking even if its predicted arrival beats one that was.
MAX_RESULTS = 5
CANDIDATE_POOL_SIZE = 10


# ── Cached loaders ──────────────────────────────────────────────────────────
# get_gtfs_data() itself lives in ui/pickers.py -- imported above -- since its
# own picker helpers need it and app.py can't be imported back into ui/pickers.py.

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


# ── Session-state helpers (From/To swap) ────────────────────────────────────

# Fields every confirmed origin/destination dict is guaranteed to carry,
# regardless of which picker path produced it -- render_from_picker()'s
# geolocation branch (gtfs_data.nearest_stops()) also attaches
# distance_km/trip_count, which the typed-search path (gtfs_data.search_stops()
# + ui/pickers._attach_route_types()) never has. Swapping through this common
# subset means the swap behaves the same no matter how each side was originally picked.
_CONFIRMED_STOP_FIELDS = ('stop_id', 'stop_ids', 'stop_name', 'stop_lat', 'stop_lon', 'route_types')


def _normalize_confirmed_stop(stop: dict | None) -> dict | None:
    """Reduce a confirmed origin/destination dict to the fields the rest of
    the app actually relies on (map markers, route legs, card labels),
    dropping selection-path-specific extras.
    """
    if stop is None:
        return None
    return {field: stop[field] for field in _CONFIRMED_STOP_FIELDS if field in stop}


def _swap_origin_dest() -> None:
    """Swap the confirmed origin/destination stops in place. Both sides are
    normalized first (see _normalize_confirmed_stop) so the swap is
    well-defined regardless of how each stop was originally selected (map tap
    vs typed search). Clears the current results since they were computed for
    the pre-swap direction -- the user re-runs Search for the reversed trip.
    """
    origin = _normalize_confirmed_stop(st.session_state.get('origin_confirmed'))
    dest = _normalize_confirmed_stop(st.session_state.get('dest_confirmed'))
    st.session_state['origin_confirmed'] = dest
    st.session_state['dest_confirmed'] = origin
    st.session_state['results'] = None


# ── Prediction wiring ────────────────────────────────────────────────────────

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


def _predict_journey_legs(journey: dict, updates: dict) -> list:
    """Predict every leg of a journey (direct or transfer -- both share the
    same journey shape via gtfs_data._direct_journey()/find_multi_leg_trips()).
    Returns a list aligned 1:1 with journey['legs']: each entry is the
    (trip, pred, raw_update) tuple _predict_leg() returns, or None if that
    leg's prediction failed.
    """
    return [
        _predict_leg(leg['trip'], leg['dest_stop_ids'], leg['search_departure_after'], updates)
        for leg in journey['legs']
    ]


def _predicted_arrival(leg_predictions: list) -> datetime | None:
    """The predicted arrival datetime at the journey's final destination --
    the last leg's scheduled arrival plus its predicted/blended delay. This
    (not scheduled time, not transfer count, not confidence) is what
    candidates are ranked by. None if the last leg's prediction failed --
    such a candidate can't be ranked and is dropped rather than guessed at.
    """
    last = leg_predictions[-1]
    if last is None:
        return None
    trip, pred, _ = last
    return trip['dest_arrival_time'] + timedelta(minutes=pred['blended_delay_minutes'])


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

mode_filter = st.segmented_control(
    'Filter by mode', ['All', 'Bus', 'Train', 'Tram'], default='All', key='mode_filter',
)
# segmented_control returns None if the user clicks the selected pill again
# (deselecting it) -- fall back to 'All' rather than leaving it unset, same
# pattern as departure_mode below.
mode_filter = mode_filter or 'All'

_swap_spacer_l, swap_col, _swap_spacer_r = st.columns([4, 1, 4])
with swap_col:
    can_swap = bool(st.session_state.get('origin_confirmed') or st.session_state.get('dest_confirmed'))
    if st.button('⇄', key='swap_origin_dest_btn', help='Swap From and To',
                 use_container_width=True, disabled=not can_swap):
        _swap_origin_dest()
        st.rerun()

col_from, col_to = st.columns(2)
with col_from:
    st.subheader('📍 From')
    origin = render_from_picker(mode_filter)

with col_to:
    st.subheader('🎯 To')
    if origin:
        dest = render_to_picker(origin, mode_filter)
    else:
        st.info('Set your origin first.')
        dest = None

now_brisbane = datetime.now(BRISBANE_TZ)

time_slots = _time_slot_options()
default_time_index = _closest_slot_index(time_slots, now_brisbane.time())
rotated_time_slots = _rotate_slots(time_slots, default_time_index)
label_to_time = {_format_time_ampm_short(t): t for t in rotated_time_slots}
time_labels = list(label_to_time.keys())

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
    selected_time_label = st.selectbox('Departure time', time_labels, index=0)
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
            # Always runs now, direct or not -- a direct trip existing is no
            # longer a reason to skip transfer alternatives that might
            # actually have a sooner predicted arrival (see the merged
            # ranking below). max_results widened to CANDIDATE_POOL_SIZE so
            # prediction (and ranking) sees more than the final display count
            # from this path too.
            transfer_journeys = gtfs_data.find_multi_leg_trips(
                origin['stop_ids'], dest['stop_ids'], departure_after, window_minutes=window_minutes,
                max_results=CANDIDATE_POOL_SIZE,
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
    dest_stop_ids = results['dest_stop_ids']
    departure_after = results['departure_after']
    st.divider()

    # Wrap direct trips into the same journey shape find_multi_leg_trips()
    # uses, so both paths can be predicted/ranked/rendered uniformly.
    # transfer_journeys is filtered to num_transfers >= 1: find_multi_leg_trips()
    # runs its own depth-0 direct check internally too (now that the old
    # `if not trips:` gate is gone, that branch is live), which would
    # otherwise duplicate every entry already covered by `trips`.
    direct_journeys = [
        gtfs_data._direct_journey(trip, dest_stop_ids, departure_after)
        for trip in trips[:CANDIDATE_POOL_SIZE]
    ]
    transfer_only_journeys = [j for j in transfer_journeys if j['num_transfers'] >= 1]
    candidate_journeys = direct_journeys + transfer_only_journeys

    if not candidate_journeys:
        st.info(
            f'No services found between these stops within the next {results["window_minutes"]} minutes, '
            'even with transfers. Try a different time or check nearby stops.'
        )
    else:
        updates, live_error = get_live_updates()
        if live_error:
            st.warning('Live GTFS-RT feed is currently unavailable — showing model predictions only.')

        stop_names = data.stops.set_index('stop_id')['stop_name']

        if any(t.get('fallback_schedule') for t in trips):
            st.info('Schedule based on projected timetable — times may vary')

        # Predict every candidate in the widened pool BEFORE truncating to
        # MAX_RESULTS, then rank purely by predicted arrival time -- not
        # scheduled time, not transfer count, and confidence is displayed
        # per-card but never factored into this sort.
        with st.spinner('Generating predictions...'):
            ranked = []
            for journey in candidate_journeys:
                leg_predictions = _predict_journey_legs(journey, updates)
                predicted_arrival = _predicted_arrival(leg_predictions)
                if predicted_arrival is not None:
                    ranked.append((predicted_arrival, journey, leg_predictions))

        ranked.sort(key=lambda r: r[0])
        display_candidates = ranked[:MAX_RESULTS]

        if not display_candidates:
            st.info('Found services, but none could be predicted right now. Try again shortly.')
        else:
            st.caption('Ranked by predicted arrival time.')

            for idx, (_predicted_arrival_dt, journey, leg_predictions) in enumerate(display_candidates):
                render_ranked_journey(idx, journey, leg_predictions, stop_names, dest_stop_ids, departure_after)

# ── Footer ──────────────────────────────────────────────────────────────

st.divider()
with st.expander('About this app'):
    st.write(
        "SEQ Transit AI is a proof-of-concept that blends TransLink's scheduled GTFS "
        'timetable with live GTFS-RT vehicle delay data and a baseline XGBoost model to '
        'estimate arrival times across South East Queensland public transport. It is a '
        'research prototype, not an official TransLink product.'
    )

training_metadata = prediction.get_training_metadata()
predictions_caption = training_window_caption(training_metadata)

st.caption(
    f'{predictions_caption} '
    'Model: XGBoost v0 baseline. Live data from TransLink GTFS-RT feeds.'
)
