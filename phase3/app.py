"""SEQ Transit AI — Streamlit POC front end.

Wires the Phase 3 data layer (gtfs_data), live GTFS-RT feed (live_gtfs), and
the v0 delay-prediction model (prediction) into a single "From -> To -> when"
trip search. No business logic lives here — presentation + wiring only.
"""
from __future__ import annotations

from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

import streamlit as st
from streamlit_geolocation import streamlit_geolocation

import gtfs_data
import live_gtfs
import map_picker
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

# Mode-filter chip options -> GTFS route_type. 'All' has no entry (means "no
# filter"). Bus/Train/Tram only, matching what SEQ actually runs -- ferry is
# excluded everywhere upstream (gtfs_data.py) and never surfaced here either.
MODE_FILTER_ROUTE_TYPE = {'Bus': 3, 'Train': 2, 'Tram': 0}

# Route-map polyline colors, one per leg, cycled if ever exceeded -- in
# practice never is, since find_multi_leg_trips()'s max_transfers=3 caps a
# journey at 4 legs.
ROUTE_MAP_LEG_COLORS = ['#2563eb', '#dc2626', '#16a34a', '#d97706']

# Final number of ranked results shown. CANDIDATE_POOL_SIZE widens each path
# (direct trips, transfer journeys) beyond MAX_RESULTS before they're merged
# and ranked by predicted arrival -- otherwise a candidate that wasn't in
# either path's own top-MAX_RESULTS-by-schedule could never surface in the
# combined ranking even if its predicted arrival beats one that was.
MAX_RESULTS = 5
CANDIDATE_POOL_SIZE = 10


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


def _attach_route_types(stops: list[dict]) -> list[dict]:
    """Enrich search_stops()-shaped dicts with route_types, so typed-search
    results can be color-coded by mode_picker the same way nearest_stops()
    candidates are.
    """
    data = get_gtfs_data()
    route_type_by_route = data.routes.set_index('route_id')['route_type'].to_dict()
    enriched = []
    for s in stops:
        route_types = set()
        for sid in s['stop_ids']:
            for r in data.stop_to_routes.get(sid, set()):
                rt = route_type_by_route.get(r)
                if rt is not None:
                    route_types.add(rt)
        enriched.append({**s, 'route_types': sorted(route_types)})
    return enriched


def _filter_by_mode(candidates: list[dict], mode_filter: str) -> list[dict]:
    """UI-level candidate filter for the mode chip above the From/To
    pickers -- doesn't touch gtfs_data.py's own filtering/BFS logic, just
    trims the list handed to map_picker.render_stop_picker(). 'All' (or a
    candidate with no route_types) passes through unfiltered.
    """
    if mode_filter == 'All':
        return candidates
    target = MODE_FILTER_ROUTE_TYPE[mode_filter]
    return [c for c in candidates if target in (c.get('route_types') or [])]


# Fields every confirmed origin/destination dict is guaranteed to carry,
# regardless of which picker path produced it -- render_from_picker()'s
# geolocation branch (gtfs_data.nearest_stops()) also attaches
# distance_km/trip_count, which the typed-search path (gtfs_data.search_stops()
# + _attach_route_types()) never has. Swapping through this common subset
# means the swap behaves the same no matter how each side was originally picked.
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
    vs typed search). Clears the current results/selected journey since they
    were computed for the pre-swap direction -- the user re-runs Search for
    the reversed trip.
    """
    origin = _normalize_confirmed_stop(st.session_state.get('origin_confirmed'))
    dest = _normalize_confirmed_stop(st.session_state.get('dest_confirmed'))
    st.session_state['origin_confirmed'] = dest
    st.session_state['dest_confirmed'] = origin
    st.session_state['results'] = None
    st.session_state['selected_journey'] = None


def _build_route_map_legs(journey: dict) -> list[dict]:
    """journey['legs'] -> map_picker.render_route_map()'s leg-dict shape:
    each leg's GTFS-shape points (None if the trip has no shape_id -- a
    valid GTFS state map_picker skips defensively), a color cycled from
    ROUTE_MAP_LEG_COLORS, and a route_badge label.
    """
    legs = []
    for i, leg in enumerate(journey['legs']):
        trip = leg['trip']
        points = gtfs_data.get_trip_shape_points(trip['trip_id'], trip['origin_stop_id'], trip['dest_stop_id'])
        legs.append({
            'points': points,
            'color': ROUTE_MAP_LEG_COLORS[i % len(ROUTE_MAP_LEG_COLORS)],
            'label': route_badge(trip),
        })
    return legs


def render_from_picker(mode_filter: str = 'All') -> dict | None:
    """The 'From' field: 'Use my location' + map picker, with a typed-search
    fallback rendered through the same map picker. Returns the confirmed stop
    dict (has stop_id/stop_ids/stop_name/stop_lat/stop_lon) once the user has
    tapped a candidate, or None beforehand. Confirms into
    st.session_state['origin_confirmed'] and offers a "Change origin" reset.

    `mode_filter` ('All'/'Bus'/'Train'/'Tram') trims the candidate list
    handed to the map picker -- session_state['origin_candidates'] itself
    stays the full, unfiltered result, so switching the filter doesn't
    require a fresh geolocation fetch.
    """
    confirmed = st.session_state.get('origin_confirmed')
    if confirmed:
        st.success(f"From: {confirmed['stop_name']} ✓")
        if st.button('Change origin', key='origin_change_btn'):
            st.session_state['origin_confirmed'] = None
            st.session_state['origin_candidates'] = None
            st.session_state.pop('origin_center', None)
            st.rerun()
        return confirmed

    st.caption('Use my location')
    location = streamlit_geolocation()
    has_location = location and location.get('latitude') is not None and location.get('longitude') is not None

    if has_location:
        lat, lon = location['latitude'], location['longitude']
        if st.session_state.get('origin_center') != (lat, lon):
            st.session_state['origin_center'] = (lat, lon)
            st.session_state['origin_candidates'] = gtfs_data.nearest_stops(lat, lon, limit=15)

        candidates = _filter_by_mode(st.session_state.get('origin_candidates') or [], mode_filter)
        if not candidates:
            st.info('No nearby stops found — search for your stop instead.')
        else:
            picked_id = map_picker.render_stop_picker(
                candidates, lat, lon, key='origin_location_map',
                mode_map=ROUTE_TYPE_MODE, default_mode=DEFAULT_ROUTE_TYPE_MODE,
            )
            if picked_id:
                chosen = next((c for c in candidates if c['stop_id'] == picked_id), None)
                if chosen:
                    st.session_state['origin_confirmed'] = chosen
                    st.rerun()
    else:
        st.info('Location unavailable — search for your stop instead.')
        query = st.text_input(
            'Search for your stop',
            placeholder='Type a stop name...',
            key='origin_typed_query',
        )
        if query and len(query.strip()) >= 2:
            matches = gtfs_data.search_stops(query, limit=15)
            candidates = _filter_by_mode(_attach_route_types(matches), mode_filter)
            print(f"[render_from_picker] typed query={query!r} mode_filter={mode_filter!r} -> "
                  f"{len(candidates)} candidates: {[c['stop_name'] for c in candidates]}")
            if not candidates:
                st.info('No matching stops found.')
            else:
                center = candidates[0]
                picked_id = map_picker.render_stop_picker(
                    candidates, center['stop_lat'], center['stop_lon'], key='origin_typed_map',
                    mode_map=ROUTE_TYPE_MODE, default_mode=DEFAULT_ROUTE_TYPE_MODE,
                )
                if picked_id:
                    chosen = next((c for c in candidates if c['stop_id'] == picked_id), None)
                    if chosen:
                        st.session_state['origin_confirmed'] = chosen
                        st.rerun()

    return None


def render_to_picker(origin_confirmed: dict, mode_filter: str = 'All') -> dict | None:
    """The 'To' field: typed search producing destination candidates, shown
    on the SAME map as the already-confirmed origin (rendered as a locked,
    non-tappable pin for context). Returns the confirmed destination dict
    (same shape render_from_picker() returns) once the user has tapped a
    candidate, or None beforehand. Confirms into
    st.session_state['dest_confirmed'] and offers a "Change destination"
    reset. Click resolution goes through the same tooltip-lookup mechanism
    as the origin picker -- no lat/lng matching.

    `mode_filter` ('All'/'Bus'/'Train'/'Tram') trims the candidate list the
    same way render_from_picker() does.
    """
    confirmed = st.session_state.get('dest_confirmed')
    if confirmed:
        st.success(f"To: {confirmed['stop_name']} ✓")
        if st.button('Change destination', key='dest_change_btn'):
            st.session_state['dest_confirmed'] = None
            st.rerun()
        return confirmed

    query = st.text_input(
        'Search for your destination',
        placeholder='Type a stop name...',
        key='dest_typed_query',
    )
    if query and len(query.strip()) >= 2:
        matches = gtfs_data.search_stops(query, limit=15)
        candidates = _filter_by_mode(_attach_route_types(matches), mode_filter)
        print(f"[render_to_picker] typed query={query!r} mode_filter={mode_filter!r} -> "
              f"{len(candidates)} candidates: {[c['stop_name'] for c in candidates]}")
        if not candidates:
            st.info('No matching stops found.')
        else:
            picked_id = map_picker.render_stop_picker(
                candidates, origin_confirmed['stop_lat'], origin_confirmed['stop_lon'], key='dest_typed_map',
                mode_map=ROUTE_TYPE_MODE, default_mode=DEFAULT_ROUTE_TYPE_MODE,
                locked_marker=origin_confirmed,
            )
            if picked_id:
                chosen = next((c for c in candidates if c['stop_id'] == picked_id), None)
                if chosen:
                    st.session_state['dest_confirmed'] = chosen
                    st.rerun()

    return None


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


def render_leave_by_banner(pred: dict) -> None:
    st.markdown(
        f'<div style="background:#2563eb18;border-radius:10px;padding:10px 16px;'
        f'margin-bottom:10px;">'
        f'<span style="font-size:1.4em;font-weight:700;color:#2563eb;">'
        f'🕒 Leave by {pred["leave_by"]}</span></div>',
        unsafe_allow_html=True,
    )


def render_card_detail(trip: dict, pred: dict, raw_update: dict | None, stop_names,
                        label_prefix: tuple[str, str] = ('From', 'To')) -> None:
    """Route badge, from/to (or board/alight) stops, delay badge, live
    tracking caption, confidence, and departure/arrival — the detail
    revealed once a card's expander is opened.

    `label_prefix` distinguishes a direct trip ('From'/'To') from a transfer
    journey leg ('Board'/'Alight') — same layout either way.
    """
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

    st.caption(
        f"Departs {prediction.format_time_ampm(trip['origin_departure_time'])} "
        f"→ Arrives {pred['estimated_arrival']}"
    )


def render_trip_card(trip: dict, pred: dict, raw_update: dict | None, stop_names,
                      dest_stop_ids: list[str], departure_after: datetime,
                      label_prefix: tuple[str, str] = ('From', 'To'), expanded: bool = False) -> None:
    """Render one leg's prediction card. The leave-by banner and plain
    English summary are always visible; everything else (route badge,
    from/to, live tracking, confidence, departure/arrival) lives inside an
    expander so collapsed cards stay compact.

    `expanded` controls the expander's initial state — True only for the
    first card in a results list.

    `dest_stop_ids`/`departure_after` are only needed to wrap this trip into
    the same one-leg-journey shape find_multi_leg_trips() produces, for the
    "Show route" button below.
    """
    render_leave_by_banner(pred)
    st.write(pred['summary'])

    if st.button('🗺️ Show route', key=f"show_route_{trip['trip_id']}"):
        st.session_state['selected_journey'] = gtfs_data._direct_journey(trip, dest_stop_ids, departure_after)
        st.rerun()

    label = f'🕐 Leave by {pred["leave_by"]} — {route_label_plain(trip)}'
    with st.expander(label, expanded=expanded):
        render_card_detail(trip, pred, raw_update, stop_names, label_prefix)


def render_transfer_journey_card(journey: dict, leg_predictions: list, stop_names,
                                  idx: int, expanded: bool) -> None:
    """Render one transfer journey (num_transfers >= 1): leave-by banner +
    journey summary always visible; per-leg detail and transfer connections
    live inside one expander. `leg_predictions` must already be computed
    (see _predict_journey_legs) -- no prediction happens here.
    """
    first_result = next((r for r in leg_predictions if r is not None), None)
    if first_result is None:
        return
    first_trip, first_pred, _ = first_result

    n = journey['num_transfers']
    transfer_note = f"{n} transfer{'' if n == 1 else 's'}, ~{journey['total_minutes']} min total"

    with st.container(border=True):
        render_leave_by_banner(first_pred)
        st.write(f"{first_pred['summary']} ({transfer_note}.)")

        if st.button('🗺️ Show route', key=f'show_route_journey_{idx}'):
            st.session_state['selected_journey'] = journey
            st.rerun()

        label = f'🕐 Leave by {first_pred["leave_by"]} — {route_label_plain(first_trip)} ({transfer_note})'
        with st.expander(label, expanded=expanded):
            for leg_idx, result in enumerate(leg_predictions):
                if result is None:
                    continue
                trip, pred, raw_update = result
                st.markdown(f'**Leg {leg_idx + 1}**')
                render_card_detail(trip, pred, raw_update, stop_names, label_prefix=('Board', 'Alight'))

                if leg_idx < len(journey['transfer_points']):
                    tp = journey['transfer_points'][leg_idx]
                    st.divider()
                    st.markdown(
                        f"🔄 **Transfer at {tp['stop_name']}** — {tp['connection_minutes']} min connection"
                    )
                    st.divider()


def render_ranked_journey(idx: int, journey: dict, leg_predictions: list, stop_names,
                           dest_stop_ids: list[str], departure_after: datetime) -> None:
    """Render one already-ranked, already-predicted candidate -- a direct
    trip (num_transfers == 0) as a single card, a transfer journey
    (num_transfers >= 1) as a multi-leg card. Ranking/truncation/prediction
    all already happened before this is called; this is display only.
    """
    if journey['num_transfers'] == 0:
        trip, pred, raw_update = leg_predictions[0]
        with st.container(border=True):
            render_trip_card(
                trip, pred, raw_update, stop_names, dest_stop_ids, departure_after,
                expanded=(idx == 0),
            )
    else:
        render_transfer_journey_card(journey, leg_predictions, stop_names, idx, expanded=(idx == 0))


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
        # Previous selection may not correspond to this new result set --
        # cleared here so the block below re-defaults to the first result.
        st.session_state['selected_journey'] = None

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

            # Previous selection may not correspond to this new result set --
            # default to the top-ranked candidate.
            if st.session_state.get('selected_journey') is None:
                st.session_state['selected_journey'] = display_candidates[0][1]

            selected_journey = st.session_state.get('selected_journey')
            origin_marker = st.session_state.get('origin_confirmed')
            dest_marker = st.session_state.get('dest_confirmed')
            if selected_journey and origin_marker and dest_marker:
                route_legs = _build_route_map_legs(selected_journey)
                print(
                    f'[route map] rendering {len(route_legs)} leg(s): '
                    f"{[(l['label'], len(l['points']) if l['points'] else 0) for l in route_legs]}"
                )
                map_picker.render_route_map(route_legs, origin_marker, dest_marker, key='route_map')

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

st.caption(
    'Predictions based on ~21 days of historical data. '
    'Model: XGBoost v0 baseline. Live data from TransLink GTFS-RT feeds.'
)
