"""From/To stop pickers for the Phase 3 Streamlit app.

get_gtfs_data() lives here (not app.py) because _attach_route_types() below
needs it and app.py can't be imported back into this module (it's the
Streamlit entry point -- importing it would re-run its whole top-level
script body). app.py imports get_gtfs_data() from here instead for its own
use, so the cached GTFS snapshot is still loaded exactly once per session
either way -- st.cache_resource keys on the function's own identity, not
which module calls it.
"""
from __future__ import annotations

import streamlit as st
from streamlit_geolocation import streamlit_geolocation

import gtfs_data
import map_picker
from ui.formatting import DEFAULT_ROUTE_TYPE_MODE, ROUTE_TYPE_MODE, format_distance

# Mode-filter chip options -> GTFS route_type. 'All' has no entry (means "no
# filter"). Bus/Train/Tram only, matching what SEQ actually runs -- ferry is
# excluded everywhere upstream (gtfs_data.py) and never surfaced here either.
MODE_FILTER_ROUTE_TYPE = {'Bus': 3, 'Train': 2, 'Tram': 0}


@st.cache_resource(show_spinner='Loading GTFS schedule data from S3...')
def get_gtfs_data():
    return gtfs_data.load_gtfs_data()


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
        distance_suffix = (
            f" ({format_distance(confirmed['distance_km'])})"
            if confirmed.get('distance_km') is not None else ''
        )
        st.success(f"From: {confirmed['stop_name']} ✓{distance_suffix}")
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
        distance_suffix = (
            f" ({format_distance(confirmed['distance_km'])})"
            if confirmed.get('distance_km') is not None else ''
        )
        st.success(f"To: {confirmed['stop_name']} ✓{distance_suffix}")
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
        matches = gtfs_data.search_stops(
            query, limit=15,
            ref_lat=origin_confirmed['stop_lat'], ref_lon=origin_confirmed['stop_lon'],
        )
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
