"""Result card rendering for the Phase 3 Streamlit app -- direct trips and
multi-leg transfer journeys, plus the per-card "🗺️ Show route" button.

render_ranked_journey() is the entry point app.py calls for each already-
ranked, already-predicted candidate; it dispatches to render_trip_card()
(direct) or render_transfer_journey_card() (transfer), both of which use
render_card_detail() for the shared per-leg detail layout. Each card's
"Show route" button no longer renders its own map -- there is exactly one
route map on the page (app.py, at the top), and clicking a card's button
just sets which journey it displays via st.session_state['selected_route_key'].
"""
from __future__ import annotations

from datetime import datetime

import streamlit as st

import gtfs_data
import prediction
from ui.formatting import CONFIDENCE_COLOR, badge_html, delay_color, route_badge, route_label_plain

# Route-map polyline colors, one per leg, cycled if ever exceeded -- in
# practice never is, since find_multi_leg_trips()'s max_transfers=3 caps a
# journey at 4 legs.
ROUTE_MAP_LEG_COLORS = ['#2563eb', '#dc2626', '#16a34a', '#d97706']


def build_route_map_legs(journey: dict) -> list[dict]:
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
                      dest_stop_ids: list[str], departure_after: datetime, journey_key: tuple,
                      label_prefix: tuple[str, str] = ('From', 'To'), expanded: bool = False) -> None:
    """Render one leg's prediction card. The leave-by banner and plain
    English summary are always visible; everything else (route badge,
    from/to, live tracking, confidence, departure/arrival) lives inside an
    expander so collapsed cards stay compact.

    `expanded` controls the expander's initial state — True only for the
    first card in a results list.

    `journey_key` is this card's stable identity in the single shared route
    map app.py renders at the top of the page. Clicking "Show route" just
    sets st.session_state['selected_route_key'] to this value and reruns --
    no map is rendered here.
    """
    render_leave_by_banner(pred)
    st.write(pred['summary'])

    if st.button('🗺️ Show route', key=f"show_route_btn_{trip['trip_id']}"):
        st.session_state['selected_route_key'] = journey_key
        st.rerun()
    if journey_key == st.session_state.get('selected_route_key'):
        st.caption('📍 shown on map above')

    label = f'🕐 Leave by {pred["leave_by"]} — {route_label_plain(trip)}'
    with st.expander(label, expanded=expanded):
        render_card_detail(trip, pred, raw_update, stop_names, label_prefix)


def render_transfer_journey_card(journey: dict, leg_predictions: list, stop_names,
                                  idx: int, expanded: bool, journey_key: tuple) -> None:
    """Render one transfer journey (num_transfers >= 1): leave-by banner +
    journey summary always visible; per-leg detail and transfer connections
    live inside one expander. `leg_predictions` must already be computed
    (see _predict_journey_legs) -- no prediction happens here.

    `journey_key` is this card's stable identity in the single shared route
    map app.py renders at the top of the page -- see render_trip_card()'s
    docstring for the same mechanism.
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

        if st.button('🗺️ Show route', key=f'show_route_btn_journey_{idx}'):
            st.session_state['selected_route_key'] = journey_key
            st.rerun()
        if journey_key == st.session_state.get('selected_route_key'):
            st.caption('📍 shown on map above')

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
                           dest_stop_ids: list[str], departure_after: datetime,
                           journey_key: tuple) -> None:
    """Render one already-ranked, already-predicted candidate -- a direct
    trip (num_transfers == 0) as a single card, a transfer journey
    (num_transfers >= 1) as a multi-leg card. Ranking/truncation/prediction
    all already happened before this is called; this is display only.

    `journey_key` is threaded through to whichever card function this
    dispatches to -- see render_trip_card()'s docstring for what it's for.
    """
    if journey['num_transfers'] == 0:
        trip, pred, raw_update = leg_predictions[0]
        with st.container(border=True):
            render_trip_card(
                trip, pred, raw_update, stop_names, dest_stop_ids, departure_after, journey_key,
                expanded=(idx == 0),
            )
    else:
        render_transfer_journey_card(
            journey, leg_predictions, stop_names, idx, expanded=(idx == 0), journey_key=journey_key,
        )
