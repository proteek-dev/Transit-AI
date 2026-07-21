"""Reusable folium-based stop picker.

Renders one marker per candidate stop, lets the user tap a pin, and resolves
the click back to a stop_id. Generic over how the candidate list was sourced
(nearest_stops() from a geolocation fix, or search_stops() from a typed
query) — callers pass in their own mode -> (emoji, label) map (app.py's
ROUTE_TYPE_MODE) rather than this module importing or redefining one, so it
stays usable from any origin/destination flow without a dependency on app.py.
"""
from __future__ import annotations

import folium
from streamlit_folium import st_folium

# folium's built-in marker colors, keyed by GTFS route_type. Only used for
# pin color; the mode emoji/label itself comes from the caller's mode_map.
_FOLIUM_COLOR_BY_MODE = {
    0: 'orange',    # tram
    1: 'purple',    # metro
    2: 'blue',      # train
    3: 'green',     # bus
    4: 'cadetblue',  # ferry
}
_DEFAULT_FOLIUM_COLOR = 'gray'


def render_stop_picker(
    candidates: list[dict],
    center_lat: float,
    center_lon: float,
    key: str,
    mode_map: dict,
    default_mode: tuple[str, str] = ('🚍', 'Transit'),
    zoom_start: int = 15,
) -> str | None:
    """Render `candidates` as pins on a folium map centered on (center_lat,
    center_lon). Returns the stop_id of the candidate the user just clicked,
    or None if nothing's been clicked yet (or candidates is empty).

    Each candidate dict needs stop_id, stop_name, stop_lat, stop_lon, and
    optionally route_types (list[int]) for mode-based icon color/label —
    candidates missing route_types just get the default marker.
    """
    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start)
    folium.Marker(
        [center_lat, center_lon],
        tooltip='You are here',
        icon=folium.Icon(color='red', icon='user', prefix='fa'),
    ).add_to(fmap)

    # Each marker's tooltip (mode + name) is unique within this candidate
    # list and doubles as the click -> stop_id lookup key, rather than
    # nearest-matching the click's lat/lng back to a candidate.
    # last_object_clicked's coordinates come from Leaflet's *rendering* of the
    # marker, and a distance-based "is this actually the center pin" guard
    # around them is fragile: it also swallows a genuine click on the single
    # most useful candidate whenever that candidate is very close to the
    # user's own position (the "You are here" pin) -- exactly the stop most
    # likely to be tapped. Reading the tooltip directly sidesteps both
    # problems, since 'You are here' just isn't a key in this lookup.
    tooltip_to_stop_id: dict[str, str] = {}

    for cand in candidates:
        route_types = cand.get('route_types') or []
        primary_mode = route_types[0] if route_types else None
        emoji, _ = mode_map.get(primary_mode, default_mode)
        mode_labels = [mode_map.get(rt, default_mode)[1] for rt in route_types] or [default_mode[1]]

        tooltip = f"{emoji} {cand['stop_name']} ({'/'.join(mode_labels)})"
        tooltip_to_stop_id[tooltip] = cand['stop_id']

        folium.Marker(
            [cand['stop_lat'], cand['stop_lon']],
            tooltip=tooltip,
            icon=folium.Icon(color=_FOLIUM_COLOR_BY_MODE.get(primary_mode, _DEFAULT_FOLIUM_COLOR), icon='info-sign'),
        ).add_to(fmap)

    map_data = st_folium(fmap, height=420, use_container_width=True, key=key)

    if not candidates:
        return None

    clicked_tooltip = (map_data or {}).get('last_object_clicked_tooltip')
    if clicked_tooltip is None:
        return None

    return tooltip_to_stop_id.get(clicked_tooltip)
