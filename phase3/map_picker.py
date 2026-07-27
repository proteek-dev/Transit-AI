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
    locked_marker: dict | None = None,
) -> str | None:
    """Render `candidates` as pins on a folium map centered on (center_lat,
    center_lon). Returns the stop_id of the candidate the user just clicked,
    or None if nothing's been clicked yet (or candidates is empty).

    Each candidate dict needs stop_id, stop_name, stop_lat, stop_lon, and
    optionally route_types (list[int]) for mode-based icon color/label —
    candidates missing route_types just get the default marker.

    `locked_marker` (optional): a confirmed stop -- same shape as a candidate
    dict -- shown as a fixed, visually distinct pin for context (e.g. an
    already-confirmed origin on a destination-picking map). It is NOT
    clickable: excluded from the tooltip -> stop_id lookup below, so tapping
    it resolves to None just like tapping empty water. When present alongside
    candidates, the map frames both via fit_bounds() rather than centering
    purely on (center_lat, center_lon), since the locked marker can be far
    from the candidates. When absent, behavior is unchanged from before:
    a plain "You are here" pin at (center_lat, center_lon).
    """
    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start)

    if locked_marker is not None:
        folium.Marker(
            [locked_marker['stop_lat'], locked_marker['stop_lon']],
            tooltip=f"🔒 {locked_marker['stop_name']} (origin)",
            icon=folium.Icon(color='darkgreen', icon='lock', prefix='fa'),
        ).add_to(fmap)
    else:
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
    # problems, since 'You are here' / the locked marker's tooltip just isn't
    # a key in this lookup.
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

    if locked_marker is not None and candidates:
        # Frame both the locked marker and the candidates -- fit_bounds()
        # overrides the initial center/zoom above. With no candidates yet,
        # that initial center is left as-is; callers pass the locked
        # marker's own coordinates for center_lat/center_lon in that case,
        # so the map still opens centered on it.
        bound_points = [[locked_marker['stop_lat'], locked_marker['stop_lon']]]
        bound_points += [[c['stop_lat'], c['stop_lon']] for c in candidates]
        lats = [p[0] for p in bound_points]
        lons = [p[1] for p in bound_points]
        fmap.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])

    map_data = st_folium(fmap, height=420, use_container_width=True, key=key)

    if not candidates:
        return None

    clicked_tooltip = (map_data or {}).get('last_object_clicked_tooltip')
    if clicked_tooltip is None:
        return None

    return tooltip_to_stop_id.get(clicked_tooltip)
