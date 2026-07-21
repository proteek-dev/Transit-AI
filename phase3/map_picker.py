"""Reusable folium-based stop picker.

Renders one marker per candidate stop, lets the user tap a pin, and resolves
the click back to a stop_id. Generic over how the candidate list was sourced
(nearest_stops() from a geolocation fix, or search_stops() from a typed
query) — callers pass in their own mode -> (emoji, label) map (app.py's
ROUTE_TYPE_MODE) rather than this module importing or redefining one, so it
stays usable from any origin/destination flow without a dependency on app.py.
"""
from __future__ import annotations

from math import atan2, cos, radians, sin, sqrt

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

# A click landing within this many km of the center marker is treated as a
# click on "You are here", not on a candidate pin.
_CENTER_CLICK_RADIUS_KM = 0.01


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(dlambda / 2) ** 2
    return 2 * r * atan2(sqrt(a), sqrt(1 - a))


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

    for cand in candidates:
        route_types = cand.get('route_types') or []
        primary_mode = route_types[0] if route_types else None
        emoji, _ = mode_map.get(primary_mode, default_mode)
        mode_labels = [mode_map.get(rt, default_mode)[1] for rt in route_types] or [default_mode[1]]

        folium.Marker(
            [cand['stop_lat'], cand['stop_lon']],
            tooltip=f"{emoji} {cand['stop_name']} ({'/'.join(mode_labels)})",
            icon=folium.Icon(color=_FOLIUM_COLOR_BY_MODE.get(primary_mode, _DEFAULT_FOLIUM_COLOR), icon='info-sign'),
        ).add_to(fmap)

    map_data = st_folium(fmap, height=420, use_container_width=True, key=key)

    if not candidates:
        return None

    clicked = (map_data or {}).get('last_object_clicked')
    if not clicked or clicked.get('lat') is None or clicked.get('lng') is None:
        return None

    if _haversine_km(clicked['lat'], clicked['lng'], center_lat, center_lon) < _CENTER_CLICK_RADIUS_KM:
        return None

    # Candidate list is capped at ~15, so a simple nearest-match against the
    # click's lat/lng is enough to resolve it back to a stop_id — no need for
    # a real spatial index.
    best = min(
        candidates,
        key=lambda c: _haversine_km(clicked['lat'], clicked['lng'], c['stop_lat'], c['stop_lon']),
    )
    return best['stop_id']
