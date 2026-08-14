"""Trip shape-point lookup for the Phase 3 Streamlit POC.

get_trip_shape_points() returns a trip's (lat, lon) polyline, trimmed to the
origin/dest-stop segment when both stop_ids are given (Session 21's fix).
"""
from __future__ import annotations

import pandas as pd

from gtfs.loader import load_gtfs_data


def get_trip_shape_points(
    trip_id: str, origin_stop_id: str | None = None, dest_stop_id: str | None = None,
) -> list[tuple] | None:
    """(lat, lon) points for trip_id's shape, in shape_pt_sequence order.

    Returns None if trip_id isn't found, or if the trip has no shape_id
    (a valid GTFS state, not a bug).

    If origin_stop_id and dest_stop_id are both given, the shape is trimmed
    to just the origin-to-destination segment: each stop's (lat, lon) is
    matched to its nearest shape point (plain Euclidean distance -- shape
    points are dense enough at this scale that haversine isn't needed), and
    the point list is sliced between the two matched indices (inclusive,
    ordered by index rather than assuming which stop comes first) so the
    rendered path is the passenger's actual ride, not the vehicle's whole
    route. Falls back to the full shape if either stop_id isn't found in
    data.stops.
    """
    data = load_gtfs_data()
    trip_rows = data.trips.loc[data.trips['trip_id'] == trip_id, 'shape_id']
    if trip_rows.empty:
        return None
    shape_id = trip_rows.iloc[0]
    if pd.isna(shape_id):
        return None
    points = data.shape_points.get(shape_id)
    if points is None or origin_stop_id is None or dest_stop_id is None:
        return points

    stop_lat_lon = data.stops.set_index('stop_id')[['stop_lat', 'stop_lon']]
    if origin_stop_id not in stop_lat_lon.index or dest_stop_id not in stop_lat_lon.index:
        return points

    origin_lat, origin_lon = stop_lat_lon.loc[origin_stop_id]
    dest_lat, dest_lon = stop_lat_lon.loc[dest_stop_id]

    def _nearest_point_idx(lat: float, lon: float) -> int:
        return min(
            range(len(points)),
            key=lambda i: (points[i][0] - lat) ** 2 + (points[i][1] - lon) ** 2,
        )

    origin_idx = _nearest_point_idx(origin_lat, origin_lon)
    dest_idx = _nearest_point_idx(dest_lat, dest_lon)
    lo, hi = min(origin_idx, dest_idx), max(origin_idx, dest_idx)
    return points[lo:hi + 1]
