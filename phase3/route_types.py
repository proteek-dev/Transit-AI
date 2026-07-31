"""Single source of truth for GTFS route_type -> mode-name mapping.

Was duplicated three times with two different key conventions: str-keyed in
notebook 05 (route_type read from routes.txt as a raw string, never cast),
int-keyed in phase3/prediction.py (feeds the trained model's categorical
schema), and gtfs_data.py's FERRY_ROUTE_TYPE constant compared against an
explicitly int-cast route_type column. All three now import from here.

SEQ doesn't run metro (route_type 1) in schedule data, so it's intentionally
absent below -- app.py's separate ROUTE_TYPE_MODE (emoji/label pairs for UI
display, a different concern from this module's mode-name strings) still
handles it via its own DEFAULT_ROUTE_TYPE_MODE fallback.
"""
from __future__ import annotations

MODE_BY_ROUTE_TYPE: dict[int, str] = {0: 'tram', 2: 'rail', 3: 'bus', 4: 'ferry'}

# notebook 05 reads routes.txt's route_type as a raw string (dtype=str,
# never cast to int) -- same mapping, string keys, so its
# feature_df['route_type'].map(...) call works unchanged.
MODE_BY_ROUTE_TYPE_STR: dict[str, str] = {str(k): v for k, v in MODE_BY_ROUTE_TYPE.items()}

FERRY_ROUTE_TYPE = next(rt for rt, mode in MODE_BY_ROUTE_TYPE.items() if mode == 'ferry')
