"""Minimal smoke test for phase3/prediction.py's predict_delay().

This formalizes a check that has only ever been run ad hoc via prompts:
does predict_delay() still work end-to-end for an ordinary in-vocab stop,
and does it still correctly flag Low confidence for a known
out-of-vocabulary destination stop_id?

PRIMARY case: stop_id '1' (Herschel Street Stop 1 near North Quay, bus,
in-vocab) -- the everyday path.

SECONDARY case: stop_id 600815 (Surfers Paradise North station, platform 1,
tram, confirmed out-of-vocabulary) -- the deliberate OOV edge-case check.
predict_delay() should report Low confidence for this one; see
prediction.py's build_features()/predict_delay() stop_id_is_oov handling.

Both cases use a route_id/stop_sequence/mode pulled from a real stop_times
row that actually serves the stop, rather than fabricated values, so
predict_delay() sees a realistic feature row.

Run from the repo root or from phase3/:
    python3 scripts/smoke_test.py
"""
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'phase3'))

import gtfs_data  # noqa: E402
import prediction  # noqa: E402


def _build_trip_info(stop_id: str) -> tuple[dict, datetime]:
    data = gtfs_data.load_gtfs_data()
    st_row = data.stop_times[data.stop_times['stop_id'] == stop_id].iloc[0]
    trip_row = data.trips[data.trips['trip_id'] == st_row['trip_id']].iloc[0]
    route_row = data.routes[data.routes['route_id'] == trip_row['route_id']].iloc[0]
    stop_row = data.stops[data.stops['stop_id'] == stop_id].iloc[0]

    departure_time = datetime.now()
    trip_info = {
        'route_id': trip_row['route_id'],
        'route_short_name': route_row['route_short_name'],
        'stop_id': stop_id,
        'stop_sequence': int(st_row['stop_sequence']),
        'route_type': int(route_row['route_type']),
        'origin_departure_time': departure_time,
        'dest_arrival_time': departure_time + timedelta(minutes=15),
        'origin_stop_name': 'origin',
        'dest_stop_name': stop_row['stop_name'],
    }
    return trip_info, departure_time


def run_case(stop_id: str, label: str) -> None:
    trip_info, departure_time = _build_trip_info(stop_id)
    result = prediction.predict_delay(trip_info, departure_time)
    print(f'--- {label} ---')
    print(f"  stop_id: {stop_id} ({trip_info['dest_stop_name']})")
    print(f"  prediction: {result['blended_delay_minutes']:+.1f} min")
    print(f"  confidence: {result['confidence']}")
    print()


if __name__ == '__main__':
    run_case('1', 'PRIMARY -- in-vocab bus stop')
    run_case('600815', 'SECONDARY -- deliberate OOV edge case (tram), expected Low confidence')
