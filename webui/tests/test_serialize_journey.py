"""_serialize_journey's leave_by fields: GTFS times are naive Brisbane
wall-clock, and leave_by_epoch must be that instant in absolute time so the
client's countdown doesn't depend on the viewer's device timezone. Shape and
stop lookups are stubbed: no GTFS.

Run from the repo root:
    python -m unittest webui/tests/test_serialize_journey.py
"""
from __future__ import annotations

import sys
import unittest
from datetime import datetime, timezone
from unittest import mock

import webui.app as webui_app

# The app's lifespan puts phase3/ on sys.path (for route_types); it doesn't run here.
if str(webui_app.PHASE3_DIR) not in sys.path:
    sys.path.insert(0, str(webui_app.PHASE3_DIR))


class LeaveByTest(unittest.TestCase):
    def test_leave_by_epoch_is_brisbane_wall_clock(self):
        trip = {
            'trip_id': 'T1', 'route_id': 'R1', 'route_type': 3,
            'origin_stop_id': 'A', 'dest_stop_id': 'B',
            'origin_departure_time': datetime(2026, 10, 8, 7, 0),  # naive Brisbane time
            'dest_arrival_time': datetime(2026, 10, 8, 7, 30),
        }
        pred = {'blended_delay_minutes': 2.0, 'confidence': 'Medium'}
        with mock.patch.object(webui_app, '_leg_shape', return_value=None), \
                mock.patch.object(webui_app, '_leg_stops', return_value=[]):
            out = webui_app._serialize_journey(
                {'num_transfers': 0}, [(trip, pred)], datetime(2026, 10, 8, 7, 32), stop_lookup={},
            )
        self.assertEqual(out['leave_by'], '06:57')
        # Brisbane is UTC+10 year-round (no DST): 06:57 local is 20:57 UTC the day before.
        # Hard-coded, not recomputed, so it pins the offset independently of the code.
        self.assertEqual(out['leave_by_epoch'], int(datetime(2026, 10, 7, 20, 57, tzinfo=timezone.utc).timestamp()))


if __name__ == '__main__':
    unittest.main()
