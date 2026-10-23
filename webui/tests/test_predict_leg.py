"""_predict_leg feeds the model the leg's scheduled arrival -- what training
derived hour_of_day/day_of_week/is_peak from -- not the search time. The
enrich + predict calls are stubbed: no GTFS, no model.

Run from the repo root:
    python -m unittest webui/tests/test_predict_leg.py
"""
from __future__ import annotations

import sys
import unittest
from datetime import datetime
from unittest import mock

import webui.app as webui_app

# The app's lifespan puts phase3/ on sys.path; it doesn't run here.
if str(webui_app.PHASE3_DIR) not in sys.path:
    sys.path.insert(0, str(webui_app.PHASE3_DIR))
import prediction  # noqa: E402 -- the same module object _predict_leg imports


class PredictLegTimeFeatureTest(unittest.TestCase):
    def test_predicts_on_scheduled_arrival_not_search_time(self):
        # Found by a 06:50 search, arrives 07:40: the model must see the peak hour.
        arrival = datetime(2026, 10, 8, 7, 40)
        trip = {'trip_id': 'T1', 'origin_departure_time': datetime(2026, 10, 8, 6, 55), 'dest_arrival_time': arrival}
        enriched = dict(trip, stop_id='B', stop_sequence=9)
        with mock.patch.object(prediction, 'enrich_trip_with_dest_stop', return_value=enriched), \
                mock.patch.object(prediction, 'predict_delay', return_value={'confidence': 'Medium'}) as predict:
            self.assertEqual(webui_app._predict_leg(trip, ['B']), (enriched, {'confidence': 'Medium'}))
        predict.assert_called_once_with(enriched, arrival, live_delay=None)


if __name__ == '__main__':
    unittest.main()
