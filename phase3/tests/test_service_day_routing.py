"""Service-day boundaries in routing: GTFS times count from the service day's
midnight, so trips timed 24:00+ belong to the PREVIOUS day's calendar (a 00:30
search must find them), and a window that runs past midnight must also check
the NEXT day's calendar. Uses a tiny in-memory GTFSData -- no S3.

Run from the repo root:
    python -m unittest phase3/tests/test_service_day_routing.py
"""
from __future__ import annotations

import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gtfs import routing  # noqa: E402
from gtfs.loader import DAY_NAMES, GTFSData  # noqa: E402

WED = datetime(2026, 10, 7)  # a Wednesday
THU = datetime(2026, 10, 8)
FRI = datetime(2026, 10, 9)

# (trip_id, service_id, departs A, arrives B) on route R1. WED-LATE runs on
# Wednesday's calendar but departs at 24:45 (Thursday 00:45 wall-clock).
BASE_TRIPS = [
    ('WED-LATE', 'WED', '24:45:00', '25:00:00'),
    ('THU-EARLY', 'THU', '00:50:00', '01:05:00'),
    ('FRI-EARLY', 'FRI', '00:15:00', '00:30:00'),
]


def _calendar_row(service_id: str, weekday: str, start_date: int) -> dict:
    row = {day: int(day == weekday) for day in DAY_NAMES}
    row.update(service_id=service_id, start_date=start_date, end_date=20261231)
    return row


def _fixture(trips=BASE_TRIPS, snapshot_date='2026-09-01', calendar_start=20260101) -> GTFSData:
    """Route R1, stops A -> B, one trip per `trips` row. The default
    snapshot_date is far from the query dates, so no 7-day fallback.
    """
    data = GTFSData()
    data.snapshot_date = snapshot_date
    data.stops = pd.DataFrame({'stop_id': ['A', 'B'], 'stop_name': ['Alpha', 'Bravo'],
                               'stop_lat': [-28.0, -28.1], 'stop_lon': [153.4, 153.4]})
    data.routes = pd.DataFrame({'route_id': ['R1'], 'route_short_name': ['1'],
                                'route_long_name': ['One'], 'route_type': [0]})
    data.trips = pd.DataFrame({'trip_id': [t[0] for t in trips], 'route_id': ['R1'] * len(trips),
                               'service_id': [t[1] for t in trips]})
    data.stop_times = pd.DataFrame({
        'trip_id': [t[0] for t in trips for _ in range(2)],
        'stop_id': ['A', 'B'] * len(trips),
        'stop_sequence': [1, 2] * len(trips),
        'arrival_time': [time for t in trips for time in (t[2], t[3])],
        'departure_time': [time for t in trips for time in (t[2], t[3])],
    })
    data.calendar = pd.DataFrame([
        _calendar_row('WED', 'wednesday', calendar_start),
        _calendar_row('THU', 'thursday', calendar_start),
        _calendar_row('FRI', 'friday', calendar_start),
    ])
    data.calendar_dates = pd.DataFrame({'service_id': [], 'date': [], 'exception_type': []}, dtype=str)
    data.stop_to_routes = {'A': {'R1'}, 'B': {'R1'}}
    data._build_route_departure_index()
    return data


def _patch_loader(test: unittest.TestCase, data: GTFSData) -> GTFSData:
    patcher = mock.patch.object(routing, 'load_gtfs_data', return_value=data)
    patcher.start()
    test.addCleanup(patcher.stop)
    return data


def _searched_dates(data: GTFSData, departure_after: datetime, window_minutes: int = 60) -> list:
    window_end = departure_after + timedelta(minutes=window_minutes)
    return [day.date for day in routing._service_days(data, departure_after, window_end)]


class ServiceDayRoutingTest(unittest.TestCase):
    def setUp(self):
        self.data = _patch_loader(self, _fixture())

    def test_find_trips_includes_previous_service_days_late_trip(self):
        trips = routing.find_trips(['A'], ['B'], THU.replace(minute=30), window_minutes=60)
        self.assertEqual([t['trip_id'] for t in trips], ['WED-LATE', 'THU-EARLY'])
        self.assertEqual(trips[0]['origin_departure_time'], THU.replace(minute=45))
        self.assertEqual(trips[0]['dest_arrival_time'], THU.replace(hour=1))

    def test_late_trip_not_found_on_wrong_calendar_day(self):
        # Tuesday's calendar has no WED service, so Wednesday 00:30 finds nothing.
        self.assertEqual(routing.find_trips(['A'], ['B'], WED.replace(minute=30), window_minutes=60), [])

    def test_lookup_leg_trip_uses_previous_service_day(self):
        trip = routing._lookup_leg_trip(self.data, 'R1', ['A'], ['B'], THU.replace(minute=30), 60)
        self.assertEqual(trip['trip_id'], 'WED-LATE')
        self.assertEqual(trip['origin_departure_time'], THU.replace(minute=45))
        self.assertFalse(trip['fallback_schedule'])

    def test_any_route_reaches_dest_sees_previous_service_day(self):
        self.assertTrue(routing._any_route_reaches_dest(self.data, ['A'], ['B'], THU.replace(minute=40), 10))

    def test_previous_day_only_searched_while_its_late_trips_can_reach(self):
        # Latest departure in the fixture is WED-LATE's B stop at 25:00 (90000 s past midnight).
        self.assertEqual(self.data.max_departure_seconds, 90000)
        self.assertEqual(_searched_dates(self.data, THU.replace(hour=1)), [WED.date(), THU.date()])
        self.assertEqual(_searched_dates(self.data, THU.replace(hour=1, minute=1)), [THU.date()])

    def test_previous_day_searched_when_max_departure_unknown(self):
        self.data.max_departure_seconds = None
        self.assertEqual(_searched_dates(self.data, THU.replace(hour=12)), [WED.date(), THU.date()])

    def test_next_day_searched_only_when_window_crosses_midnight(self):
        self.assertIn(FRI.date(), _searched_dates(self.data, THU.replace(hour=23, minute=40)))
        self.assertNotIn(FRI.date(), _searched_dates(self.data, THU.replace(hour=22)))

    # A 23:40 search's 60-minute window reaches 00:40 the next day, where
    # FRI-EARLY (Friday's calendar, 00:15) departs.
    def test_find_trips_includes_next_service_days_early_trip(self):
        trips = routing.find_trips(['A'], ['B'], THU.replace(hour=23, minute=40), window_minutes=60)
        self.assertEqual([t['trip_id'] for t in trips], ['FRI-EARLY'])
        self.assertEqual(trips[0]['origin_departure_time'], FRI.replace(minute=15))
        self.assertFalse(trips[0]['fallback_schedule'])

    def test_lookup_leg_trip_uses_next_service_day(self):
        trip = routing._lookup_leg_trip(self.data, 'R1', ['A'], ['B'], THU.replace(hour=23, minute=40), 60)
        self.assertEqual(trip['trip_id'], 'FRI-EARLY')
        self.assertEqual(trip['origin_departure_time'], FRI.replace(minute=15))

    def test_any_route_reaches_dest_sees_next_service_day(self):
        self.assertTrue(routing._any_route_reaches_dest(self.data, ['A'], ['B'], THU.replace(hour=23, minute=40), 60))


class EarliestAcrossServiceDaysTest(unittest.TestCase):
    """_lookup_leg_trip picks the earliest wall-clock departure across service
    days -- it doesn't prefer either day's calendar."""

    def lookup(self, trips):
        data = _patch_loader(self, _fixture(trips))
        return routing._lookup_leg_trip(data, 'R1', ['A'], ['B'], THU.replace(minute=30), 60)

    def test_previous_days_trip_wins_when_earlier(self):
        trip = self.lookup([('WED-LATE', 'WED', '24:45:00', '25:00:00'),
                            ('THU-EARLY', 'THU', '00:50:00', '01:05:00')])
        self.assertEqual(trip['trip_id'], 'WED-LATE')

    def test_todays_trip_wins_when_earlier(self):
        trip = self.lookup([('WED-LATE', 'WED', '24:45:00', '25:00:00'),
                            ('THU-EARLIER', 'THU', '00:35:00', '00:50:00')])
        self.assertEqual(trip['trip_id'], 'THU-EARLIER')
        self.assertEqual(trip['origin_departure_time'], THU.replace(minute=35))


class WeekAheadFallbackTest(unittest.TestCase):
    """Query date before every calendar starts but within
    FALLBACK_THIN_COVERAGE_DAYS of the snapshot: the search retries a week
    ahead -- previous-service-day trips included -- and shifts the results
    back onto the query date."""

    def setUp(self):
        self.data = _patch_loader(self, _fixture(snapshot_date='2026-10-08', calendar_start=20261010))

    def test_find_trips_falls_back_a_week_ahead(self):
        trips = routing.find_trips(['A'], ['B'], THU.replace(minute=30), window_minutes=60)
        self.assertEqual([t['trip_id'] for t in trips], ['WED-LATE', 'THU-EARLY'])
        self.assertTrue(all(t['fallback_schedule'] for t in trips))
        self.assertEqual(trips[0]['origin_departure_time'], THU.replace(minute=45))

    def test_lookup_leg_trip_falls_back_a_week_ahead(self):
        trip = routing._lookup_leg_trip(self.data, 'R1', ['A'], ['B'], THU.replace(minute=30), 60)
        self.assertEqual(trip['trip_id'], 'WED-LATE')
        self.assertTrue(trip['fallback_schedule'])
        self.assertEqual(trip['origin_departure_time'], THU.replace(minute=45))


if __name__ == '__main__':
    unittest.main()
