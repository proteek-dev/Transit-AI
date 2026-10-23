"""/live_vehicles, its feed guard and the tram feed parser, with the network
stubbed -- no S3 either (TestClient isn't used as a context manager, so the
app's lifespan never loads GTFS or the model).

Run from the repo root:
    python -m unittest webui/tests/test_live_vehicles.py
"""
from __future__ import annotations

import sys
import unittest
from unittest import mock

import requests
from fastapi.testclient import TestClient
from google.transit import gtfs_realtime_pb2

import webui.app as webui_app
import webui.live_trams as live_trams
from webui.live_trams import FeedResult

# The app's lifespan puts phase3/ on sys.path; it doesn't run here.
if str(webui_app.PHASE3_DIR) not in sys.path:
    sys.path.insert(0, str(webui_app.PHASE3_DIR))
import live_gtfs  # noqa: E402 -- the same module object /live_vehicles imports

NOW = 1_800_000_000


def _vehicle(age_seconds: int) -> dict:
    return {'lat': -28.0, 'lon': 153.4, 'speed': None, 'timestamp': NOW - age_seconds, 'current_stop_sequence': 1}


class LiveVehiclesEndpointTest(unittest.TestCase):
    def setUp(self):
        webui_app._combined_feed.reset()
        self.addCleanup(webui_app._combined_feed.reset)
        self.client = TestClient(webui_app.app)
        patches = [
            mock.patch('time.time', return_value=NOW),
            mock.patch.object(live_gtfs, 'fetch_vehicle_positions', return_value={}),
            mock.patch.object(webui_app.tram_feed, 'get', return_value=FeedResult({}, True)),
        ]
        _, self.combined, self.trams = [p.start() for p in patches]
        for p in patches:
            self.addCleanup(p.stop)

    def get(self, trip_ids):
        return self.client.get('/live_vehicles', params={'trip_ids': trip_ids})

    def test_rejects_missing_and_too_many_trip_ids(self):
        self.assertEqual(self.client.get('/live_vehicles').status_code, 400)
        self.assertEqual(self.get(',').status_code, 400)
        cap = webui_app.LIVE_VEHICLES_MAX_TRIP_IDS
        self.assertEqual(self.get(','.join(f't{i}' for i in range(cap))).status_code, 200)
        self.assertEqual(self.get(','.join(f't{i}' for i in range(cap + 1))).status_code, 400)

    def test_response_shape(self):
        self.combined.return_value = {'t1': _vehicle(30)}
        body = self.get(' t1 , t1,missing').json()  # whitespace + duplicate trimmed; unknown trip omitted
        self.assertEqual(body, {
            'vehicles': {'t1': {'lat': -28.0, 'lon': 153.4, 'timestamp': NOW - 30, 'age_seconds': 30}},
            'fetched_at': NOW,
            'degraded': False,
        })

    def test_staleness_cut_at_120_seconds(self):
        self.combined.return_value = {'fresh': _vehicle(120), 'stale': _vehicle(121)}
        body = self.get('fresh,stale').json()
        self.assertEqual(set(body['vehicles']), {'fresh'})
        self.assertEqual(body['vehicles']['fresh']['age_seconds'], 120)

    def test_fix_ahead_of_server_clock_has_zero_age(self):
        self.combined.return_value = {'t1': _vehicle(-5)}
        self.assertEqual(self.get('t1').json()['vehicles']['t1']['age_seconds'], 0)

    def test_combined_feed_wins_a_collision(self):
        self.combined.return_value = {'t1': {**_vehicle(5), 'lat': -27.0}}
        self.trams.return_value = FeedResult({'t1': {**_vehicle(5), 'lat': -28.5}, 't2': _vehicle(5)}, True)
        vehicles = self.get('t1,t2').json()['vehicles']
        self.assertEqual(vehicles['t1']['lat'], -27.0)
        self.assertIn('t2', vehicles)

    def test_combined_failure_degrades_to_trams_and_is_cached(self):
        self.combined.side_effect = RuntimeError('feed down')
        self.trams.return_value = FeedResult({'tram1': _vehicle(5)}, True)
        for _ in range(3):
            resp = self.get('tram1,bus1')
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(set(resp.json()['vehicles']), {'tram1'})
            self.assertTrue(resp.json()['degraded'])
        self.assertEqual(self.combined.call_count, 1)  # later polls inside the failure window don't retry

    def test_tram_failure_is_flagged_degraded(self):
        self.trams.return_value = FeedResult({}, False)
        self.assertTrue(self.get('t1').json()['degraded'])

    def test_response_is_not_cacheable(self):
        self.assertEqual(self.get('t1').headers.get('cache-control'), 'no-store')


class GuardedFeedTest(unittest.TestCase):
    def setUp(self):
        self.fetch = mock.Mock(return_value={'t1': _vehicle(5)})
        self.feed = live_trams.GuardedFeed(self.fetch, name='test', cache_ttl_seconds=60, failure_ttl_seconds=20)
        patcher = mock.patch('time.time', return_value=NOW)
        self.clock = patcher.start()
        self.addCleanup(patcher.stop)

    def test_success_is_cached_for_the_ttl(self):
        self.assertEqual(self.feed.get(), FeedResult({'t1': _vehicle(5)}, True))
        self.clock.return_value = NOW + 59
        self.feed.get()
        self.assertEqual(self.fetch.call_count, 1)
        self.clock.return_value = NOW + 60
        self.feed.get()
        self.assertEqual(self.fetch.call_count, 2)

    def test_failure_keeps_last_good_data_and_waits_before_retrying(self):
        self.feed.get()
        self.fetch.side_effect = RuntimeError('down')
        self.clock.return_value = NOW + 60
        self.assertEqual(self.feed.get(), FeedResult({'t1': _vehicle(5)}, False))
        self.clock.return_value = NOW + 79
        self.assertFalse(self.feed.get().ok)
        self.assertEqual(self.fetch.call_count, 2)  # inside the failure window: no retry
        self.clock.return_value = NOW + 80
        self.feed.get()
        self.assertEqual(self.fetch.call_count, 3)

    def test_concurrent_caller_gets_current_data_instead_of_waiting(self):
        self.feed.data = {'old': _vehicle(70)}
        with self.feed.lock:
            self.assertEqual(self.feed.get(), FeedResult({'old': _vehicle(70)}, True))
        self.fetch.assert_not_called()


def _tram_feed_bytes() -> bytes:
    feed = gtfs_realtime_pb2.FeedMessage()
    feed.header.gtfs_realtime_version = '2.0'

    valid = feed.entity.add(id='1').vehicle
    valid.trip.trip_id = 'tram-ok'
    valid.position.latitude, valid.position.longitude, valid.position.speed = -27.9, 153.4, 8.5
    valid.timestamp = NOW - 10
    valid.current_stop_sequence = 4

    no_trip = feed.entity.add(id='2').vehicle
    no_trip.position.latitude, no_trip.position.longitude = -27.9, 153.4
    no_trip.timestamp = NOW

    undated = feed.entity.add(id='3').vehicle
    undated.trip.trip_id = 'tram-undated'
    undated.position.latitude, undated.position.longitude = -27.9, 153.4

    unplaced = feed.entity.add(id='4').vehicle
    unplaced.trip.trip_id = 'tram-unplaced'
    unplaced.timestamp = NOW

    feed.entity.add(id='5').trip_update.trip.trip_id = 'not-a-vehicle'
    return feed.SerializeToString()


class TramFeedTest(unittest.TestCase):
    def setUp(self):
        live_trams.tram_feed.reset()
        self.addCleanup(live_trams.tram_feed.reset)

    def test_parses_only_dated_placed_trip_vehicles(self):
        resp = mock.Mock(content=_tram_feed_bytes())
        with mock.patch('time.time', return_value=NOW), \
                mock.patch.object(live_trams.requests, 'get', return_value=resp) as get:
            result = live_trams.tram_feed.get()
        self.assertTrue(result.ok)
        self.assertEqual(list(result.vehicles), ['tram-ok'])
        vehicle = result.vehicles['tram-ok']
        self.assertAlmostEqual(vehicle['lat'], -27.9, places=5)
        self.assertAlmostEqual(vehicle['speed'], 8.5, places=5)
        self.assertEqual((vehicle['timestamp'], vehicle['current_stop_sequence']), (NOW - 10, 4))
        self.assertEqual(get.call_args.kwargs['timeout'], live_trams.FETCH_TIMEOUT_SECONDS)

    def test_fetch_failure_is_not_ok_and_not_retried_at_once(self):
        with mock.patch('time.time', return_value=NOW), \
                mock.patch.object(live_trams.requests, 'get', side_effect=requests.Timeout) as get:
            self.assertEqual(live_trams.tram_feed.get(), FeedResult({}, False))
            self.assertEqual(live_trams.tram_feed.get(), FeedResult({}, False))
        get.assert_called_once()


if __name__ == '__main__':
    unittest.main()
