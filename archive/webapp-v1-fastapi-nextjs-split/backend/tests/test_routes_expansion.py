"""Integration test for /routes station-group expansion.

Hits the real app via TestClient: startup warm-up loads the live GTFS
snapshot + model from S3 (repo-root .env credentials), so expect a slow
first request. Stdlib unittest -- no pytest dependency; run from
webapp/backend/:

    python -m unittest tests.test_routes_expansion -v

Assertions are len(...) > 0 only, never specific routes, so this doesn't
churn as the timetable or model changes. Departure is pinned to midday
Brisbane time today so the result doesn't depend on when the test runs.
"""
import sys
import unittest
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402

from main import app  # noqa: E402

MIDDAY = datetime.now(ZoneInfo('Australia/Brisbane')).replace(
    hour=12, minute=0, second=0, microsecond=0, tzinfo=None,
).isoformat()


class RoutesExpansionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Context-managed so the startup warm-up runs; /routes awaits it.
        cls._client_cm = TestClient(app)
        cls.client = cls._client_cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._client_cm.__exit__(None, None, None)

    def _routes(self, from_stop_id: str, to_stop_id: str):
        return self.client.get(
            '/routes',
            params={'from_stop_id': from_stop_id, 'to_stop_id': to_stop_id, 'departure': MIDDAY},
        )

    def assert_non_empty(self, from_stop_id: str, to_stop_id: str):
        response = self._routes(from_stop_id, to_stop_id)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertGreater(len(response.json()), 0, f'{from_stop_id} -> {to_stop_id} returned no routes')

    def test_cavill_to_broadbeach_south(self):
        # 320860 is a bus bay (route 70 only); the L1 platforms are only
        # reachable through Broadbeach South's station group.
        self.assert_non_empty('600810', '320860')

    def test_broadbeach_south_to_cavill(self):
        self.assert_non_empty('320860', '600810')

    def test_surfers_paradise_to_hota(self):
        # 600808 is the single-direction platform that can't reach HOTA;
        # the one viable journey leaves from 600809 in the same group.
        self.assert_non_empty('600808', '317007')

    def test_unknown_stop_id_is_404(self):
        response = self._routes('99999999', '320860')
        self.assertEqual(response.status_code, 404, response.text)
        self.assertIn('99999999', response.json()['detail'])


if __name__ == '__main__':
    unittest.main()
