/**
 * Display-only. Values are not run through /predict — this is UI shell content, not a real prediction.
 *
 * A plausible Gold Coast → Brisbane journey (bus → G:link tram → Gold Coast
 * line rail via Helensvale) that RouteHero renders while live routes are
 * loading, failed, or came back empty, so the hero animation always mounts.
 * Confidence runs Medium/High/Medium to exercise more than one badge color.
 *
 * Times are fixed illustrative values, not "now + N minutes": anything
 * derived from the clock renders differently on the server and in the
 * browser (different timezones) and triggers a hydration mismatch.
 */
import type { RouteOption } from './types';

export function makeDemoRoute(): RouteOption {
  return {
    legs: [
      {
        stop_id: 'demo-broadbeach-south',
        route_short_name: '700',
        mode: 'bus',
        departure_time: '2026-01-01T08:10:00',
        predicted_arrival: '2026-01-01T08:25:30',
        confidence: 'Medium',
      },
      {
        stop_id: 'demo-helensvale',
        route_short_name: 'G:link',
        mode: 'tram',
        departure_time: '2026-01-01T08:30:00',
        predicted_arrival: '2026-01-01T08:52:00',
        confidence: 'High',
      },
      {
        stop_id: 'demo-central',
        route_short_name: 'Gold Coast line',
        mode: 'rail',
        departure_time: '2026-01-01T08:58:00',
        predicted_arrival: '2026-01-01T09:35:15',
        confidence: 'Medium',
      },
    ],
    total_predicted_duration_minutes: 85,
    transfer_count: 2,
  };
}
