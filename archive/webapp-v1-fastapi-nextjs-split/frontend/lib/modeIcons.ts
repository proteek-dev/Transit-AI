import type { Mode } from './types';

/**
 * Mode -> icon lookup, keyed on the exact string values GET /routes
 * actually returns (see lib/types.ts's Mode type). Deliberately has no
 * 'ferry' entry: ferry is filtered out entirely upstream in
 * phase3/gtfs/loader.py's GTFS pipeline (FERRY_ROUTE_TYPE exclusion), so
 * it can never appear in a real leg's mode field -- a case for it here
 * would be dead code implying a mode that can't occur.
 *
 * Bus/tram/rail reuse the same emoji phase3/ui/formatting.py's
 * ROUTE_TYPE_MODE already established (🚌/🚊/🚆) for visual consistency
 * with the rest of the project. 'unknown' is a genuine defensive
 * fallback -- predict_delay() emits it when a trip's route_type isn't in
 * MODE_BY_ROUTE_TYPE at all -- so it gets a plain generic marker, not a
 * transit-mode emoji implying a specific vehicle that isn't known.
 */
export const MODE_ICON: Record<Exclude<Mode, 'ferry'>, string> = {
  bus: '🚌',
  tram: '🚊',
  rail: '🚆',
  unknown: '●',
};

/** Looks up MODE_ICON, falling back to the 'unknown' marker for any mode
 * value not in the map (defensive: covers 'ferry' too, should it ever
 * appear despite the upstream exclusion, rather than throwing/rendering
 * nothing).
 */
export function iconForMode(mode: Mode): string {
  return MODE_ICON[mode as Exclude<Mode, 'ferry'>] ?? MODE_ICON.unknown;
}
