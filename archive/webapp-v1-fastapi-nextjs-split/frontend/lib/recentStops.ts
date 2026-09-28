import type { StopSearchResult } from './types';

/**
 * Recently picked stops, persisted in localStorage. distance_km is
 * deliberately excluded -- it depends on where the user was when they
 * searched, so it's meaningless on a later visit.
 *
 * Storage failures (SSR, private-mode Safari, quota, corrupt JSON) are
 * expected degraded behaviour: reads fall back to [] and writes no-op,
 * silently.
 */
export type RecentStop = Pick<
  StopSearchResult,
  'stop_id' | 'stop_ids' | 'stop_name' | 'stop_lat' | 'stop_lon'
>;

const MAX_RECENT_STOPS = 8;
// Versioned so a future shape change can move to :v2 without misreading v1 data.
const STORAGE_KEY = 'transit-ai:recent-stops:v1';

function isRecentStop(item: unknown): item is RecentStop {
  if (typeof item !== 'object' || item === null) return false;
  const s = item as Record<string, unknown>;
  return (
    typeof s.stop_id === 'string' &&
    typeof s.stop_name === 'string' &&
    typeof s.stop_lat === 'number' &&
    typeof s.stop_lon === 'number' &&
    Array.isArray(s.stop_ids) &&
    s.stop_ids.every((id) => typeof id === 'string')
  );
}

export function getRecentStops(): RecentStop[] {
  if (typeof window === 'undefined') return [];
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (raw === null) return [];
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(isRecentStop).map(toRecentStop);
  } catch {
    return [];
  }
}

export function addRecentStop(stop: StopSearchResult): void {
  if (typeof window === 'undefined') return;
  try {
    const entry = toRecentStop(stop);
    const next = [entry, ...getRecentStops().filter((s) => s.stop_id !== entry.stop_id)].slice(
      0,
      MAX_RECENT_STOPS,
    );
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
  } catch {
    // Degraded: recents just don't persist.
  }
}

export function clearRecentStops(): void {
  if (typeof window === 'undefined') return;
  try {
    window.localStorage.removeItem(STORAGE_KEY);
  } catch {
    // Degraded: nothing to clear that we can reach.
  }
}

/** Copies only the persisted fields, dropping distance_km and anything else. */
function toRecentStop(stop: RecentStop): RecentStop {
  return {
    stop_id: stop.stop_id,
    stop_ids: [...stop.stop_ids],
    stop_name: stop.stop_name,
    stop_lat: stop.stop_lat,
    stop_lon: stop.stop_lon,
  };
}
