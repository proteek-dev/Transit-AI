import type { ModelStats, RouteOption, StopSearchResult } from './types';

// NEXT_PUBLIC_ prefix required for a browser-visible env var (Next.js only
// inlines NEXT_PUBLIC_* into the client bundle). Falls back to the standard
// local FastAPI dev port so this works with zero .env setup; set
// NEXT_PUBLIC_API_BASE_URL to point at a deployed backend.
export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? 'http://localhost:8000';

export class ApiError extends Error {
  constructor(
    message: string,
    public readonly status?: number,
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

async function getJson<T>(path: string, signal?: AbortSignal): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, { signal });
  } catch (cause) {
    // Let caller-initiated aborts surface as-is so they can be told apart
    // from real network failures.
    if (signal?.aborted) throw cause;
    throw new ApiError(
      `Network error calling ${path}: ${cause instanceof Error ? cause.message : String(cause)}`,
    );
  }
  if (!response.ok) {
    const body = await response.text().catch(() => '');
    throw new ApiError(`${path} returned ${response.status}: ${body || response.statusText}`, response.status);
  }
  return response.json() as Promise<T>;
}

/**
 * GET /routes?from_stop_id=&to_stop_id=&departure= -- direct + transfer
 * journeys ranked by predicted arrival, or [] if none exist for this pair
 * (not an error; see webapp/backend/main.py's _find_ranked_routes()).
 * `departure` is an optional ISO 8601 string; omitted means "now".
 */
export async function fetchRoutes(
  fromStopId: string,
  toStopId: string,
  departure?: string,
): Promise<RouteOption[]> {
  const params = new URLSearchParams({ from_stop_id: fromStopId, to_stop_id: toStopId });
  if (departure) {
    params.set('departure', departure);
  }
  return getJson<RouteOption[]>(`/routes?${params.toString()}`);
}

type FetchStopsSearchOptions = {
  limit?: number;      // integer, 1..50; omit to accept backend default (10)
  lat?: number;        // -90..90
  lon?: number;        // -180..180
  signal?: AbortSignal;
};

function isFiniteInRange(value: number, min: number, max: number): boolean {
  return Number.isFinite(value) && value >= min && value <= max;
}

/**
 * GET /stops/search?q=&limit=&lat=&lon= -- stops matching `q`, see
 * StopSearchResult for when distance_km is populated. Inputs are validated
 * here against the backend's constraints so normal use never hits its 422.
 */
export async function fetchStopsSearch(
  q: string,
  opts?: FetchStopsSearchOptions,
): Promise<StopSearchResult[]> {
  const { limit, lat, lon, signal } = opts ?? {};

  if (typeof q !== 'string' || q.length < 1 || q.length > 100) {
    throw new ApiError('q must be a string of 1 to 100 characters');
  }
  if (limit !== undefined && !(Number.isInteger(limit) && limit >= 1 && limit <= 50)) {
    throw new ApiError('limit must be an integer from 1 to 50');
  }
  if ((lat === undefined) !== (lon === undefined)) {
    throw new ApiError('lat and lon must be provided together or both omitted');
  }
  if (lat !== undefined && !isFiniteInRange(lat, -90, 90)) {
    throw new ApiError('lat must be a finite number from -90 to 90');
  }
  if (lon !== undefined && !isFiniteInRange(lon, -180, 180)) {
    throw new ApiError('lon must be a finite number from -180 to 180');
  }

  const params = new URLSearchParams({ q });
  if (limit !== undefined) {
    params.set('limit', String(limit));
  }
  if (lat !== undefined && lon !== undefined) {
    params.set('lat', String(lat));
    params.set('lon', String(lon));
  }
  return getJson<StopSearchResult[]>(`/stops/search?${params.toString()}`, signal);
}

/** GET /model/stats -- the current promoted model's training metadata. */
export async function fetchModelStats(): Promise<ModelStats> {
  return getJson<ModelStats>('/model/stats');
}
