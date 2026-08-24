import type { ModelStats, RouteOption } from './types';

// NEXT_PUBLIC_ prefix required for a browser-visible env var (Next.js only
// inlines NEXT_PUBLIC_* into the client bundle). Falls back to the live
// Render backend for local dev so this works with zero .env setup; set
// NEXT_PUBLIC_API_BASE_URL to override (e.g. a local backend on :8000).
export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? 'https://transit-ai-isne.onrender.com';

export class ApiError extends Error {
  constructor(
    message: string,
    public readonly status?: number,
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

async function getJson<T>(path: string): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`);
  } catch (cause) {
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

/** GET /model/stats -- the current promoted model's training metadata. */
export async function fetchModelStats(): Promise<ModelStats> {
  return getJson<ModelStats>('/model/stats');
}
