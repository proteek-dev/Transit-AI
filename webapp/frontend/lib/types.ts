/**
 * Types match the live backend's actual JSON shape, confirmed by hitting
 * GET /routes and GET /model/stats directly (see webapp/backend/main.py's
 * _serialize_journey() / model_stats()) -- not guessed from the endpoint
 * names.
 */

/** GTFS mode string, per phase3/route_types.py's MODE_BY_ROUTE_TYPE
 * mapping (route_type 0/2/3/4) plus predict_delay()'s 'unknown' fallback.
 * Ferry is excluded upstream by phase3/gtfs/loader.py, so 'ferry' should
 * not appear in practice, but the type stays honest about what the field
 * can carry. Note: rail transit is reported as 'rail', not 'train'.
 */
export type Mode = 'bus' | 'tram' | 'rail' | 'ferry' | 'unknown';

export type Confidence = 'Low' | 'Medium' | 'High';

export interface RouteLeg {
  /** The stop_id this leg's predicted_arrival applies to (the alighting/destination stop). */
  stop_id: string;
  route_short_name: string;
  mode: Mode;
  /** ISO 8601, naive (no timezone suffix) local Brisbane wall-clock time. */
  departure_time: string;
  /** ISO 8601, naive, includes fractional seconds from the model's prediction. */
  predicted_arrival: string;
  confidence: Confidence;
}

export interface RouteOption {
  legs: RouteLeg[];
  total_predicted_duration_minutes: number;
  transfer_count: number;
}

export interface ModelStats {
  training_rows: number;
  days_archived: number;
  test_mae: number;
  naive_mae: number;
  pct_improvement_over_naive: number;
  model_type: string;
  training_window: {
    start: string;
    end: string;
  };
}
