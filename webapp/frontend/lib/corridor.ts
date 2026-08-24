/**
 * Static placeholder corridor for the hero visual, from the GTFS stop_name
 * lookup done earlier against phase3/gtfs/loader.py's stop index:
 *
 *   FROM_STOP_ID: Robina station's first platform-level child
 *     (stop_id='600118', stop_name='Robina station, platform 1',
 *      parent_station='place_rbnsta')
 *   TO_STOP_ID: Central station, platform 3
 *     (stop_id='600016', parent_station='place_censta')
 *
 * These are placeholders, not a verified working pair -- Robina/Central
 * each have several platform-level stop_ids (see the earlier lookup output
 * for the full list), and GTFS trips don't always call at every platform.
 * If GET /routes?from_stop_id={FROM_STOP_ID}&to_stop_id={TO_STOP_ID}
 * returns an empty array, swap in a sibling platform stop_id from the same
 * cluster (place_rbnsta / place_censta) rather than assuming the corridor
 * itself has no service.
 */
export const FROM_STOP_ID = '600118';
export const TO_STOP_ID = '600016';
