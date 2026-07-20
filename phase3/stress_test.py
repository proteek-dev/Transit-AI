"""Diagnostic stress test for the Phase 3 route-finding pipeline.

DIAGNOSTIC ONLY. This script does not modify gtfs_data.py, app.py, or any
other source file — it only calls the existing public API and prints what
it finds, including root-cause analysis for zero-result cases. Do not treat
any print statement here as a fix; it is a report.

Run: python phase3/stress_test.py
"""
from __future__ import annotations

from datetime import datetime, timedelta

from gtfs_data import (
    find_multi_leg_trips,
    find_trips,
    load_gtfs_data,
    search_stops,
)

WINDOW_MINUTES = 480


def section(title: str) -> None:
    print()
    print('=' * 100)
    print(title)
    print('=' * 100)


def subsection(title: str) -> None:
    print()
    print('-' * 90)
    print(title)
    print('-' * 90)


def _next_monday(from_date):
    """Next Monday strictly after from_date (today is itself a Monday, so
    'tomorrow (Monday)' as specified in the brief resolves to +7 days)."""
    days_ahead = (7 - from_date.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7
    return from_date + timedelta(days=days_ahead)


def pick_stop_group(query: str) -> dict | None:
    """Same heuristic as test_transfers.py / app.py's typeahead: prefer a
    'station' entry over a generic street stop sharing the name. This is
    what a user picking the FIRST sensible typeahead result would get."""
    results = search_stops(query, limit=50)
    if not results:
        return None
    exact_station = f'{query.strip().lower()} station'
    for r in results:
        if r['stop_name'].lower() == exact_station:
            return r
    station_matches = [r for r in results if 'station' in r['stop_name'].lower()]
    return station_matches[0] if station_matches else results[0]


def raw_stops_matching(data, substr: str):
    """Ground-truth stops.txt rows whose stop_name contains substr (case-insensitive).
    Bypasses search_stops()'s grouping/fuzzy logic entirely."""
    return data.stops[data.stops['stop_name'].str.contains(substr, case=False, na=False)]


def print_raw_stop_rows(df, label: str) -> None:
    print(f'  Raw stops.txt rows matching {label!r}: {len(df)}')
    for row in df.itertuples(index=False):
        parent = row.parent_station if isinstance(row.parent_station, str) and row.parent_station.strip() else '(none)'
        loc_type = row.location_type if hasattr(row, 'location_type') else '(missing col)'
        print(f'    stop_id={row.stop_id:<12} stop_name={row.stop_name!r:<35} '
              f'parent_station={parent:<12} location_type={loc_type}')


def glink_route_ids(data) -> set:
    """route_id set for G:link: route_type == 0 OR route_short_name contains 'GCL'."""
    routes = data.routes
    mask = (routes['route_type'] == 0)
    if 'route_short_name' in routes.columns:
        mask = mask | routes['route_short_name'].str.contains('GCL', case=False, na=False)
    return set(routes.loc[mask, 'route_id'])


def glink_stop_ids(data, route_ids: set) -> set:
    """Ground-truth stop_ids actually visited by G:link trips, per stop_times.txt."""
    trips_glink = data.trips[data.trips['route_id'].isin(route_ids)]
    st_glink = data.stop_times[data.stop_times['trip_id'].isin(trips_glink['trip_id'])]
    return set(st_glink['stop_id'])


def diagnose_zero(data, origin_ids: list[str], dest_ids: list[str],
                   departure_after: datetime, window_minutes: int) -> None:
    """Walk find_trips()'s filter pipeline stage by stage to find exactly
    where a pair drops to zero: shared trip_id -> stop_sequence ordering ->
    calendar/service_id -> time window."""
    st = data.stop_times
    trips = data.trips
    query_date = departure_after.date()
    active = data.active_service_ids(query_date)

    origin_st = st[st['stop_id'].isin(origin_ids)]
    dest_st = st[st['stop_id'].isin(dest_ids)]
    print(f'    origin stop_times rows: {len(origin_st)}  (stop_ids={origin_ids})')
    print(f'    dest stop_times rows:   {len(dest_st)}  (stop_ids={dest_ids})')

    shared_trip_ids = set(origin_st['trip_id']) & set(dest_st['trip_id'])
    print(f'    [1] trip_ids visiting BOTH origin and dest (any date/seq/time): {len(shared_trip_ids)}')
    if not shared_trip_ids:
        print('    -> ROOT CAUSE: no trip_id in stop_times.txt ever visits both stop sets. '
              'Not a direct route on any pattern (need a transfer, or stop_ids are wrong).')
        return

    ordered_ok = set()
    for tid in shared_trip_ids:
        o_seq = origin_st.loc[origin_st['trip_id'] == tid, 'stop_sequence'].min()
        d_seq = dest_st.loc[dest_st['trip_id'] == tid, 'stop_sequence'].max()
        if o_seq < d_seq:
            ordered_ok.add(tid)
    print(f'    [2] of those, origin_stop_sequence < dest_stop_sequence: {len(ordered_ok)}')
    if not ordered_ok:
        print('    -> ROOT CAUSE: shared trips exist but destination never comes AFTER origin '
              'in stop_sequence on any of them (wrong direction, or these are two different '
              'physical patterns that never co-occur in this order).')
        return

    trip_service = trips.set_index('trip_id')['service_id']
    active_ok = {tid for tid in ordered_ok if trip_service.get(tid) in active}
    print(f'    [3] of those, service_id ACTIVE on {query_date} ({query_date.strftime("%A")}, '
          f'{len(active)} services active that day): {len(active_ok)}')
    if not active_ok:
        sample_services = sorted({trip_service.get(tid) for tid in ordered_ok})[:10]
        print(f'    -> ROOT CAUSE: calendar filter. Correctly-ordered trips exist but their '
              f'service_id(s) {sample_services} are not active on {query_date}. '
              f'Check calendar.txt / calendar_dates.txt for these service_ids.')
        return

    day_midnight = datetime(query_date.year, query_date.month, query_date.day)
    window_end = departure_after + timedelta(minutes=window_minutes)
    in_window = set()
    sample_departures = []
    for tid in active_ok:
        dep_raw = origin_st.loc[origin_st['trip_id'] == tid, 'departure_time'].iloc[0]
        h, m, s = (int(x) for x in dep_raw.split(':'))
        dep_dt = day_midnight + timedelta(hours=h, minutes=m, seconds=s)
        sample_departures.append((tid, dep_raw))
        if departure_after <= dep_dt <= window_end:
            in_window.add(tid)
    print(f'    [4] of those, origin departure within window '
          f'[{departure_after} .. {window_end}]: {len(in_window)}')
    if not in_window:
        print(f'    -> ROOT CAUSE: time window. Active, correctly-ordered trips exist but none '
              f'depart origin in the requested window. Sample active departure_times '
              f'(raw, may exceed 24h): {sorted(sample_departures, key=lambda x: x[1])[:8]}')
        return

    print('    -> UNEXPECTED: trips exist, correctly ordered, active, in-window — '
          'find_trips() should have returned these. This points to a bug inside '
          'find_trips()/_find_trips_core() itself, not the data.')


def part1(data, departure_after: datetime) -> None:
    section('PART 1 — Diagnose Cavill Avenue -> Helensvale')

    subsection('1. gtfs_data already initialised (see load banner above)')

    subsection('2/3. search_stops() results with raw stops.txt fields')
    cavill_results = search_stops('Cavill Avenue', limit=20)
    helensvale_results = search_stops('Helensvale', limit=20)

    print(f'  search_stops("Cavill Avenue") -> {len(cavill_results)} group(s):')
    for g in cavill_results:
        print(f'    group stop_name={g["stop_name"]!r} stop_ids={g["stop_ids"]}')
    print()
    print_raw_stop_rows(raw_stops_matching(data, 'Cavill'), 'Cavill')

    print()
    print(f'  search_stops("Helensvale") -> {len(helensvale_results)} group(s):')
    for g in helensvale_results:
        print(f'    group stop_name={g["stop_name"]!r} stop_ids={g["stop_ids"]}')
    print()
    print_raw_stop_rows(raw_stops_matching(data, 'Helensvale'), 'Helensvale')

    subsection('4. Which of those stop_ids are actually served by a G:link (tram) trip?')
    glink_routes = glink_route_ids(data)
    print(f'  G:link route_ids (route_type==0 or short_name contains "GCL"): {sorted(glink_routes)}')
    glink_stops = glink_stop_ids(data, glink_routes)

    cavill_raw_ids = list(raw_stops_matching(data, 'Cavill')['stop_id'])
    helensvale_raw_ids = list(raw_stops_matching(data, 'Helensvale')['stop_id'])
    print(f'  Cavill stop_ids in G:link stop_times: '
          f'{[s for s in cavill_raw_ids if s in glink_stops]} '
          f'(of {len(cavill_raw_ids)} raw Cavill stop_ids)')
    print(f'  Helensvale stop_ids in G:link stop_times: '
          f'{[s for s in helensvale_raw_ids if s in glink_stops]} '
          f'(of {len(helensvale_raw_ids)} raw Helensvale stop_ids)')

    subsection('5/6. find_trips() for every (origin group x dest group) combination')
    print(f'  departure_after={departure_after} ({departure_after.strftime("%A")}), '
          f'window={WINDOW_MINUTES} min\n')
    any_pass = False
    for o in cavill_results:
        for d in helensvale_results:
            print(f'  [{o["stop_name"]!r} -> {d["stop_name"]!r}] '
                  f'origin_ids={o["stop_ids"]} dest_ids={d["stop_ids"]}')
            trips = find_trips(o['stop_ids'], d['stop_ids'], departure_after, WINDOW_MINUTES)
            print(f'    find_trips() -> {len(trips)} result(s)')
            if trips:
                any_pass = True
                for t in trips[:3]:
                    print(f"      {t['route_short_name'] or t['route_id']:<8} "
                          f"dep {t['origin_departure_time']:%H:%M:%S} -> "
                          f"arr {t['dest_arrival_time']:%H:%M:%S} "
                          f"fallback_schedule={t['fallback_schedule']}")
            else:
                diagnose_zero(data, o['stop_ids'], d['stop_ids'], departure_after, WINDOW_MINUTES)
            print()
    if not any_pass:
        print('  ALL origin-group x dest-group combinations for Cavill Avenue -> Helensvale '
              'returned ZERO results via find_trips().')

    subsection('7. Manual stop_times.txt query — bypass find_trips() entirely')
    st = data.stop_times
    cavill_st = st[st['stop_id'].isin(cavill_raw_ids)][['trip_id', 'stop_id', 'stop_sequence']].rename(
        columns={'stop_id': 'cavill_stop_id', 'stop_sequence': 'cavill_seq'}
    )
    helensvale_st = st[st['stop_id'].isin(helensvale_raw_ids)][['trip_id', 'stop_id', 'stop_sequence']].rename(
        columns={'stop_id': 'helensvale_stop_id', 'stop_sequence': 'helensvale_seq'}
    )
    merged = cavill_st.merge(helensvale_st, on='trip_id', how='inner')
    ordered = merged[merged['cavill_seq'] < merged['helensvale_seq']]
    print(f'  trip_ids visiting a Cavill stop_id then a Helensvale stop_id later in '
          f'stop_sequence (ANY calendar/date): {ordered["trip_id"].nunique()}')
    if not ordered.empty:
        sample = ordered.drop_duplicates('trip_id').head(10)
        trip_route = data.trips.set_index('trip_id')['route_id']
        for row in sample.itertuples(index=False):
            route_id = trip_route.get(row.trip_id, '?')
            print(f'    trip_id={row.trip_id} route_id={route_id} '
                  f'cavill_stop={row.cavill_stop_id}(seq={row.cavill_seq}) -> '
                  f'helensvale_stop={row.helensvale_stop_id}(seq={row.helensvale_seq})')
        print('  -> CONFIRMS the underlying GTFS data DOES contain trips that visit '
              'Cavill then Helensvale in the correct order. The zero-result behaviour '
              'above is therefore a filtering issue (calendar/window/stop_id selection), '
              'not missing source data.')
    else:
        print('  -> No trip in stop_times.txt EVER visits a Cavill stop_id before a '
              'Helensvale stop_id. If true, this is a genuine direct-route gap in the '
              'source GTFS data, not a bug in find_trips().')


PAIRS = [
    # (id, label, mode, origin_query, dest_query, expectation)
    # expectation: 'should_work' | 'known_gap' | 'edge_case'
    ('T1', 'Broadbeach South -> Cavill Avenue', 'Tram', 'Broadbeach South', 'Cavill Avenue', 'should_work'),
    ('T2', 'Cavill Avenue -> Broadbeach South', 'Tram', 'Cavill Avenue', 'Broadbeach South', 'should_work'),
    ('T3', 'Cavill Avenue -> Helensvale', 'Tram', 'Cavill Avenue', 'Helensvale', 'should_work'),
    ('T4', 'Helensvale -> Broadbeach South', 'Tram', 'Helensvale', 'Broadbeach South', 'should_work'),
    ('T5', 'Surfers Paradise -> HOTA', 'Tram', 'Surfers Paradise', 'HOTA', 'should_work'),
    ('R1', 'Helensvale -> Roma Street', 'Train', 'Helensvale', 'Roma Street', 'should_work'),
    ('R2', 'Roma Street -> Central', 'Train', 'Roma Street', 'Central', 'should_work'),
    ('R3', 'Varsity Lakes -> Robina', 'Train', 'Varsity Lakes', 'Robina', 'should_work'),
    ('B1', 'Southport -> Broadbeach', 'Bus', 'Southport', 'Broadbeach', 'should_work'),
    ('B2', 'Surfers Paradise -> Griffith University Gold Coast', 'Bus', 'Surfers Paradise', 'Griffith University Gold Coast', 'should_work'),
    ('X1', 'Broadbeach South -> Roma Street', 'Cross-mode (tram->train)', 'Broadbeach South', 'Roma Street', 'should_work'),
    ('X2', 'Broadbeach South -> Sunnybank', 'Cross-mode (known gap)', 'Broadbeach South', 'Sunnybank', 'known_gap'),
    ('X3', 'Robina -> Cavill Avenue', 'Cross-mode (train->tram)', 'Robina', 'Cavill Avenue', 'should_work'),
    ('X4', 'Surfers Paradise -> Varsity Lakes', 'Cross-mode (bus/tram->train)', 'Surfers Paradise', 'Varsity Lakes', 'should_work'),
    ('E1', 'Cavill Avenue -> Cavill Avenue', 'Edge case', 'Cavill Avenue', 'Cavill Avenue', 'edge_case'),
]


def part2(data, departure_after: datetime) -> list[dict]:
    section('PART 2 — Systematic stress test matrix (15 pairs)')
    print(f'departure_after={departure_after} ({departure_after.strftime("%A")}), '
          f'window={WINDOW_MINUTES} min\n')

    results = []
    for pair_id, label, mode, origin_query, dest_query, expectation in PAIRS:
        subsection(f'{pair_id}: {label}  [{mode}]  expectation={expectation}')

        origin = pick_stop_group(origin_query)
        dest = pick_stop_group(dest_query)
        if not origin or not dest:
            missing = 'origin' if not origin else 'destination'
            print(f'  NO STOP MATCH for {missing} query ({origin_query!r} / {dest_query!r})')
            results.append({
                'id': pair_id, 'label': label, 'mode': mode, 'expectation': expectation,
                'result': 'FAIL', 'kind': 'n/a', 'trip_count': 0,
                'note': f'search_stops() returned no match for {missing}',
            })
            continue

        print(f'  origin selected: {origin["stop_name"]!r} stop_ids={origin["stop_ids"]}')
        print(f'  dest selected:   {dest["stop_name"]!r} stop_ids={dest["stop_ids"]}')

        direct = find_trips(origin['stop_ids'], dest['stop_ids'], departure_after, WINDOW_MINUTES)
        print(f'  find_trips() (direct): {len(direct)} result(s)')

        journeys = []
        if not direct:
            journeys = find_multi_leg_trips(
                origin['stop_ids'], dest['stop_ids'], departure_after,
                window_minutes=WINDOW_MINUTES, max_transfers=3, max_results=5,
            )
            print(f'  find_multi_leg_trips(max_transfers=3): {len(journeys)} journey(s)')
            for j in journeys[:3]:
                route_chain = ' -> '.join(
                    leg['trip']['route_short_name'] or leg['trip']['route_id'] for leg in j['legs']
                )
                print(f'    {j["num_transfers"]} transfer(s), ~{j["total_minutes"]} min: {route_chain}')

        found_any = bool(direct) or bool(journeys)
        kind = 'direct' if direct else ('transfer' if journeys else 'none')
        trip_count = len(direct) if direct else len(journeys)

        if expectation == 'edge_case':
            verdict = 'PASS' if not found_any else 'UNEXPECTED_RESULT'
            note = 'same-stop query returned empty as expected' if not found_any else 'same-stop query unexpectedly returned results'
        elif expectation == 'known_gap':
            verdict = 'EXPECTED_FAIL' if not found_any else 'GAP_CLOSED_UNEXPECTEDLY'
            note = 'no service, as previously known' if not found_any else 'this known gap now returns results — re-verify'
        else:  # should_work
            verdict = 'PASS' if found_any else 'FAIL'
            note = '' if found_any else 'zero results for a route expected to exist'

        print(f'  VERDICT: {verdict}')

        if not found_any and expectation != 'edge_case':
            subsection(f'{pair_id} diagnostic (direct find_trips() zero-result breakdown)')
            diagnose_zero(data, origin['stop_ids'], dest['stop_ids'], departure_after, WINDOW_MINUTES)

        results.append({
            'id': pair_id, 'label': label, 'mode': mode, 'expectation': expectation,
            'result': verdict, 'kind': kind, 'trip_count': trip_count, 'note': note,
        })
        print()

    return results


def part3(data, results: list[dict]) -> None:
    section('PART 3 — Coverage summary')

    subsection('1. Results table')
    header = f'{"ID":<4} {"Mode":<26} {"Kind":<10} {"Result":<22} {"Trips":<6} Pair'
    print(header)
    print('-' * len(header))
    for r in results:
        print(f'{r["id"]:<4} {r["mode"]:<26} {r["kind"]:<10} {r["result"]:<22} '
              f'{r["trip_count"]:<6} {r["label"]}')

    subsection('2. Mode coverage — which route_types returned >=1 result?')
    for mode_label in sorted({m for _, _, m, _, _, _ in PAIRS}):
        pairs_for_mode = [r for r in results if r['mode'] == mode_label]
        any_hit = any(r['trip_count'] > 0 for r in pairs_for_mode)
        print(f'  {mode_label:<32} {"covered" if any_hit else "NO RESULTS ON ANY PAIR"}')

    subsection('3. Directionality check — A->B works but B->A does not (or vice versa)')
    by_id = {r['id']: r for r in results}
    a, b, note = 'T1', 'T2', 'Broadbeach South <-> Cavill Avenue'
    ra, rb = by_id[a], by_id[b]
    a_ok = ra['trip_count'] > 0
    b_ok = rb['trip_count'] > 0
    if a_ok != b_ok:
        print(f'  ASYMMETRY: {note} — {a}={ra["result"]} ({ra["trip_count"]} trips) vs '
              f'{b}={rb["result"]} ({rb["trip_count"]} trips)')
    else:
        print(f'  Symmetric: {note} — {a}={ra["result"]} vs {b}={rb["result"]}')
    print(f'  Informative (not a strict A<->B pair): T3 (Cavill->Helensvale)={by_id["T3"]["result"]} '
          f'vs T4 (Helensvale->Broadbeach South)={by_id["T4"]["result"]}')

    subsection('4. All FAILs with suspected root cause (see per-pair diagnostic above for detail)')
    fails = [r for r in results if r['result'] in ('FAIL', 'UNEXPECTED_RESULT', 'GAP_CLOSED_UNEXPECTEDLY')]
    if not fails:
        print('  No unexpected FAILs.')
    else:
        for r in fails:
            print(f'  {r["id"]} ({r["label"]}): {r["result"]} — {r["note"] or "see diagnostic output above"}')

    subsection('5. Overall verdict')
    hard_fails = [r for r in results if r['result'] == 'FAIL']
    print(f'  {len(hard_fails)} of {len(results)} pairs FAILed against expectation.')
    if hard_fails:
        print('  Route-finding pipeline is NOT production-ready as-is: '
              f'{", ".join(r["id"] for r in hard_fails)} return zero results for routes '
               'that should exist. See diagnostics above for root cause per pair.')
    else:
        print('  All should-work pairs returned results. Pipeline appears production-ready '
              'for the tested matrix (known gaps and edge cases behaved as expected).')


def part4(data, departure_after: datetime) -> None:
    section('PART 4 — Stop index audit')

    print(f'  Total stops in stop_to_routes index: {len(data.stop_to_routes)}')
    print(f'  Total routes in route_to_stops index: {len(data.route_to_stops)}')

    tram_route_ids = set(data.routes.loc[data.routes['route_type'] == 0, 'route_id'])
    print(f'  route_type==0 (tram) route_ids: {sorted(tram_route_ids)}')

    tram_stops_in_index = {
        stop_id for stop_id, routes in data.stop_to_routes.items()
        if routes & tram_route_ids
    }
    print(f'  Stops in stop_to_routes with a route_type=0 (tram) route: {len(tram_stops_in_index)}')
    print(f'    stop_ids: {sorted(tram_stops_in_index)}')

    subsection('Per-tram-route reachability via find_trips() (first stop -> last stop, per route pattern)')
    reachable_stops = set()
    unreachable_routes = []
    for route_id in sorted(tram_route_ids):
        stops_on_route = data.route_to_stops.get(route_id, [])
        if len(stops_on_route) < 2:
            print(f'  route_id={route_id}: fewer than 2 stops in route_to_stops ({stops_on_route}) — skipping')
            continue
        first_stop, last_stop = stops_on_route[0], stops_on_route[-1]
        trips = find_trips([first_stop], [last_stop], departure_after, WINDOW_MINUTES)
        status = 'REACHABLE' if trips else 'ZERO RESULTS'
        print(f'  route_id={route_id}: {len(stops_on_route)} stops, '
              f'first={first_stop} -> last={last_stop}: {len(trips)} trip(s) -> {status}')
        if trips:
            reachable_stops.update(stops_on_route)
        else:
            unreachable_routes.append(route_id)

    print(f'\n  Tram stops confirmed reachable via find_trips() end-to-end on their own route: '
          f'{len(reachable_stops)} of {len(tram_stops_in_index)}')
    if unreachable_routes:
        print(f'  Tram routes with ZERO end-to-end find_trips() results: {unreachable_routes}')

    subsection('Ground-truth G:link stops (from raw trips/stop_times/routes) vs stop_to_routes index')
    glink_routes = glink_route_ids(data)
    ground_truth_glink_stops = glink_stop_ids(data, glink_routes)
    missing_from_index = ground_truth_glink_stops - set(data.stop_to_routes.keys())
    print(f'  Ground-truth G:link stop_ids (from stop_times.txt): {len(ground_truth_glink_stops)}')
    print(f'  Of those, present as a key in stop_to_routes: '
          f'{len(ground_truth_glink_stops) - len(missing_from_index)}')
    if missing_from_index:
        print(f'  G:link stops MISSING from stop_to_routes index: {sorted(missing_from_index)}')
        stop_names = data.stops.set_index('stop_id')['stop_name']
        for sid in sorted(missing_from_index):
            print(f'    {sid}: stop_name={stop_names.get(sid, "?")!r}')
    else:
        print('  All G:link stops from stop_times.txt are present in stop_to_routes. Index is complete.')


def main() -> None:
    import gtfs_data as gtfs_data_module

    section('Loading static GTFS snapshot')
    data = load_gtfs_data()
    print(
        f'Loaded snapshot {data.snapshot_date}: {len(data.stops):,} stops, '
        f'{len(data.routes):,} routes, {len(data.trips):,} trips, '
        f'{len(data.stop_times):,} stop_times rows'
    )

    today = datetime.now().date()
    target_date = _next_monday(today)
    departure_after = datetime(target_date.year, target_date.month, target_date.day, 6, 0, 0)
    print(f'\nToday is {today} ({today.strftime("%A")}). '
          f'Test date resolved to next Monday: {target_date} at 06:00 AEST.')
    snapshot_date = datetime.strptime(data.snapshot_date, '%Y-%m-%d').date()
    print(f'Snapshot capture date: {snapshot_date}  '
          f'(gap to test date: {(target_date - snapshot_date).days} days — '
          f'find_trips() only applies its 1-week fallback within '
          f'{gtfs_data_module.FALLBACK_THIN_COVERAGE_DAYS} days of snapshot date)')

    part1(data, departure_after)
    results = part2(data, departure_after)
    part3(data, results)
    part4(data, departure_after)

    section('DIAGNOSTIC RUN COMPLETE — no source files were modified')


if __name__ == '__main__':
    main()
