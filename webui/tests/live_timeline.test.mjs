// Live timeline helpers -- shape projection, stop-state classification and
// the leave-by countdown -- against the real app.js: loaded into a vm with
// just enough window/document for its top level to run (no page mode, so
// neither page initialises).
//
// Run from the repo root:  node --test 'webui/tests/*.test.mjs'
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';

const source = readFileSync(new URL('../static/app.js', import.meta.url), 'utf8');
const window = { addEventListener() {}, location: { search: '' } };
const document = { body: { dataset: {} }, querySelector: () => null, querySelectorAll: () => [] };
vm.runInNewContext(source, { window, document, console });
// Arrays made inside the vm have its realm's Array prototype, which strict
// deepEqual rejects; copy results into this realm before comparing.
const plain = (value) => JSON.parse(JSON.stringify(value));
const liveLegLayout = (...args) => plain(window.TransitAI._liveLegLayout(...args));
const classifyStops = (...args) => plain(window.TransitAI._classifyStops(...args));

const DEG_PER_KM = 1 / 111;

// A straight north-south leg (lon fixed, so planar degrees = latitude
// degrees) with stops at the given distances in km from the origin.
function leg(stopKms, lengthKm) {
  const lat0 = -27.4;
  const shape = [[lat0, 153], [lat0 - lengthKm * DEG_PER_KM, 153]];
  const stops = stopKms.map((km, i) => ({
    x: i * 10, nodeIndex: i, stop: km === null ? { lat: null, lon: null } : { lat: lat0 - km * DEG_PER_KM, lon: 153 },
  }));
  return liveLegLayout(shape, stops);
}

const at = (km) => km * DEG_PER_KM;
const states = (layout, km) => classifyStops(layout.stopProgress, at(km)).map((s) => s.state);

test('approaching (at or before the origin): no state at all', () => {
  const layout = leg([0, 1, 2], 3);
  assert.deepEqual(states(layout, 0), []);
  assert.deepEqual(states(layout, 0.02), []); // within the 30 m approach band
});

test('mid-leg: passed stops, exactly one next, the rest future', () => {
  const layout = leg([0, 1, 2, 3, 4], 4);
  assert.deepEqual(states(layout, 1.5), ['passed', 'passed', 'next', 'future', 'future']);
});

test('dwelling at a stop keeps it "next" until the bus is clear of it', () => {
  const layout = leg([0, 1, 2], 2);
  assert.deepEqual(states(layout, 1.0), ['passed', 'next', 'future']);
  assert.deepEqual(states(layout, 1.05), ['passed', 'next', 'future']); // 50 m on: inside lag + pass margin
  assert.deepEqual(states(layout, 1.2), ['passed', 'passed', 'next']);
});

test('long rail leg with ~1 km station spacing classifies by distance, not leg fraction', () => {
  // 85 km leg, inner stations 1 km apart. Margins are fixed distances; a
  // fraction of the leg (3% = 2.5 km) would keep station 1 "next" at station 3.
  const layout = leg([0, 1, 2, 3, 4, 30, 85], 85);
  assert.deepEqual(states(layout, 3.2), ['passed', 'passed', 'passed', 'passed', 'next', 'future', 'future']);
});

test('short bus leg uses the same distance margins', () => {
  const layout = leg([0, 0.3, 0.6, 0.9, 1.2], 1.2);
  assert.deepEqual(states(layout, 0.45), ['passed', 'passed', 'next', 'future', 'future']);
});

test('at the end of the leg the destination is "next" until passed', () => {
  const layout = leg([0, 1, 2], 2);
  assert.deepEqual(states(layout, 2), ['passed', 'passed', 'next']);
});

test('stops without coords get no entry; the others still classify', () => {
  const layout = leg([0, null, 2, 3], 3);
  assert.deepEqual(layout.stopProgress.map((s) => s.nodeIndex), [0, 2, 3]);
  assert.deepEqual(classifyStops(layout.stopProgress, at(2.5)).map((s) => [s.nodeIndex, s.state]),
    [[0, 'passed'], [2, 'passed'], [3, 'next']]);
});

test('out-of-order stop projections are forced non-decreasing', () => {
  const layout = leg([0, 2, 1, 3], 3); // stop 2 projects behind stop 1
  const progress = layout.stopProgress.map((s) => s.progress);
  assert.ok(progress.every((p, i) => i === 0 || p >= progress[i - 1]));
  assert.deepEqual(states(layout, 2.5), ['passed', 'passed', 'passed', 'next']);
});

test('endpoint-only fallback leg (two stops)', () => {
  const layout = leg([0, 5], 5);
  assert.deepEqual(states(layout, 2.5), ['passed', 'next']);
});

// ── Contextual pill countdown (minutesUntil) ──────────────────
const minutesUntil = (...args) => window.TransitAI._minutesUntil(...args);

test('leave_by_epoch is absolute: same answer whatever the device timezone', () => {
  const epoch = Date.UTC(2026, 9, 8, 21, 0) / 1000; // 07:00 Brisbane (UTC+10), 08:00 NSW daylight time
  const now = new Date((epoch - 5 * 60) * 1000);
  assert.equal(minutesUntil({ leave_by: '07:00', leave_by_epoch: epoch }, now), 5);
  assert.equal(minutesUntil({ leave_by: '07:00', leave_by_epoch: epoch }, new Date((epoch + 61) * 1000)), -2);
});

test('falls back to the HH:MM wall-clock when leave_by_epoch is absent', () => {
  const now = new Date(2026, 9, 8, 23, 50);
  assert.equal(minutesUntil({ leave_by: '00:15' }, now), 25); // tomorrow's
  assert.equal(minutesUntil({ leave_by: 'bad' }, now), null);
  assert.equal(minutesUntil({}, now), null);
});

// nearestPointOnShape: a rough Surfers Paradise -> Broadbeach G:link shape.
const SHAPE = [[-28.0005, 153.4295], [-28.0100, 153.4300], [-28.0230, 153.4310], [-28.0330, 153.4300]];
const nearest = (lat, lon, shape = SHAPE) => plain(window.TransitAI.nearestPointOnShape(lat, lon, shape));

test('projection: the shape ends land on t=0 of the first segment and t=1 of the last', () => {
  const start = nearest(...SHAPE[0]);
  assert.equal(start.segmentIndex, 0);
  assert.equal(start.t, 0);
  assert.ok(start.distanceSq < 1e-12);
  const end = nearest(...SHAPE[SHAPE.length - 1]);
  assert.equal(end.segmentIndex, SHAPE.length - 2);
  assert.equal(end.t, 1);
  assert.ok(end.distanceSq < 1e-12);
});

test('projection: a perpendicular offset from a segment midpoint keeps t=0.5 and measures d²', () => {
  const [a, b] = [SHAPE[1], SHAPE[2]];
  const mid = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
  const len = Math.hypot(b[0] - a[0], b[1] - a[1]);
  const normal = [-(b[1] - a[1]) / len, (b[0] - a[0]) / len];
  const d = 0.001;
  const onMid = nearest(...mid);
  assert.equal(onMid.segmentIndex, 1);
  assert.ok(Math.abs(onMid.t - 0.5) < 1e-9);
  assert.ok(onMid.distanceSq < 1e-12);
  const offMid = nearest(mid[0] + d * normal[0], mid[1] + d * normal[1]);
  assert.equal(offMid.segmentIndex, 1);
  assert.ok(Math.abs(offMid.t - 0.5) < 1e-9);
  assert.ok(Math.abs(offMid.distanceSq - d * d) < d * d * 1e-6);
});

test('projection: degenerate inputs return null', () => {
  assert.equal(nearest(-28.01, 153.43, []), null);
  assert.equal(nearest(-28.01, 153.43, [SHAPE[0]]), null);
  assert.equal(nearest(NaN, 153.43), null);
});
