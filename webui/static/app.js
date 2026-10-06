// Transit AI web UI — vanilla JS, no build step. Every DOM write uses
// textContent / createElement; no HTML strings are ever parsed.

(() => {
  'use strict';

  const DEBOUNCE_MS = 200;
  const SEARCH_LIMIT = 8;
  const RECENTS_KEY = 'webui.recentStops.v2';
  const RECENTS_KEY_V1 = 'webui.recentStops.v1'; // read once by migrateRecentsV1(), then removed
  const MAX_RECENTS = 5;
  const NEAR_MATCH_NOTE = 'No exact match — showing near matches';

  const MODE_LABEL = { bus: 'Bus', tram: 'Tram', rail: 'Train', ferry: 'Ferry', unknown: 'Service' };
  const PILL_CLASS = { High: 'pill-high', Medium: 'pill-medium', Low: 'pill-low' };

  // ── Recent stops (localStorage) ─────────────────────────────
  // Every storage call is guarded: Safari private mode and blocked storage
  // throw, and a failure there should only mean "no recents".

  // Accepts both shapes: v1 {stop_id, stop_name} and v2, which adds
  // stop_lat + stop_lon. Coords are checked separately (hasCoords).
  function isRecent(item) {
    return item !== null && typeof item === 'object'
      && typeof item.stop_id === 'string' && typeof item.stop_name === 'string';
  }

  function hasCoords(item) {
    return Number.isFinite(item.stop_lat) && Number.isFinite(item.stop_lon);
  }

  // Unknown fields are kept (a later version may add e.g. mode). Coords
  // that aren't both finite numbers demote the entry to v1: the coords are
  // dropped, the entry itself is kept.
  function normalizeRecent(item) {
    if (hasCoords(item)) return { ...item };
    const { stop_lat, stop_lon, ...rest } = item;
    return rest;
  }

  // One-time v1 -> v2 copy. Runs only while the v2 key is absent and
  // removes the v1 key once v2 is written, so it runs at most once.
  function migrateRecentsV1() {
    try {
      const storage = window.localStorage;
      if (storage.getItem(RECENTS_KEY) !== null) return;
      const legacy = JSON.parse(storage.getItem(RECENTS_KEY_V1) || '[]');
      if (!Array.isArray(legacy)) return;
      const entries = legacy.filter(isRecent)
        .map(({ stop_lat, stop_lon, ...rest }) => rest) // v1 entries carry no coords
        .slice(0, MAX_RECENTS);
      if (entries.length === 0) return;
      storage.setItem(RECENTS_KEY, JSON.stringify(entries));
      storage.removeItem(RECENTS_KEY_V1);
    } catch (err) {
      // Unreadable v1 or blocked storage: skip; the next read retries.
    }
  }

  function readRecents() {
    migrateRecentsV1();
    try {
      const parsed = JSON.parse(window.localStorage.getItem(RECENTS_KEY) || '[]');
      if (!Array.isArray(parsed)) return [];
      return parsed.filter(isRecent).map(normalizeRecent).slice(0, MAX_RECENTS);
    } catch (err) {
      return [];
    }
  }

  // Always writes v2: coords whenever the picked stop has them. Re-picking a
  // stored stop keeps that entry's other fields. An entry without stop_id +
  // stop_name is never written.
  function pushRecent(stop) {
    if (!isRecent(stop)) return;
    const current = readRecents();
    const previous = current.find((s) => s.stop_id === stop.stop_id);
    const entry = { ...previous, stop_id: stop.stop_id, stop_name: stop.stop_name };
    if (hasCoords(stop)) {
      entry.stop_lat = stop.stop_lat;
      entry.stop_lon = stop.stop_lon;
    }
    const next = [entry, ...current.filter((s) => s.stop_id !== entry.stop_id)].slice(0, MAX_RECENTS);
    try {
      window.localStorage.setItem(RECENTS_KEY, JSON.stringify(next));
    } catch (err) {
      // Storage unavailable — recents just don't persist.
    }
  }

  // ── Helpers ─────────────────────────────────────────────────

  // Naive ISO 'YYYY-MM-DDTHH:MM:SS…' (Brisbane wall-clock) -> 'HH:MM'. Sliced,
  // not parsed, so the viewer's own timezone can't shift it.
  function hhmm(iso) {
    return iso.slice(11, 16);
  }

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  // ── Status line ─────────────────────────────────────────────

  const statusEl = document.querySelector('#status');

  function setStatus(message, kind) {
    statusEl.textContent = message;
    if (kind) statusEl.dataset.kind = kind;
    else delete statusEl.dataset.kind;
  }

  // Any interaction with the form clears a stale error (but not a normal
  // informational status like "Ranked by predicted arrival.").
  function clearError() {
    if (statusEl.dataset.kind === 'error') setStatus('');
  }

  // ── Stop picker (WAI-ARIA combobox, list autocomplete) ──────

  // options.onChange(): called whenever `selected` changes (set or voided).
  // options.bias(): {lat, lon} to geo-bias /stops/search with, or null.
  function makePicker(prefix, options = {}) {
    const input = document.querySelector(`#${prefix}-input`);
    const dropdown = document.querySelector(`#${prefix}-dropdown`);
    const note = document.querySelector(`#${prefix}-note`);
    const listbox = document.querySelector(`#${prefix}-listbox`);
    const notifyChange = () => { if (options.onChange) options.onChange(); };

    const picker = { selected: null };
    let items = [];
    let activeIndex = -1;
    let timer = null;
    let requestSeq = 0;

    function isOpen() {
      return !dropdown.hidden;
    }

    function open() {
      dropdown.hidden = false;
      input.setAttribute('aria-expanded', 'true');
    }

    function close() {
      dropdown.hidden = true;
      input.setAttribute('aria-expanded', 'false');
      setActive(-1);
    }

    function setActive(index) {
      const options = listbox.children;
      if (activeIndex >= 0 && options[activeIndex]) {
        options[activeIndex].setAttribute('aria-selected', 'false');
      }
      activeIndex = index;
      if (index >= 0 && options[index]) {
        options[index].setAttribute('aria-selected', 'true');
        input.setAttribute('aria-activedescendant', options[index].id);
        options[index].scrollIntoView({ block: 'nearest' });
      } else {
        input.removeAttribute('aria-activedescendant');
      }
    }

    function setNote(text) {
      note.textContent = text || '';
      note.hidden = !text;
    }

    // Replace the dropdown contents. `noteText` is a non-interactive header
    // ("Recent", the near-match label, or an empty/error message).
    function render(newItems, noteText) {
      items = newItems;
      activeIndex = -1;
      input.removeAttribute('aria-activedescendant');
      setNote(noteText);
      listbox.replaceChildren(...items.map((stop, i) => {
        const li = el('li', 'option');
        li.id = `${prefix}-option-${i}`;
        li.setAttribute('role', 'option');
        li.setAttribute('aria-selected', 'false');
        li.append(el('span', 'option-name', stop.stop_name), el('span', 'option-id', stop.stop_id));
        li.addEventListener('mouseenter', () => setActive(i));
        // Keep focus on the input so the click lands before blur closes the list.
        li.addEventListener('mousedown', (event) => event.preventDefault());
        li.addEventListener('click', () => select(stop));
        return li;
      }));
      if (items.length > 0 || noteText) open();
      else close();
    }

    function select(stop) {
      input.value = stop.stop_name;
      picker.selected = { stop_id: stop.stop_id, stop_name: stop.stop_name };
      // Keep coordinates when the stop record has them (/stops/search,
      // /stops/nearby and /stops/bbox rows do; v2 recents do, v1 ones don't).
      // A From with coords is what recentres the map (onFromChange).
      if (Number.isFinite(stop.stop_lat) && Number.isFinite(stop.stop_lon)) {
        picker.selected.lat = stop.stop_lat;
        picker.selected.lon = stop.stop_lon;
      }
      pushRecent(stop);
      close();
      clearError();
      notifyChange();
    }

    function showRecents() {
      const recents = readRecents();
      if (recents.length > 0) render(recents, 'Recent');
      else render([], '');
    }

    async function search(query) {
      const seq = ++requestSeq;
      try {
        const params = new URLSearchParams({ q: query, limit: String(SEARCH_LIMIT) });
        const bias = options.bias ? options.bias() : null; // re-read per call, so a cleared From drops it at once
        if (bias) {
          params.set('lat', String(bias.lat));
          params.set('lon', String(bias.lon));
        }
        const resp = await fetch('/stops/search?' + params.toString());
        if (!resp.ok) throw new Error('HTTP ' + resp.status);
        const stops = await resp.json();
        if (seq !== requestSeq || input.value.trim() !== query) return; // superseded
        if (stops.length === 0) {
          render([], `No stops match “${query}”.`);
          return;
        }
        // The API exposes no match score, so "near match" means no returned
        // name contains the query. Results are shown either way — label only.
        const needle = query.toLowerCase();
        const anyContains = stops.some((s) => s.stop_name.toLowerCase().includes(needle));
        render(stops, anyContains ? '' : NEAR_MATCH_NOTE);
      } catch (err) {
        if (seq !== requestSeq) return;
        render([], 'Couldn’t search stops. Try again.');
      }
    }

    input.addEventListener('focus', () => {
      clearError();
      if (input.value.trim() === '') showRecents();
    });

    input.addEventListener('input', () => {
      clearError();
      if (picker.selected) {
        picker.selected = null; // typing voids an earlier selection
        notifyChange();
      }
      clearTimeout(timer);
      const query = input.value.trim();
      if (query === '') {
        requestSeq++; // drop any in-flight search
        showRecents();
        return;
      }
      timer = setTimeout(() => search(query), DEBOUNCE_MS);
    });

    input.addEventListener('keydown', (event) => {
      const count = items.length;
      switch (event.key) {
        case 'ArrowDown':
          event.preventDefault();
          if (!isOpen()) {
            if (count > 0) open();
            else if (input.value.trim() === '') showRecents();
          } else if (count > 0) {
            setActive((activeIndex + 1) % count);
          }
          break;
        case 'ArrowUp':
          event.preventDefault();
          if (isOpen() && count > 0) {
            setActive(activeIndex <= 0 ? count - 1 : activeIndex - 1);
          }
          break;
        case 'Enter':
          // Never submit the form from a picker input.
          event.preventDefault();
          if (isOpen() && activeIndex >= 0) select(items[activeIndex]);
          else if (isOpen() && count === 1) select(items[0]);
          break;
        case 'Escape':
          if (isOpen()) {
            event.preventDefault();
            close();
          }
          break;
        case 'Tab':
          close(); // focus moves on normally
          break;
        default:
          break;
      }
    });

    input.addEventListener('blur', close);

    picker.select = select; // same path a dropdown-row click takes (used by nearby map pins)

    // Swap support. setState() writes a selection + raw input text without
    // select()'s side effects (no recents push), cancels any pending search
    // and closes the list. It returns whether the selected stop changed; the
    // caller fires notifyChange() once both pickers are written.
    picker.text = () => input.value;
    picker.setState = (selected, text) => {
      clearTimeout(timer);
      requestSeq++;
      close();
      const changed = (picker.selected ? picker.selected.stop_id : null) !== (selected ? selected.stop_id : null);
      picker.selected = selected;
      input.value = text;
      return changed;
    };
    picker.notifyChange = notifyChange;
    return picker;
  }

  // ── Route results ───────────────────────────────────────────

  function transferLabel(count) {
    if (count === 0) return 'direct';
    return `${count} transfer${count === 1 ? '' : 's'}`;
  }

  // { chip: true } (hero only) swaps the icon + "Bus 704" text for a
  // mode-tinted chip: icon + short name (mode label when there's none).
  // The mode word moves to aria-label, so it still reads "Bus 704".
  function renderLeg(leg, { chip = false } = {}) {
    const li = el('li', 'leg');
    const modeLabel = MODE_LABEL[leg.mode] || MODE_LABEL.unknown;
    const main = el('div', chip ? `leg-chip mode-${leg.mode}` : 'leg-main');
    const icon = window.TransitIcons ? window.TransitIcons.modeIcon(leg.mode, `leg-icon mode-${leg.mode}`) : null;
    if (icon) main.append(icon);
    if (chip) {
      main.setAttribute('role', 'img');
      main.setAttribute('aria-label', `${modeLabel} ${leg.route_short_name || ''}`.trim());
      main.append(el('span', 'leg-chip-label', leg.route_short_name || modeLabel));
    } else {
      main.append(
        el('span', 'leg-mode', modeLabel + ' '),
        el('span', 'leg-route', leg.route_short_name),
      );
    }
    const pill = el('span', `pill ${PILL_CLASS[leg.confidence] || 'pill-low'}`, `${leg.confidence} confidence`);
    const times = el(
      'div', 'leg-times',
      `Departs ${hhmm(leg.departure_time)} (scheduled) · arrives ~${hhmm(leg.predicted_arrival)} (predicted)`,
    );
    li.append(main, pill, times);
    return li;
  }

  function renderRoute(route) {
    const card = el('li', 'card');
    card.tabIndex = 0; // selectable by keyboard too — see the results listeners below
    const head = el('div', 'card-head');
    head.append(
      el('span', 'card-title', `Departs ${hhmm(route.legs[0].departure_time)}`),
      el('span', 'card-meta', `${route.total_predicted_duration_minutes} min · ${transferLabel(route.transfer_count)}`),
    );
    const legs = el('ol', 'legs');
    legs.append(...route.legs.map((leg) => renderLeg(leg)));
    card.append(head, legs);
    return card;
  }

  // ── Hero (the active route) ─────────────────────────────────
  // Fills the #top-pick shell in results.html. Leg rows come from
  // renderLeg(), the same builder the alternatives' cards use.

  const HERO_CONF_CLASSES = ['conf-low', 'conf-medium', 'conf-high'];

  // predicted_delay_minutes -> [pill text, status class], or null when the
  // journey has no delay figure (the whole status area is then hidden).
  function heroStatus(delay) {
    if (delay === null || delay === undefined) return null;
    if (delay === 0) return ['On time', 'status-ontime'];
    if (delay > 0) return [`+${delay} min`, 'status-late'];
    return [`${delay} min`, 'status-early']; // already carries its minus sign
  }

  function renderHero(route) {
    const hero = document.querySelector('#top-pick');
    // Backend sends "HH:MM" already; shown as-is.
    hero.querySelector('[data-hero-leave-time]').textContent = route.leave_by || '';

    const status = heroStatus(route.predicted_delay_minutes);
    const statusBox = hero.querySelector('[data-hero-status]');
    const pill = hero.querySelector('[data-hero-status-pill]');
    const caption = hero.querySelector('[data-hero-status-caption]');
    statusBox.hidden = !status;
    if (status) {
      pill.textContent = status[0];
      pill.className = `hero-status-pill ${status[1]}`;
      // live_delay is always None server-side, so this is the model alone
      // (Pass 10 swaps in RT-blended and drops the caption).
      caption.textContent = 'Model estimate';
    } else {
      pill.textContent = '';
      pill.className = 'hero-status-pill';
      caption.textContent = '';
    }

    hero.classList.remove(...HERO_CONF_CLASSES);
    const confClass = `conf-${route.confidence}`;
    if (HERO_CONF_CLASSES.includes(confClass)) hero.classList.add(confClass);

    hero.querySelector('[data-hero-journey]').textContent = [
      `${route.total_predicted_duration_minutes} min`,
      transferLabel(route.transfer_count),
      route.legs.map((leg) => MODE_LABEL[leg.mode] || MODE_LABEL.unknown).join(' → '),
    ].join(' · ');

    // One line per transfer, in the summary block. Legs can end and start
    // at different stops of one station (the router transfers within a
    // stop group), so both names show when they differ.
    const transfers = route.legs.slice(1).map((leg, i) => {
      const from = route.legs[i].to_stop.stop_name;
      const to = leg.from_stop.stop_name;
      return from === to ? `Transfer at ${from}` : `Transfer: ${from} → ${to}`;
    });
    const transfersEl = hero.querySelector('[data-hero-transfers]');
    if (transfers.length > 1) transfersEl.replaceChildren(...transfers.map((line) => el('span', null, line)));
    else transfersEl.textContent = transfers[0] || '';
    transfersEl.hidden = transfers.length === 0;

    renderWalkAnnotation(route);
    hero.querySelector('[data-hero-legs]').replaceChildren(...route.legs.map((leg) => renderLeg(leg, { chip: true })));
  }

  // ── Walk annotation (hero only) ─────────────────────────────
  // "~0.4 km · ~5 min walk to X": straight-line distance from the user's
  // location fix (<main data-origin-lat/lon>, set only when /results got a
  // valid pair) to the first leg's from_stop. Absent silently when either
  // end has no coords. Separate from leave_by, which stays departure - 3 min.

  const WALK_KMH = 5;

  // 24x24, same stroke style as icons.js.
  const WALK_SHAPES = [
    ['circle', { cx: 13, cy: 4, r: 2 }],
    ['path', { d: 'M7 11l3-3h3l3 3' }],
    ['path', { d: 'M12 8l-1.5 6' }],
    ['path', { d: 'M10.5 14L8 21' }],
    ['path', { d: 'M10.5 14l3 2.5L15 21' }],
  ];

  function walkIcon() {
    const svg = svgEl('svg', {
      viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', 'stroke-width': 2,
      'stroke-linecap': 'round', 'stroke-linejoin': 'round', 'aria-hidden': 'true',
      focusable: 'false', class: 'hero-walk-icon',
    });
    for (const [tag, attrs] of WALK_SHAPES) svg.append(svgEl(tag, attrs));
    return svg;
  }

  function haversineKm(a, b) {
    const rad = (deg) => (deg * Math.PI) / 180;
    const dLat = rad(b.lat - a.lat);
    const dLon = rad(b.lon - a.lon);
    const h = Math.sin(dLat / 2) ** 2
      + Math.cos(rad(a.lat)) * Math.cos(rad(b.lat)) * Math.sin(dLon / 2) ** 2;
    return 2 * 6371 * Math.asin(Math.sqrt(h));
  }

  function walkOrigin() {
    const { originLat, originLon } = document.querySelector('main').dataset;
    if (originLat === undefined || originLon === undefined) return null;
    const lat = Number(originLat);
    const lon = Number(originLon);
    return Number.isFinite(lat) && Number.isFinite(lon) ? { lat, lon } : null;
  }

  function renderWalkAnnotation(route) {
    const WALK_MAX_KM = 2.0; // farther than this isn't a walk to the stop; stay hidden
    const slot = document.querySelector('#top-pick [data-hero-walk]');
    const origin = walkOrigin();
    const stop = route.legs[0] && route.legs[0].from_stop;
    const rawKm = origin && stop && Number.isFinite(stop.lat) && Number.isFinite(stop.lon)
      ? haversineKm(origin, stop) : null;
    if (rawKm === null || rawKm > WALK_MAX_KM) {
      slot.hidden = true;
      slot.replaceChildren();
      return;
    }
    // Minutes come from the shown (rounded) distance so the two agree.
    const km = Math.max(0.1, Math.round(rawKm * 10) / 10);
    const minutes = Math.max(1, Math.round((km / WALK_KMH) * 60));
    slot.replaceChildren(
      walkIcon(),
      el('span', 'hero-walk-text', `~${km.toFixed(1)} km · ~${minutes} min walk to ${stop.stop_name}`),
    );
    slot.hidden = false;
  }

  // ── Contextual pill (results bottom row) ────────────────────
  // How urgent the active route is, from its leave_by. Low confidence keeps
  // the label but mutes the tone.

  // [max minutes until leave_by (inclusive), label, tone]
  const CONTEXT_PILL_RULES = [
    [-2, 'MISSED IT', 'muted'], // the 2-minute grace after leave_by is over
    [0, 'LEAVE NOW', 'danger'],
    [3, 'HEAD OUT', 'warn'],
    [10, 'GRAB YOUR KEYS', 'accent'],
    [25, 'FINISH YOUR COFFEE', 'ok'],
    [Infinity, 'PLENTY OF TIME', 'muted'],
  ];
  const CONTEXT_PILL_TONES = CONTEXT_PILL_RULES.map(([, , tone]) => `pill-context-${tone}`);
  const CONTEXT_PILL_TICK_MS = 30000; // 30s so a minute-boundary flip shows promptly
  const LOW_CONFIDENCE_TITLE = 'Low-confidence prediction — check before you go';

  // Whole minutes from `now` until leave_by ("HH:MM" Brisbane wall-clock,
  // read as the viewer's local time, like every other time here). One more
  // than 12h in the past is tomorrow's — "00:15" seen at 23:50 is 25 min
  // away. null when leave_by is missing or malformed.
  function minutesUntil(leaveBy, now) {
    const match = /^(\d{2}):(\d{2})$/.exec(leaveBy || '');
    if (!match) return null;
    const target = new Date(now);
    target.setHours(Number(match[1]), Number(match[2]), 0, 0);
    if (now - target > 12 * 60 * 60000) target.setDate(target.getDate() + 1);
    return Math.floor((target - now) / 60000);
  }

  // ── Shape projection (live vehicles) ────────────────────────
  // Snaps a GTFS-RT vehicle position onto a leg's shape. leg.shape is
  // [[lat, lon], ...] (webui/app.py _leg_shape) and is read in that order.
  //
  // Planar math: lat/lon are treated as flat x/y. Over an SEQ corridor
  // (<200 km N-S) the nearest-point error against great-circle is far
  // below GTFS-RT positional noise, so no haversine here. distanceSq is
  // in squared degrees, comparable only with other planar results.
  //
  // Pure: no DOM, globals, timers or logging. Returns
  // {lat, lon, segmentIndex, t, distanceSq} for the nearest point, where the
  // winning segment is shape[segmentIndex] -> shape[segmentIndex + 1] and t
  // is 0..1 along it. null for a non-finite lat/lon, a shape with < 2
  // points, or one whose segments are all zero-length.
  function nearestPointOnShape(lat, lon, shape) {
    if (!Number.isFinite(lat) || !Number.isFinite(lon)) return null;
    if (!Array.isArray(shape) || shape.length < 2) return null;

    let best = null;
    for (let i = 0; i < shape.length - 1; i++) {
      const [lat1, lon1] = shape[i];
      const [lat2, lon2] = shape[i + 1];
      const vLat = lat2 - lat1;
      const vLon = lon2 - lon1;
      const vv = vLat * vLat + vLon * vLon;
      if (!(vv > 0)) continue; // zero-length segment (or non-numeric point)

      const t = Math.min(1, Math.max(0, ((lat - lat1) * vLat + (lon - lon1) * vLon) / vv));
      const nLat = lat1 + t * vLat;
      const nLon = lon1 + t * vLon;
      const distanceSq = (lat - nLat) ** 2 + (lon - nLon) ** 2;
      if (best === null || distanceSq < best.distanceSq) {
        best = { lat: nLat, lon: nLon, segmentIndex: i, t, distanceSq };
      }
    }
    return best;
  }

  // ── Live vehicles (Pass 10) ─────────────────────────────────
  // Polls /live_vehicles for the active route's trips, snaps each vehicle
  // onto its leg's shape and slides that leg's timeline icon there.
  //
  // The timeline is a schematic: stops sit at equal spacing in sequence
  // order, not by geography. So a fix is placed by its progress along the
  // shape (segmentIndex + t -> cumulative length), interpolated between the
  // timeline x of the two stops it falls between (liveLegLayout).
  //
  // Declared above the page switch: load() calls stopLivePolling() before
  // its first await, so this state must already be initialised.

  const LIVE_POLL_INTERVAL_MS = 30000; // 30s poll, matches locked spec
  const LIVE_STALE_DIM_SECONDS = 90; // 0-90s: full opacity
  const LIVE_STALE_HIDE_SECONDS = 120; // 90-120s: dimmed; >120s: hidden
  const LIVE_DIM_OPACITY = 0.5; // visual for the 90-120s band
  // Planar squared-degree threshold (nearestPointOnShape's distanceSq). At
  // ~28°S, 0.02° ≈ 2.2 km north-south and ≈ 1.95 km east-west, so 0.0004
  // (0.02²) is roughly a 2 km envelope around the polyline. Anything further
  // from its own trip's shape is treated as no live position (wrong trip
  // match or GPS glitch) and the icon falls back to the Pass 9 placeholder.
  const MAX_PROJECTION_DISTSQ_DEG = 0.0004;
  const CSS_TWEEN_MS = 29500; // a little under the poll so a move finishes before the next one

  // No icon rotation: shipped, then reversed after Chrome QA — front-view
  // icons flip on N-S routes. Reasoning in pass-10.md; don't re-add it.

  let _livePollTimerId = null; // the single setInterval handle; null when not running
  let _liveCurrentTripIds = []; // the rendered route's pollable trip_ids, sorted
  // trip_id -> {leg, icon, y, cum, anchors, stopProgress, nodeEls}, rebuilt by
  // every timeline render.
  let _liveLegs = new Map();
  // trip_id -> last applied 0..1 progress. Outside _liveLegs so a re-render
  // (resize) can restore stop states without waiting for the next poll.
  let _liveLastProgress = new Map();
  // Bumped per startLivePolling(); a response for an older route is dropped.
  let _liveGeneration = 0;

  // Legs without a shape or trip_id are never polled: they keep the static
  // placeholder (first leg) or no icon (later legs).
  function isLivePollable(leg) {
    return Array.isArray(leg.shape) && leg.shape.length >= 2 && Boolean(leg.trip_id);
  }

  // Shape cumulative lengths (planar degrees, like nearestPointOnShape) and
  // the leg's stops as {progress, x} anchors, progress forced non-decreasing.
  // Stops without coords are skipped; under two usable anchors, the leg's
  // first and last stops pin the shape's two ends. stopProgress is the same
  // per-stop progress as a 0..1 fraction of the shape, keyed by timeline node.
  function liveLegLayout(shape, stops) {
    const cum = [0];
    for (let i = 1; i < shape.length; i++) {
      cum.push(cum[i - 1] + Math.hypot(shape[i][0] - shape[i - 1][0], shape[i][1] - shape[i - 1][1]));
    }
    const total = cum[cum.length - 1];
    let anchors = [];
    const stopProgress = [];
    for (const { x, stop, nodeIndex } of stops) {
      const proj = Number.isFinite(stop.lat) && Number.isFinite(stop.lon)
        ? nearestPointOnShape(stop.lat, stop.lon, shape) : null;
      if (!proj) continue;
      const progress = Math.max(shapeProgress(cum, proj), anchors.length ? anchors[anchors.length - 1].progress : 0);
      anchors.push({ progress, x });
      if (total > 0) stopProgress.push({ nodeIndex, progress: progress / total });
    }
    if (anchors.length < 2) {
      anchors = [{ progress: 0, x: stops[0].x }, { progress: total, x: stops[stops.length - 1].x }];
    }
    return { cum, anchors, stopProgress };
  }

  // A vehicle's progress along its leg as a 0..1 fraction of the shape.
  function legFraction(live, proj) {
    const total = live.cum[live.cum.length - 1];
    return total > 0 ? shapeProgress(live.cum, proj) / total : 0;
  }

  const STOP_STATES = ['state-passed', 'state-next', 'state-future'];
  const STOP_STATE_APPROACH = 0.01; // at or below: still at/before the origin (the shape[0] clamp) -- no state
  const STOP_STATE_LAG = 0.01; // classify a little behind the fix, nearer the tweening icon
  const STOP_STATE_PASS_MARGIN = 0.02; // a stop is passed once the bus is this far beyond it

  // Passed / next / future on the leg's stop nodes from the bus's fraction
  // along the shape. Classes only: CSS decides which node kinds show them.
  // Stops without coords (no stopProgress entry) keep their default look.
  function _applyStopStates(tripId, progress) {
    const live = _liveLegs.get(tripId);
    if (!live) return;
    for (const el of live.nodeEls.values()) el.classList.remove(...STOP_STATES);
    if (!(progress > STOP_STATE_APPROACH)) return;
    const compare = progress - STOP_STATE_LAG;
    let nextFound = false;
    for (const { nodeIndex, progress: stopProg } of live.stopProgress) {
      const el = live.nodeEls.get(nodeIndex);
      if (!el) continue;
      let state = 'state-future';
      if (stopProg + STOP_STATE_PASS_MARGIN < compare) state = 'state-passed';
      else if (!nextFound) {
        state = 'state-next';
        nextFound = true;
      }
      el.classList.add(state);
    }
  }

  function shapeProgress(cum, proj) {
    const i = proj.segmentIndex;
    return cum[i] + proj.t * (cum[i + 1] - cum[i]);
  }

  // Sequence-based, not geographic (see the section comment): takes the
  // projection rather than a bare lat/lon, since segmentIndex + t is what
  // locates the fix along the shape.
  function _latLonToSvgCoords(proj, live) {
    const { anchors, y } = live;
    const p = shapeProgress(live.cum, proj);
    if (p <= anchors[0].progress) return { svgX: anchors[0].x, svgY: y };
    for (let k = 1; k < anchors.length; k++) {
      const a = anchors[k - 1];
      const b = anchors[k];
      if (p <= b.progress) {
        const span = b.progress - a.progress;
        return { svgX: span > 0 ? a.x + ((p - a.progress) / span) * (b.x - a.x) : b.x, svgY: y };
      }
    }
    return { svgX: anchors[anchors.length - 1].x, svgY: y };
  }

  // The mode icon inside a <g>: Pass 9's icon is a nested <svg>, and CSS
  // transforms on a nested <svg> aren't reliably applied (Chrome), so the
  // <g> carries the translate. The icon is centred on the <g>'s origin.
  // null for a mode with no icon.
  function buildVehicleIcon(mode, restX, restY, hiddenAtRest) {
    const icon = window.TransitIcons ? window.TransitIcons.modeIcon(mode, `timeline-vehicle mode-${mode}`) : null;
    if (!icon) return null;
    icon.setAttribute('x', String(-TL_VEHICLE_SIZE / 2));
    icon.setAttribute('y', String(-TL_VEHICLE_SIZE / 2));
    icon.setAttribute('width', String(TL_VEHICLE_SIZE));
    icon.setAttribute('height', String(TL_VEHICLE_SIZE));
    const g = svgEl('g', { class: 'timeline-vehicle-live' });
    g.dataset.restX = String(restX);
    g.dataset.restY = String(restY);
    g.dataset.restHidden = String(hiddenAtRest);
    g.style.setProperty('--vehicle-tween', `${CSS_TWEEN_MS}ms`);
    g.append(icon);
    _hideVehicleIcon(g);
    return g;
  }

  // null when the route was re-rendered and the icon is gone.
  function _getTimelineLegElement(tripId) {
    const live = _liveLegs.get(tripId);
    return live && live.icon.isConnected ? live.icon : null;
  }

  // Apply without the tween: flush styles with the tween off, then restore.
  function placeWithoutTween(iconEl, apply) {
    iconEl.classList.add('is-jump');
    apply();
    iconEl.getBoundingClientRect();
    iconEl.classList.remove('is-jump');
  }

  function _setVehiclePosition(iconEl, svgX, svgY, opacity) {
    const apply = () => {
      iconEl.style.display = '';
      iconEl.style.transform = `translate(${svgX}px, ${svgY}px)`;
      iconEl.style.setProperty('--vehicle-opacity', String(opacity));
    };
    // First fix since resting: jump there rather than crawl from the origin.
    if (iconEl.dataset.live === 'true') apply();
    else placeWithoutTween(iconEl, apply);
    iconEl.dataset.live = 'true';
  }

  // Back to the Pass 9 placeholder: rest position, full opacity (hidden for
  // a later leg). Jumps, no tween. With a tripId, also drops that leg's stop
  // states so a lost fix doesn't leave stale "passed" marks.
  function _hideVehicleIcon(iconEl, tripId) {
    placeWithoutTween(iconEl, () => {
      iconEl.style.display = iconEl.dataset.restHidden === 'true' ? 'none' : '';
      iconEl.style.transform = `translate(${iconEl.dataset.restX}px, ${iconEl.dataset.restY}px)`;
      iconEl.style.setProperty('--vehicle-opacity', '1');
    });
    delete iconEl.dataset.live;
    if (tripId) {
      _liveLastProgress.delete(tripId);
      _applyStopStates(tripId, 0);
    }
  }

  async function _pollLiveVehicles() {
    if (_liveCurrentTripIds.length === 0) return;
    const generation = _liveGeneration;
    let body;
    try {
      const resp = await fetch(`/live_vehicles?trip_ids=${encodeURIComponent(_liveCurrentTripIds.join(','))}`);
      if (!resp.ok) {
        console.warn(`[live] /live_vehicles returned ${resp.status}; keeping last positions`);
        return;
      }
      body = await resp.json();
    } catch (err) {
      console.warn('[live] /live_vehicles failed; keeping last positions', err);
      return;
    }
    if (generation !== _liveGeneration) return; // a different route rendered meanwhile
    const vehicles = (body && body.vehicles) || {};

    for (const tripId of _liveCurrentTripIds) {
      const live = _liveLegs.get(tripId);
      const iconEl = _getTimelineLegElement(tripId);
      if (!live || !iconEl) continue;

      const vehicle = vehicles[tripId];
      const proj = vehicle ? nearestPointOnShape(vehicle.lat, vehicle.lon, live.leg.shape) : null;
      const age = vehicle ? vehicle.age_seconds : Infinity;
      // No fix, off its own shape, or past the server's 120s cut (defensive).
      if (!proj || proj.distanceSq > MAX_PROJECTION_DISTSQ_DEG || age > LIVE_STALE_HIDE_SECONDS) {
        _hideVehicleIcon(iconEl, tripId);
        continue;
      }
      // Negative age (clock skew) lands in the fresh band.
      const opacity = age <= LIVE_STALE_DIM_SECONDS ? 1 : LIVE_DIM_OPACITY;
      const { svgX, svgY } = _latLonToSvgCoords(proj, live);
      _setVehiclePosition(iconEl, svgX, svgY, opacity);
      const progress = legFraction(live, proj);
      _liveLastProgress.set(tripId, progress);
      _applyStopStates(tripId, progress);
    }
  }

  // Poll now (first paint shouldn't wait 30s), then every 30s. Idempotent.
  function startLiveTimer() {
    clearInterval(_livePollTimerId);
    _livePollTimerId = null;
    if (_liveCurrentTripIds.length === 0) return;
    _pollLiveVehicles();
    _livePollTimerId = setInterval(_pollLiveVehicles, LIVE_POLL_INTERVAL_MS);
  }

  // Idempotent: the same trips with the timer already running is a no-op —
  // no extra immediate poll, no timer restart.
  function startLivePolling(tripIds) {
    const next = [...new Set(tripIds)].sort();
    if (_livePollTimerId !== null && next.join(',') === _liveCurrentTripIds.join(',')) return;
    if (next.join(',') !== _liveCurrentTripIds.join(',')) _liveLastProgress = new Map();
    _liveCurrentTripIds = next;
    _liveGeneration++;
    startLiveTimer();
  }

  function stopLivePolling() {
    clearInterval(_livePollTimerId);
    _livePollTimerId = null;
  }

  // Same lifecycle as the contextual-pill timer: pagehide stops it; a
  // back/forward-cache restore (pageshow, persisted) restarts it for the
  // trips already on screen. Registered once, here, not per route.
  function _handleVisibilityChange(event) {
    if (event.type === 'pagehide') stopLivePolling();
    else if (event.type === 'pageshow' && event.persisted) startLiveTimer();
  }

  window.addEventListener('pagehide', _handleVisibilityChange);
  window.addEventListener('pageshow', _handleVisibilityChange);

  // Same namespace pattern as icons.js's window.TransitIcons.
  window.TransitAI = {
    nearestPointOnShape, startLivePolling, stopLivePolling, _handleVisibilityChange, _testPass10Projection,
  };

  // ── Page switch ─────────────────────────────────────────────
  // base.html sets <body data-page>. Exactly one page mode runs: the search
  // page never renders results, the results page never builds pickers.

  const page = document.body.dataset.page;
  if (page === 'search') initSearchPage();
  else if (page === 'results') initResultsPage();

  // ══ Search page ("/") ═══════════════════════════════════════

  function initSearchPage() {
    // 'from' | 'to' — which picker a map-pin tap fills. In memory only.
    let activePinTarget = 'from';
    let pinSelecting = false; // true while a pin tap runs a picker's select()

    // ── Map (Leaflet, loaded from unpkg by index.html) ────────
    // One full-page map, always shown. currentMapMode is one of:
    //   'empty'  — no pins: the initial SEQ overview, From cleared, or zoom < 11
    //   'nearby' — /stops/nearby around the From stop or the user's location
    //   'bbox'   — /stops/bbox for the viewport, refetched as the user pans
    // A user pan/zoom leaves 'nearby' for 'bbox'/'empty'; our own recentres
    // and fits, and Leaflet's resize handling, never do. If Leaflet failed
    // to load, the map stays an empty panel and the pickers work as before.

    const mapEl = document.querySelector('#map');
    const mapHeaderEl = document.querySelector('#map-header');
    const mapSpinner = document.querySelector('#map-spinner');
    const mapToast = document.querySelector('#map-toast');
    const SEQ_CENTER = [-27.75, 153.2]; // Brisbane <-> Gold Coast corridor
    const SEQ_ZOOM = 9;
    const BBOX_MIN_ZOOM = 11;
    const FROM_ZOOM = 15;
    const BBOX_DEBOUNCE_MS = 300;
    const BBOX_LIMIT = 50;
    const BBOX_MAX_SIDE_DEG = 0.99; // /stops/bbox rejects sides over 1 degree
    const NO_STOPS_IN_VIEW = 'No stops in view — pan or zoom out';
    const FIT_PADDING = [24, 24];
    const HIT_RADIUS = 22; // invisible marker hit area: 44px across, the touch-target size

    let map = null;
    let pinLayer = null;
    let currentMapMode = 'empty';
    let nearbyAnchor = null; // {lat, lon} the nearby pins are around; null outside 'nearby'
    let programmaticMove = false; // true while our own setView/fitBounds runs
    let resizeMovePending = false; // Leaflet's resize handling fires a moveend of its own
    let bboxTimer = null;
    // The one in-flight pin fetch (bbox or nearby — a newer one of either
    // kind aborts it), and why the last aborted fetch was aborted.
    let lastBboxFetchController = null;
    let lastBboxFetchAbortReason = null; // 'superseded' | 'mode-change'

    function mapAvailable() {
      return typeof window.L !== 'undefined';
    }

    function ensureMap() {
      if (map || !mapAvailable()) return;
      map = L.map(mapEl, {
        // Full-page map: one-finger drag pans on phones too. Wheel zoom stays
        // off. Zoom buttons go bottom-right, clear of the picker overlay.
        scrollWheelZoom: false,
        dragging: true,
        zoomControl: false,
      });
      // Leaflet's own constant attribution markup, not user data.
      L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
        maxZoom: 19,
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
      }).addTo(map);
      L.control.zoom({ position: 'bottomright' }).addTo(map);
      pinLayer = L.layerGroup().addTo(map);
      map.on('resize', () => { resizeMovePending = true; });
      map.on('moveend', onMapMoveEnd);
      moveMap(() => map.setView(SEQ_CENTER, SEQ_ZOOM, { animate: false }));
    }

    // Run one of our own view changes. animate:false makes Leaflet fire
    // moveend synchronously inside fn, so the flag is always reset.
    function moveMap(fn) {
      programmaticMove = true;
      try {
        fn();
      } finally {
        programmaticMove = false;
      }
    }

    function computeMapMode() {
      if (!map) return 'empty';
      if (nearbyAnchor) return 'nearby';
      return map.getZoom() >= BBOX_MIN_ZOOM ? 'bbox' : 'empty';
    }

    function onMapMoveEnd() {
      let userMove = false;
      if (!programmaticMove) {
        // Leaflet fires a resize's moveend 200 ms late; only a moveend that
        // isn't ours consumes the flag, so a recentre in between can't eat it.
        userMove = !resizeMovePending;
        resizeMovePending = false;
      }
      if (userMove) nearbyAnchor = null; // the user took over: browse by viewport
      applyMapMode();
    }

    // Bring pins, fetches and the pin-target header in line with the mode.
    function applyMapMode() {
      currentMapMode = computeMapMode();
      mapHeaderEl.hidden = currentMapMode === 'empty'; // pin taps only mean something with pins up
      if (currentMapMode === 'bbox') {
        scheduleBboxFetch();
        return;
      }
      clearTimeout(bboxTimer);
      if (currentMapMode === 'empty') {
        abortMapFetch('mode-change');
        if (pinLayer) pinLayer.clearLayers();
        setMapToast('');
      }
      // 'nearby': showNearby() owns the fetch and the drawing.
    }

    // The 300 ms debounce: a pan's moveend only fetches once the map settles.
    function scheduleBboxFetch() {
      clearTimeout(bboxTimer);
      bboxTimer = setTimeout(() => {
        if (computeMapMode() !== 'bbox') return;
        const bounds = map.getBounds();
        runPinFetch(
          (signal) => fetchStopsBbox(bounds, signal),
          (stops) => { if (currentMapMode === 'bbox') drawPins(stops); },
          () => {}, // a failed refresh keeps the pins already up; the next pan retries
        );
      }, BBOX_DEBOUNCE_MS);
    }

    function abortMapFetch(reason) {
      if (!lastBboxFetchController) return;
      lastBboxFetchAbortReason = reason;
      lastBboxFetchController.abort();
      lastBboxFetchController = null;
    }

    // One pin fetch at a time, aborting any older one. Pins are only
    // replaced when the new rows land, so old pins stay up meanwhile.
    async function runPinFetch(fetcher, draw, onError) {
      abortMapFetch('superseded');
      const controller = new AbortController();
      lastBboxFetchController = controller;
      setMapLoading(true);
      try {
        const stops = await fetcher(controller.signal);
        if (!controller.signal.aborted) draw(stops);
      } catch (err) {
        if (!controller.signal.aborted) onError(err);
      } finally {
        if (controller === lastBboxFetchController) lastBboxFetchController = null;
        // A superseded fetch leaves the spinner to its replacement.
        const superseded = controller.signal.aborted && lastBboxFetchAbortReason === 'superseded';
        if (!superseded) setMapLoading(false);
      }
    }

    // /stops/bbox caps each side at 1 degree; a wider view (zoom 11 on a big
    // screen) asks for the capped box around the view's centre instead.
    async function fetchStopsBbox(bounds, signal) {
      const center = bounds.getCenter();
      const halfLon = Math.min(bounds.getEast() - bounds.getWest(), BBOX_MAX_SIDE_DEG) / 2;
      const halfLat = Math.min(bounds.getNorth() - bounds.getSouth(), BBOX_MAX_SIDE_DEG) / 2;
      const params = new URLSearchParams({
        west: (center.lng - halfLon).toFixed(5),
        south: (center.lat - halfLat).toFixed(5),
        east: (center.lng + halfLon).toFixed(5),
        north: (center.lat + halfLat).toFixed(5),
        limit: String(BBOX_LIMIT),
      });
      const resp = await fetch('/stops/bbox?' + params.toString(), { signal });
      if (!resp.ok) throw new Error('HTTP ' + resp.status);
      return resp.json();
    }

    function setMapLoading(busy) {
      mapSpinner.hidden = !busy;
    }

    function setMapToast(text) {
      mapToast.textContent = text;
    }

    // Mode colours live in app.css custom properties.
    function mapColors() {
      const css = getComputedStyle(document.documentElement);
      const read = (name) => css.getPropertyValue(name).trim();
      return {
        mode: {
          bus: read('--mode-bus'),
          tram: read('--mode-tram'),
          rail: read('--mode-rail'),
          unknown: read('--mode-unknown'),
        },
        casing: read('--map-casing'),
        marker: read('--map-marker'),
      };
    }

    // Tooltip content as DOM nodes — a string would be parsed as HTML.
    function mapLabel(role, name) {
      const node = el('span', 'map-label');
      node.append(el('span', 'map-label-role', role), el('span', 'map-label-name', name));
      return node;
    }

    function distanceLabel(km) {
      return km < 0.05 ? 'under 0.1 km away' : `${km.toFixed(1)} km away`;
    }

    // Replace the pins with `stops` (bbox or nearby rows), plus the "you are
    // here" marker once location is known. Returns the drawn bounds.
    function drawPins(stops) {
      pinLayer.clearLayers();
      setMapToast(stops.length === 0 ? NO_STOPS_IN_VIEW : '');
      const origin = fromPicker.originCoords;
      const colors = mapColors();
      const bounds = L.latLngBounds([]);

      // "You are here": filled dot + pulsing ring. Never interactive, so it
      // can't swallow a tap meant for a stop pin underneath it.
      if (origin) {
        const here = [origin.lat, origin.lon];
        L.circleMarker(here, {
          radius: 14, weight: 2, color: colors.marker, fillColor: colors.marker, fillOpacity: 0.12,
          interactive: false, className: 'map-here-pulse',
        }).addTo(pinLayer);
        L.circleMarker(here, {
          radius: 6, weight: 2, color: colors.casing, fillColor: colors.marker, fillOpacity: 1, interactive: false,
        }).addTo(pinLayer);
        bounds.extend(here);
      }

      for (const stop of stops) {
        const latlng = [stop.stop_lat, stop.stop_lon];
        L.circleMarker(latlng, {
          radius: 6,
          weight: 2,
          color: colors.casing,
          fillColor: colors.mode[stop.mode] || colors.mode.unknown,
          fillOpacity: 1,
          interactive: false,
        }).addTo(pinLayer);
        const role = MODE_LABEL[stop.mode] || MODE_LABEL.unknown;
        // Bbox rows carry no distance_km: name only.
        const name = Number.isFinite(stop.distance_km)
          ? `${stop.stop_name} · ${distanceLabel(stop.distance_km)}`
          : stop.stop_name;
        L.circleMarker(latlng, { radius: HIT_RADIUS, stroke: false, fill: true, fillOpacity: 0 })
          .bindTooltip(mapLabel(role, name), { direction: 'top', offset: [0, -8] })
          .on('click', () => pinSelect(stop))
          .addTo(pinLayer);
        bounds.extend(latlng);
      }
      return bounds;
    }

    // 'nearby' around coords. A From stop recentres on itself at FROM_ZOOM
    // right away; a location fix (fit: true) fits the pins once they land,
    // as the Pass 6 overview did. Resolves when the fetch settles.
    function showNearby(coords, { fit }) {
      if (!map) return Promise.resolve();
      nearbyAnchor = coords;
      if (fit) applyMapMode();
      else moveMap(() => map.setView([coords.lat, coords.lon], FROM_ZOOM, { animate: false }));
      return runPinFetch(
        (signal) => fetchStopsNearby(coords, signal),
        (stops) => {
          if (nearbyAnchor !== coords) return; // the user panned away meanwhile
          const bounds = drawPins(stops);
          if (fit && bounds.isValid()) {
            moveMap(() => map.fitBounds(bounds, { padding: FIT_PADDING, maxZoom: 16, animate: false }));
          }
        },
        () => setFromHint('Couldn’t load nearby stops. Try again or search by name.'),
      );
    }

    // Back to the SEQ overview with no pins (the moveend lands in 'empty').
    function showEmpty() {
      if (!map) return;
      nearbyAnchor = null;
      moveMap(() => map.setView(SEQ_CENTER, SEQ_ZOOM, { animate: false }));
    }

    // ── Pickers ───────────────────────────────────────────────

    // From set with coords (typed, pin, or a v2 recent) -> nearby around it;
    // cleared -> empty overview. A v1 recent has no coords, so the map stays
    // as it is.
    function onFromChange() {
      // A manual From change (typed pick, recents, clearing the text, swap)
      // voids the location fix; a nearby-pin tap keeps it (still the GPS point).
      if (!pinSelecting) fromPicker.originCoords = null;
      if (fromPicker.selected) setFromHint('');
      retargetAfterChange('from');
      updateSwapButton();
      const coords = fromBias();
      if (coords) showNearby(coords, { fit: false });
      else if (!fromPicker.selected) showEmpty();
    }

    function onToChange() {
      retargetAfterChange('to');
      updateSwapButton();
    }

    // To is geo-biased only by a confirmed From that carries coordinates.
    function fromBias() {
      const sel = fromPicker.selected;
      return sel && Number.isFinite(sel.lat) && Number.isFinite(sel.lon) ? { lat: sel.lat, lon: sel.lon } : null;
    }

    const fromPicker = makePicker('from', { onChange: onFromChange });
    const toPicker = makePicker('to', { onChange: onToChange, bias: fromBias });
    fromPicker.originCoords = null; // in-memory only; never persisted
    const form = document.querySelector('#route-form');

    // ── Swap From <-> To ──────────────────────────────────────
    // Exchanges selections and raw input text (mid-edit text moves too),
    // writing both pickers before either onChange fires. The pin target is
    // tied to the field, not the stop, so retargeting is skipped. A changed
    // From clears originCoords (onFromChange); the recentre comes from the
    // new From's own coords.
    // Focus stays on the button; it can't disable itself, since a swap only
    // moves text between the fields.

    const swapButton = document.querySelector('#swap-button');
    let swapping = false; // true while a swap runs the pickers' onChange

    function updateSwapButton() {
      swapButton.disabled = fromPicker.text().trim() === '' && toPicker.text().trim() === '';
    }

    function swapPickers() {
      const from = { selected: fromPicker.selected, text: fromPicker.text() };
      const to = { selected: toPicker.selected, text: toPicker.text() };
      const fromChanged = fromPicker.setState(to.selected, to.text);
      const toChanged = toPicker.setState(from.selected, from.text);
      swapping = true;
      try {
        if (fromChanged) fromPicker.notifyChange();
        if (toChanged) toPicker.notifyChange();
      } finally {
        swapping = false;
      }
      renderPinTarget(); // the toggle shows each field's stop name
      clearError();
      updateSwapButton();
    }

    swapButton.addEventListener('click', swapPickers); // native button: Enter and Space both click
    for (const input of document.querySelectorAll('#from-input, #to-input')) {
      input.addEventListener('input', updateSwapButton);
    }
    updateSwapButton();

    // ── Pin target (which field a nearby-pin tap fills) ───────

    function defaultPinTarget() {
      if (!fromPicker.selected) return 'from';
      if (!toPicker.selected) return 'to';
      return 'from'; // both set: the next tap re-plans From
    }

    // Typed / recents / cleared changes. A pin tap applies its own rule in
    // pinSelect(), so this skips while one is running. A manual toggle
    // choice holds until the next selection or clear reaches here.
    function retargetAfterChange(which) {
      if (pinSelecting || swapping) return;
      const picker = which === 'from' ? fromPicker : toPicker;
      if (!picker.selected) setPinTarget(defaultPinTarget()); // cleared
      else if (which === 'from' && activePinTarget === 'from') setPinTarget('to'); // From's job is done
      else renderPinTarget(); // typed To: target stays; just refresh the names
    }

    // Nearby-pin tap: same select() a dropdown row uses, on the target
    // picker, then advance the target by the default rules.
    function pinSelect(stop) {
      const picker = activePinTarget === 'to' ? toPicker : fromPicker;
      pinSelecting = true;
      try {
        picker.select(stop);
      } finally {
        pinSelecting = false;
      }
      setPinTarget(defaultPinTarget());
    }

    function setPinTarget(target) {
      activePinTarget = target;
      renderPinTarget();
    }

    // Toggle: a two-button toolbar (aria-pressed). One Tab stop (roving
    // tabindex); arrow/Home/End move focus, Enter/Space (native button
    // activation) commits.
    const pinTargetLabel = document.querySelector('#pin-target-label');
    const pinTargetGroup = document.querySelector('#pin-target');
    const pinTargetButtons = Array.from(pinTargetGroup.querySelectorAll('.pin-target-option'));

    function renderPinTarget() {
      pinTargetLabel.textContent = `Tap a pin to set ${activePinTarget === 'to' ? 'To' : 'From'}`;
      for (const button of pinTargetButtons) {
        const target = button.dataset.target;
        const active = target === activePinTarget;
        button.setAttribute('aria-pressed', String(active));
        button.tabIndex = active ? 0 : -1;
        const sel = (target === 'to' ? toPicker : fromPicker).selected;
        const value = button.querySelector('.pin-target-value');
        value.textContent = sel ? sel.stop_name : '';
        if (sel) value.title = sel.stop_name;
        else value.removeAttribute('title');
      }
    }

    pinTargetGroup.addEventListener('click', (event) => {
      const button = event.target.closest('.pin-target-option');
      if (button) setPinTarget(button.dataset.target);
    });

    pinTargetGroup.addEventListener('keydown', (event) => {
      const index = pinTargetButtons.indexOf(document.activeElement);
      if (index < 0) return;
      let next;
      if (event.key === 'ArrowLeft' || event.key === 'ArrowUp' || event.key === 'Home') next = 0;
      else if (event.key === 'ArrowRight' || event.key === 'ArrowDown' || event.key === 'End') next = pinTargetButtons.length - 1;
      else return;
      event.preventDefault();
      pinTargetButtons[index].tabIndex = -1;
      pinTargetButtons[next].tabIndex = 0;
      pinTargetButtons[next].focus();
    });

    renderPinTarget();
    ensureMap(); // after the pickers exist: pin taps and mode changes use them

    // ── Use my location (From only) ───────────────────────────
    // User-initiated only: geolocation is never touched on page load, not
    // even to query permission state.

    const NEARBY_LIMIT = 15;
    const LOCATION_UNAVAILABLE = 'Location unavailable — search by name instead';
    const locateButton = document.querySelector('#locate-button');
    const locateDot = document.querySelector('#locate-dot');
    const fromHint = document.querySelector('#from-hint');

    function setFromHint(text) {
      fromHint.textContent = text || '';
      fromHint.hidden = !text;
    }

    function setLocating(busy) {
      locateButton.disabled = busy;
      locateButton.classList.toggle('is-loading', busy);
      if (busy) locateButton.setAttribute('aria-busy', 'true');
      else locateButton.removeAttribute('aria-busy');
    }

    function setLocationGranted(granted) {
      locateDot.hidden = !granted;
      const label = granted ? 'Use my location (location found)' : 'Use my location';
      locateButton.setAttribute('aria-label', label);
      locateButton.title = label;
    }

    // One /stops/nearby request per location grant or From change.
    async function fetchStopsNearby(coords, signal) {
      const params = new URLSearchParams({
        lat: String(coords.lat), lon: String(coords.lon), limit: String(NEARBY_LIMIT),
      });
      const resp = await fetch('/stops/nearby?' + params.toString(), { signal });
      if (!resp.ok) throw new Error('HTTP ' + resp.status);
      return resp.json();
    }

    async function onPosition(position) {
      const coords = { lat: position.coords.latitude, lon: position.coords.longitude };
      fromPicker.originCoords = coords;
      setLocationGranted(true);
      setFromHint('');
      await showNearby(coords, { fit: true }); // new pins: redraw + re-fit around the user
      setLocating(false);
    }

    function onPositionError() {
      setLocating(false); // stays clickable so the user can retry
      setFromHint(LOCATION_UNAVAILABLE);
    }

    const pin = window.TransitIcons ? window.TransitIcons.pinIcon('locate-icon') : null;
    if (pin) locateButton.prepend(pin);
    if (!('geolocation' in navigator)) {
      locateButton.disabled = true; // the only case the button is disabled at rest
      locateButton.title = 'Location isn’t supported in this browser';
    } else {
      locateButton.addEventListener('click', () => {
        if (locateButton.disabled) return;
        setLocating(true);
        setFromHint('');
        navigator.geolocation.getCurrentPosition(onPosition, onPositionError, {
          enableHighAccuracy: true,
          timeout: 10000,
          maximumAge: 0,
        });
      });
    }

    // ── Submit: navigate to /results ──────────────────────────
    // Validation stays here (#status); loading/empty/error belong to
    // /results. No departure input exists yet, so `departure` is never sent.

    form.addEventListener('submit', (event) => {
      event.preventDefault();
      if (!fromPicker.selected || !toPicker.selected) {
        setStatus('Pick a From and a To stop from the suggestions first.', 'error');
        return;
      }
      if (fromPicker.selected.stop_id === toPicker.selected.stop_id) {
        setStatus('From and To can’t be the same stop.', 'error');
        return;
      }
      let url = '/results?from=' + encodeURIComponent(fromPicker.selected.stop_id)
        + '&to=' + encodeURIComponent(toPicker.selected.stop_id);
      // The location fix rides along for the hero's walk annotation. Rounded
      // here only (3 dp, ~111m); originCoords itself keeps full precision.
      const origin = fromPicker.originCoords;
      if (origin) url += '&origin_lat=' + origin.lat.toFixed(3) + '&origin_lon=' + origin.lon.toFixed(3);
      window.location.href = url;
    });
  }

  // ══ Station timeline (results page) ═════════════════════════
  // Inline SVG built through DOM APIs (no markup strings), like icons.js.
  // The vehicle icon rests on the first leg's origin (Pass 9); the Pass 10
  // live poll (above the page switch) moves it — no timers or polling here.

  const SVG_NS = 'http://www.w3.org/2000/svg';
  const MODE_VAR = { bus: '--mode-bus', tram: '--mode-tram', rail: '--mode-rail' };
  const TL_PAD = 12; // px of line beyond the first/last circle
  const TL_BASELINE = 46; // y of the line; vehicle icon + "above" labels sit over it
  const TL_HEIGHT = 84;
  const TL_MOBILE_MIN_SPACING = 28; // ≤480px with > 8 stops: scroll instead of crushing
  const TL_LABEL_GAP = 6;
  const TL_VEHICLE_SIZE = 18;
  const TL_ROLE = { origin: 'From', destination: 'To', transfer: 'Transfer', stop: 'Stop' };

  function svgEl(tag, attrs) {
    const node = document.createElementNS(SVG_NS, tag);
    for (const [name, value] of Object.entries(attrs || {})) node.setAttribute(name, String(value));
    return node;
  }

  function modeFill(mode) {
    return `var(${MODE_VAR[mode] || '--mode-unknown'})`;
  }

  function roughKm(a, b) {
    const dy = (a.lat - b.lat) * 110.57;
    const dx = (a.lon - b.lon) * 111.32 * Math.cos((a.lat * Math.PI) / 180);
    return Math.sqrt(dx * dx + dy * dy);
  }

  // Two legs' boundary stops are one station only if their names match
  // exactly (case/space-insensitive) and they're close. Platform IDs differ
  // across modes, so stop_id can't decide this; a differing name ("X station"
  // vs "X station platform 1") keeps two nodes — over-show, never mis-merge.
  function sameStation(a, b) {
    const norm = (stop) => (stop.stop_name || '').trim().toLowerCase();
    if (!norm(a) || norm(a) !== norm(b)) return false;
    const coords = (stop) => Number.isFinite(stop.lat) && Number.isFinite(stop.lon);
    return !(coords(a) && coords(b)) || roughKm(a, b) <= 0.5;
  }

  // Route -> timeline nodes. Each leg boundary whose two stops are the same
  // station becomes ONE transfer node, two-toned (incoming | outgoing mode).
  function timelineNodes(route) {
    const nodes = [];
    route.legs.forEach((leg, legIndex) => {
      const stops = Array.isArray(leg.stops) && leg.stops.length >= 2
        ? leg.stops
        : [leg.from_stop, leg.to_stop].map((stop) => ({ ...stop, scheduled_time: null, is_transfer_point: false }));
      stops.forEach((stop, stopIndex) => {
        const prev = nodes[nodes.length - 1];
        if (stopIndex === 0 && legIndex > 0 && prev && prev.kind === 'transfer' && stop.is_transfer_point
            && sameStation(prev.stops[0], stop)) {
          prev.stops.push(stop);
          prev.modeRight = leg.mode;
          prev.legRefs.push(legIndex);
          return;
        }
        nodes.push({
          kind: stop.is_transfer_point ? 'transfer' : 'stop',
          name: stop.stop_name,
          stops: [stop],
          modeLeft: leg.mode,
          modeRight: leg.mode,
          legRefs: [legIndex], // legs this node is a stop of, parallel to `stops`
        });
      });
    });
    nodes[0].kind = 'origin';
    nodes[nodes.length - 1].kind = 'destination';
    return nodes;
  }

  function nodeTime(node) {
    const [first, second] = node.stops;
    if (second) return `Arrives ${first.scheduled_time || '—'} · departs ${second.scheduled_time || '—'}`;
    return first.scheduled_time ? `Scheduled ${first.scheduled_time}` : '';
  }

  function describeNode(node) {
    const time = nodeTime(node);
    return `${TL_ROLE[node.kind]}: ${node.name}${time ? `, ${time.toLowerCase()}` : ''}`;
  }

  function createTimeline(figure) {
    const scroller = figure.querySelector('.timeline-scroll');
    const popover = figure.querySelector('.timeline-popover');
    const popRole = popover.querySelector('.popover-role');
    const popName = popover.querySelector('.popover-name');
    const popTime = popover.querySelector('.popover-time');
    const live = figure.querySelector('.timeline-live');
    const mobile = window.matchMedia('(max-width: 480px)');

    let route = null;
    let nodes = [];
    let xs = [];
    let r = 5;
    let svg = null;
    let active = -1; // arrow-key position (aria-activedescendant)
    let pinned = false; // popover opened by tap/click/keyboard, not hover

    function hidePopover() {
      popover.hidden = true;
      pinned = false;
    }

    function setActive(index) {
      active = index;
      svg.setAttribute('aria-activedescendant', `tl-node-${index}`);
      svg.querySelectorAll('.timeline-node').forEach((g, i) => g.classList.toggle('is-active', i === index));
    }

    function showPopover(index, announce) {
      const node = nodes[index];
      popRole.textContent = TL_ROLE[node.kind];
      popName.textContent = node.name;
      popTime.textContent = nodeTime(node);
      popTime.hidden = !popTime.textContent;
      popover.hidden = false;
      const x = scroller.offsetLeft + xs[index] - scroller.scrollLeft;
      const half = popover.offsetWidth / 2;
      popover.style.left = `${Math.min(Math.max(x, half), figure.clientWidth - half)}px`;
      popover.style.top = `${scroller.offsetTop + TL_BASELINE - r - 4}px`;
      if (announce) live.textContent = describeNode(node);
    }

    function moveTo(index) {
      const next = Math.min(Math.max(index, 0), nodes.length - 1);
      setActive(next);
      // Keep the active stop in view when the timeline scrolls (mobile).
      const x = xs[next];
      if (x < scroller.scrollLeft + 24 || x > scroller.scrollLeft + scroller.clientWidth - 24) {
        scroller.scrollLeft = x - scroller.clientWidth / 2;
      }
      pinned = true;
      showPopover(next, true);
    }

    function updateFade() {
      figure.classList.toggle('at-end', scroller.scrollLeft + scroller.clientWidth >= scroller.scrollWidth - 1);
    }

    function drawNode(node, i, spacing) {
      const x = xs[i];
      const g = svgEl('g', { id: `tl-node-${i}`, class: `timeline-node is-${node.kind}`, role: 'img', 'aria-label': describeNode(node) });
      g.append(svgEl('circle', { cx: x, cy: TL_BASELINE, r: r + 4, class: 'timeline-focus' }));
      if (node.kind === 'transfer') {
        // Diamond, left half = incoming leg's mode, right half = outgoing.
        const d = r + 2;
        const y = TL_BASELINE;
        const left = svgEl('polygon', { points: `${x},${y - d} ${x},${y + d} ${x - d},${y}` });
        const right = svgEl('polygon', { points: `${x},${y - d} ${x + d},${y} ${x},${y + d}` });
        left.style.fill = modeFill(node.modeLeft);
        right.style.fill = modeFill(node.modeRight);
        g.append(left, right, svgEl('polygon', {
          points: `${x},${y - d} ${x + d},${y} ${x},${y + d} ${x - d},${y}`, class: 'timeline-diamond-edge',
        }));
      } else if (node.kind === 'destination') {
        g.append(svgEl('circle', { cx: x, cy: TL_BASELINE, r: Math.max(r - 1, 2), class: 'timeline-ring' }));
      } else {
        const dot = svgEl('circle', { cx: x, cy: TL_BASELINE, r, class: node.kind === 'origin' ? 'timeline-origin' : 'timeline-stop' });
        if (node.kind === 'origin') dot.style.fill = modeFill(node.modeRight);
        g.append(dot);
      }
      // Invisible, wider hit area for pointer accuracy on dense timelines.
      g.append(svgEl('circle', {
        cx: x, cy: TL_BASELINE, r: Math.max(r + 2, Math.min(12, spacing / 2)), class: 'timeline-hit',
      }));
      g.addEventListener('pointerenter', () => { if (!pinned) showPopover(i, false); });
      g.addEventListener('pointerleave', () => { if (!pinned) popover.hidden = true; });
      g.addEventListener('click', (event) => {
        event.stopPropagation();
        if (pinned && active === i && !popover.hidden) {
          hidePopover();
          return;
        }
        setActive(i);
        pinned = true;
        showPopover(i, false);
      });
      svg.append(g);
    }

    // One icon per leg the live poll can place. The first leg's icon is the
    // Pass 9 placeholder (always shown, resting on the origin); later legs'
    // icons rest hidden on their first stop and appear only with a live fix.
    // Each pollable leg is registered in _liveLegs for _pollLiveVehicles().
    function drawVehicles() {
      const y = TL_BASELINE - r - 3 - TL_VEHICLE_SIZE / 2; // icon centre, same spot as Pass 9
      // Per leg, its stops' timeline x in order: [{x, stop, nodeIndex}, ...].
      const legStops = route.legs.map(() => []);
      nodes.forEach((node, i) => node.legRefs.forEach((legIndex, k) => {
        legStops[legIndex].push({ x: xs[i], stop: node.stops[k], nodeIndex: i });
      }));
      route.legs.forEach((leg, legIndex) => {
        const pollable = isLivePollable(leg);
        if (legIndex > 0 && !pollable) return;
        const icon = buildVehicleIcon(leg.mode, legStops[legIndex][0].x, y, legIndex > 0);
        if (!icon) return;
        svg.append(icon);
        if (!pollable) return;
        // The leg's node <g>s, for _applyStopStates; "next" pulses in the leg's mode colour.
        const nodeEls = new Map();
        for (const { nodeIndex } of legStops[legIndex]) {
          const el = svg.querySelector(`#tl-node-${nodeIndex}`);
          if (!el) continue;
          el.style.setProperty('--next-color', modeFill(leg.mode));
          nodeEls.set(nodeIndex, el);
        }
        _liveLegs.set(leg.trip_id, { leg, icon, y, nodeEls, ...liveLegLayout(leg.shape, legStops[legIndex]) });
      });
    }

    // Labels: origin, destination and transfers only. Origin/destination are
    // placed first and always kept; each transfer tries below, then above
    // (stagger). A transfer that fits in neither row drops its label (its
    // diamond, popover and aria-label remain) — it never displaces an end.
    function drawLabels(width) {
      const rows = { below: [], above: [[xs[0] - TL_VEHICLE_SIZE / 2 - 2, xs[0] + TL_VEHICLE_SIZE / 2 + 2]] };
      const last = nodes.length - 1;
      const order = [0, last, ...nodes.map((n, i) => (n.kind === 'transfer' ? i : -1)).filter((i) => i > 0)];
      for (const i of order) {
        const text = svgEl('text', { class: 'timeline-label', 'text-anchor': 'middle', 'aria-hidden': 'true' });
        text.textContent = nodes[i].name;
        svg.append(text);
        const w = text.getComputedTextLength();
        const cx = Math.min(Math.max(xs[i], w / 2 + 2), width - w / 2 - 2);
        const span = [cx - w / 2 - TL_LABEL_GAP / 2, cx + w / 2 + TL_LABEL_GAP / 2];
        const fits = (row) => rows[row].every(([a, b]) => span[1] <= a || span[0] >= b);
        let row = ['below', 'above'].find(fits);
        if (!row) {
          if (i !== 0 && i !== last) {
            text.remove();
            continue;
          }
          row = 'below'; // ends always render, even if they must overlap
        }
        rows[row].push(span);
        text.setAttribute('x', String(cx));
        text.setAttribute('y', String(row === 'below' ? TL_BASELINE + r + 15 : TL_BASELINE - r - 9));
      }
    }

    function render(nextRoute) {
      route = nextRoute;
      hidePopover();
      active = -1;
      live.textContent = '';
      _liveLegs = new Map(); // the old icons go with the old SVG
      if (!route || !route.legs.length) {
        figure.hidden = true;
        return;
      }
      figure.hidden = false; // must be laid out before labels can be measured
      nodes = timelineNodes(route);
      const n = nodes.length;
      const avail = scroller.clientWidth;
      let spacing = n > 1 ? (avail - 2 * TL_PAD) / (n - 1) : 0;
      if (mobile.matches && n > 8) spacing = Math.max(spacing, TL_MOBILE_MIN_SPACING);
      const width = Math.max(avail, 2 * TL_PAD + spacing * (n - 1));
      // ~5px (≤480px: ~4px), shrunk only if stops would otherwise touch.
      r = Math.max(2, Math.min(mobile.matches ? 4 : 5, spacing * 0.4));
      xs = nodes.map((_, i) => TL_PAD + spacing * i);

      const transfers = route.transfer_count;
      figure.setAttribute('aria-label',
        `Timeline for ${nodes[0].name} to ${nodes[n - 1].name}, ${n} stops with ${transfers} transfer${transfers === 1 ? '' : 's'}`);
      svg = svgEl('svg', {
        class: 'timeline-svg', width, height: TL_HEIGHT, viewBox: `0 0 ${width} ${TL_HEIGHT}`,
        role: 'group', tabindex: 0, 'aria-label': 'Stops. Use the arrow keys to move between stops.',
      });
      scroller.replaceChildren(svg);
      scroller.scrollLeft = 0;
      figure.classList.toggle('is-scrollable', width > avail + 1);
      updateFade();

      svg.append(svgEl('line', { x1: TL_PAD, y1: TL_BASELINE, x2: width - TL_PAD, y2: TL_BASELINE, class: 'timeline-line' }));
      nodes.forEach((node, i) => drawNode(node, i, spacing));
      drawVehicles();
      // A re-render (resize) rebuilds every node: restore stop states from the
      // last poll rather than leaving them blank for up to 30s.
      for (const [tripId, progress] of _liveLastProgress) _applyStopStates(tripId, progress);
      drawLabels(width);

      // One Tab stop; arrows move the active stop (aria-activedescendant).
      svg.addEventListener('focus', () => {
        if (svg.matches(':focus-visible')) moveTo(active >= 0 ? active : 0);
      });
      svg.addEventListener('blur', hidePopover);
      svg.addEventListener('keydown', (event) => {
        const keys = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 };
        if (event.key in keys) moveTo((active < 0 ? 0 : active) + keys[event.key]);
        else if (event.key === 'Home') moveTo(0);
        else if (event.key === 'End') moveTo(nodes.length - 1);
        else if (event.key === 'Escape') hidePopover();
        else return;
        event.preventDefault();
      });
    }

    scroller.addEventListener('scroll', () => {
      updateFade();
      hidePopover(); // its position is stale once the strip moves
    }, { passive: true });

    document.addEventListener('click', (event) => {
      if (!figure.contains(event.target)) hidePopover();
    });

    // Width-dependent layout: redraw on resize (rAF-debounced).
    let resizeFrame = 0;
    window.addEventListener('resize', () => {
      cancelAnimationFrame(resizeFrame);
      resizeFrame = requestAnimationFrame(() => { if (route) render(route); });
    });

    return { render };
  }

  // ══ Results page ("/results") ═══════════════════════════════

  function initResultsPage() {
    const main = document.querySelector('main');
    const { from, to, departure } = main.dataset;
    const routeStrip = document.querySelector('#route-strip');
    // The <details> disclosure, and the <ol> of alternative cards inside it.
    const alternativesDisclosure = document.querySelector('#alternatives');
    const alternativesEl = document.querySelector('#alternatives-section');
    const alternativesLabel = alternativesDisclosure.querySelector('[data-alternatives-label]');
    const emptyEl = document.querySelector('#results-empty');
    const emptyMessage = document.querySelector('#results-empty-message');
    const timeline = createTimeline(document.querySelector('#timeline'));
    let routes = [];
    let activeRouteIndex = 0;

    // The Model details disclosure shows only alongside a rendered route:
    // hidden while loading and on empty/error. Absent (no model_stats) is fine.
    function setModelDetailsVisible(visible) {
      const details = document.querySelector('.model-details');
      if (details) details.hidden = !visible;
    }

    // Contextual pill: written into results.html's existing slot, never
    // created here. Cleared whenever there's no active route.
    const contextPill = document.querySelector('.context-pill-slot');

    function renderContextPill() {
      if (!contextPill) return;
      contextPill.classList.remove('pill-context', ...CONTEXT_PILL_TONES);
      contextPill.removeAttribute('title');
      contextPill.textContent = '';
      const route = routes[activeRouteIndex];
      const delta = route ? minutesUntil(route.leave_by, new Date()) : null;
      if (delta === null) return;
      const [, label, tone] = CONTEXT_PILL_RULES.find(([max]) => delta <= max);
      const low = route.confidence === 'low';
      contextPill.textContent = `● ${label}`;
      contextPill.classList.add('pill-context', `pill-context-${low ? 'muted' : tone}`);
      if (low) contextPill.title = LOW_CONFIDENCE_TITLE;
    }

    // Re-evaluates the pill as the clock moves. One timer per page view,
    // never restarted on active-route changes.
    let contextPillTimer = 0;

    function startContextPillTimer() {
      clearInterval(contextPillTimer);
      contextPillTimer = setInterval(renderContextPill, CONTEXT_PILL_TICK_MS);
    }

    window.addEventListener('pagehide', () => clearInterval(contextPillTimer));
    // A back/forward-cache restore skips load(), and pagehide cleared the
    // timer, so bring the pill up to date and restart it.
    window.addEventListener('pageshow', (event) => {
      if (!event.persisted) return;
      renderContextPill();
      startContextPillTimer();
    });

    function showEmpty(message, kind) {
      setModelDetailsVisible(false);
      routes = []; // no active route: the pill clears below
      renderContextPill();
      alternativesDisclosure.hidden = true;
      setStatus('');
      emptyMessage.textContent = message;
      if (kind) emptyEl.dataset.kind = kind;
      else delete emptyEl.dataset.kind;
      emptyEl.hidden = false;
    }

    // Alternatives: every route except the active one, in original rank
    // order, so the previous hero rotates back in where it ranks. Never
    // touches the disclosure's open state — only a fresh /routes does.
    function renderAlternatives() {
      const cards = [];
      routes.forEach((route, i) => {
        if (i === activeRouteIndex) return;
        const card = renderRoute(route);
        card.dataset.routeIndex = String(i); // card -> route, for the hero + timeline
        cards.push(card);
      });
      alternativesEl.replaceChildren(...cards);
      const count = cards.length;
      alternativesLabel.textContent = count === 0 ? 'Alternatives'
        : `${count} alternative${count === 1 ? '' : 's'}`;
      alternativesDisclosure.hidden = count === 0;
    }

    // Active route: the hero, the contextual pill and the station timeline
    // all show routes[activeRouteIndex]. The top pick (0) starts active;
    // clicking or Enter/Space on an alternative card moves it. The hero
    // isn't a card, so clicking it matches nothing here (a no-op).
    function setActiveRoute(index) {
      activeRouteIndex = index;
      renderAlternatives();
      renderHero(routes[activeRouteIndex]);
      renderContextPill();
      timeline.render(routes[activeRouteIndex]);
      startLivePolling(routes[activeRouteIndex].legs.filter(isLivePollable).map((leg) => leg.trip_id));
    }

    main.addEventListener('click', (event) => {
      const card = event.target.closest('.results > .card');
      if (card) setActiveRoute(Number(card.dataset.routeIndex));
    });

    main.addEventListener('keydown', (event) => {
      if (event.key !== 'Enter' && event.key !== ' ') return;
      if (!event.target.matches('.results > .card')) return;
      event.preventDefault();
      setActiveRoute(Number(event.target.dataset.routeIndex));
      // The focused card was just rebuilt away; keep keyboard focus in the
      // disclosure rather than dropping it to <body>.
      alternativesDisclosure.querySelector('summary').focus();
    });

    async function load() {
      stopLivePolling(); // nothing rendered yet to poll for
      setModelDetailsVisible(false);
      renderContextPill(); // no routes yet: clears the slot
      alternativesDisclosure.hidden = true;
      setStatus('Finding routes…');
      const params = new URLSearchParams({ from_stop_id: from, to_stop_id: to });
      if (departure) params.set('departure', departure);
      try {
        const resp = await fetch('/routes?' + params.toString());
        if (!resp.ok) {
          const body = await resp.json().catch(() => ({}));
          const detail = typeof body.detail === 'string' ? body.detail : resp.statusText;
          showEmpty(
            resp.status === 404
              ? `We couldn’t find one of those stops (${detail}). Try a different From or To.`
              : `Error ${resp.status}: ${detail}`,
            'error',
          );
          return;
        }
        routes = await resp.json();
        if (routes.length === 0) {
          showEmpty('No routes found for this pair. Try a different From or To.');
          return;
        }
        setStatus('Ranked by predicted arrival.');
        // Columns settle first: the timeline measures its width on render.
        setModelDetailsVisible(true);
        routeStrip.hidden = false; // visible before the timeline measures its width
        setActiveRoute(0); // routes[0] is the hero; the rest render as alternatives
        // Fresh response only: closed by default, opened for a low-confidence
        // top pick. Later active-route changes leave it as the user set it.
        alternativesDisclosure.open = routes[0].confidence === 'low';
      } catch (err) {
        showEmpty('Couldn’t reach the server. Try again.', 'error');
      }
    }

    startContextPillTimer();
    load();
  }

  // ---- Pass 10 tests ----
  // Runs automatically only with ?pass10_test=1; otherwise call
  // TransitAI._testPass10Projection() from the console.

  function _testPass10Projection() {
    // Surfers Paradise -> Broadbeach, roughly along the G:link corridor.
    const shape = [
      [-28.0005, 153.4295],
      [-28.0100, 153.4300],
      [-28.0230, 153.4310],
      [-28.0330, 153.4300],
    ];
    const last = shape.length - 1;
    const [a, b] = [shape[1], shape[2]]; // the middle segment, index 1
    const mid = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
    const d = 0.001;
    const len = Math.hypot(b[0] - a[0], b[1] - a[1]);
    const perp = [-(b[1] - a[1]) / len, (b[0] - a[0]) / len]; // unit normal to a->b

    const start = nearestPointOnShape(shape[0][0], shape[0][1], shape);
    const end = nearestPointOnShape(shape[last][0], shape[last][1], shape);
    const onMid = nearestPointOnShape(mid[0], mid[1], shape);
    const offMid = nearestPointOnShape(mid[0] + d * perp[0], mid[1] + d * perp[1], shape);

    const checks = [
      [start && start.t === 0 && start.segmentIndex === 0 && start.distanceSq < 1e-12,
        'vehicle on shape[0]: t=0, segmentIndex=0, distanceSq≈0'],
      [end && end.t === 1 && end.segmentIndex === last - 1 && end.distanceSq < 1e-12,
        'vehicle on last point: t=1, segmentIndex=length-2, distanceSq≈0'],
      [onMid && Math.abs(onMid.t - 0.5) < 1e-9 && onMid.segmentIndex === 1 && onMid.distanceSq < 1e-12,
        'vehicle at middle-segment midpoint: t≈0.5, segmentIndex=1, distanceSq≈0'],
      [offMid && Math.abs(offMid.t - 0.5) < 1e-9 && offMid.segmentIndex === 1
        && Math.abs(offMid.distanceSq - d * d) < d * d * 1e-6,
        'vehicle offset perpendicular by d: t≈0.5, distanceSq≈d²'],
      [nearestPointOnShape(-28.01, 153.43, []) === null
        && nearestPointOnShape(-28.01, 153.43, [shape[0]]) === null
        && nearestPointOnShape(NaN, 153.43, shape) === null,
        'degenerate inputs (empty shape, 1-point shape, NaN lat) return null'],
    ];

    let passed = 0;
    for (const [ok, message] of checks) {
      console.assert(ok, message);
      if (ok) passed++;
    }
    console.log(`Pass 10 projection tests: ${passed}/${checks.length} passed`);
  }

  if (window.location.search.includes('pass10_test=1')) _testPass10Projection();
})();
