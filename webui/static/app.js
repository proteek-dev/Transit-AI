// Transit AI web UI — vanilla JS, no build step. Every DOM write uses
// textContent / createElement; no HTML strings are ever parsed.

(() => {
  'use strict';

  const DEBOUNCE_MS = 200;
  const SEARCH_LIMIT = 8;
  const RECENTS_KEY = 'webui.recentStops.v1';
  const MAX_RECENTS = 5;
  const NEAR_MATCH_NOTE = 'No exact match — showing near matches';

  const MODE_LABEL = { bus: 'Bus', tram: 'Tram', rail: 'Train', ferry: 'Ferry', unknown: 'Service' };
  const PILL_CLASS = { High: 'pill-high', Medium: 'pill-medium', Low: 'pill-low' };

  // ── Recent stops (localStorage) ─────────────────────────────
  // Every storage call is guarded: Safari private mode and blocked storage
  // throw, and a failure there should only mean "no recents".

  function isRecent(item) {
    return item !== null && typeof item === 'object'
      && typeof item.stop_id === 'string' && typeof item.stop_name === 'string';
  }

  function readRecents() {
    try {
      const parsed = JSON.parse(window.localStorage.getItem(RECENTS_KEY) || '[]');
      if (!Array.isArray(parsed)) return [];
      return parsed.filter(isRecent)
        .map((s) => ({ stop_id: s.stop_id, stop_name: s.stop_name }))
        .slice(0, MAX_RECENTS);
    } catch (err) {
      return [];
    }
  }

  function pushRecent(stop) {
    const entry = { stop_id: stop.stop_id, stop_name: stop.stop_name };
    const next = [entry, ...readRecents().filter((s) => s.stop_id !== entry.stop_id)].slice(0, MAX_RECENTS);
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
      // Keep coordinates when the stop record has them (/stops/search and
      // /stops/nearby rows do; recents don't). Recents still store id + name only.
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
    return picker;
  }

  // ── Route results ───────────────────────────────────────────

  function transferLabel(count) {
    if (count === 0) return 'direct';
    return `${count} transfer${count === 1 ? '' : 's'}`;
  }

  function renderLeg(leg) {
    const li = el('li', 'leg');
    const main = el('div', 'leg-main');
    const icon = window.TransitIcons ? window.TransitIcons.modeIcon(leg.mode, `leg-icon mode-${leg.mode}`) : null;
    if (icon) main.append(icon);
    main.append(
      el('span', 'leg-mode', (MODE_LABEL[leg.mode] || MODE_LABEL.unknown) + ' '),
      el('span', 'leg-route', leg.route_short_name),
    );
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
    legs.append(...route.legs.map(renderLeg));
    card.append(head, legs);
    return card;
  }

  // ── Page switch ─────────────────────────────────────────────
  // base.html sets <body data-page>. Exactly one page mode runs: the search
  // page never renders results, the results page never builds pickers.

  const page = document.body.dataset.page;
  if (page === 'search') initSearchPage();
  else if (page === 'results') initResultsPage();

  // ══ Search page ("/") ═══════════════════════════════════════

  function initSearchPage() {
    let nearbyStops = [];
    // 'from' | 'to' — which picker a nearby-pin tap fills. In memory only.
    let activePinTarget = 'from';
    let pinSelecting = false; // true while a pin tap runs a picker's select()

    // ── Map (Leaflet, loaded from unpkg by index.html) ────────
    // One map instance, created lazily the first time the #map slot is
    // shown (Leaflet needs a visible container to measure). The search page
    // only has the nearby overview; if Leaflet failed to load, the slot
    // stays hidden and the pickers work as before.

    const mapEl = document.querySelector('#map');
    const mapHeaderEl = document.querySelector('#map-header');
    const DEFAULT_CENTER = [-27.4698, 153.0251]; // Brisbane CBD, used only if there are no coordinates
    const FIT_PADDING = [24, 24];
    const HIT_RADIUS = 22; // invisible marker hit area: 44px across, the touch-target size

    let map = null;
    let pinLayer = null;
    let renderedView = 'hidden';

    function mapAvailable() {
      return typeof window.L !== 'undefined';
    }

    function ensureMap() {
      if (map) return;
      map = L.map(mapEl, {
        // A page-embedded map shouldn't hijack scrolling: no wheel zoom, and
        // no one-finger drag on phones (zoom buttons and taps still work).
        scrollWheelZoom: false,
        dragging: !L.Browser.mobile,
      });
      // Leaflet's own constant attribution markup, not user data.
      L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
        maxZoom: 19,
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
      }).addTo(map);
      pinLayer = L.layerGroup().addTo(map);
    }

    // hidden | nearby, derived from page state only.
    function mapView() {
      if (!mapAvailable()) return 'hidden';
      return fromPicker.originCoords ? 'nearby' : 'hidden';
    }

    // Redraw the map for the current state. The only thing that shows,
    // hides or redraws it.
    function renderMap() {
      const view = mapView();
      mapHeaderEl.hidden = view !== 'nearby'; // pin-target header: nearby overview only
      if (view === 'hidden') {
        mapEl.hidden = true;
        if (pinLayer) pinLayer.clearLayers();
        renderedView = 'hidden';
        return;
      }
      const wasHidden = mapEl.hidden;
      mapEl.hidden = false;
      ensureMap();
      if (wasHidden) map.invalidateSize();
      drawNearby();
      renderedView = view;
    }

    // Redraw only if the view itself changed — e.g. picking a From from a
    // nearby pin keeps the overview as-is instead of re-fitting it.
    function syncMap() {
      if (mapView() !== renderedView) renderMap();
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

    function drawNearby() {
      pinLayer.clearLayers();
      const origin = fromPicker.originCoords;
      const colors = mapColors();
      const bounds = L.latLngBounds([]);

      // "You are here": filled dot + pulsing ring. Never interactive, so it
      // can't swallow a tap meant for a stop pin underneath it.
      const here = [origin.lat, origin.lon];
      L.circleMarker(here, {
        radius: 14, weight: 2, color: colors.marker, fillColor: colors.marker, fillOpacity: 0.12,
        interactive: false, className: 'map-here-pulse',
      }).addTo(pinLayer);
      L.circleMarker(here, {
        radius: 6, weight: 2, color: colors.casing, fillColor: colors.marker, fillOpacity: 1, interactive: false,
      }).addTo(pinLayer);
      bounds.extend(here);

      for (const stop of nearbyStops) {
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
        L.circleMarker(latlng, { radius: HIT_RADIUS, stroke: false, fill: true, fillOpacity: 0 })
          .bindTooltip(mapLabel(role, `${stop.stop_name} · ${distanceLabel(stop.distance_km)}`), {
            direction: 'top', offset: [0, -8],
          })
          .on('click', () => pinSelect(stop))
          .addTo(pinLayer);
        bounds.extend(latlng);
      }
      if (bounds.isValid()) map.fitBounds(bounds, { padding: FIT_PADDING, maxZoom: 16 });
      else map.setView(DEFAULT_CENTER, 11);
    }

    // ── Pickers ───────────────────────────────────────────────

    function onFromChange() {
      if (fromPicker.selected) setFromHint('');
      retargetAfterChange('from');
      syncMap();
    }

    function onToChange() {
      retargetAfterChange('to');
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
      if (pinSelecting) return;
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

    // Exactly one /stops/nearby request per location grant.
    async function loadNearby(coords) {
      const params = new URLSearchParams({
        lat: String(coords.lat), lon: String(coords.lon), limit: String(NEARBY_LIMIT),
      });
      try {
        const resp = await fetch('/stops/nearby?' + params.toString());
        if (!resp.ok) throw new Error('HTTP ' + resp.status);
        return await resp.json();
      } catch (err) {
        setFromHint('Couldn’t load nearby stops. Try again or search by name.');
        return [];
      }
    }

    async function onPosition(position) {
      const coords = { lat: position.coords.latitude, lon: position.coords.longitude };
      fromPicker.originCoords = coords;
      setLocationGranted(true);
      setFromHint('');
      nearbyStops = await loadNearby(coords);
      setLocating(false);
      renderMap(); // new pins: redraw + re-fit even if already in the overview
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
      window.location.href = '/results?from=' + encodeURIComponent(fromPicker.selected.stop_id)
        + '&to=' + encodeURIComponent(toPicker.selected.stop_id);
    });
  }

  // ══ Station timeline (results page) ═════════════════════════
  // Inline SVG built through DOM APIs (no markup strings), like icons.js.
  // Static in Pass 9: the vehicle icon sits on the first leg's origin; Pass
  // 10 swaps in a real GTFS-RT position — no timers or polling here.

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
          return;
        }
        nodes.push({
          kind: stop.is_transfer_point ? 'transfer' : 'stop',
          name: stop.stop_name,
          stops: [stop],
          modeLeft: leg.mode,
          modeRight: leg.mode,
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

    function drawVehicle() {
      const mode = route.legs[0].mode;
      const icon = window.TransitIcons ? window.TransitIcons.modeIcon(mode, `timeline-vehicle mode-${mode}`) : null;
      if (!icon) return;
      icon.setAttribute('x', String(xs[0] - TL_VEHICLE_SIZE / 2));
      icon.setAttribute('y', String(TL_BASELINE - r - TL_VEHICLE_SIZE - 3));
      icon.setAttribute('width', String(TL_VEHICLE_SIZE));
      icon.setAttribute('height', String(TL_VEHICLE_SIZE));
      svg.append(icon);
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
      drawVehicle();
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
    const topPickEl = document.querySelector('#top-pick');
    const alternativesSection = document.querySelector('#alternatives-section');
    const alternativesEl = document.querySelector('#alternatives');
    const emptyEl = document.querySelector('#results-empty');
    const emptyMessage = document.querySelector('#results-empty-message');
    const timeline = createTimeline(document.querySelector('#timeline'));
    let routes = [];

    function showEmpty(message, kind) {
      setStatus('');
      emptyMessage.textContent = message;
      if (kind) emptyEl.dataset.kind = kind;
      else delete emptyEl.dataset.kind;
      emptyEl.hidden = false;
    }

    // Active card: visual state plus the station timeline above the top
    // pick, which redraws for whichever card is active. The top pick starts
    // active; clicking or Enter/Space on any card moves it.
    function allCards() {
      return Array.from(main.querySelectorAll('.results > .card'));
    }

    function setActiveCard(active) {
      for (const card of allCards()) {
        const on = card === active;
        card.classList.toggle('is-active', on);
        if (on) card.setAttribute('aria-current', 'true');
        else card.removeAttribute('aria-current');
      }
      timeline.render(routes[Number(active.dataset.routeIndex)]);
    }

    main.addEventListener('click', (event) => {
      const card = event.target.closest('.results > .card');
      if (card) setActiveCard(card);
    });

    main.addEventListener('keydown', (event) => {
      if (event.key !== 'Enter' && event.key !== ' ') return;
      if (!event.target.matches('.results > .card')) return;
      event.preventDefault();
      setActiveCard(event.target);
    });

    async function load() {
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
        const cards = routes.map((route, i) => {
          const card = renderRoute(route);
          card.dataset.routeIndex = String(i); // card -> route, for the timeline
          return card;
        });
        topPickEl.replaceChildren(cards[0]);
        alternativesEl.replaceChildren(...cards.slice(1));
        routeStrip.hidden = false; // visible before the timeline measures its width
        alternativesSection.hidden = routes.length < 2;
        setActiveCard(topPickEl.firstElementChild);
      } catch (err) {
        showEmpty('Couldn’t reach the server. Try again.', 'error');
      }
    }

    load();
  }
})();
