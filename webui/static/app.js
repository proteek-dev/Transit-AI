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
  const resultsEl = document.querySelector('#results');

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

  function makePicker(prefix) {
    const input = document.querySelector(`#${prefix}-input`);
    const dropdown = document.querySelector(`#${prefix}-dropdown`);
    const note = document.querySelector(`#${prefix}-note`);
    const listbox = document.querySelector(`#${prefix}-listbox`);

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
      pushRecent(stop);
      close();
      clearError();
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
      picker.selected = null; // typing voids an earlier selection
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

  // ── Wiring ──────────────────────────────────────────────────

  const fromPicker = makePicker('from');
  const toPicker = makePicker('to');
  const form = document.querySelector('#route-form');
  const submitButton = document.querySelector('#submit');

  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    resultsEl.replaceChildren();

    if (!fromPicker.selected || !toPicker.selected) {
      setStatus('Pick a From and a To stop from the suggestions first.', 'error');
      return;
    }
    if (fromPicker.selected.stop_id === toPicker.selected.stop_id) {
      setStatus('From and To can’t be the same stop.', 'error');
      return;
    }

    setStatus('Finding routes…');
    submitButton.disabled = true;
    const params = new URLSearchParams({
      from_stop_id: fromPicker.selected.stop_id,
      to_stop_id: toPicker.selected.stop_id,
    });
    try {
      const resp = await fetch('/routes?' + params.toString());
      if (!resp.ok) {
        const body = await resp.json().catch(() => ({}));
        const detail = typeof body.detail === 'string' ? body.detail : resp.statusText;
        setStatus(`Error ${resp.status}: ${detail}`, 'error');
        return;
      }
      const routes = await resp.json();
      if (routes.length === 0) {
        setStatus('No routes found between these stops in the next 60 minutes.');
        return;
      }
      setStatus('Ranked by predicted arrival.');
      resultsEl.replaceChildren(...routes.map(renderRoute));
    } catch (err) {
      setStatus('Couldn’t reach the server. Try again.', 'error');
    } finally {
      submitButton.disabled = false;
    }
  });
})();
