'use client';

import { useEffect, useId, useRef, useState, type KeyboardEvent } from 'react';
import { fetchStopsSearch } from '@/lib/api';
import { addRecentStop, getRecentStops, type RecentStop } from '@/lib/recentStops';
import type { StopSearchResult } from '@/lib/types';

export type StopPickerProps = {
  label: string;                              // e.g. "From", "To" — rendered as visible <label>
  value: StopSearchResult | null;             // currently selected stop, controlled by parent
  onSelect: (stop: StopSearchResult | null) => void;  // null = user cleared selection
  placeholder?: string;                       // input placeholder text
  id?: string;                                // optional id for label htmlFor + input id
};

type SearchStatus = 'idle' | 'loading' | 'ok' | 'error' | 'empty';
type GeoStatus = 'idle' | 'requesting' | 'granted' | 'denied' | 'unavailable';
type Coords = { lat: number; lon: number };

const DEBOUNCE_MS = 250;
const RESULT_LIMIT = 8;
// Mirrors fetchStopsSearch's q length check so the input can't produce a
// query it would reject.
const MAX_QUERY_LENGTH = 100;
const FALLBACK_ERROR = "Couldn't search stops. Try again.";

const GEO_LABEL: Record<GeoStatus, string> = {
  idle: 'Use my location',
  requesting: 'Getting location...',
  granted: 'Location on',
  denied: 'Location blocked',
  unavailable: 'Location unavailable',
};

function formatDistance(km: number): string {
  return km < 1 ? `${Math.round(km * 1000)} m` : `${km.toFixed(1)} km`;
}

/** One dropdown row, shared by search results and recents. */
function StopRow({
  stop,
  optionId,
  active,
  divider,
  onPick,
  onHover,
}: {
  stop: StopSearchResult;
  optionId: string;
  active: boolean;
  divider: boolean;
  onPick: () => void;
  onHover: () => void;
}) {
  return (
    <li
      id={optionId}
      role="option"
      aria-selected={active}
      // Keep focus on the input so blur doesn't close the dropdown before
      // the click lands.
      onMouseDown={(e) => e.preventDefault()}
      onClick={onPick}
      onMouseEnter={onHover}
      style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        gap: 12,
        padding: '0.5rem 0.75rem',
        cursor: 'pointer',
        background: active ? '#f3f4f6' : undefined,
        borderTop: divider ? '1px solid #e5e7eb' : undefined,
      }}
    >
      <div style={{ minWidth: 0 }}>
        <div style={{ fontSize: '1em' }}>{stop.stop_name}</div>
        <div style={{ fontSize: '0.8em', color: '#666' }}>{stop.stop_id}</div>
      </div>
      {stop.distance_km !== undefined && (
        <span style={{ fontSize: '0.85em', color: '#4b5563', whiteSpace: 'nowrap' }}>
          {formatDistance(stop.distance_km)}
        </span>
      )}
    </li>
  );
}

/**
 * Debounced stop search input backed by GET /stops/search: a results
 * dropdown with keyboard navigation, an opt-in "Use my location" button that
 * biases results by distance, and loading/empty/error rows. The parent owns
 * the selected stop via `value` / `onSelect`.
 */
export default function StopPicker({ label, value, onSelect, placeholder, id }: StopPickerProps) {
  const generatedId = useId();
  const inputId = id ?? `stop-picker-${generatedId}`;
  const listboxId = `${inputId}-listbox`;

  const [query, setQuery] = useState(value?.stop_name ?? '');
  const [results, setResults] = useState<StopSearchResult[]>([]);
  const [status, setStatus] = useState<SearchStatus>('idle');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [geo, setGeo] = useState<Coords | null>(null);
  const [geoStatus, setGeoStatus] = useState<GeoStatus>('idle');
  // False only when the browser has no Geolocation API at all -- a timeout
  // or unavailable position leaves the button clickable for a retry.
  const [geoSupported, setGeoSupported] = useState(true);
  const [isOpen, setIsOpen] = useState(false);
  const [activeIndex, setActiveIndex] = useState(-1);
  // Read client-side after mount (not in the initializer) so server and
  // first client render match.
  const [recentStops, setRecentStops] = useState<RecentStop[]>([]);

  // stop_id of the last stop we reported via onSelect (or synced from
  // `value`); lets a parent-driven `value` change update the input text
  // without echoing our own selections back. Invariant: null iff the input
  // doesn't reflect a parent-supplied selection. Our own clears reset it
  // before onSelect(null), so only a parent-initiated null (e.g. a swap
  // moving our stop to the other picker) reaches the second branch.
  const [syncedStopId, setSyncedStopId] = useState<string | null>(value?.stop_id ?? null);
  if (value && value.stop_id !== syncedStopId) {
    setSyncedStopId(value.stop_id);
    setQuery(value.stop_name);
  } else if (!value && syncedStopId !== null) {
    setSyncedStopId(null);
    setQuery('');
  }

  const abortRef = useRef<AbortController | null>(null);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const blurTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);
  // Latest query for the geolocation callback, which resolves long after the
  // click that started it.
  const queryRef = useRef(query);

  function cancelPending() {
    if (debounceRef.current) {
      clearTimeout(debounceRef.current);
      debounceRef.current = null;
    }
    abortRef.current?.abort();
    abortRef.current = null;
  }

  async function runSearch(q: string, coords: Coords | null) {
    const controller = new AbortController();
    abortRef.current = controller;
    try {
      const rows = await fetchStopsSearch(q, {
        limit: RESULT_LIMIT,
        lat: coords?.lat,
        lon: coords?.lon,
        signal: controller.signal,
      });
      if (controller.signal.aborted) return;
      setResults(rows);
      setStatus(rows.length === 0 ? 'empty' : 'ok');
      setErrorMessage(null);
      setActiveIndex(-1);
      setIsOpen(true);
    } catch (err) {
      // Superseded by a newer keystroke / location grant, or unmounted.
      if (err instanceof DOMException && err.name === 'AbortError') return;
      if (controller.signal.aborted) return;
      console.error(err);
      setResults([]);
      setStatus('error');
      setErrorMessage(err instanceof Error && err.message ? err.message : FALLBACK_ERROR);
      setActiveIndex(-1);
      setIsOpen(true);
    }
  }

  useEffect(() => {
    if (typeof navigator === 'undefined' || !navigator.geolocation) {
      setGeoSupported(false);
      setGeoStatus('unavailable');
    }
    return () => {
      cancelPending();
      if (blurTimerRef.current) clearTimeout(blurTimerRef.current);
    };
  }, []);

  useEffect(() => {
    setRecentStops(getRecentStops());
  }, []);

  function handleChange(next: string) {
    const wasEmpty = query.trim() === '';
    setQuery(next);
    queryRef.current = next;
    cancelPending();

    const trimmed = next.trim();
    if (trimmed === '') {
      setStatus('idle');
      setResults([]);
      setErrorMessage(null);
      setActiveIndex(-1);
      if (!wasEmpty) {
        setSyncedStopId(null);
        onSelect(null);
      }
      return;
    }

    // Edited away from the picked stop's name: the selection no longer
    // matches what's typed, so void it.
    if (value !== null && trimmed !== value.stop_name) {
      setSyncedStopId(null);
      onSelect(null);
    }

    setStatus('loading');
    setIsOpen(true);
    const coords = geo;
    debounceRef.current = setTimeout(() => {
      debounceRef.current = null;
      void runSearch(trimmed, coords);
    }, DEBOUNCE_MS);
  }

  function selectStop(stop: StopSearchResult) {
    cancelPending();
    setSyncedStopId(stop.stop_id);
    onSelect(stop);
    addRecentStop(stop);
    setRecentStops(getRecentStops());
    setQuery(stop.stop_name);
    queryRef.current = stop.stop_name;
    setIsOpen(false);
    setActiveIndex(-1);
    inputRef.current?.blur();
  }

  function requestLocation() {
    if (!geoSupported) return;
    setGeoStatus('requesting');
    navigator.geolocation.getCurrentPosition(
      (position) => {
        const coords = { lat: position.coords.latitude, lon: position.coords.longitude };
        setGeo(coords);
        setGeoStatus('granted');
        const trimmed = queryRef.current.trim();
        if (trimmed !== '') {
          // Re-run now with the new position -- no debounce.
          cancelPending();
          setStatus('loading');
          void runSearch(trimmed, coords);
        }
      },
      (error) => {
        setGeo(null);
        setGeoStatus(error.code === error.PERMISSION_DENIED ? 'denied' : 'unavailable');
      },
      { timeout: 8000, maximumAge: 60_000 },
    );
  }

  const trimmedQuery = query.trim();
  const showRecents = trimmedQuery === '' && status === 'idle' && recentStops.length > 0;
  const showResults = trimmedQuery !== '' && status === 'ok' && results.length > 0;
  // Whichever list the dropdown renders; activeIndex indexes into it.
  const navList: StopSearchResult[] = showRecents ? recentStops : showResults ? results : [];

  function handleKeyDown(e: KeyboardEvent<HTMLInputElement>) {
    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault();
        if (!isOpen && navList.length > 0) {
          setIsOpen(true);
        } else {
          setActiveIndex((i) => Math.min(i + 1, navList.length - 1));
        }
        break;
      case 'ArrowUp':
        e.preventDefault();
        setActiveIndex((i) => Math.max(i - 1, 0));
        break;
      case 'Enter':
        if (activeIndex >= 0 && navList[activeIndex]) {
          e.preventDefault();
          selectStop(navList[activeIndex]);
        }
        break;
      case 'Escape':
        setIsOpen(false);
        setActiveIndex(-1);
        break;
    }
  }

  function handleBlur() {
    // Deferred so a row click registers before the dropdown closes.
    blurTimerRef.current = setTimeout(() => setIsOpen(false), 0);
  }

  const geoDisabled = !geoSupported || geoStatus === 'requesting';
  const dropdownOpen = isOpen && (showRecents || (trimmedQuery !== '' && status !== 'idle'));
  const statusText =
    status === 'loading'
      ? 'Searching...'
      : status === 'empty'
        ? `No stops match "${trimmedQuery}".`
        : status === 'error'
          ? errorMessage ?? FALLBACK_ERROR
          : null;

  return (
    <div>
      <label htmlFor={inputId} style={{ display: 'block', fontWeight: 600, fontSize: '0.9em', marginBottom: 4 }}>
        {label}
      </label>
      <div style={{ display: 'flex', gap: 8, alignItems: 'stretch' }}>
        <div style={{ position: 'relative', flex: 1 }}>
          <input
            ref={inputRef}
            id={inputId}
            type="text"
            role="combobox"
            aria-autocomplete="list"
            aria-expanded={dropdownOpen}
            aria-controls={listboxId}
            aria-activedescendant={
              dropdownOpen && navList.length > 0 && activeIndex >= 0 ? `${listboxId}-${activeIndex}` : undefined
            }
            autoComplete="off"
            maxLength={MAX_QUERY_LENGTH}
            value={query}
            placeholder={placeholder}
            onChange={(e) => handleChange(e.target.value)}
            onKeyDown={handleKeyDown}
            onFocus={() => {
              // Empty query: only open when there are recents to show, so a
              // first-time user doesn't get an empty dropdown flash.
              if (trimmedQuery !== '' || recentStops.length > 0) setIsOpen(true);
            }}
            onBlur={handleBlur}
            style={{
              width: '100%',
              padding: '0.5rem 0.75rem',
              border: '1px solid #d1d5db',
              borderRadius: 8,
              font: 'inherit',
            }}
          />

          {dropdownOpen && (
            <div
              style={{
                position: 'absolute',
                top: 'calc(100% + 4px)',
                left: 0,
                right: 0,
                zIndex: 10,
                background: '#fff',
                border: '1px solid #ddd',
                borderRadius: 8,
                boxShadow: '0 4px 12px rgba(0, 0, 0, 0.08)',
                overflow: 'hidden',
              }}
            >
              {navList.length > 0 ? (
                <>
                  {showRecents && (
                    <div aria-hidden style={{ padding: '0.5rem 0.75rem 0.25rem', fontSize: '0.8em', color: '#666' }}>
                      Recent
                    </div>
                  )}
                  <ul
                    id={listboxId}
                    role="listbox"
                    aria-label={showRecents ? `Recent ${label} stops` : `${label} stop results`}
                    style={{ listStyle: 'none' }}
                  >
                    {navList.map((stop, i) => (
                      <StopRow
                        key={stop.stop_id}
                        stop={stop}
                        optionId={`${listboxId}-${i}`}
                        active={i === activeIndex}
                        divider={i > 0}
                        onPick={() => selectStop(stop)}
                        onHover={() => setActiveIndex(i)}
                      />
                    ))}
                  </ul>
                </>
              ) : (
                statusText && (
                  <p
                    role={status === 'error' ? 'alert' : 'status'}
                    style={{
                      padding: '0.5rem 0.75rem',
                      fontSize: '0.9em',
                      color: status === 'error' ? '#b00020' : '#666',
                    }}
                  >
                    {statusText}
                  </p>
                )
              )}
            </div>
          )}
        </div>

        <button
          type="button"
          onClick={requestLocation}
          disabled={geoDisabled}
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: 6,
            padding: '0.5rem 0.75rem',
            border: '1px solid #d1d5db',
            borderRadius: 8,
            background: '#f9fafb',
            color: '#4b5563',
            fontSize: '0.85em',
            whiteSpace: 'nowrap',
            cursor: geoDisabled ? 'default' : 'pointer',
          }}
        >
          {geoStatus === 'granted' && (
            <span aria-hidden style={{ width: 8, height: 8, borderRadius: '50%', background: '#3AA65B' }} />
          )}
          {GEO_LABEL[geoStatus]}
        </button>
      </div>
    </div>
  );
}
