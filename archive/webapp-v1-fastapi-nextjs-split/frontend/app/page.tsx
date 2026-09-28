'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import StopPicker from '@/components/StopPicker';
import type { StopSearchResult } from '@/lib/types';

export default function Home() {
  const router = useRouter();
  const [fromStop, setFromStop] = useState<StopSearchResult | null>(null);
  const [toStop, setToStop] = useState<StopSearchResult | null>(null);
  const [validationMessage, setValidationMessage] = useState<string | null>(null);

  // A stale "same stop" warning shouldn't outlive the picks it was about.
  useEffect(() => {
    setValidationMessage(null);
  }, [fromStop, toStop]);

  function swapStops() {
    setFromStop(toStop);
    setToStop(fromStop);
  }

  function findRoute() {
    if (fromStop === null || toStop === null) return;
    if (fromStop.stop_id === toStop.stop_id) {
      setValidationMessage("From and To can't be the same stop.");
      return;
    }
    setValidationMessage(null);
    router.push(`/route?from=${fromStop.stop_id}&to=${toStop.stop_id}`);
  }

  const canSwap = fromStop !== null || toStop !== null;
  const canFind = fromStop !== null && toStop !== null;

  return (
    <main style={{ maxWidth: 720, margin: '0 auto', padding: '2rem 1rem', fontFamily: 'system-ui, sans-serif', lineHeight: 1.5, textAlign: 'center' }}>
      <h1>Transit AI</h1>
      <p style={{ color: '#666' }}>Search a route</p>

      <div style={{ maxWidth: 480, margin: '2rem auto 0', textAlign: 'left' }}>
        <StopPicker
          label="From"
          id="stop-picker-from"
          value={fromStop}
          onSelect={setFromStop}
          placeholder="Search for a stop"
        />

        <div style={{ display: 'flex', justifyContent: 'center', margin: '0.75rem 0' }}>
          <button
            type="button"
            aria-label="Swap"
            title="Swap From and To"
            onClick={swapStops}
            disabled={!canSwap}
            style={{
              width: 36,
              height: 36,
              border: '1px solid #d1d5db',
              borderRadius: '50%',
              background: '#f3f4f6',
              color: '#4b5563',
              fontSize: '1em',
              cursor: canSwap ? 'pointer' : 'default',
              opacity: canSwap ? 1 : 0.5,
            }}
          >
            ↑↓
          </button>
        </div>

        <StopPicker
          label="To"
          id="stop-picker-to"
          value={toStop}
          onSelect={setToStop}
          placeholder="Search for a stop"
        />

        <button
          type="button"
          onClick={findRoute}
          disabled={!canFind}
          style={{
            width: '100%',
            marginTop: '1.5rem',
            padding: '0.65rem 1rem',
            border: 'none',
            borderRadius: 8,
            background: canFind ? '#171717' : '#e5e7eb',
            color: canFind ? '#fff' : '#666',
            font: 'inherit',
            fontWeight: 600,
            cursor: canFind ? 'pointer' : 'default',
          }}
        >
          Find route
        </button>

        {/* Always rendered so the layout doesn't jump when a message appears. */}
        <p
          role="status"
          aria-hidden={validationMessage ? undefined : true}
          style={{ minHeight: '1.5em', marginTop: '0.5rem', fontSize: '0.9em', color: '#b00020' }}
        >
          {validationMessage}
        </p>
      </div>
    </main>
  );
}
