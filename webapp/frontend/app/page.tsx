'use client';

import { useEffect, useState } from 'react';
import RouteHero from '@/components/RouteHero';
import { ApiError, fetchModelStats, fetchRoutes } from '@/lib/api';
import { FROM_STOP_ID, TO_STOP_ID } from '@/lib/corridor';
import type { ModelStats, RouteOption } from '@/lib/types';

type LoadState<T> =
  | { status: 'loading' }
  | { status: 'error'; message: string }
  | { status: 'success'; data: T };

function errorMessage(err: unknown): string {
  return err instanceof ApiError ? err.message : String(err);
}

export default function Home() {
  const [routesState, setRoutesState] = useState<LoadState<RouteOption[]>>({ status: 'loading' });
  const [statsState, setStatsState] = useState<LoadState<ModelStats>>({ status: 'loading' });

  useEffect(() => {
    let cancelled = false;
    fetchRoutes(FROM_STOP_ID, TO_STOP_ID)
      .then((data) => {
        if (!cancelled) setRoutesState({ status: 'success', data });
      })
      .catch((err) => {
        if (!cancelled) setRoutesState({ status: 'error', message: errorMessage(err) });
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    fetchModelStats()
      .then((data) => {
        if (!cancelled) setStatsState({ status: 'success', data });
      })
      .catch((err) => {
        if (!cancelled) setStatsState({ status: 'error', message: errorMessage(err) });
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <main style={{ maxWidth: 720, margin: '0 auto', padding: '2rem 1rem', fontFamily: 'system-ui, sans-serif', lineHeight: 1.5 }}>
      <h1>Transit AI</h1>
      <p style={{ color: '#555' }}>
        Corridor: <code>{FROM_STOP_ID}</code> → <code>{TO_STOP_ID}</code>. Live predictions from the backend.
      </p>

      <section style={{ marginTop: '2rem' }}>
        <h2>Model stats</h2>
        <ModelStatsSection state={statsState} />
      </section>

      <section style={{ marginTop: '2rem' }}>
        <h2>Top route</h2>
        <RouteHeroSection state={routesState} />
      </section>
    </main>
  );
}

function ModelStatsSection({ state }: { state: LoadState<ModelStats> }) {
  if (state.status === 'loading') {
    return <p>Loading model stats...</p>;
  }
  if (state.status === 'error') {
    return <p style={{ color: '#b00020' }}>Error loading model stats: {state.message}</p>;
  }
  const s = state.data;
  return (
    <ul>
      <li>Model: {s.model_type}</li>
      <li>Training rows: {s.training_rows.toLocaleString()}</li>
      <li>Days archived: {s.days_archived}</li>
      <li>Test MAE: {s.test_mae.toFixed(3)} min</li>
      <li>Naive MAE: {s.naive_mae.toFixed(3)} min</li>
      <li>Improvement over naive: {s.pct_improvement_over_naive.toFixed(1)}%</li>
      <li>
        Training window: {s.training_window.start} to {s.training_window.end}
      </li>
    </ul>
  );
}

function RouteHeroSection({ state }: { state: LoadState<RouteOption[]> }) {
  if (state.status === 'loading') {
    return <p>Loading live predictions... this can take up to a minute on first load.</p>;
  }
  if (state.status === 'error') {
    return <p style={{ color: '#b00020' }}>Error loading routes: {state.message}</p>;
  }
  if (state.data.length === 0) {
    return (
      <p>
        No routes found for this stop pair right now (BFS returned zero results — see lib/corridor.ts for
        platform-swap notes).
      </p>
    );
  }
  return <RouteHero route={state.data[0]} />;
}
