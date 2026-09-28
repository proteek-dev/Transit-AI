'use client';

import { Suspense, useEffect, useState } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import DataFreshness from '@/components/DataFreshness';
import RouteHero, { type RouteDataStatus } from '@/components/RouteHero';
import { ApiError, fetchModelStats, fetchRoutes, fetchStopsSearch } from '@/lib/api';
import { makeDemoRoute } from '@/lib/demoRoute';
import type { ModelStats, RouteOption } from '@/lib/types';

type LoadState<T> =
  | { status: 'loading' }
  | { status: 'error'; message: string }
  | { status: 'success'; data: T };

function errorMessage(err: unknown): string {
  return err instanceof ApiError ? err.message : String(err);
}

const mainStyle = { maxWidth: 720, margin: '0 auto', padding: '2rem 1rem', fontFamily: 'system-ui, sans-serif', lineHeight: 1.5 };

// useSearchParams() must sit under a Suspense boundary or `next build` fails
// prerendering this route.
export default function RoutePage() {
  return (
    <Suspense fallback={<RoutePlaceholder />}>
      <RouteFromParams />
    </Suspense>
  );
}

function RouteFromParams() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const from = searchParams.get('from')?.trim() ?? '';
  const to = searchParams.get('to')?.trim() ?? '';

  // Missing params or a hand-typed same-stop URL can't produce a route: send
  // the user back to search. replace(), not push(), so Back doesn't return
  // to the broken URL.
  const hasValidParams = Boolean(from && to);
  const shouldRedirect = !hasValidParams || from === to;
  useEffect(() => {
    if (shouldRedirect) router.replace('/');
  }, [shouldRedirect, router]);

  // Bare heading while the redirect lands, so there's no flash of nothing.
  if (shouldRedirect) return <RoutePlaceholder />;

  // Keyed so a new from/to pair remounts with fresh loading state.
  return <RouteView key={`${from}->${to}`} fromStopId={from} toStopId={to} />;
}

function RoutePlaceholder() {
  return (
    <main style={mainStyle}>
      <h1>Transit AI</h1>
    </main>
  );
}

function RouteView({ fromStopId, toStopId }: { fromStopId: string; toStopId: string }) {
  const [routesState, setRoutesState] = useState<LoadState<RouteOption[]>>({ status: 'loading' });
  const [statsState, setStatsState] = useState<LoadState<ModelStats>>({ status: 'loading' });
  const [resolvedNames, setResolvedNames] = useState<Record<string, string>>({});

  // Name resolution is currently a no-op: /stops/search fuzzy-matches on stop_name
  // only, so q=600016 never returns the stop whose stop_id is 600016. The effect
  // stays in place so it'll start populating names automatically once a
  // /stops/{stop_id} endpoint is added on the backend. Header falls back to
  // rendering the stop_id in <code> when resolvedNames[id] is undefined.
  useEffect(() => {
    const controller = new AbortController();
    const ids = [fromStopId, toStopId];
    Promise.allSettled(
      ids.map((id) => fetchStopsSearch(id, { limit: 5, signal: controller.signal })),
    ).then((outcomes) => {
      if (controller.signal.aborted) return;
      const names: Record<string, string> = {};
      outcomes.forEach((outcome, i) => {
        if (outcome.status !== 'fulfilled') return;
        // Exact stop_id match, not the first row -- a near-miss ID can rank above it.
        const match = outcome.value.find((stop) => stop.stop_id === ids[i]);
        if (match) names[ids[i]] = match.stop_name;
      });
      setResolvedNames(names);
    });
    return () => controller.abort();
  }, [fromStopId, toStopId]);

  useEffect(() => {
    let cancelled = false;
    fetchRoutes(fromStopId, toStopId)
      .then((data) => {
        if (!cancelled) setRoutesState({ status: 'success', data });
      })
      .catch((err) => {
        if (!cancelled) setRoutesState({ status: 'error', message: errorMessage(err) });
      });
    return () => {
      cancelled = true;
    };
  }, [fromStopId, toStopId]);

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

  const [demoRoute] = useState(() => makeDemoRoute());
  const displayRoute =
    routesState.status === 'success' && routesState.data.length > 0 ? routesState.data[0] : demoRoute;
  const dataStatus: RouteDataStatus =
    routesState.status === 'success'
      ? routesState.data.length > 0
        ? 'live'
        : 'empty'
      : routesState.status;
  const routesErrorMessage = routesState.status === 'error' ? routesState.message : undefined;

  return (
    <main style={mainStyle}>
      <h1>Transit AI</h1>
      <p style={{ color: '#555' }}>
        Route: {resolvedNames[fromStopId] ?? <code>{fromStopId}</code>} →{' '}
        {resolvedNames[toStopId] ?? <code>{toStopId}</code>}. Live predictions from the backend.
      </p>

      <section style={{ marginTop: '2rem' }}>
        <h2>Model stats</h2>
        <ModelStatsSection state={statsState} />
      </section>

      <section style={{ marginTop: '2rem' }}>
        <h2>Top route</h2>
        <RouteHero route={displayRoute} status={dataStatus} errorMessage={routesErrorMessage} />
      </section>

      {statsState.status === 'success' && statsState.data.data_snapshot && (
        <DataFreshness snapshot={statsState.data.data_snapshot} />
      )}
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
