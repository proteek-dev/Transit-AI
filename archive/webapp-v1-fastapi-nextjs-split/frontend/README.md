# Transit AI — Frontend

Next.js (App Router, TypeScript) frontend for SEQ Transit AI. Users search for
a From and To stop, then see live route predictions and model stats from the
FastAPI backend in [`webapp/backend/`](../backend), with the top-ranked journey
rendered as an animated hero visual.

## Pages

- **`/`** — Search page (`app/page.tsx`). Two stop pickers (From, To), a swap
  button, and a "Find route" button. Reads no URL params. On submit, navigates
  to `/route?from=<stop_id>&to=<stop_id>` via client-side `router.push()`.
- **`/route?from=<stop_id>&to=<stop_id>`** — Route detail (`app/route/page.tsx`).
  Renders the `RouteHero` animation for the picked stops, falling back to an
  example journey when `/routes` is loading, errors, or returns no routes.
  Missing, empty, or same-stop params redirect to `/` via `router.replace()`.
  The header shows stop names when resolvable, and the `stop_id` in `<code>` as
  a fallback.

## Data layer

- **`lib/api.ts`** — `getJson()` core plus wrappers `fetchRoutes()`,
  `fetchStopsSearch()`, and `fetchModelStats()`. Errors surface as `ApiError`.
  Base URL comes from `NEXT_PUBLIC_API_BASE_URL`.
- **`lib/types.ts`** — response types, including `StopSearchResult`. Its
  `distance_km?` field is optional: the backend only sets it on the fuzzy-match
  path when both `lat` and `lon` were supplied.
- **`lib/recentStops.ts`** — localStorage-backed recent picks under
  `transit-ai:recent-stops:v1`, capped at 8, SSR-safe. Storage failures degrade
  silently to no recents.

## Run locally

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## Environment variables

| Variable | Default | Example |
|---|---|---|
| `NEXT_PUBLIC_API_BASE_URL` | `http://localhost:8000` | `https://d17e9wvem3lbaq.cloudfront.net` |

With no value set, the app talks to a local backend on port 8000.

- **`.env.local`** (git-ignored): for local dev against a local backend, set
  `NEXT_PUBLIC_API_BASE_URL=http://localhost:8000`.
- **`.env.production`** (committed): holds the CloudFront URL and is picked up
  automatically by `npm run build`. Note that `.env.local` takes precedence
  over it, so a build on a machine with `.env.local` bakes in localhost. Build
  for deploy from a clean checkout (or CI/Cloud Run).
- **`.env.example`** (committed): reference only. It shows which variables
  exist and what production values look like.

## Deployment

Migrating to Google Cloud Run — instructions pending Dockerfile.
