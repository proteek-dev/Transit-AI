# Transit AI — Frontend

Next.js (App Router, TypeScript) frontend for SEQ Transit AI's hero prediction
screen. It fetches live route predictions and model stats from the FastAPI
backend in [`webapp/backend/`](../backend) and renders the top-ranked journey
as an animated hero visual.

## Run locally

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## Backend

By default this app talks to the live Render-hosted backend. Set
`NEXT_PUBLIC_API_BASE_URL` (see `.env` / your deployment platform's env vars)
to point at a different backend, e.g. one running locally.
