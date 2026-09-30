# webapp v1 — FastAPI + Next.js split (archived)

## What this is

The first attempt at replacing the Streamlit app (`phase3/app.py`): a FastAPI
backend (`backend/`) and a separate Next.js frontend (`frontend/`), built across
Sessions 24–36.

## Why it's archived

Replaced by `webui/`, a single-process FastAPI app that serves its own HTML page.
Running two apps split across a network boundary created a maintenance seam that
kept producing drift bugs between the API and the UI.

## State at archive time

- `backend/main.py` contains an uncommitted routing fix (station-group expansion
  of `from_stop_id` / `to_stop_id`, plus a 404 for unknown stop IDs) that was
  never merged. See this project's chat history for context.
- The frontend was archived mid-refactor, with a search page and a `/route`
  page, partly staged.
- Moved here with all uncommitted and untracked changes as they were.

## Reactivation

This code imports from `phase3/` and would still work if reactivated. Nothing
in the live codebase or CI references it.

Archived: 2026-09-28 (Session 37).
