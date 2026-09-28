# SEQ Transit AI — Session 38 Summary

**Date:** Monday 2026-09-28 (AEST)
**Branch:** `work_ai`

## Outcome

Session 38 carried out the first four passes of the Streamlit-replacement plan agreed in Session 37. The FastAPI + Next.js split under `webapp/` was moved intact into `archive/webapp-v1-fastapi-nextjs-split/` and committed (Pass 1). A single-process FastAPI web UI was built in `webui/` over Passes 2–4 and committed together with the Pass 4.1 fix. During interactive testing, repeated S3 model downloads showed that Streamlit's `st.cache_resource` (used by `phase3/ml/model_io.load_model`) was evicting the cached model after minutes of idle time under uvicorn's bare mode. Pass 4.1 fixed this in `webui/app.py` without touching `phase3/`, and 15 minutes of real idle time plus an interactive browser check confirmed exactly one S3 download for the whole process lifetime.

## The five-pass plan and where it stands

The Session 37 pivot replaced the split architecture with one FastAPI process that imports `phase3/` directly and serves both the JSON API and the HTML page. The plan:

1. **Pass 1 — Archive and scaffold.** Move `webapp/backend` and `webapp/frontend` into the archive with all uncommitted state, delete the debug scripts, scaffold `webui/`. *Done.*
2. **Pass 2 — FastAPI skeleton.** Lifespan warmup of GTFS and the model, S3 metadata fetches, a freshness footer. *Done.*
3. **Pass 3 — Endpoints and plain UI.** Port the archived API with identical response shapes; a minimal HTML page. *Done.*
4. **Pass 4 — Real UI.** Styling, keyboard navigation, recents, confidence pills, mobile layout. *Done, plus Pass 4.1 (model-cache fix).*
5. **Pass 5 — Map and polish.** Leaflet map, stop markers, leg polylines, mode icons, final polish. *Not started.*

Session 38 ended with Pass 4.1 GREEN and committed.

## What landed, per pass

**Pass 1.** `git mv` of `webapp/backend` and `webapp/frontend` into `archive/webapp-v1-fastapi-nextjs-split/`, preserving the uncommitted routing group-expansion fix in `main.py`, its untracked tests, and the untracked frontend files. Added an archive README. The two `_debug_route_*.py` scripts were untracked and were deleted locally. `webui/` was scaffolded with a placeholder README.

**Pass 2.** A minimal FastAPI app with an `asynccontextmanager` lifespan that loads GTFS through `load_gtfs_data_optimized`, loads the model through `ml.model_io.load_model`, and fetches `training_metadata.json` and the feature manifest `_latest.json` from S3. `/` rendered three freshness lines: schedule snapshot, model trained (`trained_at`), and features through (the manifest's latest `source_date_range`, falling back to `data_window_end`). A temporary `/_diag/model_cache` endpoint showed which model object was cached. `requirements.txt` gained pins for fastapi 0.128.8, uvicorn[standard] 0.39.0 and jinja2 3.1.4.

**Pass 3.** `/_diag` removed. Ported `/health`, `/model/stats`, `/stops/search` and `/routes` from the archived backend with the response shapes in the archived `types.ts` (`StopSearchResult`, `RouteOption`, `RouteLeg`), including station-group expansion of `from_stop_id`/`to_stop_id` and a 404 on unknown stop IDs. A plain HTML page drove the endpoints.

**Pass 4.** CSS and JS moved into `webui/static/`. Calm styling on a 4px spacing scale, dark mode via `prefers-color-scheme`, a WAI-ARIA combobox for both stop pickers (arrows, Enter, Escape, Tab), localStorage recents under `webui.recentStops.v1` capped at 5, a near-match label when no result name contains the query, High/Medium/Low confidence pill colours, stale errors cleared on interaction, a mobile layout at ≤480px, and 44px tap targets. No frameworks, no CDN, textContent-only DOM writes.

**Pass 4.1.** `load_model` is bound by name in three modules: `ml.model_io`, `ml.inference` and `prediction`. After the first load, the lifespan replaces all three bindings with a closure that returns `app.state.model`. Each `/routes` call logs whether `load_model()` returns the cached instance and whether the `inference` binding was patched.

## Verification evidence

The Pass 4.1 boot test ran uvicorn on port 8080 with hard aborts on any retrain or upload log line; none fired. Startup logged the model object at id A = `5616044976` and the original `load_model` callable at id B = `5224649632`. Five `/routes` calls — a baseline, one after 5 minutes idle, one after 15 minutes total idle, and two interactive browser submissions — all logged `model.state id=5616044976 load_model()==app.state.model=True inference_patched=True`. "Downloaded model files from s3://" appeared exactly once in the log for the whole session.

## Corrections

The Pass 2.2 and Pass 3 boot tests reported GREEN with a download count of 1, but their checks (`/_diag/model_cache` and back-to-back `/routes` calls) only proved that the cache held across calls close together in time, not across idle time. Proteek then saw repeated S3 downloads between `/routes` calls minutes apart during interactive use. Pass 4.1 diagnosed and fixed this and added real 5- and 10-minute idle windows to the test. Learning: a caching check has to include real wall-clock idle time, not just repeated calls.

Two smaller points: Proteek's patch snippet imported `phase3.ml.model_io`, which would have created a second module object with its own cache, and patching only `model_io` would have missed the by-name binding in `ml.inference`. Both were flagged and the patch covers all three bindings under the short import names. The pre-commit secret grep matched a code comment naming `AWS_SECRET_ACCESS_KEY` (no value); it was reviewed and cleared before committing.

## Commits this session

- `d00daa1` — chore(archive): retire webapp/backend + webapp/frontend, scaffold webui/ target
- `a7c67ed` — feat(webui): single-process FastAPI web UI with cached model + phase3 GTFS
- This summary, committed as the third commit.

## Open items carried into Session 39

- **Pass 5:** Leaflet map layer (CDN-loaded), stop markers, route polylines from GTFS shapes if `phase3/` exposes them, mode SVG icons, final polish.
- Notebook 05 `LOOKBACK_DAYS=0` is still committed (unchanged this session).
- `phase3/ml/training.py` is mid-work, uncommitted (unchanged).
- `pipeline/02_train_model_capped.sh` is untracked; `pipeline/02_train_model.sh` and `.dockerignore` carry uncommitted edits (unchanged).
- `SEQ_TransitAI_Branch_Audit_2026-09-28.md` is untracked at the repo root and belongs in project files.
- EC2 rebuild-and-train instance setup, the model drift brainstorm and the other Session 37 open items carry forward.
- The Sept 27 fresh model regression (test_mae 2.093 against the Aug 14 baseline of 1.957) is unchanged.
- `AWS_DEFAULT_REGION` is still missing from `.env.example`.
- The IAM user `transit-ai-webapp-render` still needs renaming.
- Python 3.9 is past boto3's support cutoff (29 April 2026); boto3 now logs a deprecation warning on startup.

## Ways of working

One prompt at a time; show the diff and stop. No Co-Authored-By trailer or other attribution on commits. No browser automation from Claude Code — Proteek drives browser checks through Claude in Chrome. `phase3/` is read-only unless Proteek re-confirms a narrow exception. Session summaries go in `SEQ_TransitAI_Session_Summary_[N].md`. Use the roadmap-update, product-brainstorming, code-review, documentation and architecture skills whenever relevant.

## Starter prompt for Session 39

> Read this project's memory files, `SEQ_TransitAI_Session_Summary_38.md`, and any relevant earlier summaries. Session 38 ended with Pass 4.1 GREEN and committed on `work_ai`; Pass 5 is next. Confirm the goal before starting: a Leaflet map layer (CDN-loaded — the one exception to the no-CDN rule, per the Session 37 pivot decision), stop markers on both endpoints, route polylines for each leg using `phase3/` shapes if available, small SVG icons for bus, tram and train, and a final polish sweep — no server-side changes. If `phase3/` doesn't expose leg shapes, Pass 5 stops there and Pass 5.1 becomes a scoped read-only `phase3/` addition, subject to Proteek's re-confirmation of the narrow exception.
