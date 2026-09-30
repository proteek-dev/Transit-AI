# webui

This folder will hold Transit AI's web UI: a single FastAPI process that serves both a JSON API and an HTML interface, replacing the Streamlit app. Passes 2–5 will build it out in order. It imports the routing, search and prediction code from `phase3/`, and does not touch `phase3/app.py` or `phase3/ui/`, which stay as they are.
