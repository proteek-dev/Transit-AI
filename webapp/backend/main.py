"""Minimal FastAPI backend — step 1 of the FastAPI/Next.js replacement for
the Streamlit app in phase3/app.py. Additive only; phase3/ is untouched.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title='Transit-AI API')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],  # open for now; restrict once the frontend domain exists
    allow_methods=['*'],
    allow_headers=['*'],
)


@app.get('/health')
def health() -> dict:
    """No file I/O, no S3 -- just a cheap target for the external keep-alive
    cron so pinging it doesn't cost anything on every hit.
    """
    return {'status': 'ok'}


# Mirrors phase3/ml/model_io.py's MODEL_DIR/TRAINING_METADATA_PATH resolution
# (MODEL_SUBDIR_OVERRIDE, phase3/model/<subdir>/training_metadata.json)
# rather than importing that module: model_io.py unconditionally imports
# xgboost/joblib/config-with-s3fs at module load, and its only public loader
# (get_training_metadata) triggers a full model load -- S3 fetch attempt,
# and training a new model from scratch if none is found -- which is far
# more than this read-only stats endpoint needs.
PHASE3_DIR = Path(__file__).resolve().parents[2] / 'phase3'
MODEL_SUBDIR = os.environ.get('MODEL_SUBDIR_OVERRIDE', 'latest')
TRAINING_METADATA_PATH = PHASE3_DIR / 'model' / MODEL_SUBDIR / 'training_metadata.json'


def _load_training_metadata() -> dict:
    if not TRAINING_METADATA_PATH.exists():
        raise HTTPException(status_code=503, detail=f'{TRAINING_METADATA_PATH} not found')
    with open(TRAINING_METADATA_PATH) as f:
        return json.load(f)


@app.get('/model/stats')
def model_stats() -> dict:
    metadata = _load_training_metadata()
    return {
        'training_rows': sum(metadata['coverage_counts'].values()),
        'days_archived': metadata['data_window_days'],
        'test_mae': metadata['test_mae'],
        'naive_mae': metadata['naive_mae'],
        'pct_improvement_over_naive': metadata['pct_improvement_over_naive'],
        'model_type': 'XGBoost (v0)',
        'training_window': {
            'start': metadata['data_window_start'],
            'end': metadata['data_window_end'],
        },
    }
