"""Minimal FastAPI backend — step 1 of the FastAPI/Next.js replacement for
the Streamlit app in phase3/app.py. Additive only; phase3/ is untouched.
"""
from __future__ import annotations

import json
import os

import boto3
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


# training_metadata.json lives under phase3/model/, which is .gitignore'd
# and doesn't exist on Render -- so instead of reading a local file, this
# fetches the object from S3 once at process startup and caches it in
# memory. /model/stats then serves the cache; it never touches S3 or disk
# per-request.
#
# Bucket env var and object key mirror phase3/ml/model_io.py's
# _s3_model_paths()['metadata'] (bucket from phase3/config.py's
# get_s3_bucket(), which reads AWS_S3_BUCKET; key is
# phase3/model/<MODEL_SUBDIR_OVERRIDE or 'latest'>/training_metadata.json)
# without importing that module directly -- model_io.py unconditionally
# imports xgboost/joblib/config-with-s3fs at load time and its only public
# loader (get_training_metadata) triggers a full model load, which this
# read-only endpoint doesn't need.
S3_BUCKET_ENV_VAR = 'AWS_S3_BUCKET'
MODEL_SUBDIR = os.environ.get('MODEL_SUBDIR_OVERRIDE', 'latest')
S3_METADATA_KEY = f'phase3/model/{MODEL_SUBDIR}/training_metadata.json'

_training_metadata: dict | None = None
_training_metadata_fetch_error: str | None = None


def _fetch_training_metadata_from_s3() -> dict:
    bucket = os.environ.get(S3_BUCKET_ENV_VAR)
    if not bucket:
        raise RuntimeError(f'{S3_BUCKET_ENV_VAR} environment variable is not set')
    # boto3.client('s3') with no explicit credentials uses boto3's standard
    # credential chain (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY /
    # AWS_DEFAULT_REGION env vars here) -- nothing hardcoded.
    s3 = boto3.client('s3')
    response = s3.get_object(Bucket=bucket, Key=S3_METADATA_KEY)
    return json.loads(response['Body'].read())


try:
    _training_metadata = _fetch_training_metadata_from_s3()
except Exception as e:
    # Never let a startup S3/network failure crash the whole app -- surface
    # it as a 503 from /model/stats instead (see below).
    _training_metadata_fetch_error = str(e)
    print(f'Startup: failed to fetch training_metadata.json from S3 ({e}) -- /model/stats will return 503.')


@app.get('/model/stats')
def model_stats() -> dict:
    if _training_metadata is None:
        raise HTTPException(
            status_code=503,
            detail=f'Training metadata unavailable: S3 fetch failed at startup ({_training_metadata_fetch_error})',
        )
    metadata = _training_metadata
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
