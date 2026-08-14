"""Model artifact resolution and caching for the Phase 3 prediction service.

Owns MODEL_SUBDIR/MODEL_DIR/latest-vs-runs path resolution, S3 download/
upload of the three saved artifact files (xgb_v0.json, categories.joblib,
training_metadata.json), and the in-process cache so each artifact is loaded
at most once per process. Falls back to training.py's _train_and_save_model()
(imported lazily, inside _load_model_impl(), to avoid a circular import --
training.py itself imports MODEL_DIR/MODEL_PATH/etc. from this module) when
no saved model is found anywhere.

FEATURE_COLS/CATEGORICAL_COLS also live here: the one piece of schema both
training.py (fitting) and inference.py (build_features) must agree on
exactly, so both import it from this shared module rather than each
defining -- and risking drifting -- their own copy.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import xgboost as xgb

import config

# The live app always reads/writes phase3/model/latest/. pipeline/02_train_model.sh
# points a training run at an isolated phase3/model/_staging_{date}_{time}/
# subdir instead via MODEL_SUBDIR_OVERRIDE, so a live model.predict() call can
# never observe a partially-written retrain -- it promotes staging to latest/
# only after all three artifact files are confirmed present.
MODEL_SUBDIR = os.environ.get('MODEL_SUBDIR_OVERRIDE', 'latest')
# .parent.parent, not .parent: this file lives in phase3/ml/, one level
# deeper than it did before the reorg, but MODEL_DIR must still resolve to
# phase3/model/ -- the location pipeline/02_train_model.sh promotes trained
# artifacts into and the only phase3/model/ path .gitignore excludes.
MODEL_DIR = Path(__file__).parent.parent / 'model' / MODEL_SUBDIR
MODEL_PATH = MODEL_DIR / 'xgb_v0.json'
CATEGORIES_PATH = MODEL_DIR / 'categories.joblib'
TRAINING_METADATA_PATH = MODEL_DIR / 'training_metadata.json'

# Must match notebook 07 Cell 5 exactly (feature_cols after EXCLUDE_COLS).
FEATURE_COLS = ['route_id', 'stop_id', 'mode', 'stop_sequence',
                'hour_of_day', 'day_of_week', 'is_weekend', 'is_peak']
CATEGORICAL_COLS = ['route_id', 'stop_id', 'mode', 'day_of_week']

_cache: dict = {}


def _get_env():
    return config.get_s3_bucket(), config.get_s3_filesystem()


def _s3_model_prefix() -> str:
    return f'{config.get_s3_bucket()}/phase3/model/{MODEL_SUBDIR}'


def _s3_model_paths() -> dict:
    prefix = _s3_model_prefix()
    return {
        'model': f's3://{prefix}/xgb_v0.json',
        'categories': f's3://{prefix}/categories.joblib',
        'metadata': f's3://{prefix}/training_metadata.json',
    }


def _download_model_from_s3() -> bool:
    """Download the saved model files from S3 into MODEL_DIR. Returns True on success."""
    try:
        fs = config.get_s3_filesystem()
        s3_paths = _s3_model_paths()
        if not all(fs.exists(p) for p in s3_paths.values()):
            return False

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        fs.get(s3_paths['model'], str(MODEL_PATH))
        fs.get(s3_paths['categories'], str(CATEGORIES_PATH))
        fs.get(s3_paths['metadata'], str(TRAINING_METADATA_PATH))
        print(f"Downloaded model files from s3://{_s3_model_prefix()}/")
        return True
    except Exception as e:
        print(f'Could not load model from S3 ({e}) — will try local files instead.')
        return False


def _upload_model_to_s3() -> None:
    fs = config.get_s3_filesystem()
    s3_paths = _s3_model_paths()
    fs.put(str(MODEL_PATH), s3_paths['model'])
    fs.put(str(CATEGORIES_PATH), s3_paths['categories'])
    fs.put(str(TRAINING_METADATA_PATH), s3_paths['metadata'])
    print(f"Uploaded model files to s3://{_s3_model_prefix()}/")


def _load_model_impl() -> xgb.XGBRegressor:
    """S3 first (source of truth for Streamlit Cloud, which has no local model
    files), then local files (local dev fallback), training only if neither
    is available.
    """
    have_model = _download_model_from_s3()
    if not have_model:
        have_model = MODEL_PATH.exists() and CATEGORIES_PATH.exists() and TRAINING_METADATA_PATH.exists()
        if have_model:
            # Locally trained but not yet mirrored to S3 (e.g. first Cloud deploy prep) — upload now.
            _upload_model_to_s3()
    if not have_model:
        from ml.training import _train_and_save_model
        _train_and_save_model()

    model = xgb.XGBRegressor(enable_categorical=True, tree_method='hist')
    model.load_model(str(MODEL_PATH))
    categories = joblib.load(CATEGORIES_PATH)
    with open(TRAINING_METADATA_PATH) as f:
        training_metadata = json.load(f)

    _cache['categories'] = categories
    _cache['training_metadata'] = training_metadata
    return model


try:
    import streamlit as st
    load_model = st.cache_resource(
        show_spinner='Loading prediction model (training on first run can take a few minutes)...'
    )(_load_model_impl)
except ImportError:
    def load_model() -> xgb.XGBRegressor:
        """Load the trained v0 model (training + saving first if needed). Cached in memory."""
        if 'model' not in _cache:
            _cache['model'] = _load_model_impl()
        return _cache['model']


def _get_training_metadata() -> dict:
    if 'training_metadata' not in _cache:
        load_model()
    return _cache['training_metadata']


def get_training_metadata() -> dict:
    """Public accessor for training_metadata.json's contents (trained_at,
    data_window_start/end/days, MAE baselines, coverage_counts, ...) -- so
    callers like app.py don't need to reach into the private cache directly.
    """
    return _get_training_metadata()


def _get_categories() -> dict:
    if 'categories' not in _cache:
        load_model()
    return _cache['categories']
