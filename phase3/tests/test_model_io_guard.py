"""model_io's S3-failure fallback must not publish a local model to S3 or
train in-process unless explicitly opted in (MODEL_ALLOW_S3_PUBLISH /
MODEL_ALLOW_TRAIN). S3, xgboost loading and training are all stubbed.

Run from the repo root:
    python -m unittest phase3/tests/test_model_io_guard.py
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml import model_io  # noqa: E402


class ModelLoadGuardTest(unittest.TestCase):
    def setUp(self):
        tmp = Path(tempfile.mkdtemp())
        self.paths = {'MODEL_PATH': tmp / 'xgb_v0.json', 'CATEGORIES_PATH': tmp / 'categories.joblib',
                      'TRAINING_METADATA_PATH': tmp / 'training_metadata.json'}
        patches = [mock.patch.object(model_io, name, path) for name, path in self.paths.items()]
        patches += [
            mock.patch.object(model_io, '_download_model_from_s3', return_value=False),  # not in S3
            mock.patch.object(model_io, '_upload_model_to_s3'),
            mock.patch.object(model_io, '_s3_model_prefix', return_value='bucket/phase3/model/latest'),
            mock.patch.object(model_io.xgb, 'XGBRegressor'),
            mock.patch.object(model_io.joblib, 'load', return_value={}),
            mock.patch.dict(model_io._cache, clear=True),
            mock.patch.dict(os.environ),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        os.environ.pop(model_io.ALLOW_TRAIN_ENV, None)
        os.environ.pop(model_io.ALLOW_S3_PUBLISH_ENV, None)

    def _write_local_model(self):
        for path in self.paths.values():
            path.write_text('{}')

    def test_local_model_not_uploaded_without_opt_in(self):
        self._write_local_model()
        model_io._load_model_impl()
        model_io._upload_model_to_s3.assert_not_called()

    def test_local_model_uploaded_with_opt_in(self):
        self._write_local_model()
        os.environ[model_io.ALLOW_S3_PUBLISH_ENV] = '1'
        model_io._load_model_impl()
        model_io._upload_model_to_s3.assert_called_once()

    def test_no_model_anywhere_refuses_to_train_without_opt_in(self):
        fake_training = mock.MagicMock()
        with mock.patch.dict(sys.modules, {'ml.training': fake_training}):
            with self.assertRaisesRegex(RuntimeError, 'refusing to train'):
                model_io._load_model_impl()
        fake_training._train_and_save_model.assert_not_called()

    def test_s3_error_is_reported_as_unreachable_not_missing(self):
        s3_error = PermissionError('AccessDenied')
        with mock.patch.object(model_io, '_download_model_from_s3', side_effect=s3_error):
            with self.assertRaisesRegex(RuntimeError, 'S3 unreachable') as ctx:
                model_io._load_model_impl()
        self.assertIs(ctx.exception.__cause__, s3_error)

    def test_local_fallback_after_s3_error_is_logged(self):
        self._write_local_model()
        with mock.patch.object(model_io, '_download_model_from_s3', side_effect=OSError('timeout')), \
                self.assertLogs(model_io.logger, 'WARNING') as logs:
            model_io._load_model_impl()
        self.assertTrue(any('local model' in line for line in logs.output))

    def test_no_model_anywhere_trains_with_opt_in(self):
        os.environ[model_io.ALLOW_TRAIN_ENV] = '1'
        fake_training = mock.MagicMock()
        fake_training._train_and_save_model.side_effect = self._write_local_model
        with mock.patch.dict(sys.modules, {'ml.training': fake_training}):
            model_io._load_model_impl()
        fake_training._train_and_save_model.assert_called_once()


if __name__ == '__main__':
    unittest.main()
