"""
Unit tests for detection/classifier.py and network/feature_extractor.py

Run with:
    pytest tests/ -v
"""

import os
import sys
import tempfile
import types

import joblib
import numpy as np
import pytest

# Make project root importable when running from any directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from network.feature_extractor import build_feature_vector, ConnectionRateTracker


# ── feature_extractor tests ───────────────────────────────────────────────────

class TestBuildFeatureVector:
    def test_returns_six_elements(self):
        vec = build_feature_vector(0.5, 200, 0, 1)
        assert len(vec) == 6

    def test_all_floats(self):
        vec = build_feature_vector(0.5, 200, 0, 1)
        assert all(isinstance(v, float) for v in vec)

    def test_is_empty_flag_set_when_src_bytes_zero(self):
        vec = build_feature_vector(0.5, 0, 0, 0)
        is_empty_flag = vec[5]
        assert is_empty_flag == 1.0

    def test_is_empty_flag_clear_when_src_bytes_nonzero(self):
        vec = build_feature_vector(0.5, 300, 0, 0)
        is_empty_flag = vec[5]
        assert is_empty_flag == 0.0

    def test_byte_rate_computed_correctly(self):
        duration  = 2.0
        src_bytes = 100
        dst_bytes = 100
        vec       = build_feature_vector(duration, src_bytes, dst_bytes, 0)
        byte_rate = vec[4]
        expected  = (src_bytes + dst_bytes) / (duration + 1e-6)
        assert abs(byte_rate - expected) < 1e-3

    def test_zero_duration_no_division_error(self):
        """Duration of 0 must not raise ZeroDivisionError."""
        vec = build_feature_vector(0.0, 50, 0, 0)
        assert len(vec) == 6

    def test_feature_order(self):
        """Verify positional meaning of each element."""
        duration, src_bytes, dst_bytes, count = 1.0, 200, 50, 5
        vec = build_feature_vector(duration, src_bytes, dst_bytes, count)
        assert vec[0] == duration
        assert vec[1] == float(src_bytes)
        assert vec[2] == float(dst_bytes)
        assert vec[3] == float(count)
        # vec[4] = byte_rate, vec[5] = is_empty_flag


class TestConnectionRateTracker:
    def test_first_connection_returns_one(self):
        tracker = ConnectionRateTracker(window=2.0)
        count   = tracker.record("192.168.1.1")
        assert count == 1

    def test_multiple_connections_accumulate(self):
        tracker = ConnectionRateTracker(window=2.0)
        for _ in range(5):
            count = tracker.record("10.0.0.1")
        assert count == 5

    def test_different_ips_tracked_independently(self):
        tracker = ConnectionRateTracker(window=2.0)
        tracker.record("1.1.1.1")
        tracker.record("1.1.1.1")
        count_other = tracker.record("2.2.2.2")
        assert count_other == 1

    def test_window_expiry(self):
        """Connections older than `window` seconds must not be counted."""
        import time
        tracker = ConnectionRateTracker(window=0.05)   # 50 ms window
        tracker.record("5.5.5.5")
        tracker.record("5.5.5.5")
        time.sleep(0.1)                                 # let window expire
        count = tracker.record("5.5.5.5")              # only this new one
        assert count == 1


# ── classifier smoke-test with a mock model ───────────────────────────────────

class TestTrafficClassifierSmokeTest:
    """
    Builds a minimal fake model/scaler/feature_names in a temp directory and
    verifies that TrafficClassifier loads and predicts without errors.
    """

    @pytest.fixture()
    def fake_model_dir(self, tmp_path):
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression

        # Six-feature toy model
        X = np.array([
            [0.01, 300, 0, 1, 5000, 0],   # benign
            [0.50,   0, 0, 0,    0, 1],   # malicious
        ], dtype=float)
        y = [0, 1]

        scaler = StandardScaler()
        X_s    = scaler.fit_transform(X)
        model  = LogisticRegression().fit(X_s, y)

        save = tmp_path / "saved_models"
        save.mkdir()
        joblib.dump(model,  save / "lgbm_model.pkl")
        joblib.dump(scaler, save / "scaler.pkl")
        joblib.dump(
            ["duration", "src_bytes", "dst_bytes", "count", "byte_rate", "is_empty_flag"],
            save / "feature_names.pkl",
        )
        return tmp_path

    def test_predict_benign(self, fake_model_dir, monkeypatch):
        import detection.classifier as cls_module
        # Redirect the path search to our tmp dir
        monkeypatch.setattr(
            cls_module, "__file__",
            str(fake_model_dir / "detection" / "classifier.py"),
        )
        # Manually instantiate using the tmp path
        from detection.classifier import TrafficClassifier
        import unittest.mock as mock

        clf = TrafficClassifier.__new__(TrafficClassifier)
        saved = fake_model_dir / "saved_models"
        clf.model         = joblib.load(saved / "lgbm_model.pkl")
        clf.scaler        = joblib.load(saved / "scaler.pkl")
        clf.feature_names = joblib.load(saved / "feature_names.pkl")

        pred, prob = clf.predict([0.01, 300.0, 0.0, 1.0, 5000.0, 0.0])
        assert pred in (0, 1)
        assert 0.0 <= prob <= 1.0

    def test_feature_length_validation(self, fake_model_dir):
        from detection.classifier import TrafficClassifier
        import joblib

        saved = fake_model_dir / "saved_models"
        clf   = TrafficClassifier.__new__(TrafficClassifier)
        clf.model         = joblib.load(saved / "lgbm_model.pkl")
        clf.scaler        = joblib.load(saved / "scaler.pkl")
        clf.feature_names = joblib.load(saved / "feature_names.pkl")

        with pytest.raises(ValueError, match="Expected 6 features"):
            clf.predict([1.0, 2.0, 3.0])   # only 3 features — should raise