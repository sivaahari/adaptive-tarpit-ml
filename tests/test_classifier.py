"""
Unit tests for detection/classifier.py and network/feature_extractor.py

Run with:
    PYTHONPATH=. venv/bin/pytest tests/ -v
"""

import os
import sys
import time

import joblib
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from network.feature_extractor import build_feature_vector, ConnectionRateTracker


# ── build_feature_vector tests ────────────────────────────────────────────────

class TestBuildFeatureVector:
    def test_returns_five_elements(self):
        vec = build_feature_vector(0.5, 200, 0, 1)
        assert len(vec) == 5

    def test_all_floats(self):
        vec = build_feature_vector(0.5, 200, 0, 1)
        assert all(isinstance(v, float) for v in vec)

    def test_is_empty_flag_set_when_src_bytes_zero(self):
        vec = build_feature_vector(0.5, 0, 0, 0)
        is_empty_flag = vec[4]
        assert is_empty_flag == 1.0

    def test_is_empty_flag_clear_when_src_bytes_nonzero(self):
        vec = build_feature_vector(0.5, 300, 0, 0)
        is_empty_flag = vec[4]
        assert is_empty_flag == 0.0

    def test_byte_rate_computed_from_src_bytes_only(self):
        # byte_rate = src_bytes / (duration + ε) — dst_bytes excluded
        duration  = 2.0
        src_bytes = 200
        vec       = build_feature_vector(duration, src_bytes, dst_bytes=999, count=0)
        byte_rate = vec[3]
        expected  = src_bytes / (duration + 1e-6)
        assert abs(byte_rate - expected) < 1e-3

    def test_dst_bytes_excluded_from_output(self):
        """dst_bytes is accepted for API clarity but must NOT appear in the vector."""
        vec_no_dst  = build_feature_vector(1.0, 100, dst_bytes=0,      count=0)
        vec_big_dst = build_feature_vector(1.0, 100, dst_bytes=999999, count=0)
        assert vec_no_dst == vec_big_dst

    def test_zero_duration_no_division_error(self):
        vec = build_feature_vector(0.0, 50, 0, 0)
        assert len(vec) == 5

    def test_feature_order(self):
        duration, src_bytes, count = 1.0, 200, 5
        vec = build_feature_vector(duration, src_bytes, dst_bytes=0, count=count)
        assert vec[0] == duration
        assert vec[1] == float(src_bytes)
        assert vec[2] == float(count)
        # vec[3] = byte_rate, vec[4] = is_empty_flag

    def test_count_appears_in_feature_vector(self):
        vec_low  = build_feature_vector(1.0, 100, 0, count=1)
        vec_high = build_feature_vector(1.0, 100, 0, count=50)
        assert vec_high[2] == 50.0
        assert vec_low[2]  == 1.0


# ── ConnectionRateTracker tests ───────────────────────────────────────────────

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
        tracker = ConnectionRateTracker(window=0.05)
        tracker.record("5.5.5.5")
        tracker.record("5.5.5.5")
        time.sleep(0.1)
        count = tracker.record("5.5.5.5")
        assert count == 1

    def test_high_count_signals_scanner(self):
        tracker = ConnectionRateTracker(window=5.0)
        for _ in range(30):
            count = tracker.record("scanner.ip")
        assert count == 30


# ── TrafficClassifier smoke-test with a mock model ────────────────────────────

class TestTrafficClassifierSmokeTest:
    @pytest.fixture()
    def fake_model_dir(self, tmp_path):
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression

        # 5-feature toy model: [duration, src_bytes, count, byte_rate, is_empty]
        X = np.array([
            [0.01, 300, 1,  5000, 0],
            [0.50,   0, 20,    0, 1],
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
            ["duration", "src_bytes", "count", "byte_rate", "is_empty_flag"],
            save / "feature_names.pkl",
        )
        return tmp_path

    def _load_clf(self, fake_model_dir):
        from detection.classifier import TrafficClassifier
        saved = fake_model_dir / "saved_models"
        clf   = TrafficClassifier.__new__(TrafficClassifier)
        clf.model         = joblib.load(saved / "lgbm_model.pkl")
        clf.scaler        = joblib.load(saved / "scaler.pkl")
        clf.feature_names = joblib.load(saved / "feature_names.pkl")
        return clf

    def test_predict_returns_valid_output(self, fake_model_dir):
        clf        = self._load_clf(fake_model_dir)
        pred, prob = clf.predict([0.01, 300.0, 1.0, 5000.0, 0.0])
        assert pred in (0, 1)
        assert 0.0 <= prob <= 1.0

    def test_feature_length_validation_raises(self, fake_model_dir):
        clf = self._load_clf(fake_model_dir)
        with pytest.raises(ValueError, match="Expected 5 features"):
            clf.predict([1.0, 2.0, 3.0])

    def test_old_six_feature_vector_raises(self, fake_model_dir):
        """Guard against accidentally passing the old 6-element vector."""
        clf = self._load_clf(fake_model_dir)
        with pytest.raises(ValueError):
            clf.predict([0.5, 100.0, 0.0, 0.0, 500.0, 1.0])