"""
Traffic classifier wrapper around the trained LightGBM model.

Loads the model, scaler, and feature-name list saved by models/train_model.py.

Threshold
---------
THRESHOLD controls the trade-off between recall and precision on malicious
traffic.  Lower values catch more attacks (higher recall) at the cost of more
false positives.  See models/evaluation_report.txt for the ROC-AUC curve data
to inform this choice.

    THRESHOLD = 0.5   # default: balanced
    THRESHOLD = 0.35  # aggressive: catches more probes, more false positives
    THRESHOLD = 0.65  # conservative: fewer false positives, misses more attacks
"""

import os
import warnings

import joblib
import numpy as np
from loguru import logger

warnings.filterwarnings("ignore", category=UserWarning)

THRESHOLD = 0.35   # P(malicious) ≥ THRESHOLD → classified as malicious


class TrafficClassifier:
    def __init__(self):
        base_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "models", "saved_models",
        )

        model_path         = os.path.join(base_path, "lgbm_model.pkl")
        scaler_path        = os.path.join(base_path, "scaler.pkl")
        feature_names_path = os.path.join(base_path, "feature_names.pkl")

        for path in (model_path, scaler_path):
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Required file not found: {path}\n"
                    "Run 'python3 models/train_model.py' first."
                )

        self.model         = joblib.load(model_path)
        self.scaler        = joblib.load(scaler_path)
        self.feature_names = (
            joblib.load(feature_names_path)
            if os.path.exists(feature_names_path)
            else None
        )
        n = len(self.feature_names) if self.feature_names else "unknown"
        logger.info(f"ML engine loaded ({n} features, threshold={THRESHOLD}).")

    def predict(self, features: list) -> tuple[int, float]:
        """
        Classify a single connection.

        Args:
            features: list of floats in FEATURE_NAMES order:
                      [duration, src_bytes, count, byte_rate, is_empty_flag]

        Returns:
            (prediction, probability)
            prediction  : 0 = Benign, 1 = Malicious
            probability : model's P(malicious)
        """
        if self.feature_names and len(features) != len(self.feature_names):
            raise ValueError(
                f"Expected {len(self.feature_names)} features "
                f"({self.feature_names}), got {len(features)}."
            )

        X           = np.array(features, dtype=float).reshape(1, -1)
        X_scaled    = self.scaler.transform(X)
        probability = float(self.model.predict_proba(X_scaled)[0][1])
        prediction  = 1 if probability >= THRESHOLD else 0
        return prediction, probability