"""
Train the LightGBM classifier using the NSL-KDD dataset.

Features (5) — must match FEATURE_NAMES exactly and in this order.
These are the ONLY features measurable from a live asyncio TCP stream:

    [duration, src_bytes, count, byte_rate, is_empty_flag]

Why 5, not 6?
    dst_bytes was removed. In NSL-KDD it carries real signal, but at runtime
    it is always 0.0 (no server response has been sent before classification).
    Training on a feature that is structurally 0 at inference time biases the
    model and degrades confidence scores. Removing it gives an honest model.

    count is now populated at runtime via ConnectionRateTracker (see
    network/feature_extractor.py and tarpit/tarpit_engine.py).

Labels:
    'normal' → 0  (Benign)
    anything else → 1  (Malicious: DoS / Probe / R2L / U2R)

Usage:
    python3 models/train_model.py

Outputs:
    models/saved_models/lgbm_model.pkl
    models/saved_models/scaler.pkl
    models/saved_models/feature_names.pkl
    models/evaluation_report.txt        ← committed artifact, not git-ignored
"""

import os
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

# ── NSL-KDD column schema (41 features + label + difficulty) ─────────────────
NSLKDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag",
    "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
    "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count",
    "serr_rate", "srv_serr_rate", "rerr_rate", "srv_rerr_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serr_rate",
    "dst_host_srv_serr_rate", "dst_host_rerr_rate",
    "dst_host_srv_rerr_rate", "label", "difficulty",
]

# ── The 5 features measurable at runtime from a live TCP connection ───────────
# Saved alongside the model so classifier.py can validate input length.
FEATURE_NAMES = [
    "duration",       # seconds from connection open to classification decision
    "src_bytes",      # bytes received from the client in the observation window
    "count",          # connections from this IP in the past 2 s (ConnectionRateTracker)
    "byte_rate",      # src_bytes / (duration + ε) — connection speed signal
    "is_empty_flag",  # 1.0 if src_bytes == 0 (null / stealth probe indicator)
]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_nslkdd(filepath: str) -> pd.DataFrame:
    """Load a NSL-KDD .txt file (no header) and return a DataFrame."""
    df = pd.read_csv(filepath, header=None, names=NSLKDD_COLUMNS)
    df["label_binary"] = (df["label"].str.strip() != "normal").astype(int)
    return df


def engineer_features(df: pd.DataFrame):
    """
    Select and engineer the 5 runtime-measurable features.
    dst_bytes is intentionally excluded — see module docstring.
    Returns (X: DataFrame, y: Series).
    """
    df = df.copy()
    df["byte_rate"]    = df["src_bytes"] / (df["duration"] + 1e-6)
    df["is_empty_flag"] = (df["src_bytes"] == 0).astype(float)
    X = df[FEATURE_NAMES].astype(float)
    y = df["label_binary"]
    return X, y


# ── Training ──────────────────────────────────────────────────────────────────

def train() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "..", "data", "raw")
    save_dir = os.path.join(base_dir, "saved_models")
    os.makedirs(save_dir, exist_ok=True)

    train_path = os.path.join(data_dir, "KDDTrain+.txt")
    test_path  = os.path.join(data_dir, "KDDTest+.txt")

    for p in (train_path, test_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"NSL-KDD file not found: {p}\n"
                "Download from https://www.unb.ca/cic/datasets/nsl.html "
                "and place KDDTrain+.txt and KDDTest+.txt in data/raw/"
            )

    print("Loading NSL-KDD dataset …")
    train_df = load_nslkdd(train_path)
    test_df  = load_nslkdd(test_path)

    X_train, y_train = engineer_features(train_df)
    X_test,  y_test  = engineer_features(test_df)

    print(f"  Train : {len(X_train):,} samples")
    print(f"  Test  : {len(X_test):,}  samples")
    vc = y_train.value_counts()
    print(f"  Train class balance → Benign: {vc.get(0,0):,}  Malicious: {vc.get(1,0):,}")
    print(f"  Features ({len(FEATURE_NAMES)}): {FEATURE_NAMES}")

    # ── Scale ─────────────────────────────────────────────────────────────────
    scaler         = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=FEATURE_NAMES
    )
    X_test_scaled  = pd.DataFrame(
        scaler.transform(X_test), columns=FEATURE_NAMES
    )

    # ── Fit ───────────────────────────────────────────────────────────────────
    print("\nTraining LightGBM …")
    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        force_col_wise=True,
        verbose=-1,
        random_state=42,
    )
    model.fit(X_train_scaled, y_train)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    report  = classification_report(y_test, y_pred, target_names=["Benign", "Malicious"])
    cm      = confusion_matrix(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    # ── Feature importance ────────────────────────────────────────────────────
    importances = model.feature_importances_
    fi_lines = "\nFeature Importances (higher = more discriminating):\n"
    for name, score in sorted(zip(FEATURE_NAMES, importances), key=lambda x: -x[1]):
        bar = "█" * int(score / max(importances) * 30)
        fi_lines += f"  {name:<18} {bar} ({score})\n"

    # ── Print to terminal ─────────────────────────────────────────────────────
    separator = "─" * 66
    print(f"\n{separator}")
    print("Evaluation on NSL-KDD Test Set")
    print(separator)
    print(report)
    print("Confusion Matrix (rows=actual, cols=predicted):")
    print(f"  TN={cm[0,0]:>6}  FP={cm[0,1]:>6}")
    print(f"  FN={cm[1,0]:>6}  TP={cm[1,1]:>6}")
    print(f"\nROC-AUC : {roc_auc:.4f}")
    print(fi_lines)
    print(separator)

    # ── Save evaluation report as a committed artifact ────────────────────────
    report_path = os.path.join(base_dir, "evaluation_report.txt")
    with open(report_path, "w") as f:
        f.write("Adaptive ML Tarpit — Model Evaluation Report\n")
        f.write(f"Dataset  : NSL-KDD (KDDTrain+.txt / KDDTest+.txt)\n")
        f.write(f"Features : {FEATURE_NAMES}\n")
        f.write(f"Train samples : {len(X_train):,}\n")
        f.write(f"Test  samples : {len(X_test):,}\n\n")
        f.write(f"{separator}\n")
        f.write(report)
        f.write(f"\nConfusion Matrix (rows=actual, cols=predicted):\n")
        f.write(f"  TN={cm[0,0]:>6}  FP={cm[0,1]:>6}\n")
        f.write(f"  FN={cm[1,0]:>6}  TP={cm[1,1]:>6}\n")
        f.write(f"\nROC-AUC : {roc_auc:.4f}\n")
        f.write(fi_lines)
    print(f"Evaluation report saved → {report_path}")

    # ── Save model artifacts ──────────────────────────────────────────────────
    joblib.dump(model,         os.path.join(save_dir, "lgbm_model.pkl"))
    joblib.dump(scaler,        os.path.join(save_dir, "scaler.pkl"))
    joblib.dump(FEATURE_NAMES, os.path.join(save_dir, "feature_names.pkl"))
    print(f"Model artifacts saved  → {save_dir}")


if __name__ == "__main__":
    train()