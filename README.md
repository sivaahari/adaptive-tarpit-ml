# 🛡️ Adaptive ML Tarpit

> **A network defense system that classifies suspicious connections using machine learning and traps malicious clients in a high-latency drip loop — exhausting their resources while legitimate users pass through unaffected.**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/ML-LightGBM-ff69b4)](https://lightgbm.readthedocs.io/)
[![Dataset: NSL-KDD](https://img.shields.io/badge/Dataset-NSL--KDD-orange)](https://www.unb.ca/cic/datasets/nsl.html)
[![Tests](https://img.shields.io/badge/Tests-20%20passing-brightgreen)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Table of Contents

- [What Is a Tarpit?](#what-is-a-tarpit)
- [How This System Works](#how-this-system-works)
- [System Architecture](#system-architecture)
- [Machine Learning Model](#machine-learning-model)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Testing](#testing)
- [Dashboard](#dashboard)
- [Design Decisions & Limitations](#design-decisions--limitations)
- [Future Work](#future-work)
- [Disclaimer](#disclaimer)

---

## What Is a Tarpit?

A **tarpit** is a network defense technique that deliberately slows down suspicious connections instead of immediately blocking them. Rather than telling an attacker "connection refused" (which is fast and informative), a tarpit accepts the connection and then sends data at an agonizingly slow rate — keeping the attacker's socket open and tying up their connection pool for as long as possible.

Traditional tarpits apply this slowdown to everyone, which can harm legitimate users. This project uses **machine learning** to make the decision: classify the connection first, then act. Legitimate traffic gets a normal response in milliseconds; detected probes enter the drip loop.

---

## How This System Works

```
Incoming TCP connection
         │
         ▼
┌─────────────────────┐
│   asyncio server    │  ← non-blocking, handles hundreds of
│   (port 8080)       │    concurrent trapped connections
└─────────┬───────────┘
          │
          ▼
  Read first 4KB (1s timeout)
  Record: duration, src_bytes
          │
          ▼
┌─────────────────────┐
│  Feature Extractor  │  ← builds the 6-element feature vector
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  LightGBM           │  ← run in ThreadPoolExecutor
│  Classifier         │    (never blocks the event loop)
└─────────┬───────────┘
          │
    ┌─────┴──────┐
    │            │
  P < 0.5      P ≥ 0.5
    │            │
    ▼            ▼
HTTP 200 OK   Drip Loop
(immediate)   (1 null byte every 10s, forever)
    │            │
    └─────┬──────┘
          │
          ▼
   Log to SQLite DB
   (async, off event loop)
```

---

## System Architecture

The project is built on Python's `asyncio` for high-concurrency I/O, with CPU-bound work (ML inference, database writes) offloaded to a thread pool so the event loop is never blocked.

### Key Design Choices

**Asyncio over threads:** Each tarpitted connection is a lightweight coroutine — not a thread or process. This allows hundreds of simultaneous trapped sessions with minimal CPU and RAM overhead.

**ThreadPoolExecutor for blocking work:** LightGBM inference and SQLite writes are synchronous. Calling them directly inside an `async` function would stall every other connection. Both are dispatched via `loop.run_in_executor()`.

**Semaphore cap (500 sessions):** Without a cap, a flood of incoming connections could exhaust file descriptors and RAM — turning the tarpit into a self-inflicted DoS. The semaphore ensures the system remains stable under attack.

**Behavioral classification, not IP blocklisting:** Static IP blocklists are easily bypassed (rotate IPs, use botnets). This system looks at *how* a connection behaves in its first second — payload size, timing, whether any data was sent at all — rather than *who* is connecting.

---

## Machine Learning Model

### Dataset: NSL-KDD

The classifier is trained on the **NSL-KDD dataset**, the standard benchmark for network intrusion detection research, published by the Canadian Institute for Cybersecurity. It contains labeled connection records across five categories:

| Label | Description |
|-------|-------------|
| `normal` | Legitimate traffic → class **0** (Benign) |
| `DoS` | Denial of Service attacks → class **1** (Malicious) |
| `Probe` | Port scans and network sweeps → class **1** (Malicious) |
| `R2L` | Remote-to-Local unauthorized access → class **1** (Malicious) |
| `U2R` | User-to-Root privilege escalation → class **1** (Malicious) |

**Training set:** 125,973 samples (53.5% benign, 46.5% malicious)  
**Test set:** 22,544 samples

### Features

Only 6 of NSL-KDD's 41 features are used — specifically the ones that can be observed at connection time from a single TCP stream:

| # | Feature | How it's obtained | What it signals |
|---|---------|-------------------|-----------------|
| 0 | `duration` | Measured live | Time from connection open to first byte |
| 1 | `src_bytes` | Measured live | Bytes sent by the client |
| 2 | `dst_bytes` | NSL-KDD (0 at runtime) | Bytes sent by the server |
| 3 | `count` | NSL-KDD (0 at runtime*) | Connections to same host in past 2s |
| 4 | `byte_rate` | Engineered | `(src + dst bytes) / duration` |
| 5 | `is_empty_flag` | Engineered | `1.0` if `src_bytes == 0` (stealth probe) |

> \*`count` is set to 0.0 at inference time. `ConnectionRateTracker` in `network/feature_extractor.py` implements the sliding-window counter needed to populate this properly — see [Future Work](#future-work).

### Algorithm: LightGBM

LightGBM (Light Gradient Boosting Machine) was chosen because:
- Extremely fast inference (microseconds per prediction) — critical for a real-time network tool
- Handles class imbalance well with minimal tuning
- Interpretable via feature importance scores
- Small model footprint (the `.pkl` file is under 1MB)

### Evaluation Results

```
─── Evaluation on NSL-KDD Test Set ───────────────────────────────
              precision    recall  f1-score   support

      Benign       0.65      0.97      0.78      9711
   Malicious       0.96      0.61      0.75     12833

    accuracy                           0.76     22544
   macro avg       0.81      0.79      0.76     22544
weighted avg       0.83      0.76      0.76     22544

Confusion Matrix (rows=actual, cols=predicted):
  TN=  9414  FP=   297
  FN=  5028  TP=  7805

ROC-AUC: 0.9737
──────────────────────────────────────────────────────────────────
```

**Interpreting these results honestly:**

The **ROC-AUC of 0.9737** shows the model has strong discriminative ability — it reliably separates benign from malicious traffic. The **recall of 0.61 on malicious traffic** (missing ~5,000 attacks in the test set) is a threshold effect, not a model quality problem. The default 0.5 threshold is conservative; lowering it captures more attacks at the cost of more false positives. With only 6 of 41 available features, this performance is a realistic reflection of what can be measured from a single connection.

### Adjusting the Decision Threshold

In `detection/classifier.py`, `THRESHOLD` controls sensitivity:

```python
THRESHOLD = 0.35   # lower = more aggressive, catches more probes
                   # higher = more conservative, fewer false positives
```

---

## Project Structure

```
adaptive-tarpit-ml/
│
├── main.py                        # Entry point — starts the asyncio server
│
├── detection/
│   ├── __init__.py
│   └── classifier.py              # LightGBM wrapper: loads model, runs predict()
│
├── tarpit/
│   ├── __init__.py
│   └── tarpit_engine.py           # Core engine: connection handler + drip loop
│
├── models/
│   ├── train_model.py             # Training script — loads NSL-KDD, evaluates, saves
│   └── saved_models/              # lgbm_model.pkl, scaler.pkl, feature_names.pkl
│                                  # (git-ignored — run train_model.py to generate)
│
├── logging_system/
│   ├── __init__.py
│   └── database.py                # SQLite event logger with indexed queries
│
├── network/
│   ├── __init__.py
│   └── feature_extractor.py       # build_feature_vector() + ConnectionRateTracker
│
├── visualization/
│   └── dashboard.py               # 5-panel PNG dashboard from live DB logs
│
├── data/
│   ├── raw/                       # KDDTrain+.txt, KDDTest+.txt (git-ignored)
│   └── tarpit_logs.db             # Runtime SQLite log (git-ignored)
│
├── tests/
│   ├── __init__.py
│   ├── test_classifier.py         # Unit tests: feature vector, rate tracker, classifier
│   └── test_tarpit.py             # Async unit tests: engine behavior under all cases
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.10 or higher
- Linux (tested on Kali Linux / Ubuntu via WSL)
- `sudo` access (required to run the server on privileged ports via `iptables`)

### Step 1 — Clone the repository

```bash
git clone https://github.com/sivaahari/adaptive-tarpit-ml.git
cd adaptive-tarpit-ml
```

### Step 2 — Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Download the NSL-KDD dataset

The dataset is freely available from the Canadian Institute for Cybersecurity:

```
https://www.unb.ca/cic/datasets/nsl.html
```

Download `KDDTrain+.txt` and `KDDTest+.txt` and place them here:

```
data/raw/KDDTrain+.txt
data/raw/KDDTest+.txt
```

### Step 5 — Create required `__init__.py` files

```bash
touch detection/__init__.py tarpit/__init__.py \
      logging_system/__init__.py network/__init__.py \
      tests/__init__.py
```

---

## Usage

### 1. Train the model

```bash
PYTHONPATH=. python3 models/train_model.py
```

This will:
- Load and preprocess the NSL-KDD dataset
- Engineer the 6 runtime features
- Train the LightGBM classifier
- Print a full evaluation report on the held-out test set
- Save `lgbm_model.pkl`, `scaler.pkl`, and `feature_names.pkl` to `models/saved_models/`

Verify the model files were saved:

```bash
ls models/saved_models/
# lgbm_model.pkl  scaler.pkl  feature_names.pkl
```

### 2. Launch the tarpit server

```bash
sudo PYTHONPATH=. python3 main.py
```

The server listens on `0.0.0.0:8080`. You should see:

```
INFO  | ML engine loaded successfully.
INFO  | Tarpit live on ('0.0.0.0', 8080)  (max 500 concurrent sessions)
INFO  | Press Ctrl-C to stop.
```

### 3. Route real traffic to the tarpit (optional)

By default the tarpit only intercepts connections explicitly sent to port 8080. To intercept traffic arriving on standard HTTP port 80, use `iptables` to redirect it:

```bash
# Redirect port 80 → 8080
sudo iptables -t nat -A PREROUTING -p tcp --dport 80 -j REDIRECT --to-port 8080

# To undo:
sudo iptables -t nat -D PREROUTING -p tcp --dport 80 -j REDIRECT --to-port 8080
```

> **Warning:** This affects all TCP traffic on port 80 on the machine. Only run this on an isolated test environment or a dedicated honeypot host.

### 4. Test it manually

Open a second terminal with the server running:

```bash
# Benign — sends real HTTP data, should get HTTP 200 OK immediately
curl http://localhost:8080

# Malicious — sends zero bytes (stealth port probe), should hang indefinitely
nc -z localhost 8080

# Malicious — service fingerprinting scan
nmap -sV localhost -p 8080
```

Watch the server terminal for classification output:

```
INFO    | BENIGN [127.0.0.1:54321] 87B confidence=0.9998 → ALLOWED
WARNING | PROBE  [127.0.0.1:54322] confidence=0.9991 → TARPITTED
```

---

## Testing

The test suite covers 20 cases across feature extraction, the rate tracker, classifier input validation, and async engine behavior.

```bash
PYTHONPATH=. venv/bin/pytest tests/ -v
```

Expected output:

```
tests/test_classifier.py::TestBuildFeatureVector::test_returns_six_elements          PASSED
tests/test_classifier.py::TestBuildFeatureVector::test_all_floats                    PASSED
tests/test_classifier.py::TestBuildFeatureVector::test_is_empty_flag_set_...         PASSED
tests/test_classifier.py::TestBuildFeatureVector::test_is_empty_flag_clear_...       PASSED
tests/test_classifier.py::TestBuildFeatureVector::test_byte_rate_computed_correctly  PASSED
tests/test_classifier.py::TestBuildFeatureVector::test_zero_duration_no_div_error    PASSED
tests/test_classifier.py::TestBuildFeatureVector::test_feature_order                 PASSED
tests/test_classifier.py::TestConnectionRateTracker::test_first_connection_...       PASSED
tests/test_classifier.py::TestConnectionRateTracker::test_multiple_connections_...   PASSED
tests/test_classifier.py::TestConnectionRateTracker::test_different_ips_...          PASSED
tests/test_classifier.py::TestConnectionRateTracker::test_window_expiry              PASSED
tests/test_classifier.py::TestTrafficClassifierSmokeTest::test_predict_benign        PASSED
tests/test_classifier.py::TestTrafficClassifierSmokeTest::test_feature_length_...    PASSED
tests/test_tarpit.py::TestIntelligentTarpit::test_benign_connection_gets_200         PASSED
tests/test_tarpit.py::TestIntelligentTarpit::test_malicious_connection_is_...        PASSED
tests/test_tarpit.py::TestIntelligentTarpit::test_inference_called_with_six_...      PASSED
tests/test_tarpit.py::TestIntelligentTarpit::test_empty_payload_sets_is_empty_flag   PASSED
tests/test_tarpit.py::TestIntelligentTarpit::test_nonempty_payload_clears_...        PASSED
tests/test_tarpit.py::TestIntelligentTarpit::test_exception_in_classifier_...        PASSED
tests/test_tarpit.py::TestIntelligentTarpit::test_semaphore_initialized_to_...       PASSED

20 passed in 3.33s
```

---

## Dashboard

After collecting traffic, generate a 5-panel visualization from the SQLite logs:

```bash
PYTHONPATH=. python3 visualization/dashboard.py
xdg-open visualization/tarpit_stats.png
```

The dashboard includes:

| Panel | What it shows |
|-------|--------------|
| Traffic Classification | Bar chart of benign vs. malicious connection counts |
| ML Confidence Distribution | Histogram of `P(Malicious)` scores split by verdict |
| Top 10 Malicious IPs | Horizontal bar chart of most active offending IPs |
| Action Breakdown | Pie chart of ALLOWED vs. TARPIT\_DELAY actions |
| Connections per Minute | Time-series of traffic volume over the session |

![Dashboard](visualization/tarpit_stats.png)

---

## Design Decisions & Limitations

| Area | Decision / Honest Limitation |
|------|------------------------------|
| **Asyncio** | Chosen for I/O concurrency — coroutines are far cheaper than threads for holding open hundreds of slow connections simultaneously |
| **ML inference off-loop** | `run_in_executor` ensures LightGBM never blocks the event loop; without this, every tarpitted connection would stall all others during inference |
| **Connection cap (500)** | Prevents the tarpit from becoming its own DoS target; connections beyond the cap are dropped immediately |
| **`count` = 0.0 at runtime** | The scan-rate feature requires cross-connection state; `ConnectionRateTracker` is built but not yet wired into the engine |
| **`dst_bytes` = 0 at runtime** | No response is sent before classification, so server-side bytes are always zero at decision time |
| **Port 8080** | Requires `iptables` redirect to intercept real-world traffic on port 80/443; the tarpit alone intercepts nothing on standard ports without this step |
| **6 of 41 features** | Only features measurable from a single asyncio stream connection are used; this limits recall but keeps the system deployable without raw packet capture |
| **No TLS inspection** | Encrypted traffic payload cannot be inspected at this layer; `is_empty_flag` and timing features still apply |
| **NSL-KDD is a benchmark dataset** | Real-world traffic distributions differ; the model should be fine-tuned on live captured traffic for production deployment |

---

## Future Work

These are concrete, implementable improvements that would meaningfully strengthen the system:

**Wire in `ConnectionRateTracker`**
`network/feature_extractor.py` already implements a sliding-window counter. Connecting it into `tarpit_engine.py` would populate the `count` feature at runtime, improving detection of rapid port sweeps.

**Online retraining pipeline**
Build a feedback loop: confirmed malicious connections logged to SQLite can be periodically used to retrain the model, adapting to new attack patterns over time.

**Configurable parameters via config file**
Currently `THRESHOLD`, `MAX_CONCURRENT`, `DRIP_INTERVAL`, and `PORT` are hardcoded constants. A `config.yaml` or `argparse` interface would make the tool configurable without editing source.

**Deceptive response generation**
Instead of sending null bytes, tarpitted connections could receive plausible-looking fake server banners (SSH, FTP, HTTP) to waste more attacker time and harvest fingerprinting attempts.

**GeoIP enrichment in the dashboard**
Map source IPs to countries and ASNs using a local MaxMind GeoLite2 database and render attack origins on a world map panel.

**Prometheus metrics endpoint**
Expose live counters (connections/sec, tarpit occupancy, classifier latency) in Prometheus format for integration with Grafana dashboards.

---

## Disclaimer

This tool is intended for **defensive security research and educational purposes only**. Deploy exclusively on networks you own or have explicit written authorization to test. The `iptables` redirect commands require root access — always run in an isolated virtual machine or dedicated honeypot environment, never on a shared or production system.