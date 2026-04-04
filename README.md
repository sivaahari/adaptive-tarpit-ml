# 🛡️ Adaptive ML Tarpit

**Adaptive Intelligent Tarpit with ML Classification** — a network defense tool that detects suspicious connections using a LightGBM classifier trained on the NSL-KDD dataset, and traps malicious clients in a high-latency drip loop to exhaust their resources.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/ML-LightGBM-ff69b4?logo=lightgbm&logoColor=white)](https://lightgbm.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## What it does

Traditional firewalls block suspicious IPs reactively. This system classifies *behavior*: it intercepts a connection, observes the first ~1 second of activity, extracts 6 features, and runs a binary ML classifier to decide whether to serve a normal response or silently trap the connection in a drip loop (sending 1 null byte every 10 seconds to keep the socket open while wasting the attacker's resources).

```
Client connects
      │
      ▼
asyncio server reads first 4KB (1s timeout)
      │
      ▼
6 features extracted → LightGBM classifier
      │
      ├─ Benign (P < 0.5)  ──► HTTP 200 OK
      │
      └─ Malicious (P ≥ 0.5) ──► Drip loop (1 byte / 10s forever)
```

---

## ML Model

### Training Data

The classifier is trained on the **NSL-KDD dataset** (the standard benchmark for network intrusion detection). NSL-KDD contains labeled records of both normal traffic and four attack categories: DoS, Probe, R2L, and U2R.

Binary labeling: `normal` → 0, everything else → 1.

### Features Used

| # | Feature | Source | Notes |
|---|---------|--------|-------|
| 0 | `duration` | measured | seconds from connection open to classification |
| 1 | `src_bytes` | measured | bytes received from client |
| 2 | `dst_bytes` | NSL-KDD | 0 at classification time (pre-response) |
| 3 | `count` | NSL-KDD | connections to same host in past 2 s |
| 4 | `byte_rate` | engineered | `(src + dst bytes) / duration` |
| 5 | `is_empty_flag` | engineered | `1.0` if `src_bytes == 0` |

**Honest limitation:** `dst_bytes` and `count` are drawn from NSL-KDD at training time but cannot be fully measured at runtime from a single connection without cross-connection state tracking. `count` is fixed at 0.0 at inference time. `ConnectionRateTracker` in `network/feature_extractor.py` provides the scaffolding to fix this in a future iteration.

---

## Project Structure

```
adaptive-tarpit-ml/
├── main.py                      # Entry point
├── detection/
│   └── classifier.py            # LightGBM wrapper
├── tarpit/
│   └── tarpit_engine.py         # Async connection handler + drip loop
├── models/
│   ├── train_model.py           # Training script (NSL-KDD)
│   └── saved_models/            # lgbm_model.pkl, scaler.pkl (git-ignored)
├── logging_system/
│   └── database.py              # SQLite event logger
├── network/
│   └── feature_extractor.py     # Feature vector builder + rate tracker
├── visualization/
│   └── dashboard.py             # Multi-panel PNG report
├── data/
│   └── raw/                     # KDDTrain+.txt, KDDTest+.txt (not in repo)
├── tests/
│   ├── test_classifier.py       # Unit tests: features + classifier
│   └── test_tarpit.py           # Async unit tests: engine behavior
└── requirements.txt
```

---

## Installation & Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/sivaahari/adaptive-tarpit-ml.git
cd adaptive-tarpit-ml
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Download the NSL-KDD dataset

The NSL-KDD dataset is freely available from the Canadian Institute for Cybersecurity:

```
https://www.unb.ca/cic/datasets/nsl.html
```

Download `KDDTrain+.txt` and `KDDTest+.txt` and place them in `data/raw/`:

```
data/raw/KDDTrain+.txt
data/raw/KDDTest+.txt
```

### 3. Train the model

```bash
python3 models/train_model.py
```

This loads the NSL-KDD data, trains LightGBM, evaluates on the held-out test set, and prints a classification report before saving the model.

### 4. Run the tests

```bash
pytest tests/ -v
```

### 5. Launch the tarpit

The server listens on port 8080. On Linux, you can redirect incoming traffic to it using `iptables`:

```bash
# Start the tarpit
sudo PYTHONPATH=. ./venv/bin/python3 main.py

# In a separate terminal: redirect port 80 traffic to port 8080
# (requires root; adjust the interface/port as needed)
sudo iptables -t nat -A PREROUTING -p tcp --dport 80 -j REDIRECT --to-port 8080
```

Without the `iptables` redirect, the tarpit only intercepts connections explicitly sent to port 8080.

### 6. Generate the dashboard

After collecting some traffic, generate the visualization:

```bash
python3 visualization/dashboard.py
```

---

## Design Decisions & Known Limitations

| Area | Decision / Limitation |
|------|-----------------------|
| **Asyncio** | Chosen for concurrency — each tarpitted connection is a lightweight coroutine, not a thread |
| **ML inference off-loop** | `run_in_executor` ensures LightGBM inference never blocks the event loop |
| **Connection cap** | Semaphore at 500 concurrent sessions prevents this tool from becoming its own DoS vector |
| **`count` feature** | Set to 0.0 at runtime. `ConnectionRateTracker` is implemented but not yet wired into the engine |
| **`dst_bytes`** | Always 0 at classification time (no response has been sent yet) |
| **Port 8080** | Requires `iptables` NAT redirect to intercept real-world traffic on privileged ports |
| **No TLS inspection** | Encrypted traffic cannot be inspected at payload level |

---

## Disclaimer

This tool is intended for **defensive security research** and educational purposes. Only deploy on networks you own or have explicit written permission to test. The `iptables` commands above require root access — run in an isolated environment.