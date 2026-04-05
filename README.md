```markdown
# 🛡️ Adaptive ML Tarpit

**An intelligent, machine learning-powered network defense system that classifies incoming connections in real time and selectively traps malicious traffic in a high-latency drip loop — while allowing legitimate users to pass through unaffected.**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/ML-LightGBM-ff69b4)](https://lightgbm.readthedocs.io/)
[![Dataset: NSL-KDD](https://img.shields.io/badge/Dataset-NSL--KDD-orange)](https://www.unb.ca/cic/datasets/nsl.html)
[![Tests](https://img.shields.io/badge/Tests-20-passing-brightgreen)](https://github.com/sivaahari/adaptive-tarpit-ml/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

Traditional tarpits apply indiscriminate delays that can degrade user experience. **Adaptive ML Tarpit** solves this by using a lightweight **LightGBM** classifier to make per-connection decisions based on observable behavioral features.

- **Benign connections** receive an immediate, normal response.
- **Suspicious connections** (probes, scans, or attacks) are accepted but trapped in a slow "drip loop" that consumes attacker resources while tying up their connection pool.

Built with **asyncio** for high concurrency, offloaded ML inference, and a resource-safe semaphore cap, this project demonstrates production-minded defensive security engineering.

It serves as both a practical honeypot component and a showcase of applying machine learning to real-time network defense with honest evaluation and clean architecture.

---

## Key Features

- **Real-time behavioral classification** using only runtime-observable features
- **Live connection-rate tracking** (`count` feature) for detecting port scans and floods
- **Non-blocking architecture**: ML inference and database writes offloaded via `ThreadPoolExecutor`
- **Resource safety**: Semaphore caps concurrent tarpitted sessions at 500 to prevent self-DoS
- **Persistent logging**: Async SQLite backend with indexed queries
- **Comprehensive testing**: 20+ unit and async tests covering feature extraction, classification, and engine behavior
- **Visualization dashboard**: Automated multi-panel analysis of captured traffic
- **Transparent evaluation**: Full model metrics, confusion matrix, and feature importance included

---

## How It Works

```mermaid
flowchart TD
    A[Incoming TCP Connection] --> B[asyncio Server]
    B --> C[Read initial payload (1s timeout)]
    C --> D[Extract features: duration, src_bytes, count, byte_rate, is_empty_flag]
    D --> E[LightGBM Classification]
    E -->|Benign| F[Immediate HTTP 200 OK]
    E -->|Malicious| G[Slow Drip Loop<br>(null byte every 10s)]
    F --> H[Log to SQLite]
    G --> H
```

Classification happens within the first second using five carefully chosen features that can be measured from a single TCP stream without packet capture or TLS termination.

---

## Machine Learning Model

### Dataset
Trained on the **NSL-KDD** dataset (a refined version of the classic KDD Cup 1999 data), widely used for intrusion detection research.

- **Training samples**: 125,973
- **Test samples**: 22,544
- Binary labels: `normal` → Benign (0); all others (DoS, Probe, R2L, U2R) → Malicious (1)

### Features (5 runtime-measurable)
The model uses only features that are available at classification time:

| Feature          | Description                                      | Signal Detected                  |
|------------------|--------------------------------------------------|----------------------------------|
| `duration`       | Time from connection open to decision            | Connection timing                |
| `src_bytes`      | Bytes received from client                       | Payload size                     |
| `count`          | Connections from same IP in last 2 seconds       | Scan / flood rate (live tracked) |
| `byte_rate`      | `src_bytes / duration`                           | Connection speed                 |
| `is_empty_flag`  | 1 if no data received                            | Stealth / null probes            |

**Note**: `dst_bytes` was intentionally excluded because it is structurally zero before any server response is sent. This ensures training and inference remain consistent.

### Algorithm & Performance
- **Model**: LightGBM (fast inference, low memory footprint)
- **Evaluation on NSL-KDD test set**:

```
              precision    recall  f1-score   support
      Benign       0.66      0.97      0.79      9711
   Malicious       0.97      0.63      0.76     12833
    accuracy                           0.77     22544
   macro avg       0.81      0.80      0.77     22544
weighted avg       0.83      0.77      0.77     22544

ROC-AUC : 0.9580
```

Feature importance highlights `src_bytes` and `count` as the strongest predictors.

The decision threshold (`THRESHOLD = 0.35` by default) can be adjusted for more aggressive or conservative behavior.

---

## Architecture Highlights

- **Concurrency**: Pure `asyncio` + `StreamReader/Writer` for lightweight handling of hundreds of trapped connections.
- **Offloading**: Blocking operations (model prediction, SQLite writes) dispatched to a `ThreadPoolExecutor`.
- **Safety**: `asyncio.Semaphore` prevents resource exhaustion under heavy load.
- **Modularity**: Clean separation between detection, tarpit engine, feature extraction, logging, and visualization.
- **Testing**: Rigorous coverage of edge cases, including exception resilience and feature vector integrity.

---

## Project Structure

```
adaptive-tarpit-ml/
├── main.py                     # Entry point — starts the asyncio server
├── detection/
│   └── classifier.py           # LightGBM wrapper and prediction logic
├── tarpit/
│   └── tarpit_engine.py        # Core connection handler and drip loop
├── models/
│   ├── train_model.py          # Training + evaluation pipeline
│   └── saved_models/           # Trained artifacts (generated)
├── logging_system/
│   └── database.py             # SQLite persistence layer
├── network/
│   └── feature_extractor.py    # Feature vector builder + rate tracker
├── visualization/
│   └── dashboard.py            # Generates traffic analysis PNG
├── tests/                      # Unit & async tests
├── data/
│   ├── raw/                    # NSL-KDD dataset files
│   └── tarpit_logs.db          # Runtime logs (git-ignored)
├── requirements.txt
└── README.md
```

---

## Installation & Setup

### Prerequisites
- Python 3.10+
- Linux environment (tested on Ubuntu/Kali; requires `sudo` for port redirection in production-like tests)
- Isolated test environment or VM recommended

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/sivaahari/adaptive-tarpit-ml.git
   cd adaptive-tarpit-ml
   ```

2. **Create virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NSL-KDD dataset**
   Place `KDDTrain+.txt` and `KDDTest+.txt` in `data/raw/`.  
   (Dataset originally from the Canadian Institute for Cybersecurity.)

5. **Train the model**
   ```bash
   PYTHONPATH=. python3 models/train_model.py
   ```

6. **(Optional) Create package init files** (if missing)
   ```bash
   touch detection/__init__.py tarpit/__init__.py logging_system/__init__.py \
         network/__init__.py tests/__init__.py
   ```

---

## Usage

### Launch the Tarpit Server
```bash
sudo PYTHONPATH=. python3 main.py
```

The server listens on `0.0.0.0:8080` by default.

### Redirect Traffic (optional, for testing on standard ports)
```bash
# Redirect port 80 traffic to the tarpit
sudo iptables -t nat -A PREROUTING -p tcp --dport 80 -j REDIRECT --to-port 8080
```

**Warning**: Use only in isolated environments.

### Manual Testing
```bash
# Benign traffic (should receive immediate response)
curl http://localhost:8080

# Stealth probe (likely tarpitted)
nc -z localhost 8080

# Service scan (likely tarpitted)
nmap -sV localhost -p 8080
```

Watch the console for classification logs and check `visualization/tarpit_stats.png` after running the dashboard.

### Generate Dashboard
```bash
PYTHONPATH=. python3 visualization/dashboard.py
```

---

## Testing

Run the full test suite:
```bash
PYTHONPATH=. pytest tests/ -v
```

All 20 tests cover feature engineering, rate tracking, classification edge cases, and async engine behavior.

---

## Design Decisions & Limitations

This project prioritizes **practical deployability** and **transparency**:

- Features are limited to what a single TCP stream can observe (no raw packet capture or deep TLS inspection).
- `count` is now fully live via `ConnectionRateTracker`.
- Model performance reflects real constraints (no future knowledge leakage).
- NSL-KDD is a benchmark; production use should include fine-tuning on your own traffic.

**Known limitations** (addressed transparently in code and docs):
- No TLS termination
- Threshold-based decisions (tunable)
- Requires root for privileged port redirection

---

## Future Enhancements

- Configurable parameters via YAML/CLI
- Periodic model retraining from logged data
- Deceptive responses (fake banners) for tarpitted connections
- Prometheus metrics endpoint
- GeoIP enrichment in the dashboard
- Docker + docker-compose deployment

---

## Disclaimer

This tool is provided for **educational, research, and defensive security purposes only**. Use exclusively on systems you own or have explicit authorization to monitor. Misuse may violate applicable laws.

Always deploy in isolated environments. The project is licensed under the MIT License.

---

**Built with curiosity and a focus on practical security engineering.**

Feedback, issues, and contributions are welcome!
```