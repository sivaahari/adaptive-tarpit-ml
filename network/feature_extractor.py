"""
Feature extractor for live asyncio connections.

Converts raw connection observations into the 5-element feature vector
consumed by TrafficClassifier.predict().

Feature order (must match models/train_model.py :: FEATURE_NAMES):
    [duration, src_bytes, count, byte_rate, is_empty_flag]

Why 5 features, not 6?
    dst_bytes was removed from the feature set. In NSL-KDD it has real signal,
    but at classification time (before a server response is sent) it is always
    0. Training on a feature that is structurally 0 at inference biases the
    model. Removing it makes training and inference consistent.

Note on `count`
---------------
In NSL-KDD, `count` is the number of connections to the same destination host
in the past 2 seconds — a scan-rate signal.  ConnectionRateTracker (below)
maintains that state and is wired into tarpit_engine.py so every call to
predict() receives a real, live value for this feature.
"""

import time
from collections import defaultdict, deque
from threading import Lock


# ── Feature extraction ────────────────────────────────────────────────────────

def build_feature_vector(
    duration:    float,
    src_bytes:   int,
    dst_bytes:   int,   # accepted for API clarity but NOT passed to the model
    count:       int,
) -> list[float]:
    """
    Assemble the 5-feature vector from raw connection observations.

    Args:
        duration  : seconds from connection open to classification decision
        src_bytes : bytes received from the remote client
        dst_bytes : bytes sent by the server — accepted but excluded from the
                    model (always 0 pre-response; including it biases inference)
        count     : connections from this IP in the past 2 seconds
                    (populated by ConnectionRateTracker)

    Returns:
        list of 5 floats in FEATURE_NAMES order:
        [duration, src_bytes, count, byte_rate, is_empty_flag]
    """
    byte_rate = src_bytes / (duration + 1e-6)
    is_empty  = 1.0 if src_bytes == 0 else 0.0

    return [
        float(duration),
        float(src_bytes),
        float(count),
        float(byte_rate),
        is_empty,
    ]


# ── Cross-connection rate tracker ─────────────────────────────────────────────

class ConnectionRateTracker:
    """
    Thread-safe sliding-window connection counter per source IP.

    Tracks how many connections each IP has opened within the past `window`
    seconds.  The returned count provides the `count` feature, which is one of
    the strongest signals for detecting port scanners and DoS tools.

    Wired into IntelligentTarpit in tarpit/tarpit_engine.py.

    Usage:
        tracker = ConnectionRateTracker(window=2.0)

        # On each incoming connection from ip:
        count = tracker.record(ip)

        # Pass count to build_feature_vector(...)
    """

    def __init__(self, window: float = 2.0):
        self._window  = window
        self._buckets: dict[str, deque] = defaultdict(deque)
        self._lock    = Lock()

    def record(self, ip: str) -> int:
        """
        Record a new connection from `ip` and return the total within the
        sliding window (including this connection).
        """
        now = time.monotonic()
        with self._lock:
            dq = self._buckets[ip]
            while dq and now - dq[0] > self._window:
                dq.popleft()
            dq.append(now)
            return len(dq)

    def purge_stale(self) -> None:
        """Remove entries for IPs quiet for 10× the window.  Call periodically."""
        now = time.monotonic()
        with self._lock:
            stale = [
                ip for ip, dq in self._buckets.items()
                if dq and now - dq[-1] > self._window * 10
            ]
            for ip in stale:
                del self._buckets[ip]