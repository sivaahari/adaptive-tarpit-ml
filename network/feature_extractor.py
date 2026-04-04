"""
Feature extractor for live asyncio connections.

Converts raw connection observations (timing, byte counts) into the 6-element
feature vector consumed by TrafficClassifier.predict().

Feature order (must match models/train_model.py :: FEATURE_NAMES):
    [duration, src_bytes, dst_bytes, count, byte_rate, is_empty_flag]

Note on `count`
---------------
In NSL-KDD, `count` is the number of connections to the same destination host
in the past 2 seconds — a scan-rate signal.  At single-connection time we
cannot compute this without cross-connection state.  ConnectionRateTracker
(below) maintains that state so the engine can pass a real value.
"""

import time
from collections import defaultdict, deque
from threading import Lock


# ── Feature extraction ────────────────────────────────────────────────────────

def build_feature_vector(
    duration:    float,
    src_bytes:   int,
    dst_bytes:   int,
    count:       int,
) -> list[float]:
    """
    Assemble the 6-feature vector from raw observations.

    Args:
        duration  : seconds from connection open to classification decision
        src_bytes : bytes received from the remote client
        dst_bytes : bytes sent by the server (0 if called before any response)
        count     : connections from the same IP in the past 2 seconds

    Returns:
        list of 6 floats ready to pass to TrafficClassifier.predict()
    """
    byte_rate    = (src_bytes + dst_bytes) / (duration + 1e-6)
    is_empty     = 1.0 if src_bytes == 0 else 0.0

    return [
        float(duration),
        float(src_bytes),
        float(dst_bytes),
        float(count),
        float(byte_rate),
        is_empty,
    ]


# ── Cross-connection rate tracker ─────────────────────────────────────────────

class ConnectionRateTracker:
    """
    Thread-safe sliding-window counter.

    Tracks how many connections each IP has opened in the past `window`
    seconds.  This provides the `count` feature that signals port scanning.

    Usage (in tarpit_engine.py)
    ---------------------------
        tracker = ConnectionRateTracker(window=2.0)

        # When a connection arrives from ip:
        count = tracker.record(ip)

        # Pass count into build_feature_vector(...)
    """

    def __init__(self, window: float = 2.0):
        self._window  = window
        self._buckets: dict[str, deque] = defaultdict(deque)
        self._lock    = Lock()

    def record(self, ip: str) -> int:
        """
        Record a new connection from `ip` and return the total count
        of connections from that IP within the sliding window.
        """
        now = time.monotonic()
        with self._lock:
            dq = self._buckets[ip]
            # Evict timestamps outside the window
            while dq and now - dq[0] > self._window:
                dq.popleft()
            dq.append(now)
            return len(dq)

    def purge_stale(self) -> None:
        """Remove entries for IPs that have gone quiet.  Call periodically."""
        now = time.monotonic()
        with self._lock:
            stale = [
                ip for ip, dq in self._buckets.items()
                if dq and now - dq[-1] > self._window * 10
            ]
            for ip in stale:
                del self._buckets[ip]