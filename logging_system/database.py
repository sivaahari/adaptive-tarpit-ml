"""
SQLite persistence layer for tarpit connection logs.

Schema
------
connections
  id          INTEGER PRIMARY KEY AUTOINCREMENT
  timestamp   TEXT     ISO-8601 UTC timestamp
  src_ip      TEXT     attacker / client IP
  dst_port    INTEGER  port the connection arrived on
  prediction  INTEGER  0=Benign, 1=Malicious
  probability REAL     ML confidence score (malicious)
  action_taken TEXT    ALLOWED | TARPIT_DELAY
"""

import os
import sqlite3
from datetime import datetime, timezone

from loguru import logger


class TarpitDB:
    def __init__(self):
        base      = os.path.dirname(os.path.abspath(__file__))
        self.path = os.path.normpath(os.path.join(base, "..", "data", "tarpit_logs.db"))
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._init_schema()

    # ── Schema ────────────────────────────────────────────────────────────────

    def _init_schema(self) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS connections (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp    TEXT    NOT NULL,
                    src_ip       TEXT    NOT NULL,
                    dst_port     INTEGER NOT NULL,
                    prediction   INTEGER NOT NULL,
                    probability  REAL    NOT NULL,
                    action_taken TEXT    NOT NULL
                )
            """)
            # Index for fast IP-based queries (e.g. "how many times has this IP hit us?")
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_src_ip
                ON connections (src_ip)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp
                ON connections (timestamp)
            """)

    # ── Write ─────────────────────────────────────────────────────────────────

    def log_event(
        self,
        src_ip: str,
        port: int,
        prediction: int,
        probability: float,
        action: str,
    ) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.path, isolation_level=None) as conn:
            conn.execute("""
                INSERT INTO connections
                    (timestamp, src_ip, dst_port, prediction, probability, action_taken)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (ts, src_ip, int(port), int(prediction), float(probability), action))
        label = "Malicious" if prediction == 1 else "Benign"
        logger.debug(f"DB ← {src_ip}:{port} classified as {label} ({probability:.4f})")

    # ── Read helpers (used by dashboard) ──────────────────────────────────────

    def fetch_all(self):
        """Return all connection records as a list of dicts."""
        with sqlite3.connect(self.path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM connections ORDER BY timestamp DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def count_by_ip(self, limit: int = 20):
        """Return top IPs by connection count."""
        with sqlite3.connect(self.path) as conn:
            rows = conn.execute("""
                SELECT src_ip, COUNT(*) AS hits,
                       SUM(prediction) AS malicious_hits
                FROM connections
                GROUP BY src_ip
                ORDER BY hits DESC
                LIMIT ?
            """, (limit,)).fetchall()
        return rows