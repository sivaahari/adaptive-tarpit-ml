"""
Core tarpit engine.

Key design points:
  - ML inference and DB writes are offloaded to a ThreadPoolExecutor so they
    never block the asyncio event loop.
  - A Semaphore caps simultaneous trapped sessions, preventing resource
    exhaustion (an uncapped tarpit is itself a DoS vector).
  - ConnectionRateTracker provides the live `count` feature — how many times
    this IP has connected in the past 2 seconds — a primary scan indicator.
  - Feature vector (5 elements) matches FEATURE_NAMES in train_model.py exactly.
    dst_bytes is intentionally excluded: it is always 0 pre-response, which
    would bias the model if included at inference time.
  - Exceptions are logged, not silently swallowed.
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from loguru import logger

from logging_system.database import TarpitDB
from network.feature_extractor import build_feature_vector, ConnectionRateTracker

# ── Tuneable constants ────────────────────────────────────────────────────────
MAX_CONCURRENT   = 500    # semaphore cap; prevents self-DoS from flood attacks
READ_TIMEOUT     = 1.0    # seconds to wait for the client's first byte
DRIP_INTERVAL    = 10.0   # seconds between each null-byte drip to a tarpitted conn
EXECUTOR_WORKERS = 4      # thread-pool size for blocking I/O (inference + SQLite)


class IntelligentTarpit:
    def __init__(self, delay_base: float = DRIP_INTERVAL, classifier=None):
        self.delay_base   = delay_base
        self.classifier   = classifier
        self.db           = TarpitDB()
        self._semaphore   = asyncio.Semaphore(MAX_CONCURRENT)
        self._executor    = ThreadPoolExecutor(max_workers=EXECUTOR_WORKERS)
        self._rate_tracker = ConnectionRateTracker(window=2.0)

    # ── Public entry-point (passed to asyncio.start_server) ──────────────────

    async def handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """
        Acquires a semaphore slot before processing so that a connection flood
        cannot exhaust file descriptors or RAM.  Excess connections are dropped.
        """
        if self._semaphore.locked():
            writer.close()
            return

        async with self._semaphore:
            await self._process(reader, writer)

    # ── Internal processing ───────────────────────────────────────────────────

    async def _process(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        start_time = time.monotonic()
        addr       = writer.get_extra_info("peername") or ("unknown", 0)
        ip, port   = addr[0], addr[1]

        try:
            # ── Record connection and get live rate count ──────────────────────
            # `count` = how many connections this IP has opened in the past 2s.
            # This is the scan-rate signal — a key indicator for port sweeps.
            count = self._rate_tracker.record(ip)

            # ── Read initial payload ───────────────────────────────────────────
            try:
                data      = await asyncio.wait_for(reader.read(4096), timeout=READ_TIMEOUT)
                src_bytes = len(data)
            except asyncio.TimeoutError:
                src_bytes = 0

            duration = time.monotonic() - start_time

            # ── Build feature vector ───────────────────────────────────────────
            # Order MUST match FEATURE_NAMES in models/train_model.py:
            # [duration, src_bytes, count, byte_rate, is_empty_flag]
            features = build_feature_vector(
                duration=duration,
                src_bytes=src_bytes,
                dst_bytes=0,    # not passed to model; kept in signature for clarity
                count=count,
            )

            # ── Classify (offloaded — never blocks the event loop) ─────────────
            loop = asyncio.get_event_loop()
            prediction, prob = await loop.run_in_executor(
                self._executor, self.classifier.predict, features
            )

            # ── Log to DB (offloaded) ──────────────────────────────────────────
            action = "TARPIT_DELAY" if prediction == 1 else "ALLOWED"
            await loop.run_in_executor(
                self._executor, self.db.log_event, ip, port, prediction, prob, action
            )

            # ── Act ────────────────────────────────────────────────────────────
            if prediction == 1:
                logger.warning(
                    f"PROBE  [{ip}:{port}] count={count} "
                    f"src_bytes={src_bytes} confidence={prob:.4f} → TARPITTED"
                )
                await self._drip(writer)
            else:
                logger.info(
                    f"BENIGN [{ip}:{port}] count={count} "
                    f"src_bytes={src_bytes} confidence={1-prob:.4f} → ALLOWED"
                )
                writer.write(
                    b"HTTP/1.1 200 OK\r\n"
                    b"Content-Type: text/plain\r\n"
                    b"Content-Length: 8\r\n"
                    b"\r\n"
                    b"Welcome.\n"
                )
                await writer.drain()

        except Exception as exc:
            logger.debug(f"Connection [{ip}:{port}] closed: {exc}")
        finally:
            if not writer.is_closing():
                writer.close()

    # ── Drip loop ─────────────────────────────────────────────────────────────

    async def _drip(self, writer: asyncio.StreamWriter) -> None:
        """
        Sends one null byte every DRIP_INTERVAL seconds to keep the attacker's
        connection open without sending meaningful data.  Exits cleanly when the
        remote end disconnects.
        """
        while not writer.is_closing():
            await asyncio.sleep(self.delay_base)
            try:
                writer.write(b"\0")
                await writer.drain()
            except (ConnectionResetError, BrokenPipeError):
                break