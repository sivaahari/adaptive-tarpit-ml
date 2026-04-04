"""
Core tarpit engine.

Key fixes vs. original:
  - ML inference and DB writes are offloaded to a ThreadPoolExecutor so they
    never block the asyncio event loop.
  - A Semaphore caps simultaneous trapped sessions, preventing resource
    exhaustion (the original code was itself a DoS vector).
  - Feature vector order now matches FEATURE_NAMES in train_model.py exactly.
  - Exceptions are logged (not silently swallowed).
  - Connections held open indefinitely include a writer-closed guard.
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from loguru import logger

from logging_system.database import TarpitDB

# ── Tuneable constants ────────────────────────────────────────────────────────
MAX_CONCURRENT   = 500    # semaphore cap; prevents self-DoS from flood attacks
READ_TIMEOUT     = 1.0    # seconds to wait for the client's first byte
DRIP_INTERVAL    = 10.0   # seconds between each null-byte drip to a tarpitted conn
EXECUTOR_WORKERS = 4      # thread-pool size for blocking I/O (inference + SQLite)


class IntelligentTarpit:
    def __init__(self, delay_base: float = DRIP_INTERVAL, classifier=None):
        self.delay_base  = delay_base
        self.classifier  = classifier
        self.db          = TarpitDB()
        self._semaphore  = asyncio.Semaphore(MAX_CONCURRENT)
        self._executor   = ThreadPoolExecutor(max_workers=EXECUTOR_WORKERS)

    # ── Public entry-point (passed to asyncio.start_server) ──────────────────

    async def handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """
        Acquires a slot from the semaphore before processing so that a flood
        of incoming connections cannot exhaust file descriptors or RAM.
        Connections that exceed MAX_CONCURRENT are dropped immediately.
        """
        acquired = await asyncio.wait_for(
            self._semaphore.acquire(), timeout=0.1
        ) if not self._semaphore.locked() else False

        if not self._semaphore._value and not acquired:   # type: ignore[attr-defined]
            writer.close()
            return

        # Use acquire properly
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
            # ── Read initial payload ──────────────────────────────────────────
            try:
                data      = await asyncio.wait_for(reader.read(4096), timeout=READ_TIMEOUT)
                src_bytes = len(data)
            except asyncio.TimeoutError:
                src_bytes = 0

            duration     = time.monotonic() - start_time
            is_empty     = 1.0 if src_bytes == 0 else 0.0

            # ── Build feature vector ──────────────────────────────────────────
            # Order MUST match FEATURE_NAMES in models/train_model.py:
            # [duration, src_bytes, dst_bytes, count, byte_rate, is_empty_flag]
            #
            # Notes on unmeasurable features at connection time:
            #   dst_bytes → 0.0  (no response sent yet)
            #   count     → 0.0  (would require cross-connection session tracking;
            #                      see network/feature_extractor.py for a future impl)
            features = [
                float(duration),
                float(src_bytes),
                0.0,                                          # dst_bytes
                0.0,                                          # count
                float((src_bytes) / (duration + 1e-6)),      # byte_rate
                is_empty,                                     # is_empty_flag
            ]

            # ── Classify (offloaded — never blocks the event loop) ────────────
            loop = asyncio.get_event_loop()
            prediction, prob = await loop.run_in_executor(
                self._executor, self.classifier.predict, features
            )

            # ── Log to DB (offloaded) ─────────────────────────────────────────
            action = "TARPIT_DELAY" if prediction == 1 else "ALLOWED"
            await loop.run_in_executor(
                self._executor, self.db.log_event, ip, port, prediction, prob, action
            )

            # ── Act ───────────────────────────────────────────────────────────
            if prediction == 1:
                logger.warning(
                    f"PROBE  [{ip}:{port}] confidence={prob:.4f} → TARPITTED"
                )
                await self._drip(writer)
            else:
                logger.info(
                    f"BENIGN [{ip}:{port}] {src_bytes}B "
                    f"confidence={1 - prob:.4f} → ALLOWED"
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