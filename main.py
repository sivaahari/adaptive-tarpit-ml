"""
Adaptive ML Tarpit — entry point.

Usage:
    sudo PYTHONPATH=. python3 main.py
"""

import asyncio
from loguru import logger

from detection.classifier import TrafficClassifier
from tarpit.tarpit_engine import IntelligentTarpit, MAX_CONCURRENT

HOST = "0.0.0.0"
PORT = 8080


async def start_tarpit_server() -> None:
    logger.info("Loading ML classifier …")
    classifier = TrafficClassifier()

    tarpit = IntelligentTarpit(delay_base=10.0, classifier=classifier)

    server = await asyncio.start_server(
        tarpit.handle_connection, HOST, PORT
    )
    addrs = ", ".join(str(s.getsockname()) for s in server.sockets)
    logger.info(f"Tarpit live on {addrs}  (max {MAX_CONCURRENT} concurrent sessions)")
    logger.info("Press Ctrl-C to stop.")

    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    try:
        asyncio.run(start_tarpit_server())
    except KeyboardInterrupt:
        logger.info("Tarpit deactivated.")