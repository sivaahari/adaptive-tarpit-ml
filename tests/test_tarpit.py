"""
Async unit tests for tarpit/tarpit_engine.py

Run with:
    pytest tests/ -v
"""

import asyncio
import sys
import os
import unittest.mock as mock

import pytest
import pytest_asyncio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tarpit.tarpit_engine import IntelligentTarpit, MAX_CONCURRENT


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_mock_classifier(prediction: int, prob: float):
    """Return a mock classifier that always returns (prediction, prob)."""
    clf = mock.MagicMock()
    clf.predict.return_value = (prediction, prob)
    return clf


def _make_stream_pair(payload: bytes = b"GET / HTTP/1.1\r\n\r\n"):
    """
    Create a fake (reader, writer) pair that immediately returns `payload`
    then EOF, and a writer that records what was written to it.
    """
    reader = mock.AsyncMock(spec=asyncio.StreamReader)
    reader.read = mock.AsyncMock(return_value=payload)

    writer = mock.MagicMock(spec=asyncio.StreamWriter)
    writer.get_extra_info.return_value = ("127.0.0.1", 12345)
    writer.is_closing.return_value     = False
    writer.drain                       = mock.AsyncMock()

    return reader, writer


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestIntelligentTarpit:
    @pytest.fixture()
    def benign_tarpit(self):
        clf = _make_mock_classifier(prediction=0, prob=0.001)
        with mock.patch("tarpit.tarpit_engine.TarpitDB"):
            return IntelligentTarpit(classifier=clf)

    @pytest.fixture()
    def malicious_tarpit(self):
        clf = _make_mock_classifier(prediction=1, prob=0.999)
        with mock.patch("tarpit.tarpit_engine.TarpitDB"):
            return IntelligentTarpit(classifier=clf)

    @pytest.mark.asyncio
    async def test_benign_connection_gets_200(self, benign_tarpit):
        reader, writer = _make_stream_pair(b"GET / HTTP/1.1\r\n\r\n")
        await benign_tarpit._process(reader, writer)
        written = b"".join(
            call.args[0] for call in writer.write.call_args_list
        )
        assert b"HTTP/1.1 200 OK" in written

    @pytest.mark.asyncio
    async def test_malicious_connection_is_closed_eventually(self, malicious_tarpit):
        reader, writer = _make_stream_pair(b"")   # empty payload → probe
        # Override _drip to return immediately so the test doesn't hang
        malicious_tarpit._drip = mock.AsyncMock()
        await malicious_tarpit._process(reader, writer)
        malicious_tarpit._drip.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_inference_called_with_six_features(self, benign_tarpit):
        reader, writer = _make_stream_pair(b"HELLO")
        await benign_tarpit._process(reader, writer)
        # classifier.predict is called by run_in_executor; check via mock
        assert benign_tarpit.classifier.predict.called
        features = benign_tarpit.classifier.predict.call_args[0][0]
        assert len(features) == 6, "Feature vector must have exactly 6 elements"

    @pytest.mark.asyncio
    async def test_empty_payload_sets_is_empty_flag(self, benign_tarpit):
        reader, writer = _make_stream_pair(b"")
        await benign_tarpit._process(reader, writer)
        features = benign_tarpit.classifier.predict.call_args[0][0]
        is_empty_flag = features[5]
        assert is_empty_flag == 1.0

    @pytest.mark.asyncio
    async def test_nonempty_payload_clears_is_empty_flag(self, benign_tarpit):
        reader, writer = _make_stream_pair(b"GET / HTTP/1.1")
        await benign_tarpit._process(reader, writer)
        features      = benign_tarpit.classifier.predict.call_args[0][0]
        is_empty_flag = features[5]
        assert is_empty_flag == 0.0

    @pytest.mark.asyncio
    async def test_exception_in_classifier_does_not_crash_server(self):
        clf = mock.MagicMock()
        clf.predict.side_effect = RuntimeError("model exploded")
        with mock.patch("tarpit.tarpit_engine.TarpitDB"):
            tarpit = IntelligentTarpit(classifier=clf)
        reader, writer = _make_stream_pair(b"data")
        # Should not raise — exception is caught and logged
        await tarpit._process(reader, writer)
        writer.close.assert_called()

    def test_semaphore_initialized_to_max_concurrent(self, benign_tarpit):
        assert benign_tarpit._semaphore._value == MAX_CONCURRENT  # type: ignore[attr-defined]