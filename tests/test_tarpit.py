"""
Async unit tests for tarpit/tarpit_engine.py

Run with:
    PYTHONPATH=. venv/bin/pytest tests/ -v
"""

import asyncio
import sys
import os
import unittest.mock as mock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tarpit.tarpit_engine import IntelligentTarpit, MAX_CONCURRENT


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_mock_classifier(prediction: int, prob: float):
    clf = mock.MagicMock()
    clf.predict.return_value = (prediction, prob)
    return clf


def _make_stream_pair(payload: bytes = b"GET / HTTP/1.1\r\n\r\n"):
    reader = mock.AsyncMock(spec=asyncio.StreamReader)
    reader.read = mock.AsyncMock(return_value=payload)

    writer = mock.MagicMock(spec=asyncio.StreamWriter)
    writer.get_extra_info.return_value = ("127.0.0.1", 12345)
    writer.is_closing.return_value     = False
    writer.drain                       = mock.AsyncMock()

    return reader, writer


def _make_tarpit(prediction: int, prob: float) -> IntelligentTarpit:
    clf = _make_mock_classifier(prediction, prob)
    with mock.patch("tarpit.tarpit_engine.TarpitDB"):
        return IntelligentTarpit(classifier=clf)


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestIntelligentTarpit:

    @pytest.mark.asyncio
    async def test_benign_connection_gets_200(self):
        tarpit = _make_tarpit(prediction=0, prob=0.001)
        reader, writer = _make_stream_pair(b"GET / HTTP/1.1\r\n\r\n")
        await tarpit._process(reader, writer)
        written = b"".join(call.args[0] for call in writer.write.call_args_list)
        assert b"HTTP/1.1 200 OK" in written

    @pytest.mark.asyncio
    async def test_malicious_connection_enters_drip(self):
        tarpit = _make_tarpit(prediction=1, prob=0.999)
        reader, writer = _make_stream_pair(b"")
        tarpit._drip = mock.AsyncMock()
        await tarpit._process(reader, writer)
        tarpit._drip.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_inference_called_with_five_features(self):
        tarpit = _make_tarpit(prediction=0, prob=0.001)
        reader, writer = _make_stream_pair(b"HELLO")
        await tarpit._process(reader, writer)
        assert tarpit.classifier.predict.called
        features = tarpit.classifier.predict.call_args[0][0]
        assert len(features) == 5, \
            f"Feature vector must have exactly 5 elements, got {len(features)}"

    @pytest.mark.asyncio
    async def test_empty_payload_sets_is_empty_flag(self):
        tarpit = _make_tarpit(prediction=0, prob=0.001)
        reader, writer = _make_stream_pair(b"")
        await tarpit._process(reader, writer)
        features      = tarpit.classifier.predict.call_args[0][0]
        is_empty_flag = features[4]  # index 4 in 5-feature vector
        assert is_empty_flag == 1.0

    @pytest.mark.asyncio
    async def test_nonempty_payload_clears_is_empty_flag(self):
        tarpit = _make_tarpit(prediction=0, prob=0.001)
        reader, writer = _make_stream_pair(b"GET / HTTP/1.1")
        await tarpit._process(reader, writer)
        features      = tarpit.classifier.predict.call_args[0][0]
        is_empty_flag = features[4]
        assert is_empty_flag == 0.0

    @pytest.mark.asyncio
    async def test_count_feature_is_nonzero_on_repeat_connections(self):
        """
        ConnectionRateTracker is now wired in — repeated connections from the
        same IP should produce an increasing count in the feature vector.
        """
        tarpit = _make_tarpit(prediction=0, prob=0.001)
        reader1, writer1 = _make_stream_pair(b"data")
        reader2, writer2 = _make_stream_pair(b"data")

        await tarpit._process(reader1, writer1)
        await tarpit._process(reader2, writer2)

        # Second call should have count >= 2 (same IP 127.0.0.1)
        features_second = tarpit.classifier.predict.call_args_list[-1][0][0]
        count_feature   = features_second[2]  # index 2 = count
        assert count_feature >= 2

    @pytest.mark.asyncio
    async def test_exception_in_classifier_does_not_crash_server(self):
        clf = mock.MagicMock()
        clf.predict.side_effect = RuntimeError("model exploded")
        with mock.patch("tarpit.tarpit_engine.TarpitDB"):
            tarpit = IntelligentTarpit(classifier=clf)
        reader, writer = _make_stream_pair(b"data")
        await tarpit._process(reader, writer)
        writer.close.assert_called()

    def test_semaphore_initialized_to_max_concurrent(self):
        tarpit = _make_tarpit(prediction=0, prob=0.001)
        assert tarpit._semaphore._value == MAX_CONCURRENT  # type: ignore[attr-defined]

    def test_rate_tracker_initialized(self):
        tarpit = _make_tarpit(prediction=0, prob=0.001)
        assert tarpit._rate_tracker is not None

    @pytest.mark.asyncio
    async def test_no_six_feature_vector_ever_sent(self):
        """Regression: ensure the old 6-feature vector is never constructed."""
        tarpit = _make_tarpit(prediction=0, prob=0.001)
        for payload in [b"", b"GET /", b"A" * 500]:
            reader, writer = _make_stream_pair(payload)
            await tarpit._process(reader, writer)

        for call in tarpit.classifier.predict.call_args_list:
            features = call[0][0]
            assert len(features) != 6, \
                "Old 6-feature vector detected — dst_bytes must be removed"