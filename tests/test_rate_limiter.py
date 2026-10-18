"""Unit tests for client request tracking, budget caps, and availability."""

from __future__ import annotations

import pytest

from router.clients.mock import MockFrontierClient, MockLocalClient
from router.config import AppConfig


def test_mock_clients_availability():
    loc = MockLocalClient()
    front = MockFrontierClient()
    assert loc.is_available()
    assert front.is_available()


def test_request_counter_tracking():
    client = MockFrontierClient()
    assert client.requests_made == 0
    client.generate("prompt 1")
    assert client.requests_made == 1
    client.generate("prompt 2")
    assert client.requests_made == 2


def test_frontier_client_daily_cap(config: AppConfig):
    from router.clients.frontier import GeminiFrontierClient

    # Set daily request cap to 2
    config.rate_limit.daily_request_cap = 2
    client = GeminiFrontierClient(config, api_key="fake-key-for-test")
    client._requests_made_count = 2

    with pytest.raises(RuntimeError, match="Daily request cap"):
        client.generate("Any prompt")
