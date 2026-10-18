"""Shared pytest fixtures for llm-router."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure src is on sys.path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eval.data_build import build_fixture_tasks
from router.clients.cache import ResponseCache
from router.clients.mock import MockFrontierClient, MockLocalClient
from router.config import AppConfig


@pytest.fixture
def config() -> AppConfig:
    """Return default application configuration."""
    return AppConfig()


@pytest.fixture
def mock_local() -> MockLocalClient:
    """Return a deterministic mock local client."""
    return MockLocalClient()


@pytest.fixture
def mock_frontier() -> MockFrontierClient:
    """Return a deterministic mock frontier client."""
    return MockFrontierClient()


@pytest.fixture
def tmp_cache(tmp_path: Path) -> ResponseCache:
    """Return a cache instance backed by a temporary directory."""
    return ResponseCache(tmp_path / "cache")


@pytest.fixture
def fixture_tasks():
    """Return the 10-task offline fixture suite."""
    return build_fixture_tasks()
