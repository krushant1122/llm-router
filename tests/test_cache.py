"""Unit tests for on-disk response cache and key hashing."""

from __future__ import annotations

from router.clients.cache import ResponseCache
from router.models import CacheKey, ModelResponse


def test_cache_put_get(tmp_cache: ResponseCache):
    key = CacheKey(model="qwen", prompt="What is 2+2?", temperature=0.0)
    resp = ModelResponse(text="4", model_name="qwen", input_tokens=4, output_tokens=1)

    assert not tmp_cache.has(key)
    tmp_cache.put(key, resp)
    assert tmp_cache.has(key)

    retrieved = tmp_cache.get(key)
    assert retrieved is not None
    assert retrieved.text == "4"
    assert retrieved.input_tokens == 4


def test_cache_miss(tmp_cache: ResponseCache):
    key = CacheKey(model="gemini", prompt="Non-existent prompt")
    assert tmp_cache.get(key) is None


def test_cache_temperature_distinction(tmp_cache: ResponseCache):
    k0 = CacheKey(model="m", prompt="p", temperature=0.0)
    k1 = CacheKey(model="m", prompt="p", temperature=0.7)

    tmp_cache.put(k0, ModelResponse(text="deterministic", model_name="m"))
    tmp_cache.put(k1, ModelResponse(text="stochastic", model_name="m"))

    assert tmp_cache.get(k0).text == "deterministic"
    assert tmp_cache.get(k1).text == "stochastic"


def test_cache_key_hash_stability():
    k1 = CacheKey(model="gemini-3.1-pro-preview", prompt="Hello", temperature=0.0)
    k2 = CacheKey(model="gemini-3.1-pro-preview", prompt="Hello", temperature=0.0)
    k3 = CacheKey(model="gemini-3.8-flash", prompt="Hello", temperature=0.0)

    assert k1.to_hash() == k2.to_hash()
    assert k1.to_hash() != k3.to_hash()
