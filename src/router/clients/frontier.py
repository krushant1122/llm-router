"""Gemini frontier client with rate limiting, retry, fallback, and budget control.

Uses the ``google-genai`` SDK to call Gemini models. Implements:
- Model fallback: if the primary model returns 429/404/quota errors, switch
  to the configured fallback model for the remainder of the session.
- Rate limiting: enforces ``requests_per_minute`` with sleep-based throttling.
- Retry: exponential backoff with jitter on 429 / 5xx errors.
- Budget cap: aborts if ``daily_request_cap`` is exceeded.
- Cost estimation: computes equivalent dollar cost from token counts and pricing.
"""

from __future__ import annotations

import logging
import os
import random
import time
from typing import Any

from router.clients.base import FrontierClient
from router.config import AppConfig
from router.models import ModelResponse

logger = logging.getLogger(__name__)

# Retryable error signals (checked against exception message / status)
_RETRYABLE_KEYWORDS = frozenset({"429", "500", "502", "503", "resource exhausted", "quota"})
_FALLBACK_KEYWORDS = frozenset({"429", "404", "not found", "quota"})


class GeminiFrontierClient(FrontierClient):
    """Gemini frontier client with production-grade resilience.

    Args:
        config: Application configuration containing model names, pricing,
            rate limits, and budget caps.
        api_key: Explicit API key. If ``None``, reads from the env var specified in config.
    """

    def __init__(self, config: AppConfig, api_key: str | None = None) -> None:
        self._config = config
        self._api_key = api_key or os.environ.get(config.frontier.api_key_env, "")

        # Model selection
        frontier_cfg = config.frontier
        self._primary_model: str = frontier_cfg.primary
        self._fallback_model: str = frontier_cfg.fallback
        self._active_model: str = self._primary_model
        self._fell_back: bool = False

        # Rate limiting state
        self._rpm_limit: int = config.rate_limit.requests_per_minute
        self._request_timestamps: list[float] = []

        # Budget tracking
        self._daily_cap: int = config.rate_limit.daily_request_cap
        self._requests_made_count: int = 0

        # Retry config
        self._max_retries: int = config.rate_limit.retry_max_attempts
        self._base_delay: float = config.rate_limit.retry_base_delay
        self._max_delay: float = config.rate_limit.retry_max_delay

        # Lazy SDK client — created on first use so import errors are deferred
        self._client: Any = None

    # ── FrontierClient interface ────────────────────────────────────────

    @property
    def model_name(self) -> str:
        """Return the currently active model identifier."""
        return self._active_model

    @property
    def requests_made(self) -> int:
        """Total API requests made this session."""
        return self._requests_made_count

    def is_available(self) -> bool:
        """Check that GEMINI_API_KEY is set (non-empty)."""
        return bool(self._api_key)

    def _get_pricing(self, model: str) -> tuple[float, float, float]:
        """Return (input_price_per_token, output_price_per_token, thinking_price_per_token)."""
        pricing = self._config.pricing.models.get(model)
        if pricing:
            return (
                pricing.input_per_1m / 1_000_000,
                pricing.output_per_1m / 1_000_000,
                pricing.thinking_per_1m / 1_000_000,
            )
        # Default fallback pricing if model not in pricing table
        return (1.25 / 1_000_000, 10.0 / 1_000_000, 3.75 / 1_000_000)

    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> ModelResponse:
        """Generate a response from Gemini with retry, rate limiting, and fallback.

        Raises:
            RuntimeError: If the daily request cap has been exceeded.
        """
        # Budget guard
        if self._requests_made_count >= self._daily_cap:
            raise RuntimeError(
                f"Daily request cap ({self._daily_cap}) exceeded. "
                f"Made {self._requests_made_count} requests today."
            )

        # Rate-limit: sleep if necessary
        self._enforce_rate_limit()

        # Retry loop with exponential backoff
        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = self._call_api(prompt, temperature, max_tokens)
                self._requests_made_count += 1
                self._request_timestamps.append(time.monotonic())
                return response
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                err_msg = str(exc).lower()

                # Check if we should fall back to another model
                if not self._fell_back and any(kw in err_msg for kw in _FALLBACK_KEYWORDS):
                    logger.warning(
                        "Primary model %s error: %s — falling back to %s",
                        self._primary_model,
                        exc,
                        self._fallback_model,
                    )
                    self._active_model = self._fallback_model
                    self._fell_back = True
                    # Try immediately with fallback model
                    continue

                # Check if the error is retryable
                if any(kw in err_msg for kw in _RETRYABLE_KEYWORDS):
                    if attempt < self._max_retries:
                        delay = self._backoff_delay(attempt)
                        logger.warning(
                            "Retryable error (attempt %d/%d): %s — sleeping %.1fs",
                            attempt + 1,
                            self._max_retries,
                            exc,
                            delay,
                        )
                        time.sleep(delay)
                        continue

                # Non-retryable error — break out
                logger.error("Non-retryable Gemini error: %s", exc)
                break

        # All retries exhausted — return an error response
        assert last_error is not None
        logger.error("Gemini call failed after %d attempts: %s", self._max_retries + 1, last_error)
        self._requests_made_count += 1
        return ModelResponse(
            text=f"ERROR: {last_error}",
            model_name=self._active_model,
            input_tokens=0,
            output_tokens=0,
            latency_seconds=0.0,
            equivalent_cost_usd=0.0,
        )

    # ── Internal helpers ────────────────────────────────────────────────

    def _get_client(self) -> Any:
        """Lazily initialise the google-genai SDK client."""
        if self._client is None:
            from google import genai  # noqa: PLC0415

            self._client = genai.Client(api_key=self._api_key)
        return self._client

    def _call_api(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> ModelResponse:
        """Execute a single Gemini API call and parse the response."""
        from google.genai import types  # noqa: PLC0415

        client = self._get_client()

        t0 = time.perf_counter()
        response = client.models.generate_content(
            model=self._active_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )
        latency = time.perf_counter() - t0

        # Extract text
        text: str = response.text or ""

        # Extract token counts from usage_metadata
        usage = getattr(response, "usage_metadata", None)
        input_tokens: int = getattr(usage, "prompt_token_count", 0) or 0
        candidates_tokens: int = getattr(usage, "candidates_token_count", 0) or 0
        thinking_tokens: int = getattr(usage, "thoughts_token_count", 0) or 0
        output_tokens: int = candidates_tokens + thinking_tokens

        # Compute equivalent cost
        in_p, out_p, think_p = self._get_pricing(self._active_model)
        equivalent_cost = (
            (input_tokens * in_p) + (candidates_tokens * out_p) + (thinking_tokens * think_p)
        )

        logger.info(
            "Gemini [%s] %d→%d tokens (thinking=%d), %.2fs, $%.6f",
            self._active_model,
            input_tokens,
            output_tokens,
            thinking_tokens,
            latency,
            equivalent_cost,
        )

        return ModelResponse(
            text=text,
            model_name=self._active_model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            thinking_tokens=thinking_tokens,
            latency_seconds=latency,
            equivalent_cost_usd=equivalent_cost,
        )

    def _enforce_rate_limit(self) -> None:
        """Sleep if we'd exceed the requests-per-minute limit."""
        if not self._request_timestamps:
            return

        now = time.monotonic()
        window_start = now - 60.0

        # Prune timestamps older than 60 seconds
        self._request_timestamps = [ts for ts in self._request_timestamps if ts > window_start]

        if len(self._request_timestamps) >= self._rpm_limit:
            oldest_in_window = self._request_timestamps[0]
            sleep_time = oldest_in_window - window_start + 0.1
            if sleep_time > 0:
                logger.info("Rate limit: sleeping %.1fs", sleep_time)
                time.sleep(sleep_time)

    def _backoff_delay(self, attempt: int) -> float:
        """Compute exponential backoff delay with jitter."""
        base = self._base_delay * (2**attempt)
        jitter = base * (0.5 + random.random())  # noqa: S311
        return min(jitter, self._max_delay)
