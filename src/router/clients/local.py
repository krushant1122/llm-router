"""Ollama HTTP client for local LLM inference.

Communicates with a locally-running Ollama server via its REST API
at http://localhost:11434. Supports multi-sample generation for
self-consistency routing and optional memory tracking via psutil.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from router.clients.base import LocalClient
from router.models import ModelResponse

logger = logging.getLogger(__name__)


def _sample_ollama_memory_mb() -> float | None:
    """Sample peak RSS memory used by Ollama processes.

    Looks for processes named 'ollama' or 'ollama_llama_server' and
    returns their combined RSS in megabytes. Returns None if psutil
    is unavailable or no Ollama process is found.
    """
    try:
        import psutil  # noqa: PLC0415 — optional dependency
    except ImportError:
        return None

    total_rss_bytes: int = 0
    target_names = {"ollama", "ollama_llama_server"}
    try:
        for proc in psutil.process_iter(["name", "memory_info"]):
            proc_name: str | None = proc.info.get("name")  # type: ignore[union-attr]
            if proc_name and proc_name.lower() in target_names:
                mem_info = proc.info.get("memory_info")  # type: ignore[union-attr]
                if mem_info is not None:
                    total_rss_bytes += mem_info.rss
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        pass

    return total_rss_bytes / (1024 * 1024) if total_rss_bytes > 0 else None


class OllamaLocalClient(LocalClient):
    """Ollama HTTP client for local model inference.

    Args:
        model: The Ollama model tag (e.g., ``"qwen2.5:7b"``).
        base_url: Base URL for the Ollama API server.
        timeout: HTTP request timeout in seconds. Model inference can be
            slow on CPU, so this defaults to a generous 300s.
        track_memory: Whether to sample Ollama process memory after each call.
    """

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        base_url: str = "http://localhost:11434",
        timeout: float = 300.0,
        track_memory: bool = True,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._track_memory = track_memory
        self._http = httpx.Client(base_url=self._base_url, timeout=self._timeout)

    # ── LocalClient interface ───────────────────────────────────────────

    @property
    def model_name(self) -> str:
        """Return the Ollama model tag."""
        return self._model

    def is_available(self) -> bool:
        """Ping ``GET /api/tags`` to check server availability."""
        try:
            resp = self._http.get("/api/tags", timeout=5.0)
            return resp.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPError):
            return False

    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        n: int = 1,
    ) -> list[ModelResponse]:
        """Generate ``n`` independent responses from Ollama.

        Each sample is a separate API call (Ollama serves one request at
        a time anyway). For self-consistency, use ``n > 1`` with
        ``temperature > 0``.
        """
        responses: list[ModelResponse] = []
        for i in range(n):
            resp = self._single_generate(prompt, temperature, max_tokens, sample_idx=i)
            responses.append(resp)
        return responses

    # ── Internal helpers ────────────────────────────────────────────────

    def _single_generate(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        sample_idx: int = 0,
    ) -> ModelResponse:
        """Execute a single Ollama ``/api/generate`` call."""
        payload: dict[str, Any] = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        t0 = time.perf_counter()
        try:
            http_resp = self._http.post("/api/generate", json=payload)
            http_resp.raise_for_status()
            data: dict[str, Any] = http_resp.json()
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            elapsed = time.perf_counter() - t0
            logger.error("Ollama connection error (sample %d): %s", sample_idx, exc)
            return ModelResponse(
                text="",
                model=self._model,
                input_tokens=0,
                output_tokens=0,
                latency_seconds=elapsed,
                metadata={"error": str(exc), "sample_idx": sample_idx},
            )
        except httpx.HTTPStatusError as exc:
            elapsed = time.perf_counter() - t0
            logger.error(
                "Ollama HTTP %s (sample %d): %s",
                exc.response.status_code,
                sample_idx,
                exc,
            )
            return ModelResponse(
                text="",
                model=self._model,
                input_tokens=0,
                output_tokens=0,
                latency_seconds=elapsed,
                metadata={
                    "error": str(exc),
                    "status_code": exc.response.status_code,
                    "sample_idx": sample_idx,
                },
            )

        latency = time.perf_counter() - t0

        # Parse Ollama response fields
        text: str = data.get("response", "")
        eval_count: int = data.get("eval_count", 0)
        prompt_eval_count: int = data.get("prompt_eval_count", 0)
        eval_duration_ns: int = data.get("eval_duration", 0)
        total_duration_ns: int = data.get("total_duration", 0)

        # Build metadata dict
        metadata: dict[str, Any] = {
            "sample_idx": sample_idx,
            "eval_duration_ms": eval_duration_ns / 1_000_000 if eval_duration_ns else 0,
            "total_duration_ms": total_duration_ns / 1_000_000 if total_duration_ns else 0,
        }

        # Compute tokens-per-second if we have timing data
        if eval_duration_ns > 0 and eval_count > 0:
            tokens_per_sec = eval_count / (eval_duration_ns / 1e9)
            metadata["tokens_per_second"] = round(tokens_per_sec, 1)

        # Optionally sample memory usage
        if self._track_memory:
            mem_mb = _sample_ollama_memory_mb()
            if mem_mb is not None:
                metadata["ollama_memory_mb"] = round(mem_mb, 1)

        return ModelResponse(
            text=text,
            model=self._model,
            input_tokens=prompt_eval_count,
            output_tokens=eval_count,
            latency_seconds=latency,
            metadata=metadata,
        )

    def __del__(self) -> None:
        """Close the HTTP client on garbage collection."""
        try:
            self._http.close()
        except Exception:  # noqa: BLE001
            pass
