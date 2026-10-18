"""Abstract base classes for LLM clients.

Defines the contract for local (Ollama) and frontier (Gemini) model clients.
Both produce ModelResponse objects with standardized metadata for routing decisions.
"""

from abc import ABC, abstractmethod

from router.models import ModelResponse


class LocalClient(ABC):
    """Abstract base class for local LLM clients (e.g., Ollama).

    Local clients support generating multiple responses for self-consistency
    voting via the ``n`` parameter. They run on the user's hardware with
    no API cost.
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        n: int = 1,
    ) -> list[ModelResponse]:
        """Generate n responses from the local model.

        For self-consistency routing, call with ``n > 1`` and ``temperature > 0``
        to get diverse samples for majority voting.

        Args:
            prompt: The input prompt to send to the model.
            temperature: Sampling temperature. 0.0 for greedy decoding.
            max_tokens: Maximum tokens to generate per response.
            n: Number of independent responses to generate.

        Returns:
            A list of ``n`` ModelResponse objects, one per sample.
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check whether the local model server is reachable and ready.

        Returns:
            True if the server is running and responsive.
        """
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the canonical model identifier (e.g., 'qwen2.5:7b')."""
        ...


class FrontierClient(ABC):
    """Abstract base class for frontier / cloud LLM clients (e.g., Gemini).

    Frontier clients produce a single high-quality response per call and
    track usage for budget accounting.
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> ModelResponse:
        """Generate a single response from the frontier model.

        Args:
            prompt: The input prompt to send to the model.
            temperature: Sampling temperature. 0.0 for greedy decoding.
            max_tokens: Maximum tokens to generate.

        Returns:
            A single ModelResponse with cost and token metadata.
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check whether the frontier API is reachable (e.g., API key is set).

        Returns:
            True if the service can be called.
        """
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the canonical model identifier (e.g., 'gemini-2.5-flash')."""
        ...

    @property
    @abstractmethod
    def requests_made(self) -> int:
        """Return the total number of API requests made during this session."""
        ...
