"""Deterministic mock clients for offline testing.

These clients return predictable, prompt-dependent responses so that the
routing pipeline can be tested end-to-end without a running Ollama server
or Gemini API key. The mock frontier client intentionally returns different
answers than the mock local client (e.g., different multiple-choice picks)
to exercise routing logic that compares outputs.
"""

from __future__ import annotations

import re

from router.clients.base import FrontierClient, LocalClient
from router.models import ModelResponse


def _classify_prompt(prompt: str) -> str:
    """Heuristically classify a prompt into a category.

    Returns one of: ``"math"``, ``"code"``, ``"multiple_choice"``, ``"general"``.
    """
    lower = prompt.lower()

    # Multiple-choice: contains option markers like (A), A., A)
    if re.search(r"\b[ABCD]\)", prompt) or re.search(r"\([ABCD]\)", prompt):
        return "multiple_choice"

    # Math: contains digits with arithmetic operators or common math words
    if re.search(r"\d\s*[+\-*/^%]\s*\d", prompt) or any(
        kw in lower for kw in ("calculate", "compute", "solve", "equation", "sum of", "integral")
    ):
        return "math"

    # Code: contains function definitions or code-like keywords
    if any(kw in prompt for kw in ("def ", "function ", "class ", "import ", "return ")):
        return "code"

    return "general"


def _try_eval_simple_math(prompt: str) -> str | None:
    """Attempt to evaluate simple arithmetic expressions in the prompt.

    Returns the result as a string if a safe expression is found, else ``None``.
    """
    # Look for patterns like "2 + 3", "100 * 7", "45 / 9"
    match = re.search(r"(\d+(?:\.\d+)?)\s*([+\-*/])\s*(\d+(?:\.\d+)?)", prompt)
    if match:
        a, op, b = match.group(1), match.group(2), match.group(3)
        try:
            a_f, b_f = float(a), float(b)
            result = {
                "+": a_f + b_f,
                "-": a_f - b_f,
                "*": a_f * b_f,
                "/": a_f / b_f if b_f != 0 else float("inf"),
            }[op]
            # Return as int if the result is a whole number
            if result == int(result):
                return str(int(result))
            return str(result)
        except (ValueError, ZeroDivisionError):
            return None
    return None


# ═══════════════════════════════════════════════════════════════════════
#  Mock Local Client
# ═══════════════════════════════════════════════════════════════════════


class MockLocalClient(LocalClient):
    """Returns deterministic responses based on prompt content for offline testing.

    - Math prompts → a plausible but possibly wrong answer (always returns ``"17"``
      unless it's a trivially evaluable expression, in which case it returns a
      slightly wrong answer to simulate local model weakness).
    - Code prompts → a simple function stub.
    - Multiple-choice → always ``"A"``.
    - General prompts → a fixed mock string.

    For ``n > 1`` samples, responses are varied by appending ``" (variation N)"``.
    """

    def __init__(self, model: str = "mock-local-7b") -> None:
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model

    def is_available(self) -> bool:
        return True

    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        n: int = 1,
    ) -> list[ModelResponse]:
        category = _classify_prompt(prompt)
        base_text = self._make_response(prompt, category)
        input_tokens = len(prompt.split())

        responses: list[ModelResponse] = []
        for i in range(n):
            text = base_text if i == 0 else f"{base_text} (variation {i + 1})"
            output_tokens = len(text.split())
            responses.append(
                ModelResponse(
                    text=text,
                    model=self._model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    latency_seconds=0.1,
                    metadata={"category": category, "sample_idx": i},
                )
            )
        return responses

    @staticmethod
    def _make_response(prompt: str, category: str) -> str:
        if "sub_tasks" in prompt or "decompose" in prompt.lower():
            return '{"sub_tasks": [{"id": 1, "text": "Solve first part", "depends_on": [], "context_facts": []}, {"id": 2, "text": "Solve second part", "depends_on": [1], "context_facts": []}]}'

        if category == "math":
            # Try to evaluate, but give a slightly wrong answer to simulate
            # a weaker local model
            result = _try_eval_simple_math(prompt)
            if result is not None:
                # Intentionally off by 1 to simulate local model errors
                try:
                    return str(int(result) + 1)
                except ValueError:
                    return str(float(result) + 0.5)
            return "17"

        if category == "code":
            return 'def solution():\n    return "TODO: implement"'

        if category == "multiple_choice":
            return "A"

        return "This is a mock local response."


# ═══════════════════════════════════════════════════════════════════════
#  Mock Frontier Client
# ═══════════════════════════════════════════════════════════════════════


class MockFrontierClient(FrontierClient):
    """Returns high-quality deterministic responses for offline testing.

    - Math prompts → correct answer (via eval for simple expressions, else ``"42"``).
    - Code prompts → plausible correct-looking code.
    - Multiple-choice → ``"B"`` (intentionally different from local's ``"A"``).
    - General → a detailed mock response string.

    Token counts are inflated relative to the local mock to simulate the
    richer tokenisation of frontier models, and a small non-zero
    ``equivalent_cost`` is always included.
    """

    def __init__(self, model: str = "mock-frontier-pro") -> None:
        self._model = model
        self._requests_made_count: int = 0

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def requests_made(self) -> int:
        return self._requests_made_count

    def is_available(self) -> bool:
        return True

    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> ModelResponse:
        self._requests_made_count += 1
        category = _classify_prompt(prompt)
        text = self._make_response(prompt, category)

        input_tokens = len(prompt.split()) * 2
        output_tokens = len(text.split()) * 3
        thinking_tokens = output_tokens // 4
        # Small but non-zero cost for budget tracking tests
        equivalent_cost = (input_tokens * 0.15 + output_tokens * 0.60) / 1_000_000

        return ModelResponse(
            text=text,
            model=self._model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_seconds=0.5,
            thinking_tokens=thinking_tokens,
            equivalent_cost=equivalent_cost,
            metadata={"category": category},
        )

    @staticmethod
    def _make_response(prompt: str, category: str) -> str:
        if "sub_tasks" in prompt or "decompose" in prompt.lower():
            return '{"sub_tasks": [{"id": 1, "text": "Solve first part", "depends_on": [], "context_facts": []}, {"id": 2, "text": "Solve second part", "depends_on": [1], "context_facts": []}]}'

        if category == "math":
            result = _try_eval_simple_math(prompt)
            if result is not None:
                return result  # Correct answer
            return "42"

        if category == "code":
            return (
                "def solution(data: list) -> list:\n"
                '    """Solve the problem efficiently."""\n'
                "    result = sorted(set(data))\n"
                "    return result"
            )

        if category == "multiple_choice":
            return "B"

        return "This is a comprehensive mock frontier response with detailed analysis."
