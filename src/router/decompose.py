"""Prompt decomposition for hybrid local/frontier solving.

Complex prompts are broken into independent sub-tasks.  Each sub-task
is first attempted by the local model; only sub-tasks that fail quality
checks are escalated to the frontier — minimising token spend while
preserving answer quality.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from router.clients.base import FrontierClient, LocalClient
from router.config import AppConfig
from router.models import (
    DecompositionResult,
    ModelResponse,
    SubTask,
    Task,
)
from router.signals.verifier import verifier_signal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

DECOMPOSE_PROMPT = """\
Analyze the following prompt and break it into independent sub-tasks.
Return a JSON object with this EXACT schema:
{{
  "sub_tasks": [
    {{
      "id": 1,
      "text": "the sub-task description",
      "depends_on": [],
      "context_facts": ["any specific facts from the original prompt needed"]
    }}
  ]
}}

If the prompt cannot be meaningfully decomposed, return:
{{"sub_tasks": [{{"id": 1, "text": "<original prompt>", "depends_on": [], "context_facts": []}}]}}

Prompt to decompose:
{prompt}

JSON response:"""


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def parse_decomposition(text: str, original_prompt: str) -> DecompositionResult:
    """Parse a JSON decomposition from model output, with graceful fallback."""
    try:
        data: Any = None

        # 1. Look for markdown code fence with JSON
        code_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if code_match:
            try:
                data = json.loads(code_match.group(1))
            except Exception:
                pass

        # 2. Look for outermost balanced { ... } containing sub_tasks
        if not isinstance(data, dict):
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    data = json.loads(text[start : end + 1])
                except Exception:
                    pass

        # 3. Try parsing text directly
        if not isinstance(data, dict):
            try:
                candidate = json.loads(text.strip())
                if isinstance(candidate, dict):
                    data = candidate
            except Exception:
                pass

        if not isinstance(data, dict) or "sub_tasks" not in data:
            raise ValueError("No valid dictionary containing 'sub_tasks' found")

        raw_list = data.get("sub_tasks")
        if not isinstance(raw_list, list):
            raise ValueError("'sub_tasks' is not a list")

        sub_tasks: list[SubTask] = []
        for st in raw_list:
            if not isinstance(st, dict):
                continue
            sub_tasks.append(
                SubTask(
                    id=st.get("id", len(sub_tasks) + 1),
                    text=st.get("text", ""),
                    depends_on=st.get("depends_on", []),
                    context_facts=st.get("context_facts", []),
                )
            )

        if not sub_tasks:
            raise ValueError("No valid sub-tasks found in sub_tasks list")

        decomposed = len(sub_tasks) > 1
        return DecompositionResult(
            original_prompt=original_prompt,
            sub_tasks=sub_tasks,
            decomposed=decomposed,
        )

    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
        logger.warning(
            "Decomposition parse failed: %s. Falling back to no decomposition.",
            exc,
        )
        return DecompositionResult(
            original_prompt=original_prompt,
            sub_tasks=[SubTask(id=1, text=original_prompt, depends_on=[], context_facts=[])],
            decomposed=False,
        )


# ---------------------------------------------------------------------------
# Context / merge helpers
# ---------------------------------------------------------------------------


def build_minimal_context(sub_task: SubTask, all_results: dict[int, str]) -> str:
    """Build minimal context for a sub-task.

    Includes only the sub-task text, its ``context_facts``, and the
    results of any declared dependencies.
    """
    parts: list[str] = []
    if sub_task.context_facts:
        parts.append("Context: " + "; ".join(sub_task.context_facts))
    for dep_id in sub_task.depends_on:
        if dep_id in all_results:
            parts.append(f"Result from step {dep_id}: {all_results[dep_id]}")
    parts.append(sub_task.text)
    return "\n".join(parts)


def merge_results(sub_results: dict[int, str], original_prompt: str) -> str:
    """Merge sub-task results into a single final answer."""
    if len(sub_results) == 1:
        return list(sub_results.values())[0]

    parts: list[str] = []
    for task_id in sorted(sub_results.keys()):
        parts.append(sub_results[task_id])
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Decomposer
# ---------------------------------------------------------------------------


class Decomposer:
    """Decomposes complex prompts and solves sub-tasks adaptively.

    Parameters
    ----------
    config:
        Application configuration.
    local:
        Client for the local model.
    frontier:
        Client for the frontier model.
    """

    def __init__(
        self,
        config: AppConfig,
        local: LocalClient,
        frontier: FrontierClient,
    ) -> None:
        self.config = config
        self.local = local
        self.frontier = frontier

    def decompose(self, prompt: str) -> DecompositionResult:
        """Ask the local model to decompose *prompt* into sub-tasks."""
        decompose_prompt = DECOMPOSE_PROMPT.format(prompt=prompt)
        responses: list[ModelResponse] = self.local.generate(decompose_prompt, temperature=0.0)
        if responses:
            return parse_decomposition(responses[0].text, prompt)
        return DecompositionResult(
            original_prompt=prompt,
            sub_tasks=[SubTask(id=1, text=prompt, depends_on=[], context_facts=[])],
            decomposed=False,
        )

    def solve_with_decomposition(self, task: Task) -> tuple[str, dict[str, Any]]:
        """Solve *task* using decomposition.

        Each sub-task is first attempted locally.  Only those that fail
        the verifier signal are escalated to the frontier.

        Returns
        -------
        A ``(answer, metadata)`` tuple.  If no decomposition occurred
        ``answer`` is an empty string and the caller should fall back to
        normal routing.
        """
        result = self.decompose(task.prompt)

        if not result.decomposed:
            return "", {
                "decomposed": False,
                "decomposition": result.model_dump(),
            }

        sub_results: dict[int, str] = {}
        frontier_tokens: int = 0
        local_tokens: int = 0
        frontier_tasks: int = 0

        # Process sub-tasks in dependency order (by id)
        for st in sorted(result.sub_tasks, key=lambda s: s.id):
            context = build_minimal_context(st, sub_results)

            # --- try local first ---
            local_responses = self.local.generate(context, temperature=0.0)
            if local_responses:
                local_tokens += local_responses[0].input_tokens + local_responses[0].output_tokens

                # Quick quality check
                ver = verifier_signal(local_responses[0].text, task.category.value)
                if ver.score < 0.5:  # local answer looks OK
                    sub_results[st.id] = local_responses[0].text
                    continue

            # --- escalate to frontier with minimal context ---
            frontier_resp: ModelResponse = self.frontier.generate(context)
            frontier_tokens += (
                frontier_resp.input_tokens
                + frontier_resp.output_tokens
                + frontier_resp.thinking_tokens
            )
            frontier_tasks += 1
            sub_results[st.id] = frontier_resp.text

        answer = merge_results(sub_results, task.prompt)

        metadata: dict[str, Any] = {
            "decomposed": True,
            "decomposition": result.model_dump(),
            "sub_task_count": len(result.sub_tasks),
            "frontier_sub_tasks": frontier_tasks,
            "frontier_tokens_from_decomposition": frontier_tokens,
            "local_tokens_from_decomposition": local_tokens,
        }
        return answer, metadata
