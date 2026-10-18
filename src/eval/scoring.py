"""Quality scoring functions for all evaluation categories.

Supports:
- GSM8K: exact numeric matching (extracts last number)
- ARC-Challenge: multiple choice letter matching (A/B/C/D)
- MBPP: Python execution in a sandboxed subprocess with timeout against unit test asserts
- Multi-Part: per-sub-question exact match / containment accuracy
"""

from __future__ import annotations

import logging
import re
import subprocess
import tempfile
from pathlib import Path

from router.models import Task, TaskCategory

logger = logging.getLogger(__name__)


def extract_number(text: str) -> str | None:
    """Extract the last integer or decimal from a string."""
    clean = text.replace(",", "")
    numbers = re.findall(r"-?\d+(?:\.\d+)?", clean)
    return numbers[-1] if numbers else None


def extract_letter(text: str) -> str | None:
    """Extract a single-letter choice from text."""
    # Look for patterns like "Answer: B", "(B)", "**B**", or standalone "B"
    patterns = [
        r"(?:answer|choice)\s*(?:is)?\s*[:\s]*\(?([A-Da-d])\)?",
        r"\*\*([A-Da-d])\*\*",
        r"\b([A-Da-d])\)",
        r"\b([A-Da-d])\b",
    ]
    for pat in patterns:
        matches = re.findall(pat, text)
        if matches:
            return matches[-1].upper()
    return None


def score_gsm8k(answer: str, reference: str) -> float:
    """Score math responses using numeric equivalence."""
    pred = extract_number(answer)
    ref = extract_number(reference)
    if pred is None or ref is None:
        return 0.0
    try:
        return 1.0 if abs(float(pred) - float(ref)) < 1e-4 else 0.0
    except ValueError:
        return 0.0


def score_arc(answer: str, reference: str) -> float:
    """Score multiple choice responses against reference letter."""
    pred = extract_letter(answer)
    ref = extract_letter(reference) or reference.strip().upper()
    if pred is None or not ref:
        return 0.0
    return 1.0 if pred == ref else 0.0


def score_code(answer: str, test_list: list[str], timeout: int = 5) -> float:
    """Execute generated Python code against test assertions in a subprocess."""
    if not test_list:
        return 0.0

    # Extract code from Markdown code block if present
    code_match = re.search(r"```(?:python)?\s*\n(.*?)```", answer, re.DOTALL)
    code = code_match.group(1) if code_match else answer

    # Build executable Python script
    script = code + "\n\n" + "\n".join(test_list) + "\n"

    tmp_file = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(script)
            tmp_file = Path(f.name)

        res = subprocess.run(
            ["python3", str(tmp_file)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return 1.0 if res.returncode == 0 else 0.0
    except subprocess.TimeoutExpired:
        logger.warning("Code evaluation timed out after %ds", timeout)
        return 0.0
    except Exception as exc:
        logger.warning("Code evaluation execution error: %s", exc)
        return 0.0
    finally:
        if tmp_file and tmp_file.exists():
            try:
                tmp_file.unlink()
            except OSError:
                pass


def score_multi_part(answer: str, individual_answers: list[str]) -> float:
    """Score multi-part question answers on a per-sub-question basis."""
    if not individual_answers:
        return 0.0

    lines = [ln.strip() for ln in answer.strip().split("\n") if ln.strip()]
    full_text = answer.lower()

    correct_count = 0
    for ref in individual_answers:
        ref_norm = ref.strip().lower()

        # Check numeric equivalence if reference is numeric
        ref_num = extract_number(ref_norm)
        if ref_num is not None:
            ans_nums = re.findall(r"-?\d+(?:\.\d+)?", answer.replace(",", ""))
            if any(abs(float(n) - float(ref_num)) < 1e-4 for n in ans_nums):
                correct_count += 1
                continue

        # Check if reference string is present in lines or full text
        found = False
        for line in lines:
            if ref_norm in line.lower():
                found = True
                break
        if not found and ref_norm in full_text:
            found = True

        if found:
            correct_count += 1

    return correct_count / len(individual_answers)


def score_task(task: Task, answer: str) -> float:
    """Dispatch task to its corresponding scoring function."""
    if not answer or answer.startswith("ERROR:"):
        return 0.0

    cat = task.category
    if cat == TaskCategory.GSM8K:
        return score_gsm8k(answer, task.reference_answer)
    elif cat == TaskCategory.ARC_CHALLENGE:
        return score_arc(answer, task.reference_answer)
    elif cat == TaskCategory.MBPP:
        tests = task.metadata.get("test_list", [])
        return score_code(answer, tests)
    elif cat == TaskCategory.MULTI_PART:
        refs = task.metadata.get("individual_answers", [])
        return score_multi_part(answer, refs)
    return 0.0
