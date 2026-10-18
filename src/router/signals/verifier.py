"""Answer verification routing signal.

Runs lightweight *post-hoc* checks on a model's answer:
  - format validity (does the answer match expected shape?)
  - arithmetic consistency (do inline equations check out?)
  - code parseability (does extracted Python parse without syntax errors?)

The fraction of **failed** checks becomes the escalation score.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

from router.models import SignalResult

# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def check_format_validity(answer: str, category: str) -> tuple[bool, str]:
    """Check whether the answer has the expected structural format."""
    if category == "gsm8k":
        numbers = re.findall(r"-?\d+\.?\d*", answer)
        return (len(numbers) > 0, f"found {len(numbers)} numbers")
    elif category == "arc_challenge":
        letters = re.findall(r"\b[A-Da-d]\b", answer)
        return (len(letters) > 0, f"found {len(letters)} letter choices")
    elif category == "mbpp":
        return ("def " in answer or "return" in answer, "has function structure")
    return (True, "no format check")


def check_arithmetic(answer: str) -> tuple[bool, str]:
    """Verify simple inline arithmetic expressions (e.g. ``3 + 4 = 7``)."""
    expressions = re.findall(r"(\d+)\s*([+\-*/])\s*(\d+)\s*=\s*(\d+)", answer)
    for a_str, op, b_str, result_str in expressions:
        try:
            a, b, result = float(a_str), float(b_str), float(result_str)
            expected = {
                "+": a + b,
                "-": a - b,
                "*": a * b,
                "/": a / b if b != 0 else float("inf"),
            }[op]
            if abs(expected - result) > 0.01:
                return (False, f"{a_str}{op}{b_str}={result_str} but expected {expected}")
        except (ValueError, ZeroDivisionError):
            pass
    return (True, "arithmetic checks passed")


def check_code_executable(code: str, timeout: int = 5) -> tuple[bool, str]:
    """Check that extracted Python code parses without syntax errors.

    Only the code inside fenced blocks is checked.  Execution is **not**
    attempted — we only call ``ast.parse``.
    """
    # Extract code from fenced blocks if present
    code_match = re.search(r"```(?:python)?\n(.*?)```", code, re.DOTALL)
    if code_match:
        code = code_match.group(1)

    if "def " not in code and "print" not in code:
        return (True, "no executable code detected")

    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
            tmp.write(code)
            tmp.flush()
            tmp_path = Path(tmp.name)

        result = subprocess.run(
            [
                "python3",
                "-c",
                f'import ast; ast.parse(open("{tmp_path}").read())',
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        tmp_path.unlink(missing_ok=True)

        if result.returncode == 0:
            return (True, "code parses successfully")
        return (False, f"syntax error: {result.stderr[:200]}")
    except subprocess.TimeoutExpired:
        return (False, "code check timed out")
    except Exception as exc:
        return (True, f"code check skipped: {exc}")


# ---------------------------------------------------------------------------
# Public signal
# ---------------------------------------------------------------------------


def verifier_signal(answer: str, category: str) -> SignalResult:
    """Run all applicable checks and return an escalation score.

    The score equals the fraction of checks that **failed**.
    """
    checks: list[tuple[bool, str]] = []

    # 1. Format validity
    checks.append(check_format_validity(answer, category))

    # 2. Arithmetic for math answers
    if category == "gsm8k":
        checks.append(check_arithmetic(answer))

    # 3. Code parseability
    if category == "mbpp":
        checks.append(check_code_executable(answer))

    if not checks:
        return SignalResult(name="verifier", score=0.0, metadata={"checks": []})

    failed = sum(1 for passed, _ in checks if not passed)
    score = failed / len(checks)

    return SignalResult(
        name="verifier",
        score=min(max(score, 0.0), 1.0),
        metadata={
            "checks": [{"passed": p, "detail": d} for p, d in checks],
            "failed_count": failed,
            "total_checks": len(checks),
        },
    )
