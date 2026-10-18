"""Confidence routing signal via self-consistency.

Generates multiple samples from the local model, measures agreement,
and converts that into an *escalation* score: low agreement → high
score → should escalate.
"""

from __future__ import annotations

import re
from collections import Counter

from router.models import ModelResponse, SignalResult

# ---------------------------------------------------------------------------
# Answer normalisation helpers
# ---------------------------------------------------------------------------


def normalize_answer(text: str) -> str:
    """Normalize an answer string for exact-match comparison.

    * For numeric answers (e.g. GSM8K) the last number in the text is
      extracted.
    * For multiple-choice (e.g. ARC) the last single letter a–d is
      extracted.
    * Otherwise the full lowered/stripped text is returned.
    """
    text = text.strip().lower()

    # Extract final numeric answer (for math)
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    if numbers:
        return numbers[-1]

    # Extract letter choice (for MC)
    letters = re.findall(r"\b([a-d])\b", text)
    if letters:
        return letters[-1]

    return text


def token_overlap_similarity(a: str, b: str) -> float:
    """Dice-coefficient token overlap between two strings."""
    tokens_a = set(a.lower().split())
    tokens_b = set(b.lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    return 2 * len(intersection) / (len(tokens_a) + len(tokens_b))


# ---------------------------------------------------------------------------
# Agreement measurement
# ---------------------------------------------------------------------------


def measure_agreement(responses: list[ModelResponse], objective: bool = True) -> float:
    """Measure inter-sample agreement.

    Parameters
    ----------
    responses:
        Two or more generations from the same model.
    objective:
        If ``True`` (e.g. math / MC) use exact-match majority voting.
        If ``False`` use pairwise token overlap (for open-ended tasks).

    Returns
    -------
    A float in [0, 1] where 1.0 means perfect agreement.
    """
    if len(responses) <= 1:
        return 1.0

    if objective:
        normalized = [normalize_answer(r.text) for r in responses]
        counts = Counter(normalized)
        most_common_count = counts.most_common(1)[0][1]
        return most_common_count / len(normalized)

    # Pairwise token overlap for subjective / open-ended tasks
    similarities: list[float] = []
    for i in range(len(responses)):
        for j in range(i + 1, len(responses)):
            similarities.append(token_overlap_similarity(responses[i].text, responses[j].text))
    return sum(similarities) / max(len(similarities), 1)


# ---------------------------------------------------------------------------
# Public signal
# ---------------------------------------------------------------------------


def confidence_signal(
    responses: list[ModelResponse],
    objective: bool = True,
) -> SignalResult:
    """Compute the confidence routing signal.

    High agreement → high confidence → *low* need to escalate, so the
    returned score is ``1 - agreement``.
    """
    agreement = measure_agreement(responses, objective=objective)

    # Invert: high agreement ⇒ low escalation score
    score = 1.0 - agreement

    return SignalResult(
        name="confidence",
        score=min(max(score, 0.0), 1.0),
        metadata={
            "agreement": agreement,
            "num_samples": len(responses),
            "objective": objective,
        },
    )
