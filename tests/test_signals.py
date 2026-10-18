"""Unit tests for individual routing signals."""

from __future__ import annotations

from pathlib import Path

from router.models import ModelResponse
from router.signals.confidence import (
    confidence_signal,
    normalize_answer,
)
from router.signals.difficulty import DifficultyClassifier
from router.signals.verifier import (
    check_arithmetic,
    check_code_executable,
    check_format_validity,
)


def test_difficulty_simple_vs_complex():
    clf = DifficultyClassifier()
    res_easy = clf.score("What is 2 + 2?")
    assert 0.0 <= res_easy.score <= 1.0
    assert res_easy.name == "difficulty"

    res_hard = clf.score(
        "```python\ndef solve(x):\n    # First step by step derive recurrence\n    return x**2\n```\nProve convergence."
    )
    assert res_hard.score > res_easy.score


def test_difficulty_features_extraction():
    clf = DifficultyClassifier()
    feats = clf.extract_features("1. Calculate 5 * 10.\n2. Prove theorem.")
    assert feats["length"] > 0
    assert feats["numeric_density"] > 0
    assert feats["num_sub_questions"] >= 2


def test_difficulty_training_and_joblib(tmp_path: Path):
    clf = DifficultyClassifier()
    prompts = [
        "What is 1+1?",
        "What is 2+2?",
        "```python\ndef complex_func():\n    pass\n```\nStep by step prove.",
        "```python\ndef hard():\n    pass\n```\nDerive asymptotics.",
    ]
    labels = [0, 0, 1, 1]
    save_file = tmp_path / "model.joblib"
    clf.train(prompts, labels, save_file)
    assert save_file.exists()

    loaded_clf = DifficultyClassifier(save_file)
    score = loaded_clf.score("```python\ndef solve(): pass\n```")
    assert 0.0 <= score.score <= 1.0


def test_normalize_answer():
    assert normalize_answer("The result is 42.") == "42"
    assert normalize_answer("Option (B)") == "b"
    assert normalize_answer("3.1415") == "3.1415"


def test_confidence_self_consistency():
    # 1. Full agreement -> low score (0 escalation needed)
    resps_high = [
        ModelResponse(text="The answer is 42", model_name="local"),
        ModelResponse(text="Final answer: 42", model_name="local"),
        ModelResponse(text="It is 42", model_name="local"),
    ]
    sig_high = confidence_signal(resps_high, objective=True)
    assert sig_high.score == 0.0  # agreement = 1.0, score = 1 - 1.0 = 0.0

    # 2. Total disagreement -> high score
    resps_low = [
        ModelResponse(text="The answer is 10", model_name="local"),
        ModelResponse(text="The answer is 20", model_name="local"),
        ModelResponse(text="The answer is 30", model_name="local"),
    ]
    sig_low = confidence_signal(resps_low, objective=True)
    assert sig_low.score > 0.5


def test_verifier_arithmetic_and_format():
    # Math format validity
    valid, _ = check_format_validity("Answer is 100", "gsm8k")
    assert valid
    invalid, _ = check_format_validity("I do not know", "gsm8k")
    assert not invalid

    # Arithmetic checks
    ok, _ = check_arithmetic("10 + 5 = 15")
    assert ok
    bad, _ = check_arithmetic("10 + 5 = 16")
    assert not bad


def test_verifier_code_execution():
    valid_code = "def add(a, b):\n    return a + b"
    passed, msg = check_code_executable(valid_code)
    assert passed

    broken_code = "def bad_syntax(:\n    return"
    passed, msg = check_code_executable(broken_code)
    assert not passed
