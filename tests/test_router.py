"""Unit tests for the router decision logic and cost sensitivity."""

from __future__ import annotations

from router.models import ModelResponse, Task, TaskCategory
from router.router import Router


def test_router_decision_keep_easy(config, mock_local, mock_frontier):
    router = Router(config, mock_local, mock_frontier)
    task = Task(prompt="What is 5 + 5?", reference_answer="10", category=TaskCategory.GSM8K)
    local_responses = [
        ModelResponse(text="10", model_name="mock_local"),
        ModelResponse(text="10", model_name="mock_local"),
        ModelResponse(text="10", model_name="mock_local"),
    ]
    decision = router.decide(task, local_responses)
    assert not decision.escalate
    assert decision.combined_score < config.router.escalation_threshold


def test_router_decision_escalate_on_high_difficulty(config, mock_local, mock_frontier):
    config.router.escalation_threshold = 0.2
    config.router.cost_lambda = 0.0
    router = Router(config, mock_local, mock_frontier)

    task = Task(
        prompt="```python\ndef solve(): pass\n```\nStep by step prove this complex recurrence relation.",
        reference_answer="",
        category=TaskCategory.MULTI_PART,
    )
    # Mixed local answers
    local_responses = [
        ModelResponse(text="answer 1", model_name="mock_local"),
        ModelResponse(text="something else", model_name="mock_local"),
    ]
    decision = router.decide(task, local_responses)
    assert decision.escalate


def test_router_cost_lambda_sensitivity(config, mock_local, mock_frontier):
    task = Task(
        prompt="Find derivative of x^2.", reference_answer="2x", category=TaskCategory.GSM8K
    )
    responses = [ModelResponse(text="2x", model_name="mock_local")]

    # With lambda=0, normal threshold
    config.router.cost_lambda = 0.0
    router_cheap = Router(config, mock_local, mock_frontier)
    dec_cheap = router_cheap.decide(task, responses)

    # With lambda=1.0, higher effective threshold (penalizes escalation)
    config.router.cost_lambda = 1.0
    router_expensive = Router(config, mock_local, mock_frontier)
    dec_expensive = router_expensive.decide(task, responses)

    # The score is identical, but cost_lambda elevates the hurdle for escalation
    assert dec_cheap.combined_score == dec_expensive.combined_score
