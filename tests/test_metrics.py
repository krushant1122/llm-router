"""Unit tests for metric computation, confidence intervals, and savings."""

from __future__ import annotations

from eval.metrics import bootstrap_ci, compute_metrics
from router.models import RoutingDecision, TaskResult


def test_bootstrap_ci_coverage():
    scores = [1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0]
    low, high = bootstrap_ci(scores, n_bootstrap=500, ci=0.95, seed=42)
    mean_val = sum(scores) / len(scores)
    assert low <= mean_val <= high
    assert 0.0 <= low <= high <= 1.0


def test_compute_metrics_summary():
    results = [
        TaskResult(
            task_id="t1",
            system_name="proposed",
            answer="42",
            quality_score=1.0,
            total_frontier_input_tokens=100,
            total_frontier_output_tokens=50,
            total_frontier_thinking_tokens=20,
            total_local_tokens=200,
            equivalent_cost_usd=0.0012,
            latency_seconds=1.2,
            peak_memory_mb=450.0,
            routing_decision=RoutingDecision(escalate=True, combined_score=0.8),
        ),
        TaskResult(
            task_id="t2",
            system_name="proposed",
            answer="10",
            quality_score=1.0,
            total_frontier_input_tokens=0,
            total_frontier_output_tokens=0,
            total_frontier_thinking_tokens=0,
            total_local_tokens=150,
            equivalent_cost_usd=0.0,
            latency_seconds=0.5,
            peak_memory_mb=460.0,
            routing_decision=RoutingDecision(escalate=False, combined_score=0.2),
        ),
    ]

    m = compute_metrics(
        results,
        frontier_only_tokens=1000,
        frontier_only_quality=1.0,
    )

    assert m.system_name == "proposed"
    assert m.quality_score == 1.0
    assert m.total_frontier_tokens == 170  # 100 + 50 + 20
    assert m.escalation_rate == 0.5  # 1 out of 2 escalated
    assert m.peak_memory_mb == 460.0
    assert m.frontier_tokens_saved_pct == 83.0  # (1000 - 170) / 1000 = 83%
    assert m.quality_retained_pct == 100.0
