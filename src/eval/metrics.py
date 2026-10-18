"""Metric aggregation, bootstrap confidence intervals, and savings calculations.

Computes:
- Quality score with 95% bootstrap confidence interval
- Frontier input, output, thinking tokens
- Total equivalent dollar cost at Google's published list prices
- Mean and 95th percentile latency
- Peak resident memory (RSS in MB)
- Escalation rate
- Frontier token savings % relative to frontier_only
- Quality retention % relative to frontier_only
"""

from __future__ import annotations

import numpy as np

from router.models import SystemMetrics, TaskResult


def bootstrap_ci(
    scores: list[float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Calculate bootstrap confidence interval for mean score."""
    if not scores:
        return (0.0, 0.0)
    if len(scores) == 1:
        return (scores[0], scores[0])

    rng = np.random.RandomState(seed)
    n = len(scores)
    boot_means = np.empty(n_bootstrap)
    arr = np.asarray(scores, dtype=np.float64)

    for i in range(n_bootstrap):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means[i] = np.mean(sample)

    alpha = (1.0 - ci) / 2.0
    low = float(np.percentile(boot_means, alpha * 100.0))
    high = float(np.percentile(boot_means, (1.0 - alpha) * 100.0))
    return (round(low, 4), round(high, 4))


def compute_metrics(
    results: list[TaskResult],
    frontier_only_tokens: int | None = None,
    frontier_only_quality: float | None = None,
) -> SystemMetrics:
    """Aggregate individual TaskResults into a SystemMetrics summary."""
    if not results:
        return SystemMetrics(system_name="unknown")

    name = results[0].system_name
    scores = [r.quality_score for r in results]
    quality = float(np.mean(scores)) if scores else 0.0
    ci_low, ci_high = bootstrap_ci(scores)

    frontier_in = sum(r.total_frontier_input_tokens for r in results)
    frontier_out = sum(r.total_frontier_output_tokens for r in results)
    frontier_think = sum(r.total_frontier_thinking_tokens for r in results)
    total_frontier = frontier_in + frontier_out + frontier_think

    total_cost = sum(r.equivalent_cost_usd for r in results)
    local_tokens = sum(r.total_local_tokens for r in results)

    latencies = [r.latency_seconds for r in results if r.latency_seconds > 0]
    mean_lat = float(np.mean(latencies)) if latencies else 0.0
    p95_lat = float(np.percentile(latencies, 95)) if latencies else 0.0

    peak_mem = max((r.peak_memory_mb for r in results), default=0.0)

    escalations = sum(1 for r in results if r.routing_decision and r.routing_decision.escalate)
    esc_rate = escalations / len(results) if results else 0.0

    # Savings relative to frontier_only
    tokens_saved_pct = 0.0
    if frontier_only_tokens and frontier_only_tokens > 0:
        saved = (frontier_only_tokens - total_frontier) / frontier_only_tokens
        tokens_saved_pct = round(saved * 100.0, 1)

    # Quality retained relative to frontier_only
    quality_retained_pct = 100.0
    if frontier_only_quality and frontier_only_quality > 0:
        retained = (quality / frontier_only_quality) * 100.0
        quality_retained_pct = round(retained, 1)

    return SystemMetrics(
        system_name=name,
        quality_score=round(quality, 4),
        quality_ci_low=ci_low,
        quality_ci_high=ci_high,
        frontier_input_tokens=frontier_in,
        frontier_output_tokens=frontier_out,
        frontier_thinking_tokens=frontier_think,
        total_frontier_tokens=total_frontier,
        equivalent_cost_usd=round(total_cost, 6),
        mean_latency=round(mean_lat, 3),
        p95_latency=round(p95_lat, 3),
        local_tokens=local_tokens,
        peak_memory_mb=round(peak_mem, 1),
        escalation_rate=round(esc_rate, 4),
        frontier_tokens_saved_pct=tokens_saved_pct,
        quality_retained_pct=quality_retained_pct,
        task_count=len(results),
    )
