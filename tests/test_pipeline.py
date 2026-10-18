"""End-to-end integration test: Full pipeline on offline fixture tasks with mock clients."""

from __future__ import annotations

from pathlib import Path

from eval.metrics import compute_metrics
from eval.plots import plot_pareto_curve, plot_token_comparison
from eval.report import generate_report
from eval.scoring import score_task
from router.baselines.systems import (
    FrontierOnlySystem,
    LocalOnlySystem,
    ProposedSystem,
    RuleBasedSystem,
)
from router.clients.cache import ResponseCache
from router.pipeline import Pipeline


def test_full_pipeline_offline(config, mock_local, mock_frontier, fixture_tasks, tmp_path: Path):
    """Run full pipeline across all baselines on 10-task fixture without internet or API keys."""
    cache = ResponseCache(tmp_path / "cache")

    systems = {
        "local_only": LocalOnlySystem(config, mock_local, mock_frontier, cache),
        "frontier_only": FrontierOnlySystem(config, mock_local, mock_frontier, cache),
        "rule_based": RuleBasedSystem(config, mock_local, mock_frontier, cache),
        "proposed": ProposedSystem(config, mock_local, mock_frontier, cache),
    }

    pipeline = Pipeline(config, systems, fixture_tasks)
    results = pipeline.run()

    # 1. Verify all 4 systems ran all 10 fixture tasks
    assert len(results) == 4
    for sys_name, res_list in results.items():
        assert len(res_list) == len(fixture_tasks)
        for r in res_list:
            assert r.system_name == sys_name
            assert r.answer != ""

    # 2. Score task results
    for sys_name, res_list in results.items():
        for r, t in zip(res_list, fixture_tasks):
            r.quality_score = score_task(t, r.answer)
            assert 0.0 <= r.quality_score <= 1.0

    # 3. Frontier only should have consumed frontier tokens, local only zero
    frontier_only_res = results["frontier_only"]
    f_tokens = sum(
        r.total_frontier_input_tokens
        + r.total_frontier_output_tokens
        + r.total_frontier_thinking_tokens
        for r in frontier_only_res
    )
    assert f_tokens > 0

    local_only_res = results["local_only"]
    loc_frontier_tokens = sum(
        r.total_frontier_input_tokens
        + r.total_frontier_output_tokens
        + r.total_frontier_thinking_tokens
        for r in local_only_res
    )
    assert loc_frontier_tokens == 0

    # 4. Metrics aggregation and report generation
    metrics_list = [compute_metrics(res, frontier_only_tokens=f_tokens) for res in results.values()]
    assert len(metrics_list) == 4

    out_dir = tmp_path / "eval_output"
    pipeline.save_results(out_dir)
    assert (out_dir / "meta.json").exists()
    assert (out_dir / "proposed.jsonl").exists()

    plot_pareto_curve(metrics_list, out_dir)
    assert (out_dir / "pareto_curve.png").exists()
    assert (out_dir / "pareto_data.csv").exists()

    plot_token_comparison(metrics_list, out_dir)
    assert (out_dir / "token_comparison.png").exists()

    rep = generate_report(metrics_list, out_dir)
    assert "Evaluation Results" in rep
    assert (out_dir / "results.md").exists()


def test_pipeline_max_requests_cap(
    config, mock_local, mock_frontier, fixture_tasks, tmp_path: Path
):
    cache = ResponseCache(tmp_path / "cache")
    systems = {"local_only": LocalOnlySystem(config, mock_local, mock_frontier, cache)}
    pipeline = Pipeline(config, systems, fixture_tasks)

    # Cap at 3 requests
    results = pipeline.run(max_requests=3)
    assert len(results["local_only"]) == 3
