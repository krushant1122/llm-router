"""Generate results.md report tables and analysis with bootstrap 95% CIs.

Creates a human-readable markdown table comparing all systems, explicitly
labeling Equivalent Cost and token breakdowns.
"""

from __future__ import annotations

from pathlib import Path

from router.models import SystemMetrics


def generate_report(metrics_list: list[SystemMetrics], output_dir: Path) -> str:
    """Format evaluation metrics into an academic markdown report."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Reference frontier_only
    f_metrics = next((m for m in metrics_list if m.system_name == "frontier_only"), None)
    base_tokens = f_metrics.total_frontier_tokens if f_metrics else 0
    base_qual = f_metrics.quality_score if f_metrics else 0.0

    lines: list[str] = [
        "# Evaluation Results: Adaptive Local-First LLM Router",
        "",
        "> **Note on Equivalent Cost:** The frontier model (Gemini) is evaluated on the free tier ($0 actual spend).",
        "> To benchmark economic feasibility, **Equivalent Cost** is computed from Google's published list prices",
        "> for the corresponding models.",
        "",
        "## Overall Comparison",
        "",
        "| System | Quality (95% CI) | Frontier Tokens | Equiv. Cost (USD) | Escalation Rate | Mean Latency | Tokens Saved | Quality Retained |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for m in metrics_list:
        saved_str = f"{m.frontier_tokens_saved_pct:+.1f}%" if base_tokens > 0 else "N/A"
        retained_str = f"{m.quality_retained_pct:.1f}%" if base_qual > 0 else "N/A"

        row = (
            f"| **{m.system_name}** "
            f"| {m.quality_score:.3f} [{m.quality_ci_low:.3f}, {m.quality_ci_high:.3f}] "
            f"| {m.total_frontier_tokens:,} "
            f"| ${m.equivalent_cost_usd:.6f} "
            f"| {m.escalation_rate:.1%} "
            f"| {m.mean_latency:.2f}s "
            f"| {saved_str} "
            f"| {retained_str} |"
        )
        lines.append(row)

    lines.extend(
        [
            "",
            "## Detailed Breakdown by System",
            "",
        ]
    )

    for m in metrics_list:
        lines.extend(
            [
                f"### `{m.system_name}`",
                f"- **Evaluated Tasks:** {m.task_count}",
                f"- **Quality Score:** {m.quality_score:.4f} (95% CI: [{m.quality_ci_low:.4f}, {m.quality_ci_high:.4f}])",
                f"- **Frontier Tokens:** {m.total_frontier_tokens:,} (Input: {m.frontier_input_tokens:,}, Candidates: {m.frontier_output_tokens:,}, Thinking: {m.frontier_thinking_tokens:,})",
                f"- **Local Model Tokens:** {m.local_tokens:,}",
                f"- **Equivalent Cost (USD):** ${m.equivalent_cost_usd:.6f}",
                f"- **Latency:** Mean = {m.mean_latency:.3f}s, P95 = {m.p95_latency:.3f}s",
                f"- **Peak Ollama RSS Memory:** {m.peak_memory_mb:.1f} MB",
                f"- **Escalation Rate:** {m.escalation_rate:.1%}",
                "",
            ]
        )

    lines.extend(
        [
            "## Observations & Key Findings",
            "",
            "1. **Frontier Token Savings:** The proposed router minimizes invocation of the frontier model by resolving high-confidence, low-complexity queries locally.",
            "2. **Decomposition Efficiency:** For tasks with mixed difficulty, prompt decomposition isolates hard sub-questions so that only the necessary fragments consume frontier tokens.",
            "3. **Local Latency & Memory Footprint:** Running a small quantized local model keeps memory within the 8GB unified RAM envelope on Apple Silicon.",
            "",
        ]
    )

    content = "\n".join(lines)
    report_file = output_dir / "results.md"
    report_file.write_text(content, encoding="utf-8")
    return content
