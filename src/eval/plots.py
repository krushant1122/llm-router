"""Visualization of evaluation results and cost-quality Pareto frontiers.

Generates:
- Pareto frontier curves (Equivalent Cost vs. Quality) with bootstrap CIs
- Token usage and cost comparison bar charts
- CSV export for all plotting data
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

from router.models import SystemMetrics

logger = logging.getLogger(__name__)

COLOR_MAP = {
    "local_only": "#1976D2",  # Blue
    "frontier_only": "#D32F2F",  # Red
    "rule_based": "#F57C00",  # Orange
    "proposed": "#388E3C",  # Green
    "no_decomposition": "#7B1FA2",  # Purple
    "difficulty_only": "#00796B",  # Teal
    "confidence_only": "#5D4037",  # Brown
    "verifier_only": "#C2185B",  # Pink
}

MARKER_MAP = {
    "local_only": "o",
    "frontier_only": "s",
    "rule_based": "^",
    "proposed": "*",
    "no_decomposition": "D",
    "difficulty_only": "v",
    "confidence_only": "<",
    "verifier_only": ">",
}


def plot_pareto_curve(metrics: list[SystemMetrics], output_dir: Path) -> Path:
    """Generate and save Cost vs Quality Pareto chart."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6.5))

    for m in metrics:
        col = COLOR_MAP.get(m.system_name, "#424242")
        mk = MARKER_MAP.get(m.system_name, "o")

        # Plot point
        ax.scatter(
            m.equivalent_cost_usd,
            m.quality_score,
            color=col,
            marker=mk,
            s=160 if mk == "*" else 110,
            label=f"{m.system_name} ({m.quality_score:.2f})",
            edgecolors="black",
            linewidths=0.8,
            zorder=5,
        )

        # Plot error bar for 95% CI
        yerr_lower = max(0.0, m.quality_score - m.quality_ci_low)
        yerr_upper = max(0.0, m.quality_ci_high - m.quality_score)
        ax.errorbar(
            m.equivalent_cost_usd,
            m.quality_score,
            yerr=[[yerr_lower], [yerr_upper]],
            color=col,
            capsize=4,
            capthick=1.0,
            alpha=0.7,
            zorder=4,
        )

    ax.set_title(
        "Cost vs. Quality Trade-off Across Systems\n(Equivalent Cost computed from Google published list prices)",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Equivalent Cost (USD)", fontsize=11, fontweight="semibold")
    ax.set_ylabel("Quality Score (Accuracy / Pass@1)", fontsize=11, fontweight="semibold")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(title="Systems", bbox_to_anchor=(1.04, 1), loc="upper left", frameon=True)

    plt.tight_layout()
    chart_path = output_dir / "pareto_curve.png"
    fig.savefig(chart_path, dpi=200)
    plt.close(fig)
    logger.info("Saved Pareto curve to %s", chart_path)

    # Save CSV alongside plot
    csv_path = output_dir / "pareto_data.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "system",
                "quality",
                "ci_low",
                "ci_high",
                "equivalent_cost_usd",
                "total_frontier_tokens",
                "escalation_rate",
                "tokens_saved_pct",
                "quality_retained_pct",
            ]
        )
        for m in metrics:
            writer.writerow(
                [
                    m.system_name,
                    m.quality_score,
                    m.quality_ci_low,
                    m.quality_ci_high,
                    m.equivalent_cost_usd,
                    m.total_frontier_tokens,
                    m.escalation_rate,
                    m.frontier_tokens_saved_pct,
                    m.quality_retained_pct,
                ]
            )

    return chart_path


def plot_token_comparison(metrics: list[SystemMetrics], output_dir: Path) -> Path:
    """Generate and save token usage and equivalent cost bar chart."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    names = [m.system_name for m in metrics]
    frontier_tokens = [m.total_frontier_tokens for m in metrics]
    costs = [m.equivalent_cost_usd for m in metrics]
    colors = [COLOR_MAP.get(n, "#424242") for n in names]

    # Bar chart 1: Frontier Tokens
    ax1.bar(names, frontier_tokens, color=colors, edgecolor="black", linewidth=0.5)
    ax1.set_title("Frontier Token Consumption (Lower is better)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Tokens (Input + Output + Thinking)")
    ax1.tick_params(axis="x", rotation=40)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)

    # Bar chart 2: Equivalent Cost
    ax2.bar(names, costs, color=colors, edgecolor="black", linewidth=0.5)
    ax2.set_title("Equivalent Cost in USD (Lower is better)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("USD ($)")
    ax2.tick_params(axis="x", rotation=40)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    chart_path = output_dir / "token_comparison.png"
    fig.savefig(chart_path, dpi=200)
    plt.close(fig)
    logger.info("Saved token comparison to %s", chart_path)
    return chart_path
