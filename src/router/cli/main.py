"""Command Line Interface for the Adaptive Local-First LLM Router.

Supported commands:
  router data build
  router train-difficulty
  router run --system proposed --split test
  router eval --all
  router report <results_dir>
  router ask "your prompt"
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from router.clients.cache import ResponseCache
from router.clients.local import OllamaLocalClient
from router.clients.mock import MockFrontierClient, MockLocalClient
from router.config import AppConfig, load_config
from router.models import Task, TaskCategory
from router.signals.difficulty import DifficultyClassifier

app = typer.Typer(name="router", help="Adaptive Local-First LLM Router CLI", no_args_is_help=True)
data_app = typer.Typer(name="data", help="Data operations")
app.add_typer(data_app, name="data")

console = Console()
logger = logging.getLogger("router")


def _get_clients(config: AppConfig):
    """Instantiate local and frontier clients with safe offline mock fallbacks."""
    # 1. Local client
    local_client = OllamaLocalClient(
        model=config.local.primary,
        base_url=config.local.ollama_url,
        timeout=config.local.timeout_seconds,
    )
    if not local_client.is_available():
        console.print(
            "[yellow]Notice: Ollama not reachable at http://localhost:11434. Using MockLocalClient.[/yellow]"
        )
        local_client = MockLocalClient()

    # 2. Frontier client
    api_key = os.environ.get(config.frontier.api_key_env, "")
    if api_key:
        try:
            from router.clients.frontier import GeminiFrontierClient

            frontier_client = GeminiFrontierClient(config=config, api_key=api_key)
        except Exception as exc:
            console.print(
                f"[yellow]Notice: Could not initialize Gemini SDK ({exc}). Using MockFrontierClient.[/yellow]"
            )
            frontier_client = MockFrontierClient()
    else:
        console.print(
            "[yellow]Notice: GEMINI_API_KEY environment variable not set. Using MockFrontierClient.[/yellow]"
        )
        frontier_client = MockFrontierClient()

    return local_client, frontier_client


def _get_systems(config: AppConfig, local_client, frontier_client, cache: ResponseCache):
    """Create all candidate and baseline systems."""
    from router.baselines.systems import (
        ConfidenceOnlySystem,
        DifficultyOnlySystem,
        FrontierOnlySystem,
        LocalOnlySystem,
        ProposedSystem,
        RouterNoDecompositionSystem,
        RuleBasedSystem,
        VerifierOnlySystem,
    )

    clf_path = config.paths.models_dir / "difficulty_classifier.joblib"
    diff_clf = DifficultyClassifier(clf_path if clf_path.exists() else None)

    return {
        "local_only": LocalOnlySystem(config, local_client, frontier_client, cache),
        "frontier_only": FrontierOnlySystem(config, local_client, frontier_client, cache),
        "rule_based": RuleBasedSystem(config, local_client, frontier_client, cache),
        "proposed": ProposedSystem(
            config, local_client, frontier_client, cache, difficulty_classifier=diff_clf
        ),
        "no_decomposition": RouterNoDecompositionSystem(
            config, local_client, frontier_client, cache, difficulty_classifier=diff_clf
        ),
        "difficulty_only": DifficultyOnlySystem(
            config, local_client, frontier_client, cache, difficulty_classifier=diff_clf
        ),
        "confidence_only": ConfidenceOnlySystem(config, local_client, frontier_client, cache),
        "verifier_only": VerifierOnlySystem(config, local_client, frontier_client, cache),
    }


@data_app.command("build")
@app.command("data-build", hidden=True)
def data_build(
    config_file: Path = typer.Option(
        Path("config.yaml"), "--config", "-c", help="Path to config.yaml"
    ),
) -> None:
    """Build evaluation dataset tasks and write to data/tasks.jsonl."""
    from eval.data_build import build_all_tasks, save_tasks

    config = load_config(config_file)
    tasks = build_all_tasks(config)
    out_path = config.paths.data_dir / "tasks.jsonl"
    save_tasks(tasks, out_path)
    console.print(
        f"[green]Successfully generated {len(tasks)} benchmark tasks into {out_path}[/green]"
    )


@app.command("train-difficulty")
def train_difficulty(
    config_file: Path = typer.Option(
        Path("config.yaml"), "--config", "-c", help="Path to config.yaml"
    ),
) -> None:
    """Train logistic regression difficulty classifier on the dev split."""
    from eval.data_build import load_tasks

    config = load_config(config_file)
    data_path = config.paths.data_dir / "tasks.jsonl"
    if not data_path.exists():
        console.print("[yellow]Benchmark data not found. Running data build first...[/yellow]")
        from eval.data_build import build_all_tasks, save_tasks

        save_tasks(build_all_tasks(config), data_path)

    tasks = load_tasks(data_path)
    dev_tasks = [t for t in tasks if t.split == "dev"]
    if not dev_tasks:
        dev_tasks = tasks[: max(5, len(tasks) // 3)]

    prompts = [t.prompt for t in dev_tasks]
    # Label 1 for complex tasks (MBPP, multi-part, or long prompt)
    labels = [
        1
        if t.category in (TaskCategory.MBPP, TaskCategory.MULTI_PART) or len(t.prompt.split()) > 40
        else 0
        for t in dev_tasks
    ]

    clf = DifficultyClassifier()
    save_file = config.paths.models_dir / "difficulty_classifier.joblib"
    clf.train(prompts, labels, save_file)
    console.print(
        f"[green]Trained difficulty classifier on {len(dev_tasks)} dev samples -> {save_file}[/green]"
    )


@app.command("run")
def run_system(
    system: str = typer.Option("proposed", "--system", "-s", help="System to run"),
    split: str = typer.Option("test", "--split", help="Dataset split (dev or test)"),
    max_requests: int = typer.Option(0, "--max-requests", "-m", help="Stop after N requests"),
    config_file: Path = typer.Option(
        Path("config.yaml"), "--config", "-c", help="Path to config.yaml"
    ),
) -> None:
    """Run a specific router system on benchmark tasks."""
    from eval.data_build import load_tasks
    from router.pipeline import Pipeline

    config = load_config(config_file)
    data_path = config.paths.data_dir / "tasks.jsonl"
    if not data_path.exists():
        console.print("[yellow]Tasks file missing. Generating tasks...[/yellow]")
        from eval.data_build import build_all_tasks, save_tasks

        save_tasks(build_all_tasks(config), data_path)

    tasks = [t for t in load_tasks(data_path) if t.split == split]
    local_client, frontier_client = _get_clients(config)
    cache = ResponseCache(config.paths.cache_dir)
    systems = _get_systems(config, local_client, frontier_client, cache)

    if system not in systems:
        console.print(
            f"[red]Error: System '{system}' not available. Choices: {list(systems.keys())}[/red]"
        )
        raise typer.Exit(1)

    pipeline = Pipeline(config, {system: systems[system]}, tasks)
    cap = max_requests if max_requests > 0 else None
    _ = pipeline.run(max_requests=cap)

    run_id = time.strftime("run_%Y%m%d_%H%M%S")
    out_dir = config.paths.results_dir / run_id
    pipeline.save_results(out_dir)
    console.print(f"[green]Run finished! Results saved to {out_dir}[/green]")


@app.command("eval")
def eval_cmd(
    all_systems: bool = typer.Option(True, "--all/--no-all", help="Evaluate all baselines"),
    max_requests: int = typer.Option(0, "--max-requests", "-m", help="Max tasks per system"),
    config_file: Path = typer.Option(Path("config.yaml"), "--config", "-c", help="Config file"),
) -> None:
    """Run end-to-end evaluation across all systems, score answers, generate plots and report."""
    from eval.data_build import load_tasks
    from eval.metrics import compute_metrics
    from eval.plots import plot_pareto_curve, plot_token_comparison
    from eval.report import generate_report
    from eval.scoring import score_task
    from router.pipeline import Pipeline

    config = load_config(config_file)
    data_path = config.paths.data_dir / "tasks.jsonl"
    if not data_path.exists():
        from eval.data_build import build_all_tasks, save_tasks

        save_tasks(build_all_tasks(config), data_path)

    test_tasks = [t for t in load_tasks(data_path) if t.split == "test"]
    if not test_tasks:
        test_tasks = load_tasks(data_path)

    local_client, frontier_client = _get_clients(config)
    cache = ResponseCache(config.paths.cache_dir)
    systems = _get_systems(config, local_client, frontier_client, cache)

    cap = max_requests if max_requests > 0 else None
    pipeline = Pipeline(config, systems, test_tasks)
    eval_results = pipeline.run(max_requests=cap)

    # Score each system
    for name, task_results in eval_results.items():
        for r, t in zip(task_results, test_tasks):
            r.quality_score = score_task(t, r.answer)

    # First get frontier_only reference numbers
    frontier_only_results = eval_results.get("frontier_only", [])
    f_tokens = sum(
        r.total_frontier_input_tokens
        + r.total_frontier_output_tokens
        + r.total_frontier_thinking_tokens
        for r in frontier_only_results
    )
    f_quality = (
        float(sum(r.quality_score for r in frontier_only_results) / len(frontier_only_results))
        if frontier_only_results
        else 0.0
    )

    # Compute metrics for all systems
    metrics_list = []
    for name, task_results in eval_results.items():
        m = compute_metrics(
            task_results,
            frontier_only_tokens=f_tokens,
            frontier_only_quality=f_quality,
        )
        metrics_list.append(m)

    run_id = time.strftime("eval_%Y%m%d_%H%M%S")
    out_dir = config.paths.results_dir / run_id
    pipeline.save_results(out_dir)

    plot_pareto_curve(metrics_list, out_dir)
    plot_token_comparison(metrics_list, out_dir)
    _ = generate_report(metrics_list, out_dir)

    # Render summary table in CLI
    table = Table(title="Evaluation Results Summary", show_header=True, header_style="bold magenta")
    table.add_column("System")
    table.add_column("Quality")
    table.add_column("Frontier Tokens")
    table.add_column("Equiv. Cost ($)")
    table.add_column("Esc. Rate")
    table.add_column("Tokens Saved")

    for m in metrics_list:
        table.add_row(
            m.system_name,
            f"{m.quality_score:.3f}",
            f"{m.total_frontier_tokens:,}",
            f"${m.equivalent_cost_usd:.6f}",
            f"{m.escalation_rate:.1%}",
            f"{m.frontier_tokens_saved_pct:+.1f}%",
        )

    console.print(table)
    console.print(f"\n[green]Detailed report and plots saved to {out_dir}[/green]")


@app.command("report")
def report_cmd(
    results_dir: Path = typer.Argument(
        ..., help="Path to results directory containing jsonl files"
    ),
) -> None:
    """Regenerate markdown report and plots from an existing results directory."""
    from eval.metrics import compute_metrics
    from eval.plots import plot_pareto_curve, plot_token_comparison
    from eval.report import generate_report
    from router.models import TaskResult

    if not results_dir.exists():
        console.print(f"[red]Directory not found: {results_dir}[/red]")
        raise typer.Exit(1)

    jsonl_files = list(results_dir.glob("*.jsonl"))
    if not jsonl_files:
        console.print(f"[yellow]No .jsonl log files found in {results_dir}[/yellow]")
        raise typer.Exit(1)

    all_results = {}
    for jf in jsonl_files:
        sys_name = jf.stem
        records = []
        with open(jf, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(TaskResult.model_validate_json(line))
        all_results[sys_name] = records

    f_res = all_results.get("frontier_only", [])
    f_tokens = sum(
        r.total_frontier_input_tokens
        + r.total_frontier_output_tokens
        + r.total_frontier_thinking_tokens
        for r in f_res
    )
    f_quality = float(sum(r.quality_score for r in f_res) / len(f_res)) if f_res else 0.0

    metrics_list = [
        compute_metrics(recs, frontier_only_tokens=f_tokens, frontier_only_quality=f_quality)
        for recs in all_results.values()
    ]

    plot_pareto_curve(metrics_list, results_dir)
    plot_token_comparison(metrics_list, results_dir)
    generate_report(metrics_list, results_dir)
    console.print(f"[green]Report regenerated in {results_dir}[/green]")


@app.command("ask")
def ask_cmd(
    prompt: str = typer.Argument(..., help="The prompt to route and answer"),
    config_file: Path = typer.Option(Path("config.yaml"), "--config", "-c", help="Config file"),
) -> None:
    """Prompt the router interactively. Displays routing decision, tokens, and answer."""
    config = load_config(config_file)
    local_client, frontier_client = _get_clients(config)
    cache = ResponseCache(config.paths.cache_dir)
    systems = _get_systems(config, local_client, frontier_client, cache)

    proposed = systems["proposed"]
    task = Task(prompt=prompt, reference_answer="", category=TaskCategory.MULTI_PART)

    t0 = time.perf_counter()
    res = proposed.solve(task)
    dur = time.perf_counter() - t0

    console.print("\n" + "=" * 60)
    console.print("[bold cyan]Adaptive LLM Router Decision[/bold cyan]")
    console.print("=" * 60)
    if res.routing_decision:
        status = (
            "[red]ESCALATED TO FRONTIER[/red]"
            if res.routing_decision.escalate
            else "[green]RESOLVED LOCALLY[/green]"
        )
        console.print(f"Decision: {status}")
        console.print(f"Reason:   {res.routing_decision.reason}")
        if res.routing_decision.signals:
            for s in res.routing_decision.signals:
                console.print(f"  • Signal {s.name:<12}: {s.score:.3f}")
    console.print(f"Latency:        {dur:.2f}s")
    console.print(f"Local Tokens:   {res.total_local_tokens}")
    console.print(
        f"Frontier Tokens: {res.total_frontier_input_tokens + res.total_frontier_output_tokens + res.total_frontier_thinking_tokens}"
    )
    console.print(f"Equivalent Cost: ${res.equivalent_cost_usd:.6f}")
    console.print("-" * 60)
    console.print("[bold]Answer:[/bold]")
    console.print(res.answer)
    console.print("=" * 60 + "\n")


if __name__ == "__main__":
    app()
