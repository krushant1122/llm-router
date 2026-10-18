"""End-to-end orchestration pipeline for evaluating router systems.

Runs each configured system against a task dataset, logs per-task telemetry
(latency, memory, token usage, equivalent costs, routing decisions), and caches
model calls so evaluation reruns are reproducible, cheap, and resumable.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import time
from pathlib import Path

from router.baselines.systems import BaseSystem
from router.config import AppConfig
from router.models import Task, TaskResult

logger = logging.getLogger(__name__)


def get_git_commit() -> str:
    """Return the short git commit hash or 'unknown'."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return res.stdout.strip() if res.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def get_config_hash(config: AppConfig) -> str:
    """Return a deterministic short hash of the active configuration."""
    raw = config.model_dump_json()
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


class Pipeline:
    """Evaluates multiple router systems on a collection of benchmark tasks."""

    def __init__(
        self,
        config: AppConfig,
        systems: dict[str, BaseSystem],
        tasks: list[Task],
    ) -> None:
        self.config = config
        self.systems = systems
        self.tasks = tasks
        self.results: dict[str, list[TaskResult]] = {}

    def run(
        self,
        system_names: list[str] | None = None,
        max_requests: int | None = None,
    ) -> dict[str, list[TaskResult]]:
        """Run systems on tasks.

        Args:
            system_names: Names of systems to run. Defaults to all systems.
            max_requests: Maximum number of tasks to evaluate per system.

        Returns:
            Dictionary mapping system name to list of TaskResults.
        """
        targets = system_names or list(self.systems.keys())

        for name in targets:
            if name not in self.systems:
                logger.warning("System '%s' not recognized; skipping.", name)
                continue

            system = self.systems[name]
            results: list[TaskResult] = []
            logger.info("Starting run for system: %s (%d tasks)", name, len(self.tasks))

            for i, task in enumerate(self.tasks):
                if max_requests is not None and i >= max_requests:
                    logger.info("Reached max_requests cap (%d) for system %s.", max_requests, name)
                    break

                try:
                    res = system.solve(task)
                    results.append(res)
                    logger.debug(
                        "[%s] Task %d/%d (%s) finished in %.2fs",
                        name,
                        i + 1,
                        len(self.tasks),
                        task.id,
                        res.latency_seconds,
                    )
                except Exception as exc:
                    logger.error("[%s] Task %s failed: %s", name, task.id, exc, exc_info=True)
                    results.append(
                        TaskResult(
                            task_id=task.id,
                            system_name=name,
                            answer=f"ERROR: {exc}",
                        )
                    )

            self.results[name] = results

        return self.results

    def save_results(self, output_dir: Path) -> None:
        """Save results to disk in JSONL format with run metadata."""
        output_dir.mkdir(parents=True, exist_ok=True)

        meta = {
            "git_commit": get_git_commit(),
            "config_hash": get_config_hash(self.config),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "seed": self.config.seed,
            "tasks_count": len(self.tasks),
            "systems": list(self.results.keys()),
        }
        (output_dir / "meta.json").write_text(json.dumps(meta, indent=2))

        for name, task_results in self.results.items():
            log_file = output_dir / f"{name}.jsonl"
            with open(log_file, "w", encoding="utf-8") as f:
                for r in task_results:
                    f.write(r.model_dump_json() + "\n")

        logger.info("Saved run results for %d systems to %s", len(self.results), output_dir)
