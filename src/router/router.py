"""Core routing policy.

Combines difficulty, confidence, and verifier signals into a single
escalation decision: should the current task be routed to the frontier
model or can the local answer be trusted?
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from router.clients.base import FrontierClient, LocalClient
from router.config import AppConfig
from router.models import ModelResponse, RoutingDecision, SignalResult, Task
from router.signals.confidence import confidence_signal
from router.signals.difficulty import DifficultyClassifier
from router.signals.verifier import verifier_signal

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class Router:
    """Adaptive routing policy.

    Parameters
    ----------
    config:
        Application configuration (thresholds, cost λ, …).
    local:
        Client for the local model.
    frontier:
        Client for the frontier model.
    difficulty_classifier:
        Optional pre-initialised classifier; a default heuristic-only
        instance is created if not supplied.
    enabled_signals:
        Which signal names to evaluate.  Defaults to all three.
    """

    # Default signal weights used by ``combine_signals``
    _SIGNAL_WEIGHTS: dict[str, float] = {
        "difficulty": 0.3,
        "confidence": 0.4,
        "verifier": 0.3,
    }

    def __init__(
        self,
        config: AppConfig,
        local: LocalClient,
        frontier: FrontierClient,
        difficulty_classifier: DifficultyClassifier | None = None,
        enabled_signals: list[str] | None = None,
    ) -> None:
        self.config = config
        self.local = local
        self.frontier = frontier
        self.difficulty = difficulty_classifier or DifficultyClassifier()
        self.enabled_signals: list[str] = enabled_signals or [
            "difficulty",
            "confidence",
            "verifier",
        ]

    # ------------------------------------------------------------------
    # Signal computation
    # ------------------------------------------------------------------

    def compute_signals(
        self,
        task: Task,
        local_responses: list[ModelResponse],
    ) -> list[SignalResult]:
        """Evaluate all enabled signals for the given task."""
        signals: list[SignalResult] = []

        if "difficulty" in self.enabled_signals:
            signals.append(self.difficulty.score(task.prompt))

        if "confidence" in self.enabled_signals and local_responses:
            objective = task.category in ("gsm8k", "arc_challenge")
            signals.append(confidence_signal(local_responses, objective=objective))

        if "verifier" in self.enabled_signals and local_responses:
            signals.append(verifier_signal(local_responses[0].text, task.category.value))

        return signals

    # ------------------------------------------------------------------
    # Signal combination
    # ------------------------------------------------------------------

    def combine_signals(self, signals: list[SignalResult]) -> float:
        """Weighted average of signal scores."""
        if not signals:
            return 0.0

        total_weight = sum(self._SIGNAL_WEIGHTS.get(s.name, 1.0) for s in signals)
        weighted_sum = sum(s.score * self._SIGNAL_WEIGHTS.get(s.name, 1.0) for s in signals)
        return weighted_sum / total_weight if total_weight > 0 else 0.0

    # ------------------------------------------------------------------
    # Decision
    # ------------------------------------------------------------------

    def decide(
        self,
        task: Task,
        local_responses: list[ModelResponse],
    ) -> RoutingDecision:
        """Decide whether to escalate *task* to the frontier model.

        Returns
        -------
        A ``RoutingDecision`` with the boolean verdict, underlying
        signal values, combined score, and a human-readable reason
        string.
        """
        signals = self.compute_signals(task, local_responses)
        combined = self.combine_signals(signals)

        threshold = self.config.router.escalation_threshold

        # Cost-aware adjustment: higher λ → prefer local (raise bar)
        effective_threshold = threshold + self.config.router.cost_lambda * 0.2

        escalate = combined > effective_threshold

        reason = f"combined={combined:.3f} vs threshold={effective_threshold:.3f}"
        decision = RoutingDecision(
            escalate=escalate,
            signals=signals,
            combined_score=round(combined, 6),
            reason=reason,
        )

        logger.info(
            "Routing decision: escalate=%s, %s, signals=%s",
            escalate,
            reason,
            [(s.name, f"{s.score:.3f}") for s in signals],
        )
        return decision
