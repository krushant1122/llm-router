"""Baseline systems and router variants implementing a unified interface.

All systems take a Task, solve it, and return a TaskResult with full telemetry:
tokens (frontier and local, input/output/thinking), latency, equivalent dollar
cost, and Ollama peak resident memory.
"""

from __future__ import annotations

import logging
import re
import time
from abc import ABC, abstractmethod

from router.clients.base import FrontierClient, LocalClient
from router.clients.cache import ResponseCache
from router.config import AppConfig
from router.decompose import Decomposer
from router.models import (
    CacheKey,
    DecompositionResult,
    ModelResponse,
    RoutingDecision,
    Task,
    TaskResult,
)
from router.router import Router
from router.signals.difficulty import DifficultyClassifier

logger = logging.getLogger(__name__)


def _sample_ollama_memory_mb() -> float:
    """Sample current resident memory (RSS in MB) of Ollama processes."""
    try:
        import psutil

        total_rss = 0
        for proc in psutil.process_iter(["name", "memory_info"]):
            try:
                name = (proc.info.get("name") or "").lower()
                if "ollama" in name:
                    total_rss += proc.info["memory_info"].rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return total_rss / (1024 * 1024)
    except Exception:
        return 0.0


class BaseSystem(ABC):
    """Common interface for all router systems and baselines."""

    name: str

    @abstractmethod
    def solve(self, task: Task) -> TaskResult:
        """Solve a given task and return the telemetry-rich TaskResult."""
        pass


class LocalOnlySystem(BaseSystem):
    """Always routes to the local model. Frontier token usage is strictly 0."""

    name = "local_only"

    def __init__(
        self,
        config: AppConfig,
        local: LocalClient,
        frontier: FrontierClient,
        cache: ResponseCache,
    ) -> None:
        self.config = config
        self.local = local
        self.frontier = frontier
        self.cache = cache

    def solve(self, task: Task) -> TaskResult:
        t0 = time.perf_counter()
        mem_start = _sample_ollama_memory_mb()

        cache_key = CacheKey(
            model=self.local.model_name,
            prompt=task.prompt,
            temperature=0.0,
            max_tokens=2048,
        )

        resp = self.cache.get(cache_key)
        if resp is None:
            responses = self.local.generate(task.prompt, temperature=0.0)
            resp = (
                responses[0]
                if responses
                else ModelResponse(text="", model_name=self.local.model_name)
            )
            self.cache.put(cache_key, resp)

        latency = time.perf_counter() - t0
        mem_end = _sample_ollama_memory_mb()

        decision = RoutingDecision(
            escalate=False,
            combined_score=0.0,
            reason="local_only baseline: always keep local",
        )

        return TaskResult(
            task_id=task.id,
            system_name=self.name,
            answer=resp.text,
            local_response=resp,
            routing_decision=decision,
            total_frontier_input_tokens=0,
            total_frontier_output_tokens=0,
            total_frontier_thinking_tokens=0,
            total_local_tokens=resp.input_tokens + resp.output_tokens,
            equivalent_cost_usd=0.0,
            latency_seconds=latency,
            peak_memory_mb=max(mem_start, mem_end),
        )


class FrontierOnlySystem(BaseSystem):
    """Always routes to the frontier model. Quality ceiling baseline."""

    name = "frontier_only"

    def __init__(
        self,
        config: AppConfig,
        local: LocalClient,
        frontier: FrontierClient,
        cache: ResponseCache,
    ) -> None:
        self.config = config
        self.local = local
        self.frontier = frontier
        self.cache = cache

    def solve(self, task: Task) -> TaskResult:
        t0 = time.perf_counter()

        cache_key = CacheKey(
            model=self.frontier.model_name,
            prompt=task.prompt,
            temperature=0.0,
            max_tokens=4096,
        )

        resp = self.cache.get(cache_key)
        if resp is None:
            resp = self.frontier.generate(task.prompt, temperature=0.0)
            self.cache.put(cache_key, resp)

        latency = time.perf_counter() - t0

        decision = RoutingDecision(
            escalate=True,
            combined_score=1.0,
            reason="frontier_only baseline: always escalate",
        )

        return TaskResult(
            task_id=task.id,
            system_name=self.name,
            answer=resp.text,
            frontier_response=resp,
            routing_decision=decision,
            total_frontier_input_tokens=resp.input_tokens,
            total_frontier_output_tokens=resp.output_tokens,
            total_frontier_thinking_tokens=resp.thinking_tokens,
            total_local_tokens=0,
            equivalent_cost_usd=resp.equivalent_cost_usd,
            latency_seconds=latency,
            peak_memory_mb=0.0,
        )


class RuleBasedSystem(BaseSystem):
    """Naive static rules router: length, code fence, keywords, sub-questions."""

    name = "rule_based"

    def __init__(
        self,
        config: AppConfig,
        local: LocalClient,
        frontier: FrontierClient,
        cache: ResponseCache,
    ) -> None:
        self.config = config
        self.local = local
        self.frontier = frontier
        self.cache = cache
        self.rule_cfg = config.rule_based

    def should_escalate(self, prompt: str) -> tuple[bool, str]:
        # Rule 1: Length in approx words/tokens
        words = prompt.split()
        if len(words) > self.rule_cfg.max_token_length:
            return True, f"length {len(words)} > {self.rule_cfg.max_token_length}"

        # Rule 2: Code fences
        if self.rule_cfg.code_fence_escalate and "```" in prompt:
            return True, "contains code fence"

        # Rule 3: Keywords
        lower_prompt = prompt.lower()
        for kw in self.rule_cfg.keywords:
            if kw.lower() in lower_prompt:
                return True, f"contains keyword '{kw}'"

        # Rule 4: Numbered sub-questions
        sub_q = len(re.findall(r"(?:^|\n)\s*(?:\d+[.)]|[a-z][.)]|[-*])", prompt))
        if sub_q > self.rule_cfg.max_sub_questions:
            return True, f"sub-questions count {sub_q} > {self.rule_cfg.max_sub_questions}"

        return False, "no rule triggered"

    def solve(self, task: Task) -> TaskResult:
        t0 = time.perf_counter()
        mem_start = _sample_ollama_memory_mb()

        escalate, reason = self.should_escalate(task.prompt)
        decision = RoutingDecision(
            escalate=escalate,
            combined_score=1.0 if escalate else 0.0,
            reason=f"rule_based: {reason}",
        )

        local_resp = None
        frontier_resp = None
        total_local_tok = 0
        total_front_in = 0
        total_front_out = 0
        total_front_think = 0
        cost = 0.0

        if escalate:
            cache_key = CacheKey(
                model=self.frontier.model_name,
                prompt=task.prompt,
                temperature=0.0,
            )
            frontier_resp = self.cache.get(cache_key)
            if frontier_resp is None:
                frontier_resp = self.frontier.generate(task.prompt)
                self.cache.put(cache_key, frontier_resp)
            answer = frontier_resp.text
            total_front_in = frontier_resp.input_tokens
            total_front_out = frontier_resp.output_tokens
            total_front_think = frontier_resp.thinking_tokens
            cost = frontier_resp.equivalent_cost_usd
        else:
            cache_key = CacheKey(
                model=self.local.model_name,
                prompt=task.prompt,
                temperature=0.0,
            )
            local_resp = self.cache.get(cache_key)
            if local_resp is None:
                resps = self.local.generate(task.prompt, temperature=0.0)
                local_resp = (
                    resps[0] if resps else ModelResponse(text="", model_name=self.local.model_name)
                )
                self.cache.put(cache_key, local_resp)
            answer = local_resp.text
            total_local_tok = local_resp.input_tokens + local_resp.output_tokens

        latency = time.perf_counter() - t0
        mem_end = _sample_ollama_memory_mb()

        return TaskResult(
            task_id=task.id,
            system_name=self.name,
            answer=answer,
            local_response=local_resp,
            frontier_response=frontier_resp,
            routing_decision=decision,
            total_frontier_input_tokens=total_front_in,
            total_frontier_output_tokens=total_front_out,
            total_frontier_thinking_tokens=total_front_think,
            total_local_tokens=total_local_tok,
            equivalent_cost_usd=cost,
            latency_seconds=latency,
            peak_memory_mb=max(mem_start, mem_end),
        )


class ProposedSystem(BaseSystem):
    """The full proposed adaptive router with prompt decomposition."""

    name = "proposed"

    def __init__(
        self,
        config: AppConfig,
        local: LocalClient,
        frontier: FrontierClient,
        cache: ResponseCache,
        difficulty_classifier: DifficultyClassifier | None = None,
        enabled_signals: list[str] | None = None,
        enable_decomposition: bool = True,
    ) -> None:
        self.config = config
        self.local = local
        self.frontier = frontier
        self.cache = cache
        self.router = Router(
            config=config,
            local=local,
            frontier=frontier,
            difficulty_classifier=difficulty_classifier,
            enabled_signals=enabled_signals,
        )
        self.decomposer = Decomposer(config, local, frontier)
        self.enable_decomposition = enable_decomposition

    def solve(self, task: Task) -> TaskResult:
        t0 = time.perf_counter()
        mem_start = _sample_ollama_memory_mb()

        # Step 1: Sample local model for confidence self-consistency
        k = self.config.router.confidence_samples
        temp = self.config.router.confidence_temperature

        local_cache_key = CacheKey(
            model=self.local.model_name,
            prompt=task.prompt,
            temperature=temp,
            max_tokens=k,
        )

        local_responses: list[ModelResponse] = []
        cached_resp = self.cache.get(local_cache_key)
        if cached_resp is not None:
            local_responses = [cached_resp]
        else:
            local_responses = self.local.generate(
                task.prompt,
                temperature=temp,
                n=k,
            )
            if local_responses:
                self.cache.put(local_cache_key, local_responses[0])

        total_local_tokens = sum(r.input_tokens + r.output_tokens for r in local_responses)

        # Step 2: Compute signals & decision
        decision = self.router.decide(task, local_responses)

        answer = ""
        total_front_in = 0
        total_front_out = 0
        total_front_think = 0
        cost = 0.0
        decomp_result = None
        frontier_resp = None

        if not decision.escalate:
            # KEEP LOCAL: use best local response
            answer = local_responses[0].text if local_responses else ""
        else:
            # ESCALATE: check decomposition if enabled
            if self.enable_decomposition:
                decomp_answer, meta = self.decomposer.solve_with_decomposition(task)
                if meta.get("decomposed", False) and decomp_answer:
                    answer = decomp_answer
                    decomp_raw = meta.get("decomposition")
                    if isinstance(decomp_raw, dict):
                        decomp_result = DecompositionResult.model_validate(decomp_raw)
                    elif isinstance(decomp_raw, DecompositionResult):
                        decomp_result = decomp_raw
                    else:
                        decomp_result = None
                    total_local_tokens += meta.get("local_tokens_from_decomposition", 0)
                    total_front_in += meta.get("frontier_tokens_from_decomposition", 0) // 2
                    total_front_out += meta.get("frontier_tokens_from_decomposition", 0) // 2
                    # Compute equivalent cost
                    pricing = self.config.pricing.models.get(self.frontier.model_name)
                    if pricing:
                        cost = (total_front_in / 1_000_000) * pricing.input_per_1m + (
                            total_front_out / 1_000_000
                        ) * pricing.output_per_1m
                else:
                    # Fallback to direct frontier query
                    frontier_cache_key = CacheKey(
                        model=self.frontier.model_name,
                        prompt=task.prompt,
                        temperature=0.0,
                    )
                    frontier_resp = self.cache.get(frontier_cache_key)
                    if frontier_resp is None:
                        frontier_resp = self.frontier.generate(task.prompt)
                        self.cache.put(frontier_cache_key, frontier_resp)

                    answer = frontier_resp.text
                    total_front_in = frontier_resp.input_tokens
                    total_front_out = frontier_resp.output_tokens
                    total_front_think = frontier_resp.thinking_tokens
                    cost = frontier_resp.equivalent_cost_usd
            else:
                # Direct frontier query
                frontier_cache_key = CacheKey(
                    model=self.frontier.model_name,
                    prompt=task.prompt,
                    temperature=0.0,
                )
                frontier_resp = self.cache.get(frontier_cache_key)
                if frontier_resp is None:
                    frontier_resp = self.frontier.generate(task.prompt)
                    self.cache.put(frontier_cache_key, frontier_resp)

                answer = frontier_resp.text
                total_front_in = frontier_resp.input_tokens
                total_front_out = frontier_resp.output_tokens
                total_front_think = frontier_resp.thinking_tokens
                cost = frontier_resp.equivalent_cost_usd

        latency = time.perf_counter() - t0
        mem_end = _sample_ollama_memory_mb()

        return TaskResult(
            task_id=task.id,
            system_name=self.name,
            answer=answer,
            local_response=local_responses[0] if local_responses else None,
            frontier_response=frontier_resp,
            routing_decision=decision,
            decomposition=decomp_result,
            total_frontier_input_tokens=total_front_in,
            total_frontier_output_tokens=total_front_out,
            total_frontier_thinking_tokens=total_front_think,
            total_local_tokens=total_local_tokens,
            equivalent_cost_usd=cost,
            latency_seconds=latency,
            peak_memory_mb=max(mem_start, mem_end),
        )


class RouterNoDecompositionSystem(ProposedSystem):
    """Ablation: Router with all signals, but skips decomposition on escalation."""

    name = "no_decomposition"

    def __init__(
        self,
        config: AppConfig,
        local: LocalClient,
        frontier: FrontierClient,
        cache: ResponseCache,
        difficulty_classifier: DifficultyClassifier | None = None,
    ) -> None:
        super().__init__(
            config=config,
            local=local,
            frontier=frontier,
            cache=cache,
            difficulty_classifier=difficulty_classifier,
            enable_decomposition=False,
        )


class DifficultyOnlySystem(ProposedSystem):
    """Ablation: Router with only the difficulty classifier signal."""

    name = "difficulty_only"

    def __init__(
        self,
        config: AppConfig,
        local: LocalClient,
        frontier: FrontierClient,
        cache: ResponseCache,
        difficulty_classifier: DifficultyClassifier | None = None,
    ) -> None:
        super().__init__(
            config=config,
            local=local,
            frontier=frontier,
            cache=cache,
            difficulty_classifier=difficulty_classifier,
            enabled_signals=["difficulty"],
            enable_decomposition=False,
        )


class ConfidenceOnlySystem(ProposedSystem):
    """Ablation: Router with only the confidence / self-consistency signal."""

    name = "confidence_only"

    def __init__(
        self,
        config: AppConfig,
        local: LocalClient,
        frontier: FrontierClient,
        cache: ResponseCache,
    ) -> None:
        super().__init__(
            config=config,
            local=local,
            frontier=frontier,
            cache=cache,
            enabled_signals=["confidence"],
            enable_decomposition=False,
        )


class VerifierOnlySystem(ProposedSystem):
    """Ablation: Router with only the answer verifier signal."""

    name = "verifier_only"

    def __init__(
        self,
        config: AppConfig,
        local: LocalClient,
        frontier: FrontierClient,
        cache: ResponseCache,
    ) -> None:
        super().__init__(
            config=config,
            local=local,
            frontier=frontier,
            cache=cache,
            enabled_signals=["verifier"],
            enable_decomposition=False,
        )
