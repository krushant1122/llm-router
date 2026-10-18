"""Data models for the LLM Router pipeline.

These Pydantic v2 models define the core data structures flowing through
the routing, evaluation, and decomposition pipeline.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from enum import Enum

from pydantic import BaseModel, Field, model_validator

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TaskCategory(str, Enum):
    """Benchmark task categories."""

    GSM8K = "gsm8k"
    ARC_CHALLENGE = "arc_challenge"
    MBPP = "mbpp"
    MULTI_PART = "multi_part"


# ---------------------------------------------------------------------------
# Input models
# ---------------------------------------------------------------------------


class Task(BaseModel):
    """A single evaluation task."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    prompt: str
    reference_answer: str
    category: TaskCategory
    split: str = "test"  # "dev" or "test"
    metadata: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# LLM response models
# ---------------------------------------------------------------------------


class ModelResponse(BaseModel):
    """Response from an LLM invocation."""

    text: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    latency_seconds: float = 0.0
    model_name: str = ""
    model: str = ""
    equivalent_cost_usd: float = 0.0
    equivalent_cost: float = 0.0
    metadata: dict = Field(default_factory=dict)

    model_config = {"extra": "allow"}

    @model_validator(mode="after")
    def _sync_aliases(self) -> ModelResponse:
        if not self.model_name and self.model:
            self.model_name = self.model
        elif not self.model and self.model_name:
            self.model = self.model_name
        if not self.equivalent_cost_usd and self.equivalent_cost:
            self.equivalent_cost_usd = self.equivalent_cost
        elif not self.equivalent_cost and self.equivalent_cost_usd:
            self.equivalent_cost = self.equivalent_cost_usd
        return self


# ---------------------------------------------------------------------------
# Routing models
# ---------------------------------------------------------------------------


class SignalResult(BaseModel):
    """Result from a single routing signal."""

    name: str
    score: float = Field(ge=0.0, le=1.0)
    metadata: dict = Field(default_factory=dict)


class RoutingDecision(BaseModel):
    """Aggregated routing decision."""

    escalate: bool
    signals: list[SignalResult] = Field(default_factory=list)
    combined_score: float = 0.0
    reason: str = ""


# ---------------------------------------------------------------------------
# Decomposition models
# ---------------------------------------------------------------------------


class SubTask(BaseModel):
    """A single sub-task produced by decomposition."""

    id: int
    text: str
    depends_on: list[int] = Field(default_factory=list)
    requires_frontier: bool = False
    context_facts: list[str] = Field(default_factory=list)


class DecompositionResult(BaseModel):
    """Result of decomposing a prompt into sub-tasks."""

    original_prompt: str
    sub_tasks: list[SubTask]
    decomposed: bool = True  # False if fallback to no decomposition


# ---------------------------------------------------------------------------
# Result / metrics models
# ---------------------------------------------------------------------------


class TaskResult(BaseModel):
    """Complete result for a single task evaluation."""

    task_id: str
    system_name: str
    answer: str = ""
    correct: bool | None = None
    quality_score: float = 0.0
    local_response: ModelResponse | None = None
    frontier_response: ModelResponse | None = None
    routing_decision: RoutingDecision | None = None
    decomposition: DecompositionResult | None = None
    total_frontier_input_tokens: int = 0
    total_frontier_output_tokens: int = 0
    total_frontier_thinking_tokens: int = 0
    total_local_tokens: int = 0
    equivalent_cost_usd: float = 0.0
    latency_seconds: float = 0.0
    peak_memory_mb: float = 0.0
    timestamp: str = Field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))


class SystemMetrics(BaseModel):
    """Aggregated metrics for a system across all evaluated tasks."""

    system_name: str
    quality_score: float = 0.0
    quality_ci_low: float = 0.0
    quality_ci_high: float = 0.0
    frontier_input_tokens: int = 0
    frontier_output_tokens: int = 0
    frontier_thinking_tokens: int = 0
    total_frontier_tokens: int = 0
    equivalent_cost_usd: float = 0.0
    mean_latency: float = 0.0
    p95_latency: float = 0.0
    local_tokens: int = 0
    peak_memory_mb: float = 0.0
    escalation_rate: float = 0.0
    frontier_tokens_saved_pct: float = 0.0  # vs frontier_only
    quality_retained_pct: float = 0.0  # vs frontier_only
    task_count: int = 0


# ---------------------------------------------------------------------------
# Cache models
# ---------------------------------------------------------------------------


class CacheKey(BaseModel):
    """Key for caching LLM responses."""

    model: str
    prompt: str
    temperature: float = 0.0
    max_tokens: int = 0

    def to_hash(self) -> str:
        """Return a short SHA-256 hex digest for this cache key."""
        content = f"{self.model}|{self.prompt}|{self.temperature}|{self.max_tokens}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
