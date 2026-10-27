"""Configuration models for the LLM Router.

All configuration is defined via Pydantic v2 models and loaded from
a YAML file (default: config.yaml in the project root).
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------


class LocalModelConfig(BaseModel):
    """Configuration for locally-hosted models via Ollama."""

    primary: str = "qwen2.5:1.5b-instruct"
    fallback: str = "qwen2.5:3b-instruct"
    ollama_url: str = "http://localhost:11434"
    timeout_seconds: float = 120.0


class FrontierModelConfig(BaseModel):
    """Configuration for frontier (cloud) models."""

    provider: str = "gemini"
    primary: str = "gemini-3.1-pro-preview"  # strongest Pro on free tier
    fallback: str = "gemini-3.8-flash"  # fallback if quota/not-found
    api_key_env: str = "GEMINI_API_KEY"


class ModelPricing(BaseModel):
    """USD per 1M tokens for a single model."""

    input_per_1m: float
    output_per_1m: float
    thinking_per_1m: float = 0.0


class PricingConfig(BaseModel):
    """USD per 1M tokens, from Google's published list."""

    models: dict[str, ModelPricing] = Field(
        default_factory=lambda: {
            "gemini-3.1-pro-preview": ModelPricing(
                input_per_1m=1.25,
                output_per_1m=10.0,
                thinking_per_1m=3.75,
            ),
            "gemini-3.8-flash": ModelPricing(
                input_per_1m=0.10,
                output_per_1m=0.40,
                thinking_per_1m=0.25,
            ),
        }
    )


class RateLimitConfig(BaseModel):
    """Rate-limiting parameters for the frontier API."""

    requests_per_minute: int = 10
    daily_request_cap: int = 500
    retry_max_attempts: int = 5
    retry_base_delay: float = 1.0
    retry_max_delay: float = 60.0


class RouterConfig(BaseModel):
    """Core routing parameters."""

    escalation_threshold: float = 0.6  # escalate if combined signal > this
    cost_lambda: float = 0.5  # quality-cost tradeoff
    confidence_samples: int = 3
    confidence_temperature: float = 0.7


class RuleBasedConfig(BaseModel):
    """Parameters for rule-based routing signals."""

    max_token_length: int = 200
    code_fence_escalate: bool = True
    keywords: list[str] = Field(default_factory=lambda: ["prove", "derive", "step by step"])
    max_sub_questions: int = 2


class EvalConfig(BaseModel):
    """Evaluation harness parameters."""

    tasks_per_category: int = 30
    dev_ratio: float = 0.3
    seed: int = 42
    categories: list[str] = Field(
        default_factory=lambda: ["gsm8k", "arc_challenge", "mbpp", "multi_part"]
    )


class PathsConfig(BaseModel):
    """File-system paths used by the project."""

    data_dir: Path = Path("data")
    results_dir: Path = Path("results")
    cache_dir: Path = Path(".cache")
    models_dir: Path = Path("models")


class EdgeConfig(BaseModel):
    """Configuration for the edge router model (ONNX + quantized)."""

    model_id: str = "sentence-transformers/all-MiniLM-L6-v2"
    fallback_model_id: str = "BAAI/bge-small-en-v1.5"
    head_hidden_dim: int = 64
    onnx_opset: int = 17
    artifacts_dir: Path = Path("artifacts/edge")
    training_data_path: Path = Path("data/edge_train_tasks.jsonl")
    signal_weight: float = 0.0  # 0.0 means off by default (preserves baseline behavior)
    intra_op_num_threads: int = 0  # 0 = auto
    pi_profile_threads: int = 4


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


class AppConfig(BaseModel):
    """Root configuration object aggregating all sub-configs."""

    local: LocalModelConfig = LocalModelConfig()
    frontier: FrontierModelConfig = FrontierModelConfig()
    pricing: PricingConfig = PricingConfig()
    rate_limit: RateLimitConfig = RateLimitConfig()
    router: RouterConfig = RouterConfig()
    rule_based: RuleBasedConfig = RuleBasedConfig()
    eval: EvalConfig = EvalConfig()
    paths: PathsConfig = PathsConfig()
    edge: EdgeConfig = EdgeConfig()
    seed: int = 42


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_config(path: Path = Path("config.yaml")) -> AppConfig:
    """Load configuration from a YAML file, falling back to defaults.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        A fully-populated ``AppConfig`` instance.
    """
    if path.exists():
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return AppConfig(**data)
    return AppConfig()
