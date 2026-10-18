# Architectural & Design Decisions

This document records all architectural choices, assumptions, and design defaults selected for the **Adaptive Local-First LLM Router** project.

---

## 1. Local Model & Execution Constraints
- **Default Local Model:** `qwen2.5:1.5b-instruct` (fallback: `qwen2.5:3b-instruct`).
  - *Rationale:* Targeted for Apple Silicon M1 (8GB unified RAM). 1.5B/3B Q4 quantized models require ~1.2GB–2.2GB resident memory, comfortably fitting alongside OS processes without swap thrashing.
- **Single Process Policy:** The router never loads more than one local model into Ollama simultaneously. Multi-sample self-consistency runs sequentially (`n` calls to `/api/generate`) rather than spawning parallel model instances.
- **Resident Memory Measurement:** Process RSS is measured via `psutil` by querying processes named `ollama` or `ollama_llama_server`. This represents actual host RAM allocated to the local inference runtime.

---

## 2. Frontier Model & Gemini Free-Tier Resilience
- **Model Choice:** `gemini-3.1-pro-preview` as the primary reasoning model, with dynamic fallback to `gemini-3.8-flash` if 429 quota exhaustion or 404 model deprecation occurs.
- **Provider-Agnostic Interface:** All frontier providers implement `FrontierClient` (`router.clients.base`). Swapping Gemini for Anthropic or OpenAI requires implementing a single class.
- **Token Accounting:**
  - Input tokens = `usage_metadata.prompt_token_count`
  - Output tokens = `usage_metadata.candidates_token_count + (usage_metadata.thoughts_token_count or 0)`
  - Thinking/reasoning tokens are logged and tracked as a distinct metric column.
- **Equivalent Cost Metric:**
  - Actual expenditure is $0 (free tier).
  - Financial metric reported across all tables and charts is **Equivalent Cost (USD)**, calculated from Google's published list prices:
    - `gemini-3.1-pro-preview`: \$1.25 / 1M input tokens, \$10.00 / 1M output tokens, \$3.75 / 1M thinking tokens.
    - `gemini-3.8-flash`: \$0.10 / 1M input tokens, \$0.40 / 1M output tokens, \$0.25 / 1M thinking tokens.
- **Rate-Limiting & Budget Control:**
  - Client-side rate limiter enforces max 10 requests per minute with sliding-window sleep backpressure.
  - Exponential backoff with random jitter (0.5x to 1.5x) on HTTP 429 and 5xx errors.
  - `--max-requests` flag provides a hard session cap to cleanly terminate evaluations before hitting daily quotas.

---

## 3. Caching & Reproducibility
- **On-Disk Response Cache:** Keyed by SHA-256 hash of `(model, prompt, temperature, max_tokens)`.
- **Resumability:** Aborted evaluation runs can be resumed immediately. The pipeline queries the disk cache before executing any HTTP request, skipping redundant external calls.
- **Seed & Hash Tagging:** Every evaluation run writes `meta.json` containing the git commit SHA, configuration hash, random seed, and execution timestamp.

---

## 4. Routing Policy & Signal Weights
- **Signal Aggregation:**
  - Difficulty signal weight: `0.3`
  - Confidence (self-consistency) signal weight: `0.4`
  - Verifier signal weight: `0.3`
- **Cost-Aware Hurdle (`cost_lambda`):**
  - The escalation threshold is adjusted as `effective_threshold = threshold + (cost_lambda * 0.2)`.
  - Increasing $\lambda$ elevates the difficulty/uncertainty hurdle required to trigger a frontier call, strictly prioritizing local execution.

---

## 5. Prompt Decomposition & Minimal Context Isolation
- **Strict JSON Schema:** Prompts are decomposed via local instructions returning `{ "sub_tasks": [{"id": ..., "text": ..., "depends_on": [...], "context_facts": [...]}] }`.
- **Zero-Context-Leak Isolation:** Sub-tasks that fail local verification are escalated to the frontier with *only* their required `context_facts` and output from preceding dependent steps. The full original prompt is never sent if sub-tasks can be solved in isolation.
- **Graceful Fallback:** Any JSON parsing error or malformed model output immediately falls back to single-prompt execution.

---

## 6. Evaluation Datasets & Synthetic Offline Mode
- **Split Distribution:** 30% dev split (for difficulty classifier training and parameter tuning), 70% test split (strictly held out for evaluation reporting).
- **Default Sample Count:** 30 tasks per category (120 total), calibrated to complete comfortably within free-tier rate limits.
- **Offline Determinism:** If HuggingFace datasets are unreachable or network access is restricted, the data generator creates synthetic benchmark tasks mirroring GSM8K math, ARC reasoning, MBPP coding, and multi-part queries. A 10-task offline fixture suite is included for zero-dependency test execution.
