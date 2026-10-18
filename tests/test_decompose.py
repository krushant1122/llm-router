"""Unit tests for prompt decomposition, schema validation, and fallback."""

from __future__ import annotations

from router.decompose import (
    Decomposer,
    build_minimal_context,
    merge_results,
    parse_decomposition,
)
from router.models import SubTask


def test_parse_decomposition_valid_json():
    json_text = """{
      "sub_tasks": [
        {"id": 1, "text": "Compute 5 + 3", "depends_on": [], "context_facts": []},
        {"id": 2, "text": "Multiply result by 2", "depends_on": [1], "context_facts": ["factor=2"]}
      ]
    }"""
    res = parse_decomposition(json_text, "Original prompt")
    assert res.decomposed
    assert len(res.sub_tasks) == 2
    assert res.sub_tasks[0].id == 1
    assert res.sub_tasks[1].depends_on == [1]


def test_parse_decomposition_invalid_json_fallback():
    garbage = "I am an LLM and I cannot produce valid JSON."
    res = parse_decomposition(garbage, "Original prompt")
    assert not res.decomposed
    assert len(res.sub_tasks) == 1
    assert res.sub_tasks[0].text == "Original prompt"


def test_build_minimal_context_only_includes_dependencies():
    st = SubTask(id=3, text="Final step", depends_on=[1], context_facts=["gravity=9.8"])
    history = {1: "Step 1 output: 10", 2: "Step 2 output: 20"}
    ctx = build_minimal_context(st, history)

    assert "gravity=9.8" in ctx
    assert "Step 1 output: 10" in ctx
    assert "Final step" in ctx
    # Step 2 was not declared in depends_on -> must NOT be leaked
    assert "Step 2 output: 20" not in ctx


def test_merge_results():
    parts = {1: "First part answer.", 2: "Second part answer."}
    merged = merge_results(parts, "prompt")
    assert "First part answer." in merged
    assert "Second part answer." in merged


def test_decomposer_offline(config, mock_local, mock_frontier):
    decomposer = Decomposer(config, mock_local, mock_frontier)
    res = decomposer.decompose("Answer: 1. 2+2? 2. Capital of Spain?")
    assert len(res.sub_tasks) >= 1
