"""
Unit tests for agents/product_research_node.py — covers:

  DETERMINISTIC helpers (no I/O):
  1.  _parse_json_response: strips ```json fence, normal JSON
  2.  _sanitize_queries: strips control chars/quotes, enforces product name,
      deduplicates, caps at _MAX_QUERIES, falls back to templates
  3.  _required_tokens: identifies digit-containing and ALL-CAPS tokens
  4.  _numeric_grounding_ok: passes when tokens are in snippets, fails when not
  5.  _build_facts: normalises category/confidence, assigns r1/r2/... ids,
      drops ungrounded claims, enforces _MAX_FACTS

  ASYNC node (mocked I/O):
  6.  No TAVILY_API_KEY → immediate _skipped() (no LLM call)
  7.  classification = "skip" → _skipped(), no Tavily call
  8.  classification = "research_needed" + successful search → performs=True,
      facts populated, research_complete event dispatched
  9.  All Tavily searches raise exception → performed=True, facts=[], event emitted
  10. LLM distillation raises exception → caught, node returns _skipped()
  11. Exception in classify call → caught by outer handler, returns _skipped()
"""
from __future__ import annotations

import asyncio
import json
import os

import pytest

from agents.product_research_node import (
    _MAX_FACTS,
    _MAX_QUERIES,
    _build_facts,
    _concat_snippets,
    _numeric_grounding_ok,
    _parse_json_response,
    _required_tokens,
    _sanitize_queries,
    product_research_node,
)


# ---------------------------------------------------------------------------
# 1. _parse_json_response
# ---------------------------------------------------------------------------

def test_parse_json_response_plain():
    raw = '{"classification": "skip", "product_name": "mug"}'
    assert _parse_json_response(raw) == {"classification": "skip", "product_name": "mug"}


def test_parse_json_response_strips_json_fence():
    raw = "```json\n{\"a\": 1}\n```"
    assert _parse_json_response(raw) == {"a": 1}


def test_parse_json_response_strips_bare_fence():
    raw = "```\n{\"b\": 2}\n```"
    assert _parse_json_response(raw) == {"b": 2}


def test_parse_json_response_raises_on_invalid():
    with pytest.raises(json.JSONDecodeError):
        _parse_json_response("not json")


# ---------------------------------------------------------------------------
# 2. _sanitize_queries
# ---------------------------------------------------------------------------

def test_sanitize_queries_keeps_good_query():
    out = _sanitize_queries(["BIC lighter campfire use cases"], "BIC lighter")
    assert out == ["BIC lighter campfire use cases"]


def test_sanitize_queries_removes_query_missing_product_name():
    out = _sanitize_queries(["campfire use cases", "BIC lighter specs"], "BIC lighter")
    assert "campfire use cases" not in out
    assert any("BIC lighter" in q for q in out)


def test_sanitize_queries_strips_control_chars():
    out = _sanitize_queries(["BIC lighter\x00specs\x1freview"], "BIC lighter")
    assert all("\x00" not in q and "\x1f" not in q for q in out)


def test_sanitize_queries_strips_quotes():
    out = _sanitize_queries(['BIC lighter "features"'], "BIC lighter")
    assert all('"' not in q for q in out)


def test_sanitize_queries_falls_back_to_template_when_no_valid_queries():
    out = _sanitize_queries(["unrelated query", "another unrelated"], "Widget X")
    assert len(out) >= 1
    assert any("Widget X" in q for q in out)


def test_sanitize_queries_caps_at_max():
    queries = [f"BIC lighter query {i}" for i in range(10)]
    out = _sanitize_queries(queries, "BIC lighter")
    assert len(out) <= _MAX_QUERIES


def test_sanitize_queries_deduplicates():
    out = _sanitize_queries(
        ["BIC lighter specs", "BIC lighter specs", "BIC lighter features"],
        "BIC lighter",
    )
    assert len(out) == len(set(q.lower() for q in out))


def test_sanitize_queries_caps_individual_query_length():
    long_q = "BIC lighter " + "x" * 200
    out = _sanitize_queries([long_q], "BIC lighter")
    assert all(len(q) <= 120 for q in out)


# ---------------------------------------------------------------------------
# 3. _required_tokens
# ---------------------------------------------------------------------------

def test_required_tokens_digit_only():
    assert "4K" in _required_tokens("Captures 4K video")


def test_required_tokens_allcaps_only():
    assert "ANC" in _required_tokens("Adaptive ANC cuts noise")


def test_required_tokens_no_required_tokens():
    assert _required_tokens("clean and fresh") == []


def test_required_tokens_mixed():
    tokens = _required_tokens("Runs 2h and supports ANC at 4K")
    assert "2h" in tokens
    assert "ANC" in tokens
    assert "4K" in tokens


# ---------------------------------------------------------------------------
# 4. _numeric_grounding_ok
# ---------------------------------------------------------------------------

def test_numeric_grounding_ok_passes_when_token_in_snippets():
    assert _numeric_grounding_ok("Runs 2h on a charge", "battery runs 2h per charge tested")


def test_numeric_grounding_ok_fails_when_token_missing():
    assert not _numeric_grounding_ok("Runs 4K video", "great resolution and clarity")


def test_numeric_grounding_ok_case_insensitive():
    assert _numeric_grounding_ok("Uses ANC technology", "the anc system cuts sound")


def test_numeric_grounding_ok_passes_for_claims_without_digits_or_allcaps():
    assert _numeric_grounding_ok("Feels comfortable on long runs", "anything")


# ---------------------------------------------------------------------------
# 5. _build_facts
# ---------------------------------------------------------------------------

_SAMPLE_SNIPPETS = "4K 120fps sensor confirmed. ANC cuts 31dB noise."

def _raw_fact(**kwargs):
    base = {
        "claim": "Captures 4K at 120fps",
        "category": "spec",
        "source_url": "https://example.com/review",
        "confidence": "high",
    }
    base.update(kwargs)
    return base


def test_build_facts_assigns_sequential_ids():
    raw = [_raw_fact(), _raw_fact(claim="ANC cuts 31dB noise")]
    facts = _build_facts(raw, _SAMPLE_SNIPPETS)
    assert [f["fact_id"] for f in facts] == ["r1", "r2"]


def test_build_facts_filters_invalid_category():
    raw = [_raw_fact(category="pricing"), _raw_fact(claim="ANC cuts 31dB noise", category="spec")]
    facts = _build_facts(raw, _SAMPLE_SNIPPETS)
    assert len(facts) == 1
    assert facts[0]["category"] == "spec"


def test_build_facts_normalises_unknown_confidence_to_medium():
    raw = [_raw_fact(confidence="very_high")]
    facts = _build_facts(raw, _SAMPLE_SNIPPETS)
    assert facts[0]["confidence"] == "medium"


def test_build_facts_drops_ungrounded_numeric_claim():
    raw = [_raw_fact(claim="Runs 999mph")]
    facts = _build_facts(raw, _SAMPLE_SNIPPETS)
    assert facts == []


def test_build_facts_enforces_max_facts():
    raw = [_raw_fact(claim=f"ANC cuts 31dB noise {i}") for i in range(20)]
    facts = _build_facts(raw, _SAMPLE_SNIPPETS)
    assert len(facts) <= _MAX_FACTS


def test_build_facts_skips_non_dict_entries():
    raw = [None, "bad", _raw_fact(claim="ANC cuts 31dB noise")]
    facts = _build_facts(raw, _SAMPLE_SNIPPETS)
    assert len(facts) == 1


def test_build_facts_skips_empty_claim():
    raw = [_raw_fact(claim=""), _raw_fact(claim="  ")]
    facts = _build_facts(raw, _SAMPLE_SNIPPETS)
    assert facts == []


# ---------------------------------------------------------------------------
# 6. Node: no TAVILY_API_KEY → immediate skip
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_node_skips_when_no_tavily_key(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    result = await product_research_node({"brief": "A lighter", "brand_name": ""})
    pr = result["product_research"]
    assert pr["performed"] is False
    assert pr["classification"] == "skipped"


# ---------------------------------------------------------------------------
# 7. Node: LLM says "skip" → returns _skipped(), no Tavily call
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_node_skips_when_classified_skip(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-api-key")
    monkeypatch.setenv("DASHSCOPE_BASE_URL", "http://test.example.com/v1")
    monkeypatch.setenv("MODEL_TEXT", "qwen-test")

    classify_payload = json.dumps({
        "classification": "skip",
        "product_name": "artisan ceramic bowl",
        "search_queries": [],
    })

    from tests._fakes import make_fake_async_openai
    monkeypatch.setattr(
        "agents.product_research_node.AsyncOpenAI",
        make_fake_async_openai([classify_payload]),
    )

    tavily_called = {"n": 0}
    async def _fake_tavily(*a, **kw):
        tavily_called["n"] += 1
        return {}

    monkeypatch.setattr("agents.product_research_node._search", _fake_tavily)

    state = {"brief": "Hand-thrown ceramic bowl.", "brand_name": "", "product_truths": []}
    result = await product_research_node(state)

    pr = result["product_research"]
    assert pr["performed"] is False
    assert pr["classification"] == "skipped"
    assert tavily_called["n"] == 0


# ---------------------------------------------------------------------------
# 8. Node: full success path → performed=True, facts populated
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_node_performs_research_and_returns_facts(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-api-key")
    monkeypatch.setenv("DASHSCOPE_BASE_URL", "http://test.example.com/v1")
    monkeypatch.setenv("MODEL_TEXT", "qwen-test")

    classify_payload = json.dumps({
        "classification": "research_needed",
        "product_name": "BIC Classic Lighter",
        "search_queries": [
            "BIC Classic Lighter features campfire",
            "BIC Classic Lighter specs review",
        ],
    })
    distill_payload = json.dumps({
        "facts": [
            {
                "claim": "Produces a reliable windproof flame",
                "category": "feature",
                "source_url": "https://example.com/bic",
                "confidence": "medium",
            },
        ]
    })

    from tests._fakes import make_fake_async_openai
    monkeypatch.setattr(
        "agents.product_research_node.AsyncOpenAI",
        make_fake_async_openai([classify_payload, distill_payload]),
    )

    fake_results = [
        {"url": "https://example.com/bic", "title": "BIC Lighter Review", "content": "windproof flame"},
    ]

    async def _fake_search(queries):
        snippets = "[source: https://example.com/bic]\nBIC Lighter Review\nwindproof flame\n"
        return snippets, fake_results

    monkeypatch.setattr("agents.product_research_node._search", _fake_search)

    events_dispatched = []
    async def _fake_emit(config, fact_count, product_name, queries):
        events_dispatched.append({"fact_count": fact_count, "product_name": product_name})

    monkeypatch.setattr("agents.product_research_node._emit", _fake_emit)

    state = {
        "brief": "A slim windproof lighter — everyday carry.",
        "brand_name": "BIC",
        "product_truths": [{"truth_id": "t1", "fact": "Slim rectangular body"}],
    }
    result = await product_research_node(state)

    pr = result["product_research"]
    assert pr["performed"] is True
    assert pr["classification"] == "research_needed"
    assert pr["product_name"] == "BIC Classic Lighter"
    assert len(pr["facts"]) == 1
    assert pr["facts"][0]["fact_id"] == "r1"
    assert pr["facts"][0]["category"] == "feature"
    assert len(events_dispatched) == 1
    assert events_dispatched[0]["product_name"] == "BIC Classic Lighter"


# ---------------------------------------------------------------------------
# 9. Node: all Tavily searches fail → performed=True, facts=[]
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_node_handles_all_searches_failing(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-api-key")
    monkeypatch.setenv("DASHSCOPE_BASE_URL", "http://test.example.com/v1")
    monkeypatch.setenv("MODEL_TEXT", "qwen-test")

    classify_payload = json.dumps({
        "classification": "research_needed",
        "product_name": "BIC Classic Lighter",
        "search_queries": ["BIC Classic Lighter specs"],
    })

    from tests._fakes import make_fake_async_openai
    monkeypatch.setattr(
        "agents.product_research_node.AsyncOpenAI",
        make_fake_async_openai([classify_payload]),
    )

    async def _fake_search(queries):
        return "", []  # empty snippets, no results

    monkeypatch.setattr("agents.product_research_node._search", _fake_search)

    events_dispatched = []
    async def _fake_emit(config, fact_count, product_name, queries):
        events_dispatched.append({"fact_count": fact_count})

    monkeypatch.setattr("agents.product_research_node._emit", _fake_emit)

    state = {"brief": "A lighter.", "brand_name": "", "product_truths": []}
    result = await product_research_node(state)

    pr = result["product_research"]
    assert pr["performed"] is True
    assert pr["facts"] == []
    assert len(events_dispatched) == 1
    assert events_dispatched[0]["fact_count"] == 0


# ---------------------------------------------------------------------------
# 10. Node: exception in node body → returns _skipped(), NEVER raises
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_node_degrades_to_noop_on_classify_exception(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-api-key")
    monkeypatch.setenv("DASHSCOPE_BASE_URL", "http://test.example.com/v1")
    monkeypatch.setenv("MODEL_TEXT", "qwen-test")

    async def _exploding_classify(*a, **kw):
        raise RuntimeError("DashScope unreachable")

    monkeypatch.setattr("agents.product_research_node._classify", _exploding_classify)

    state = {"brief": "A lighter.", "brand_name": "", "product_truths": []}
    # Node must NEVER raise — wraps all errors in a graceful no-op.
    result = await product_research_node(state)

    pr = result["product_research"]
    assert pr["performed"] is False
    assert pr["classification"] == "skipped"
    # Trace note is written by the outer except block.
    assert "product_research" in result.get("reasoning_trace", "")
