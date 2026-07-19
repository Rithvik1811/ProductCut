"""
Tests for brand_research_node and brand-name threading through concept_agent.

No real HTTP or LLM calls — all external I/O is injected via monkeypatch
following the same patterns used throughout this test suite.

Coverage:
  - brand_research_node: no-op when brand_url absent
  - brand_research_node: happy path (fetch + summarize)
  - brand_research_node: graceful degrade on fetch failure
  - brand_research_node: graceful degrade on HTTP error
  - _fetch_page_text: strips <script>, <style>, tags, collapses whitespace
  - _fetch_page_text: truncates at _MAX_PAGE_CHARS
  - _brand_identity_block: empty when no brand info
  - _brand_identity_block: CTA mandate appears when brand_name present
  - _brand_identity_block: brand context block appears when brand_context present
  - _build_system_prompt: brand block injected into prompt when provided
  - _build_system_prompt: brand block absent from prompt when not provided
  - _build_user_content: brand line appears at top when brand_name present
  - _build_user_content: brand context section present when brand_context provided
  - _build_user_content: brand fields absent when not provided
  - generate_script_variants: brand_name/brand_context reach the LLM message
  - concept_agent_node: reads brand_name and brand_context from state
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.brand_research_node import (
    _fetch_page_text_httpx,
    _get_page_text,
    brand_research_node,
    _MAX_PAGE_CHARS,
)
from agents.concept_agent import (
    _abrupt_cta_problem,
    _brand_identity_block,
    _build_system_prompt,
    _build_user_content,
    generate_script_variants,
    concept_agent_node,
    DEFAULT_TARGET_LENGTH_SEC,
    FRAMEWORKS,
)
from tests._fakes import FakeOpenAIClient, make_fake_async_openai


# ---------------------------------------------------------------------------
# Minimal product truth fixtures reused from concept agent tests
# ---------------------------------------------------------------------------

_TRUTHS = [
    {"truth_id": "t0", "fact": "1-litre stainless steel vacuum flask", "category": "form_factor", "source": "photo_1"},
    {"truth_id": "t1", "fact": "matte midnight-blue powder coat", "category": "color", "source": "photo_1"},
    {"truth_id": "t2", "fact": "double-wall insulation keeps cold 24 h", "category": "construction_detail", "source": "photo_1"},
]


def _beats(n: int = 5, total: int = DEFAULT_TARGET_LENGTH_SEC) -> list[dict]:
    step = total // n
    beats = []
    t = 0
    for i in range(n):
        end = t + step if i < n - 1 else total
        beats.append({"t_start": t, "t_end": end, "line": f"beat {i+1}"})
        t = end
    return beats


def _variant(vid: str = "v1", framework: str = FRAMEWORKS[0]) -> dict:
    return {
        "variant_id": vid,
        "text": "some ad text",
        "framework": framework,
        "hook_type": "concrete claim",
        "emotional_trigger": "desire",
        "grounding_truth_ids": ["t0", "t2"],
        "beats": _beats(),
        "target_length_sec": DEFAULT_TARGET_LENGTH_SEC,
    }


def _four_variants() -> list[dict]:
    return [_variant(f"v{i+1}", FRAMEWORKS[i]) for i in range(4)]


def _valid_json(variants: list[dict] | None = None) -> str:
    return json.dumps({"script_variants": variants or _four_variants()})


# ---------------------------------------------------------------------------
# _fetch_page_text_httpx (httpx fallback path)
# ---------------------------------------------------------------------------

def test_fetch_page_text_strips_script_and_style_tags():
    html = "<html><script>var x=1;</script><style>.a{}</style><body><p>Hello world</p></body></html>"
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.text = html

    with patch("agents.brand_research_node.httpx.get", return_value=mock_resp):
        result = _fetch_page_text_httpx("https://example.com")

    assert "var x=1" not in result
    assert ".a{}" not in result
    assert "Hello world" in result


def test_fetch_page_text_strips_html_tags():
    html = "<div class='hero'><h1>Best Water Bottle</h1><p>Stay hydrated.</p></div>"
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.text = html

    with patch("agents.brand_research_node.httpx.get", return_value=mock_resp):
        result = _fetch_page_text_httpx("https://example.com")

    assert "<" not in result
    assert "Best Water Bottle" in result
    assert "Stay hydrated" in result


def test_fetch_page_text_truncates_at_max_chars():
    html = "A" * (_MAX_PAGE_CHARS + 5000)
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.text = html

    with patch("agents.brand_research_node.httpx.get", return_value=mock_resp):
        result = _fetch_page_text_httpx("https://example.com")

    assert len(result) == _MAX_PAGE_CHARS


def test_fetch_page_text_collapses_whitespace():
    html = "<p>too   many\n\n\nspaces</p>"
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.text = html

    with patch("agents.brand_research_node.httpx.get", return_value=mock_resp):
        result = _fetch_page_text_httpx("https://example.com")

    assert "  " not in result
    assert "too many spaces" in result


# ---------------------------------------------------------------------------
# brand_research_node — no-op paths
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_brand_research_node_no_brand_url_returns_empty():
    result = await brand_research_node({"job_id": "j1"})
    assert result == {}


@pytest.mark.asyncio
async def test_brand_research_node_empty_brand_url_returns_empty():
    result = await brand_research_node({"job_id": "j1", "brand_url": ""})
    assert result == {}


# ---------------------------------------------------------------------------
# brand_research_node — fetch failure degrades gracefully
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_brand_research_node_fetch_error_returns_empty():
    with patch("agents.brand_research_node._get_page_text",side_effect=Exception("timeout")):
        result = await brand_research_node({"job_id": "j1", "brand_url": "https://example.com"})
    assert result == {}


@pytest.mark.asyncio
async def test_brand_research_node_http_error_returns_empty():
    import httpx as _httpx

    exc = _httpx.HTTPStatusError("404", request=MagicMock(), response=MagicMock())
    with patch("agents.brand_research_node._get_page_text",side_effect=exc):
        result = await brand_research_node({"job_id": "j1", "brand_url": "https://example.com"})
    assert result == {}


# ---------------------------------------------------------------------------
# brand_research_node — happy path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_brand_research_node_happy_path_writes_brand_context(monkeypatch):
    fake_summary = "HydroFlask makes premium insulated water bottles for outdoor enthusiasts."

    monkeypatch.setenv("DASHSCOPE_API_KEY", "fake-key")
    monkeypatch.setenv("DASHSCOPE_BASE_URL", "https://fake.api")
    monkeypatch.setenv("MODEL_TEXT", "qwen-max")

    with (
        patch("agents.brand_research_node._get_page_text", new=AsyncMock(return_value="some page text")),
        patch("agents.brand_research_node.AsyncOpenAI", make_fake_async_openai([fake_summary])),
    ):
        result = await brand_research_node({
            "job_id": "j1",
            "brand_url": "https://hydroflask.com",
            "brand_name": "HydroFlask",
        })

    assert "brand_context" in result
    assert result["brand_context"] == fake_summary


@pytest.mark.asyncio
async def test_brand_research_node_no_brand_name_still_works(monkeypatch):
    fake_summary = "Premium outdoor gear brand."
    monkeypatch.setenv("DASHSCOPE_API_KEY", "fake-key")
    monkeypatch.setenv("DASHSCOPE_BASE_URL", "https://fake.api")
    monkeypatch.setenv("MODEL_TEXT", "qwen-max")

    with (
        patch("agents.brand_research_node._get_page_text", new=AsyncMock(return_value="page text")),
        patch("agents.brand_research_node.AsyncOpenAI", make_fake_async_openai([fake_summary])),
    ):
        result = await brand_research_node({"job_id": "j1", "brand_url": "https://example.com"})

    assert result.get("brand_context") == fake_summary


# ---------------------------------------------------------------------------
# _brand_identity_block
# ---------------------------------------------------------------------------

def test_brand_identity_block_empty_when_no_brand():
    assert _brand_identity_block("", "") == ""


def test_brand_identity_block_contains_brand_name_cta_rule():
    block = _brand_identity_block("HydroFlask", "")
    assert "HydroFlask" in block
    assert "CTA" in block.upper() or "cta" in block.lower()
    # generic CTAs should be forbidden
    assert "Get yours" in block or "generic" in block.lower()


def test_brand_identity_block_contains_brand_context():
    context = "Outdoor-focused. Premium. Sustainability-first."
    block = _brand_identity_block("", context)
    assert context in block


def test_brand_identity_block_contains_both():
    block = _brand_identity_block("HydroFlask", "Premium outdoor brand.")
    assert "HydroFlask" in block
    assert "Premium outdoor brand." in block


# ---------------------------------------------------------------------------
# _build_system_prompt — brand block integration
# ---------------------------------------------------------------------------

def test_build_system_prompt_brand_block_present_when_provided():
    prompt = _build_system_prompt(18, brand_name="HydroFlask", brand_context="")
    assert "BRAND IDENTITY" in prompt
    assert "HydroFlask" in prompt


def test_build_system_prompt_brand_block_absent_when_not_provided():
    prompt = _build_system_prompt(18)
    assert "BRAND IDENTITY" not in prompt


def test_build_system_prompt_brand_context_in_prompt():
    prompt = _build_system_prompt(18, brand_name="", brand_context="Eco-first, trail-ready.")
    assert "Eco-first, trail-ready." in prompt


def test_build_system_prompt_no_brand_name_no_context_leaves_prompt_clean():
    prompt = _build_system_prompt(18)
    # no stray brand placeholder text
    assert "brand_name" not in prompt
    assert "brand_context" not in prompt


# ---------------------------------------------------------------------------
# _build_user_content — brand fields in user message
# ---------------------------------------------------------------------------

def test_build_user_content_brand_name_at_top():
    content = _build_user_content("great bottle", _TRUTHS, None, brand_name="HydroFlask")
    lines = content.splitlines()
    assert lines[0] == "Brand: HydroFlask"


def test_build_user_content_brand_context_section():
    content = _build_user_content("great bottle", _TRUTHS, None, brand_context="Premium outdoor brand.")
    assert "Brand context:" in content
    assert "Premium outdoor brand." in content


def test_build_user_content_no_brand_fields_absent():
    content = _build_user_content("great bottle", _TRUTHS, None)
    assert "Brand:" not in content
    assert "Brand context:" not in content


def test_build_user_content_brief_still_present_with_brand():
    content = _build_user_content("great bottle", _TRUTHS, None, brand_name="HydroFlask")
    assert "great bottle" in content


# ---------------------------------------------------------------------------
# generate_script_variants — brand params reach LLM messages
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_generate_script_variants_brand_in_messages(monkeypatch):
    captured_messages: list[list[dict]] = []

    class _CapturingClient(FakeOpenAIClient):
        async def create(self, model, messages, **kw):
            captured_messages.append(messages)
            return await super().create(model, messages, **kw)

    client = _CapturingClient([_valid_json()])
    monkeypatch.setenv("MODEL_TEXT", "qwen-max")

    await generate_script_variants(
        "insulated water bottle",
        _TRUTHS,
        brand_name="HydroFlask",
        brand_context="Premium. Outdoor. Durable.",
        client=client,
    )

    assert captured_messages, "LLM was never called"
    system_msg = captured_messages[0][0]["content"]
    user_msg = captured_messages[0][1]["content"]

    assert "HydroFlask" in system_msg, "brand_name missing from system prompt"
    assert "Premium. Outdoor. Durable." in system_msg or "Premium. Outdoor. Durable." in user_msg, \
        "brand_context missing from LLM messages"
    assert "Brand: HydroFlask" in user_msg, "brand_name missing from user message"


@pytest.mark.asyncio
async def test_generate_script_variants_no_brand_clean_messages(monkeypatch):
    captured_messages: list[list[dict]] = []

    class _CapturingClient(FakeOpenAIClient):
        async def create(self, model, messages, **kw):
            captured_messages.append(messages)
            return await super().create(model, messages, **kw)

    client = _CapturingClient([_valid_json()])
    monkeypatch.setenv("MODEL_TEXT", "qwen-max")

    await generate_script_variants("insulated water bottle", _TRUTHS, client=client)

    system_msg = captured_messages[0][0]["content"]
    assert "BRAND IDENTITY" not in system_msg
    assert "Brand:" not in captured_messages[0][1]["content"]


# ---------------------------------------------------------------------------
# concept_agent_node — reads brand fields from state
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_concept_agent_node_passes_brand_from_state(monkeypatch):
    captured: dict = {}

    async def _fake_generate(**kwargs):
        captured.update(kwargs)
        return _four_variants()

    monkeypatch.setattr("agents.concept_agent.generate_script_variants", _fake_generate)

    await concept_agent_node({
        "job_id": "j1",
        "brief": "great bottle",
        "product_truths": _TRUTHS,
        "brand_name": "HydroFlask",
        "brand_context": "Outdoor premium brand.",
        "reasoning_trace": "",
    })

    assert captured.get("brand_name") == "HydroFlask"
    assert captured.get("brand_context") == "Outdoor premium brand."


@pytest.mark.asyncio
async def test_concept_agent_node_no_brand_fields_default_to_empty(monkeypatch):
    captured: dict = {}

    async def _fake_generate(**kwargs):
        captured.update(kwargs)
        return _four_variants()

    monkeypatch.setattr("agents.concept_agent.generate_script_variants", _fake_generate)

    await concept_agent_node({
        "job_id": "j1",
        "brief": "great bottle",
        "product_truths": _TRUTHS,
        "reasoning_trace": "",
    })

    assert captured.get("brand_name") == ""
    assert captured.get("brand_context") == ""


# ---------------------------------------------------------------------------
# CTA bridge validator behaviour with branded CTAs
#
# The prompt tells the LLM: "NEVER write 'Get yours' — use 'So get your
# HydroFlask.' or 'That's what HydroFlask is built for.'" We must verify
# that those forms PASS the deterministic _abrupt_cta_problem guard so they
# are not filtered out before reaching the Critic Chain.
# ---------------------------------------------------------------------------

def _beats_with_cta(cta_line: str) -> list[dict]:
    """5-beat sequence whose last beat carries the given CTA line."""
    return [
        {"t_start": 0,  "t_end": 3,  "line": "It holds everything you pack."},
        {"t_start": 3,  "t_end": 6,  "line": "Double-wall keeps cold 24 hours."},
        {"t_start": 6,  "t_end": 10, "line": "Matte midnight-blue powder coat."},
        {"t_start": 10, "t_end": 14, "line": "It still looks new a year later."},
        {"t_start": 14, "t_end": 18, "line": cta_line},
    ]


def test_abrupt_cta_bare_brand_name_flagged():
    # "Get HydroFlask" — 2 words, no connective, no back-reference → abrupt
    beats = _beats_with_cta("Get HydroFlask")
    assert _abrupt_cta_problem(beats) is not None


def test_abrupt_cta_connective_brand_name_passes():
    # "So get your HydroFlask." — starts with "so " → passes bridge check
    beats = _beats_with_cta("So get your HydroFlask.")
    assert _abrupt_cta_problem(beats) is None


def test_abrupt_cta_reference_brand_name_passes():
    # "That's what HydroFlask is built for." — "That's what" is a connective
    beats = _beats_with_cta("That's what HydroFlask is built for.")
    assert _abrupt_cta_problem(beats) is None


def test_abrupt_cta_long_branded_cta_passes():
    # 9 words — above ABRUPT_CTA_MAX_WORDS=8 so check is skipped entirely
    beats = _beats_with_cta("Get your HydroFlask before the next batch sells out.")
    assert _abrupt_cta_problem(beats) is None


def test_abrupt_cta_back_reference_with_brand_passes():
    # "It's your HydroFlask." — "It" matches _CTA_BRIDGE_REFERENCE_RE
    beats = _beats_with_cta("It's your HydroFlask.")
    assert _abrupt_cta_problem(beats) is None


# ---------------------------------------------------------------------------
# End-to-end: brand name survives generate_script_variants output
#
# The fake LLM returns variants whose CTA beat contains the brand name.
# We verify the returned variants still carry that brand name, i.e. the
# validator did not strip or reject them just because they mention the brand.
# ---------------------------------------------------------------------------

def _four_branded_variants() -> list[dict]:
    # Each variant has a distinct hook_type and emotional_trigger (required by _split_valid_invalid).
    # Hooks: concrete claim uses contrast marker; pov/curiosity gap/in-media-res use 2nd-person + concrete noun.
    # CTAs: all include brand name with a bridge connective or are 9+ words — passes _abrupt_cta_problem.
    # grounding_truth_ids covers all 3 mandatory tiers: t0=form_factor, t1=color, t2=construction_detail.
    data = [
        ("v1", FRAMEWORKS[0], "concrete claim",                 "desire",    "It will not let you down.",                    "So get your HydroFlask."),
        ("v2", FRAMEWORKS[1], "pov",                            "curiosity", "Grab your flask. Ready when you are.",         "That is what HydroFlask is built for."),
        ("v3", FRAMEWORKS[2], "curiosity gap",                  "relief",    "Your flask holds cold. All day. Every time.",  "It is your HydroFlask ready for anything."),
        ("v4", FRAMEWORKS[3], "relatable moment / in-media-res","FOMO",      "You fill it once. It keeps up all day.",       "Get your HydroFlask before the next batch sells out today."),
    ]
    return [
        {
            "variant_id": vid,
            "text": f"ad text ending with {cta}",
            "framework": fw,
            "hook_type": ht,
            "emotional_trigger": trigger,
            "grounding_truth_ids": ["t0", "t1", "t2"],
            "beats": [
                {"t_start": 0,  "t_end": 3,  "line": hook},
                {"t_start": 3,  "t_end": 9,  "line": "Cold stays cold for 24 hours."},
                {"t_start": 9,  "t_end": 16, "line": "Matte blue looks the same a year later."},
                {"t_start": 16, "t_end": 23, "line": "Double-wall keeps every degree inside."},
                {"t_start": 23, "t_end": 30, "line": cta},
            ],
            "target_length_sec": DEFAULT_TARGET_LENGTH_SEC,
        }
        for vid, fw, ht, trigger, hook, cta in data
    ]


@pytest.mark.asyncio
async def test_generate_script_variants_branded_cta_survives_validation(monkeypatch):
    """Brand name in CTA passes the validator — all 4 branded variants returned."""
    client = FakeOpenAIClient([json.dumps({"script_variants": _four_branded_variants()})])
    monkeypatch.setenv("MODEL_TEXT", "qwen-max")

    results = await generate_script_variants(
        "insulated water bottle",
        _TRUTHS,
        brand_name="HydroFlask",
        brand_context="Premium outdoor hydration.",
        client=client,
    )

    assert len(results) == 4, f"Expected 4 variants, got {len(results)}"
    cta_beats = [v["beats"][-1]["line"] for v in results]
    assert all("HydroFlask" in line for line in cta_beats), \
        f"Brand name missing from some CTA beats: {cta_beats}"


@pytest.mark.asyncio
async def test_generate_script_variants_brand_name_in_output_text(monkeypatch):
    """The 'text' field of returned variants contains the brand name."""
    client = FakeOpenAIClient([json.dumps({"script_variants": _four_branded_variants()})])
    monkeypatch.setenv("MODEL_TEXT", "qwen-max")

    results = await generate_script_variants(
        "insulated water bottle",
        _TRUTHS,
        brand_name="HydroFlask",
        client=client,
    )

    assert all("HydroFlask" in v["text"] for v in results), \
        "Brand name missing from variant text fields"


@pytest.mark.asyncio
async def test_concept_agent_node_output_contains_brand_name_in_cta(monkeypatch):
    """Full node: state with brand_name → output script_variants have brand in CTA."""
    monkeypatch.setenv("MODEL_TEXT", "qwen-max")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "fake")
    monkeypatch.setenv("DASHSCOPE_BASE_URL", "https://fake.api")

    monkeypatch.setattr(
        "agents.concept_agent.AsyncOpenAI",
        make_fake_async_openai([json.dumps({"script_variants": _four_branded_variants()})]),
    )

    result = await concept_agent_node({
        "job_id": "j1",
        "brief": "insulated water bottle",
        "product_truths": _TRUTHS,
        "brand_name": "HydroFlask",
        "brand_context": "Premium outdoor hydration brand.",
        "reasoning_trace": "",
    })

    variants = result.get("script_variants", [])
    assert len(variants) == 4, f"Expected 4 variants, got {len(variants)}"
    cta_beats = [v["beats"][-1]["line"] for v in variants]
    assert all("HydroFlask" in line for line in cta_beats), \
        f"Brand name not in CTA beats after full node execution: {cta_beats}"


# ---------------------------------------------------------------------------
# Tavily extraction path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_page_text_uses_tavily_when_key_set(monkeypatch):
    """When TAVILY_API_KEY is set, _get_page_text delegates to _extract_with_tavily."""
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-fake")

    with patch(
        "agents.brand_research_node._extract_with_tavily",
        new=AsyncMock(return_value="clean tavily content"),
    ) as mock_tavily:
        result = await _get_page_text("https://example.com")

    mock_tavily.assert_awaited_once_with("https://example.com")
    assert result == "clean tavily content"


@pytest.mark.asyncio
async def test_get_page_text_uses_httpx_fallback_when_no_key(monkeypatch):
    """When TAVILY_API_KEY is absent, _get_page_text falls back to httpx."""
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    with patch(
        "agents.brand_research_node._fetch_page_text_httpx",
        return_value="httpx scraped content",
    ) as mock_httpx:
        result = await _get_page_text("https://example.com")

    mock_httpx.assert_called_once_with("https://example.com")
    assert result == "httpx scraped content"


@pytest.mark.asyncio
async def test_brand_research_node_tavily_happy_path(monkeypatch):
    """End-to-end with Tavily: clean content → LLM → brand_context written."""
    fake_summary = "Peak Design makes premium camera gear with minimalist design."
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-fake")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "fake-key")
    monkeypatch.setenv("DASHSCOPE_BASE_URL", "https://fake.api")
    monkeypatch.setenv("MODEL_TEXT", "qwen-max")

    with (
        patch(
            "agents.brand_research_node._extract_with_tavily",
            new=AsyncMock(return_value="Peak Design website content here."),
        ),
        patch("agents.brand_research_node.AsyncOpenAI", make_fake_async_openai([fake_summary])),
    ):
        result = await brand_research_node({
            "job_id": "j1",
            "brand_url": "https://peakdesign.com",
            "brand_name": "Peak Design",
        })

    assert result.get("brand_context") == fake_summary


@pytest.mark.asyncio
async def test_brand_research_node_tavily_failure_degrades(monkeypatch):
    """If Tavily raises, node degrades gracefully — does not crash the job."""
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-fake")

    with patch(
        "agents.brand_research_node._extract_with_tavily",
        new=AsyncMock(side_effect=Exception("Tavily rate limit")),
    ):
        result = await brand_research_node({
            "job_id": "j1",
            "brand_url": "https://example.com",
        })

    assert result == {}


@pytest.mark.asyncio
async def test_brand_research_node_empty_tavily_result_skips_llm(monkeypatch):
    """Empty string from Tavily → node skips LLM and returns {}."""
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-fake")

    with patch(
        "agents.brand_research_node._extract_with_tavily",
        new=AsyncMock(return_value="   "),
    ):
        result = await brand_research_node({
            "job_id": "j1",
            "brand_url": "https://example.com",
        })

    assert result == {}
