"""
Brand Research Node — runs before product_truth_extractor.

If `brand_url` is in state, extracts the page content and calls the LLM to
produce a concise brand identity summary (tone, positioning, target customer,
key differentiators). Stores the result in state["brand_context"] so the
Concept Agent can write on-brand scripts with the correct CTA voice.

If `brand_url` is absent, returns {} immediately (no-op — zero cost).

Page extraction strategy (in priority order):
  1. Tavily Extract API (TAVILY_API_KEY set) — returns clean, structured text
     with JavaScript-rendered content, no HTML parsing needed.
  2. httpx fallback (no TAVILY_API_KEY) — raw HTML fetch + regex strip.
     Works without a Tavily account but misses JS-heavy pages.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re

import httpx
from openai import AsyncOpenAI

from agents._retry import create_completion
from graph.state import ProductCutState

logger = logging.getLogger("productcut.agents.brand_research_node")

_MAX_PAGE_CHARS = 8_000  # truncate scraped text to avoid flooding context


async def _extract_with_tavily(url: str) -> str:
    """Use Tavily Extract API to get clean page text.

    Returns the raw_content string from the first successful result, truncated
    to _MAX_PAGE_CHARS. Raises on network or API failure (caller handles).
    """
    from tavily import AsyncTavilyClient  # local import — only needed when key is set
    client = AsyncTavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    response = await client.extract(urls=[url])
    results = response.get("results", [])
    if not results:
        raise ValueError(f"Tavily returned no results for {url}")
    return results[0].get("raw_content", "")[:_MAX_PAGE_CHARS]


def _fetch_page_text_httpx(url: str) -> str:
    """Fallback: fetch URL and return readable text with HTML stripped."""
    resp = httpx.get(
        url, timeout=20.0, follow_redirects=True,
        headers={"User-Agent": "Mozilla/5.0 (compatible; ProductCut/1.0)"},
    )
    resp.raise_for_status()
    html = resp.text
    html = re.sub(r"<(script|style)[^>]*>.*?</(script|style)>", "", html,
                  flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:_MAX_PAGE_CHARS]


async def _get_page_text(url: str) -> str:
    """Extract page text using Tavily if available, httpx otherwise."""
    if os.environ.get("TAVILY_API_KEY"):
        logger.info("brand_research_node: using Tavily Extract for %s", url)
        return await _extract_with_tavily(url)
    logger.info("brand_research_node: no TAVILY_API_KEY — falling back to httpx for %s", url)
    return await asyncio.to_thread(_fetch_page_text_httpx, url)


async def brand_research_node(state: ProductCutState) -> dict:
    """LangGraph node: extract brand page and write brand_context to state."""
    brand_url = state.get("brand_url", "")
    brand_name = state.get("brand_name", "")

    if not brand_url:
        logger.info("brand_research_node: no brand_url — skipping")
        return {}

    try:
        page_text = await _get_page_text(brand_url)
    except Exception as exc:
        logger.warning("brand_research_node: failed to fetch %s: %s", brand_url, exc)
        return {}

    if not page_text.strip():
        logger.warning("brand_research_node: extracted empty text from %s — skipping LLM", brand_url)
        return {}

    brand_label = f'brand "{brand_name}"' if brand_name else "this brand"
    system = (
        "You are a brand strategist. Given a webpage's visible text content, write a "
        "concise brand identity summary in 120 words or fewer. Cover: what the brand sells, "
        "their tone of voice (formal/casual/playful/premium/etc.), their key differentiators, "
        "their target customer, and any notable taglines or positioning claims. "
        "Output only the summary, no headers, no bullet points."
    )
    user = (
        f"Webpage for {brand_label} ({brand_url}):\n\n"
        f"{page_text}\n\n"
        "Brand identity summary:"
    )

    client = AsyncOpenAI(
        api_key=os.environ["DASHSCOPE_API_KEY"],
        base_url=os.environ["DASHSCOPE_BASE_URL"],
        timeout=45.0,
    )
    try:
        try:
            brand_context = await create_completion(
                client,
                model=os.environ.get("MODEL_TEXT", "qwen-max"),
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0.2,
            )
        except Exception as exc:
            logger.warning(
                "brand_research_node: LLM summarization failed (%s); degrading to no brand_context", exc
            )
            return {}
    finally:
        await client.close()

    logger.info("brand_research_node: summarized context for %s (%d chars)", brand_url, len(brand_context))
    return {"brand_context": brand_context}
