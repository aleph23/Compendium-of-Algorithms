#!/usr/bin/env python3
"""
research.py
-----------
Pre-assembly research agent for the Compendium of Algorithms.

Runs BEFORE assemble_tome.py on every biweekly cycle.  Produces
research_context.json, which the assembler injects into generation prompts
so every page cites current literature.

What this script does
---------------------
1. ARXIV CRAWL          — query arXiv for papers published in the last 18 days
                          for each known topic category; extract titles, abstracts,
                          authors, arXiv IDs.

2. PAPERS WITH CODE     — pull recent SOTA benchmark results for relevant tasks.

3. emergent ARCH SCAN   — ask Agent (with web_search) to identify architectures
                          or techniques published recently that are NOT yet in our
                          topics_registry.  Outputs candidate entries the maintainer
                          can copy-paste into the registry.

4. WRITE CONTEXT FILE   — serialise everything to research_context.json, which
                          assemble_tome.py reads at generation time.

Usage
-----
  python scripts/research.py                 # full run
  python scripts/research.py --days 30       # extend look-back window
  python scripts/research.py --skip-arxiv    # skip arXiv (e.g. API down)
  python scripts/research.py --skip-emergent # skip the expensive LLM scan
"""

from argparse import ArgumentParser
import json
import os
import re
import sys
import time
from urllib import error, parse, request

import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path

import anthropic

# Paths
ROOT = Path(__file__).parent.parent
CONTEXT_FILE = ROOT / "research_context.json"

sys.path.insert(0, str(Path(__file__).parent))
from topics_registry import TOPICS, TOPICS_BY_ID, CATEGORY_ORDER, CATEGORY_LABELS

# API client
API_KEY = os.environ.get("LLM_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
if not API_KEY:
    sys.exit("ERROR: LLM_API_KEY environment variable not set.")

client = anthropic.Anthropic(api_key=API_KEY)
RESEARCH_MODEL = "claude-opus-4-7"   

# arXiv category mapping
# Maps our internal categories → arXiv search terms
ARXIV_QUERIES: dict[str, list[str]] = {
    "foundational": ["neural network", "perceptron", "backpropagation", "bayesian", "monte carlo"],
    "convolutional": ["convolutional neural network", "CNN", "image recognition"],
    "recurrent": ["recurrent neural network", "LSTM", "long short-term memory", "time-series model"],
    "attention": ["attention mechanism", "transformer", "language model"],
    "generative": ["generative model", "diffusion", "GAN", "adversarial network", "VAE", "auto-encoder", "u-net", "flow matching"],
    "graph": ["graph network", "GNN", "geometric deep learning"],
    "reinforcement": ["reinforcement learning", "policy gradient", "deep Q"],
    "modern": ["mixture of experts", "state space model", "vision transformer", "SOTA"],
}

# Also run a cross-cutting query for anything new we might have missed
ARXIV_WILDCARD_QUERIES = [
    "novel neural architecture 2024 2025 2026",
    "new deep learning architecture",
]

ARXIV_MAX_RESULTS = 12    # per query
PAPERS_WITH_CODE_N = 6    # top N tasks to fetch from PWC

# 1. ARXIV CRAWL
ARXIV_BASE = "https://export.arxiv.org/api/query"
ARXIV_NS = "http://www.w3.org/2005/Atom"

parser = ArgumentParser(description="Compendium of Algorithms — Research Agent")
parser.add_argument("--days", type=int, default=18, help="Look-back window in days (default: 14 for biweekly cadence)")
parser.add_argument("--skip-arxiv", action="store_true", help="Skip the arXiv crawl phase")
parser.add_argument("--skip-pwc", action="store_true", help="Skip the Papers With Code phase")
args = parser.parse_args()

def _arxiv_search(query: str, max_results: int = ARXIV_MAX_RESULTS,
                  days_back: int = 14) -> list[dict]:
    """Query arXiv and return a list of paper dicts."""
    since = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y%m%d")

    params = parse.urlencode({
        "search_query": f"all:{query}",
        "sortBy": "submittedDate",
        "sortOrder": "descending",
        "max_results": max_results,
    })
    url = f"{ARXIV_BASE}?{params}"

    try:
        with request.urlopen(url, timeout=20) as resp:
            xml_bytes = resp.read()
    except (error.URLError, TimeoutError) as exc:
        print(f"  ⚠️ arXiv fetch failed for '{query}': {exc}")
        return []

    root   = ET.fromstring(xml_bytes)
    papers = []

    for entry in root.findall(f"{{{ARXIV_NS}}}entry"):
        published = entry.findtext(f"{{{ARXIV_NS}}}published", "")
        # Filter to look-back window
        if published[:8].replace("-", "") < since:
            continue

        arxiv_id = entry.findtext(f"{{{ARXIV_NS}}}id", "").split("/abs/")[-1]
        title = re.sub(r"\s+", " ", entry.findtext(f"{{{ARXIV_NS}}}title", "")).strip()
        abstract = re.sub(r"\s+", " ", entry.findtext(f"{{{ARXIV_NS}}}summary", "")).strip()
        authors = [
            a.findtext(f"{{{ARXIV_NS}}}name", "")
            for a in entry.findall(f"{{{ARXIV_NS}}}author")
        ][:6]  # cap at 6

        papers.append({
            "arxiv_id": arxiv_id,
            "title": title,
            "authors": authors,
            "published": published[:10],
            "abstract": abstract[:600] + ("…" if len(abstract) > 600 else ""),
            "url": f"https://arxiv.org/abs/{arxiv_id}",
        })

    return papers


def run_arxiv_crawl(days_back: int = 18) -> dict[str, list[dict]]:
    """
    Returns {category: [paper, ...]} for all categories,
    plus a "wildcard" key for cross-cutting finds.
    """
    print("\n📡 Phase 1: arXiv crawl")
    results: dict[str, list[dict]] = {}

    all_queries = list(ARXIV_QUERIES.items())
    for cat, queries in all_queries:
        cat_papers: list[dict] = []
        for q in queries:
            print(f" searching [{cat}] → '{q}'")
            papers = _arxiv_search(q, days_back=days_back)
            cat_papers.extend(papers)
            time.sleep(2)   # arXiv rate-limit courtesy

        # Deduplicate by arxiv_id
        seen: set[str] = set()
        unique = []
        for p in cat_papers:
            if p["arxiv_id"] not in seen:
                seen.add(p["arxiv_id"])
                unique.append(p)

        results[cat] = unique
        print(f" → {len(unique)} unique paper(s) for '{cat}'")

    # Wildcard pass
    wildcard: list[dict] = []
    for q in ARXIV_WILDCARD_QUERIES:
        print(f" wildcard search → '{q}'")
        wildcard.extend(_arxiv_search(q, days_back=days_back))
        time.sleep(1)
    results["_wildcard"] = wildcard
    print(f" → {len(wildcard)} wildcard paper(s)")

    return results


# 2. PAPERS WITH CODE
PWC_TASKS_URL = "https://paperswithcode.com/api/v1/tasks/?format=json&page_size=50"
PWC_PAPERS_URL = "https://paperswithcode.com/api/v1/papers/?format=json&ordering=-published&page_size=5"

TASK_KEYWORDS = [
    "language modelling", "text generation", "image generation", "semantic segmentation",
    "training optimization", "inference engine", "novel architecture", "perplexity", "music generation", "image-to-video"
]

def _pwc_fetch(url: str) -> dict | None:
    try:
        req = request.Request(url, headers={"User-Agent": "Compendium/1.0"})
        with request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read())
    except Exception as exc:
        print(f" ⚠️  PWC fetch failed: {exc}")
        return None


def run_pwc_crawl() -> list[dict]:
    """Return recent high-profile papers from Papers With Code."""
    print("\n📡 Phase 2: Papers With Code")
    data = _pwc_fetch(PWC_PAPERS_URL)
    if not data:
        return []

    papers = []
    for item in data.get("results", [])[:PAPERS_WITH_CODE_N]:
        papers.append({
            "title": item.get("title", ""),
            "url": item.get("url_pdf") or item.get("url_abs", ""),
            "published": item.get("published", ""),
            "stars": item.get("stars", 0),
            "abstract": (item.get("abstract") or "")[:500],
        })

    print(f" → {len(papers)} paper(s) from Papers With Code")
    return papers

# 3. emergent ARCHITECTURE SCAN (Agent + web_search)
def run_emergent_scan(known_ids: list[str]) -> dict:
    """
    Ask Agent (with web_search) to identify architectures published recently
    that are NOT in our registry.  Returns a dict with:
      - summary: prose paragraph
      - candidates: list of dicts ready to paste into topics_registry.py
    """
    print("\n🔍 Phase 3: emergent architecture scan (Agent + web search)")

    known_titles = [TOPICS_BY_ID[i]["title"] for i in known_ids if i in TOPICS_BY_ID]
    known_str = "\n".join(f"  - {t}" for t in known_titles)

    prompt = f"""You are a research assistant for the *Compendium of Algorithms* — a living technical reference book on AI neural-network architectures.

Your task: identify AI/ML neural-network architectures or foundational techniques that have been proposed, published, or achieved significant traction in the past 6 months, and that are NOT yet in our registry. AND identify any new methods, techniques, or optimizations we have not yet catalogued.

## Already covered (do NOT suggest these)
{known_str}

## Instructions
1. Use your web_search tool to find recent arXiv papers, blog posts, and announcements about new neural-network architectures or new methods for existing architectures.
2. Focus on architectures that are **structurally novel** — new layer types, new training paradigms, new inductive biases — as well as incremental scaling of existing models.
3. For each candidate you find, output a JSON entry in this exact schema:
   {{
     "id": "kebab-case-id",
     "title": "Full Human-Readable Title",
     "era": "YYYY or YYYY–YYYY",
     "category": "one of: foundational | convolutional | recurrent | attention | generative | graph | reinforcement | modern",
     "sidebar_order": <integer — place it after the last entry in its category>,
     "depends_on": ["id-of-prerequisite", ...],
     "summary": "One sentence — the core insight.",
     "source_urls": ["https://arxiv.org/abs/...", ...]
   }}

## Output format
Respond with a JSON object (no markdown fences) with two keys:
- "summary": a 2–3 sentence prose paragraph describing what you found
- "candidates": an array of the above schema objects ("candidates": "nothing new" if nothing genuinely new)

Search broadly before concluding. Be selective — only include architectures with clear structural novelty or genuinely new methods/optimizations and at least one credible source.  Regard sources without code, algorithm, or some means of method reproduction with high skepticism.
"""

    try:
        msg = client.messages.create(
            model=RESEARCH_MODEL,
            max_tokens=8196,
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            messages=[{"role": "user", "content": prompt}],
        )

        # Collect text blocks from the response (may interleave with tool_use blocks)
        text_parts = [
            getattr(block, "text")
            for block in msg.content
            if getattr(block, "type", None) == "text" and getattr(block, "text", "").strip()
        ]
        raw = "\n".join(text_parts).strip()

        # Strip accidental markdown fences
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)

        result = json.loads(raw)
        candidates = result.get("candidates", [])
        summary = result.get("summary", "")

        print(f" → {len(candidates)} emergent candidate(s) identified")
        if summary:
            print(f" ℹ️ {summary[:200]}…" if len(summary) > 200 else f" ℹ️ {summary}")

        return {"summary": summary, "candidates": candidates}

    except json.JSONDecodeError as exc:
        print(f" ⚠️ Could not parse emergent-scan response as JSON: {exc}")
        return {"summary": "Parse error — see logs.", "candidates": []}
    except Exception as exc:
        print(f" ⚠️ emergent scan failed: {exc}")
        return {"summary": str(exc), "candidates": []}


# 4. TOPIC-LEVEL CONTEXT ENRICHMENT
def build_per_topic_context(arxiv_results: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """
    For each topic, collect the arXiv papers most likely to be relevant.
    Simple keyword matching — good enough for prompt injection.
    """
    per_topic: dict[str, list[dict]] = {}

    for topic in TOPICS:
        tid = topic["id"]
        cat = topic["category"]
        keywords = set(topic["title"].lower().split())
        keywords.update(topic["summary"].lower().split())

        # Score papers in this category + wildcard pool
        pool = arxiv_results.get(cat, []) + arxiv_results.get("_wildcard", [])
        scored: list[tuple[int, dict]] = []

        for paper in pool:
            haystack = (paper["title"] + " " + paper["abstract"]).lower()
            score = sum(1 for kw in keywords if len(kw) > 4 and kw in haystack)
            if score > 0:
                scored.append((score, paper))

        # Top 3 most relevant
        scored.sort(key=lambda x: -x[0])
        per_topic[tid] = [p for _, p in scored[:3]]

    return per_topic

# 5. WRITE CONTEXT FILE
def write_context(
    arxiv_results: dict[str, list[dict]],
    pwc_results: list[dict],
    emergent: dict,
    per_topic: dict[str, list[dict]],
    days_back: int,
):
    context = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "coverage_days": days_back,
        "coverage_from": (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y-%m-%d"),
        "coverage_to": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "arxiv_by_category": arxiv_results,
        "papers_with_code": pwc_results,
        "emergent": {
            "summary": emergent.get("summary", ""),
            "candidates": emergent.get("candidates", []),
        },
        "per_topic": per_topic,
    }
    CONTEXT_FILE.write_text(json.dumps(context, indent=2, ensure_ascii=False))
    print(f"\n✅ Research context written → {CONTEXT_FILE.relative_to(ROOT)}")

    # Print emergent candidates as copy-pasteable registry entries
    candidates = emergent.get("candidates", [])
    if candidates:
        print("\n" + "═" * 70)
        print("🆕 emergent ARCHITECTURE CANDIDATES")
        print("Review, then copy-paste into scripts/topics_registry.py")
        print("═" * 70)
        for c in candidates:
            urls = c.pop("source_urls", [])
            print(f"\n# Sources: {', '.join(urls)}")
            print(f" {json.dumps(c, indent=4)},")
        print("═" * 70 + "\n`")


# MAIN
def main():
    print(f"🔬 Research agent starting — {args.days}-day look-back window\n")

    # Phase 1: arXiv
    arxiv_results: dict[str, list[dict]] = {}
    if not args.skip_arxiv:
        arxiv_results = run_arxiv_crawl(days_back=args.days)
    else:
        print("⏭️ Skipping arXiv crawl")

    # Phase 2: Papers With Code
    pwc_results: list[dict] = []
    if not args.skip_pwc:
        pwc_results = run_pwc_crawl()
    else:
        print("⏭️ Skipping Papers With Code")

    # Phase 3: emergent scan
    emergent: dict = {"summary": "", "candidates": []}
    known_ids = [t["id"] for t in TOPICS]
    run_emergent_scan(known_ids)

    # Phase 4: Per-topic context mapping
    print("\n🗺️ Phase 4: Mapping papers to topics")
    per_topic = build_per_topic_context(arxiv_results)
    covered = sum(1 for v in per_topic.values() if v)
    print(f" → {covered}/{len(TOPICS)} topics have at least one recent paper")

    # Phase 5: Write output
    write_context(arxiv_results, pwc_results, emergent, per_topic, args.days)

if __name__ == "__main__":
    main()
