#!/usr/bin/env python3
"""
assemble_tome.py
----------------
Uses an Anthropic-API-compatible API to generate/refresh Starlight-ready Markdown pages.

Run modes
---------
  python assemble_tome.py            # incremental: only changed/new topics
  python assemble_tome.py --all      # force-regenerate every topic
  python assemble_tome.py --topic resnet  # regenerate one topic by id

Environment
-----------
  LLM_API_KEY   Anthropic-API-compatible key (set in GitHub Actions secrets)
"""

import argparse
import json
import os
import re
import sys
import time
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import anthropic

# Paths
ROOT          = Path(__file__).parent.parent
CONTENT_DIR   = ROOT / "src" / "content" / "docs"
STATE_FILE    = ROOT / ".assembler_state.json"   # tracks content hashes
CONTEXT_FILE  = ROOT / "research_context.json"   # written by research.py

# Import topic registry
sys.path.insert(0, str(Path(__file__).parent))
from topics_registry import TOPICS, TOPICS_BY_ID, CATEGORY_ORDER, CATEGORY_LABELS

# Anthropic client
API_KEY = os.environ.get("LLM_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
if not API_KEY:
    sys.exit("ERROR: LLM_API_KEY environment variable not set.")

client = anthropic.Anthropic(api_key=API_KEY)
MODEL      = "claude-sonnet-4-7"      # Use the best available model for quality
MAX_TOKENS = 32000
RETRY_SLEEP = 20                    # seconds between rate-limit retries
MAX_RETRIES = 3


# RESEARCH CONTEXT  (produced by research.py, consumed here)
def load_research_context() -> dict:
    """Load research_context.json if present; return empty structure otherwise."""
    if CONTEXT_FILE.exists():
        ctx = json.loads(CONTEXT_FILE.read_text())
        generated = ctx.get("generated_at", "unknown")
        covered   = ctx.get("coverage_from", "?")
        to        = ctx.get("coverage_to",   "?")
        print(f"📖 Research context loaded  (generated {generated[:10]}, "
              f"covering {covered} → {to})")
        return ctx
    else:
        print("⚠️  No research_context.json found — "
              "run research.py first for best results. "
              "Proceeding with model-knowledge only.")
        return {}


def format_research_block(topic: dict, ctx: dict) -> str:
    """
    Build a 'Recent Research Intelligence' section for the generation prompt.
    Returns an empty string if no relevant context exists.
    """
    if not ctx:
        return ""

    tid     = topic["id"]
    lines   = []

    # ── Per-topic recent papers ───────────────────────────────────────────────
    papers = ctx.get("per_topic", {}).get(tid, [])
    if papers:
        lines.append("## Recent Research Intelligence")
        lines.append(
            "The following papers were published in the last "
            f"{ctx.get('coverage_days', 14)} days and are likely relevant to this topic. "
            "Cite them by URL where appropriate, and integrate any novel findings or "
            "corrections into the page content.\n"
        )
        for p in papers:
            authors_str = ", ".join(p.get("authors", []))
            lines.append(
                f"- **{p['title']}** ({p.get('published', '')})\n"
                f"  Authors: {authors_str}\n"
                f"  URL: {p.get('url', '')}\n"
                f"  Abstract excerpt: {p.get('abstract', '')}\n"
            )

    # ── emergent candidates note ──────────────────────────────────────────────
    emergent_summary = ctx.get("emergent", {}).get("summary", "")
    if emergent_summary:
        lines.append(
            "\n:::note\n"
            f"**Broader landscape note:** {emergent_summary}\n"
            "Consider a brief forward-reference in the 'See Also' section if relevant.\n"
            ":::\n"
        )

    return "\n".join(lines) if lines else ""


# PROMPT TEMPLATE
def build_prompt(topic: dict, all_topics: list[dict], research_ctx: dict = {}, target_dir: str = "frontier", is_bootstrap: bool = False) -> str:
    """Build the full generation prompt for one architecture topic."""

    dep_titles = [
        TOPICS_BY_ID[d]["title"]
        for d in topic.get("depends_on", [])
        if d in TOPICS_BY_ID
    ]
    dep_str = ", ".join(dep_titles) if dep_titles else "none"

    research_block = format_research_block(topic, research_ctx)

    bootstrap_instruction = ""
    if is_bootstrap:
        bootstrap_instruction = (
            "\n**BOOTSTRAP INSTRUCTION:** This is the foundational instantiation of the Compendium. "
            "Establish a highly authoritative, timeless, and encyclopedic baseline tone. "
            "Do NOT mention that the book is new, a work in progress, or being written—act as if this is a finished, prestigious reference manual.\n"
        )

    prompt_template = f"""As a professional technical author write a chapter of the *Compendium of Algorithms: The Formal Specification and Structural Composition of Probabilistic Computational Systems and Their Occasional Interdependence* — a living reference on AI neural-network architectures. Your audience ranges from new engineers to seasoned ML researchers needing reference material. Write with precision and clarity.

## Assignment
Write a complete Starlight-compatible Markdown page for the following architecture:

- **Title**: {topic["title"]}
- **Era**: {topic["era"]}
- **Category**: {topic["category"]}
- **One-line summary**: {topic["summary"]}
- **Builds on**: {dep_str}

## Required Sections (in this exact order)

### 1. Frontmatter
Emit YAML frontmatter block:
```yaml
---
title: "{topic["title"]}"
description: "<one crisp sentence>"
sidebar:
  order: {topic["sidebar_order"]}
  badge:
    text: "{topic["era"]}"
    variant: note
---
```

### 2. Overview
5 paragraphs covering: the problem it solves, historical context, and why it matters today.

### 3. Architecture at a Glance  (Mermaid structural diagram)
Emit a **Mermaid flowchart** showing all major components and all of their connections.
Use `graph TD` orientation.  Label each node with its widely recognized name.
Wrap in a fenced code block: ```mermaid ... ```

### 4. Operational Flow  (Mermaid sequence or state diagram)
Emit a **Mermaid sequenceDiagram or stateDiagram-v2** tracing one full pass through the architecture step by step.  Multiple diagrams are allowed if needed for clarity.

### 5. Layer Breakdown
For each distinct layer-type in this architecture, create a sub-section `#### LayerName` containing:
- **Purpose**: one sentence
- **Inputs / Outputs**: shape notation (e.g., `(B, T, d_model)`)
- **Learnable parameters**: list with shapes
- **Key hyperparameters**: list
- **Effective methods**: list with `code` examples

### 6. Core Equations
For EACH key equation:
1. Display the equation in LaTeX inside a `$$...$$` block.
2. Immediately follow with a **Symbol Key** table:

| Symbol | Plain-English meaning |
|--------|-----------------------|
| symbol | meaning |

3. Then write a **Plain-English Paragraph** — one cohesive paragraph that describes exactly what the equation computes, with every symbol's name in parentheses after the corresponding English word.  Example style: "The output (y-hat) is computed by multiplying the input vector (x) by the weight matrix (W) and adding the bias (b)..."

### 7. Complexity Analysis
Table with rows: Time Complexity, Space Complexity, Typical Parameter Count, Typical FLOP Count (per forward pass).

### 8. Strengths & Limitations
Two bullet lists.

### 9. Key Milestones
Timeline of important papers or model releases related to this architecture (3–7 items) with links.
Format: `- **YYYY** — ![link](*Paper title*) — one-line impact`

### 10. Reference Material
Off-site reference material, implementation documentation, code examples/snippets.

### 11. See Also
Cross-links to related topics in this book. Use Starlight relative links:
`Topic Title`

## Style Rules
- Err of the side of too much information
- Always embed documentation links as much and as often as possible.
- All new research must be appropriately sited and linked.
- Use `:::note`, `:::tip`, `:::caution` admonitions sparingly for genuinely important callouts.
- All math MUST be valid KaTeX (used by Starlight).
- All Mermaid MUST be valid Mermaid v10+ syntax. Avoid parentheses in node labels; use square brackets.
- Do not use HTML tags.
- Write the full Markdown document from frontmatter to the last line. No preamble, no explanation outside the document itself.
- You can have some fun with 19th Century language, so long as it does not muddle the information.
{bootstrap_instruction}

{research_block}
"""
    return prompt_template.replace("{target_dir}", target_dir)

# STATE MANAGEMENT
def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}

def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2))

def topic_hash(topic: dict) -> str:
    """Fingerprint a topic definition — changes when we bump its content spec."""
    blob = json.dumps(topic, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]

def needs_update(topic: dict, state: dict) -> bool:
    tid = topic["id"]
    return state.get(tid, {}).get("hash") != topic_hash(topic)


# CONTENT GENERATION
def call_llm(prompt: str) -> str:
    """Call Agent with retries on rate-limit errors."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            msg = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text
        except anthropic.RateLimitError:
            if attempt < MAX_RETRIES:
                print(f"  ⏳ Rate limited — retrying in {RETRY_SLEEP}s ({attempt}/{MAX_RETRIES})")
                time.sleep(RETRY_SLEEP * attempt)
            else:
                raise
        except anthropic.APIStatusError as e:
            print(f"  ⚠️  API error: {e.status_code} — {e.message}")
            raise


def write_topic_page(topic: dict, research_ctx: dict = {}, target_dir: str = "frontier", is_bootstrap: bool = False) -> Path:
    """Generate and write a Markdown page for one topic. Returns the output path."""
    print(f"  📝 Generating: {topic['title']}")

    prompt   = build_prompt(topic, TOPICS, research_ctx, target_dir, is_bootstrap)
    markdown = call_llm(prompt)

    # Strip any accidental leading/trailing whitespace or code fences
    markdown = markdown.strip()
    if markdown.startswith("```"):
        markdown = re.sub(r"^```[a-z]*\n?", "", markdown)
        markdown = re.sub(r"\n?```$", "", markdown)

    # CANON GUARD — hard tripwire
    # The assembler usually ONLY writes inside frontier/.
    # Any path resolving into received-canon/ requires the explicit target.
    # Use cmd argument '--target received-canon' for updating the main book.
    base_root = CONTENT_DIR / target_dir

    # Ensure output directory under target/
    category_dir = base_root / topic["category"]
    category_dir.mkdir(parents=True, exist_ok=True)

    out_path = category_dir / f"{topic['id']}.md"

    # Paranoid resolution check — catches symlink tricks too
    try:
        resolved = out_path.resolve()
        canon_resolved = (CONTENT_DIR / "received-canon").resolve()
        if target_dir != "received-canon":
            assert not str(resolved).startswith(str(canon_resolved)), (
                f"CANON GUARD VIOLATION: attempted write to received-canon path: {resolved}"
            )
    except AssertionError as guard_err:
        sys.exit(f"\n🚫 {guard_err}\n"
                 "The assembler must never modify received-canon without explicit override. Aborting.")

    out_path.write_text(markdown, encoding="utf-8")
    print(f"  ✅  Written → {out_path.relative_to(ROOT)}")
    return out_path


# INDEX GENERATION
def write_category_index(category: str, topics_in_cat: list[dict], target_dir: str = "frontier"):
    """Write a category landing page listing all topics."""
    label = CATEGORY_LABELS.get(category, category.title())
    target_label = target_dir.replace("-", " ").title()
    lines = [
        "---",
        f'title: "{label}"',
        f'description: "All {label} architectures in {target_label}."',
        "---",
        "",
        f"# {label}",
        "",
        "| Architecture | Era | Summary |",
        "|---|---|---|",
    ]
    for t in sorted(topics_in_cat, key=lambda x: x["sidebar_order"]):
        lines.append(
            f"| [{t['title']}](/{target_dir}/{t['category']}/{t['id']}/) "
            f"| {t['era']} | {t['summary']} |"
        )

    cat_dir = CONTENT_DIR / target_dir / category
    cat_dir.mkdir(parents=True, exist_ok=True)
    (cat_dir / "index.md").write_text("\n".join(lines), encoding="utf-8")


def write_home_index(target_dir: str = "frontier"):
    """Write the root index page."""
    lines = [
        "---",
        'title: "Compendium of Algorithms"',
        'description: "The Formal Specification and Structural Composition of Probabilistic Computational Systems and Their Occasional Interdependence. "',
        "template: splash",
        "hero:",
        '  title: "Compendium of Algorithms"',
        '  tagline: "Every major neural-network architecture — used, excused, and imaged — from  perceptron to tomorrow."',
        "  actions:",
        '    - text: "Start Reading →"',
        '      link: /foundations/perceptron/',
        '      variant: primary',
        "---",
        "",
        "## What's inside",
        "",
        "Each page contains:",
        "",
        "- **Narrative overview** - historical context and motivation",
        "- **Structural diagram** - Mermaid flowchart of all components",
        "- **Operational flow** - step-by-step data-pass diagram",
        "- **Layer breakdown** - every layer: purpose, shapes, parameters, hyperparameters, tuning",
        "- **Core equations** - LaTeX math + symbol key table + plain-English paragraph",
        "- **Complexity analysis** - time, space, parameter counts",
        "- **Strengths & limitations**",
        "- **Key milestones** - landmark papers and releases",
        "- **Reference material** - Implementation documentation and usage code snippets"
        "",
        "## Architectures covered",
        "",
    ]
    for cat in CATEGORY_ORDER:
        label = CATEGORY_LABELS.get(cat, cat)
        cat_topics = [t for t in TOPICS if t["category"] == cat]
        lines.append(f"### {label}")
        for t in sorted(cat_topics, key=lambda x: x["sidebar_order"]):
            lines.append(f"- [{t['title']}](/{target_dir}/{t['category']}/{t['id']}/) — {t['era']}")
        lines.append("")

    (CONTENT_DIR / "index.md").write_text("\n".join(lines), encoding="utf-8")
    print("  ✅  Home index written")


# MAIN
def parse_args():
    p = argparse.ArgumentParser(description="Compendium of Algorithms Assembler")
    p.add_argument("--all",   action="store_true", help="Regenerate every topic")
    p.add_argument("--topic", metavar="ID",        help="Regenerate a single topic by id")
    p.add_argument("--target", choices=["frontier", "received-canon"], default="frontier",
                   help="Target directory for output (default: frontier)")
    p.add_argument("--bootstrap", action="store_true", help="Inject foundational tone instructions for initial creation")
    return p.parse_args()


def main():
    args   = parse_args()
    state  = load_state()
    CONTENT_DIR.mkdir(parents=True, exist_ok=True)
    target_dir = args.target

    # Load research context (produced by research.py)
    research_ctx = load_research_context()

    # Determine work queue
    if args.topic:
        if args.topic not in TOPICS_BY_ID:
            sys.exit(f"ERROR: unknown topic id '{args.topic}'. Check topics_registry.py")
        queue = [TOPICS_BY_ID[args.topic]]
    elif args.all:
        queue = TOPICS
    else:
        queue = [t for t in TOPICS if needs_update(t, state)]

    if not queue:
        print("✨ Everything is up-to-date. Nothing to regenerate.")
    else:
        print(f"🚀 Assembler starting — {len(queue)} topic(s) to generate\n")
        for topic in queue:
            try:
                write_topic_page(topic, research_ctx, target_dir, args.bootstrap)
                state[topic["id"]] = {
                    "hash":      topic_hash(topic),
                    "generated": datetime.now(timezone.utc).isoformat(),
                    "title":     topic["title"],
                }
                save_state(state)          # save after each page (crash-safe)
                time.sleep(1)              # polite pacing
            except Exception as exc:
                print(f"  ❌  Failed: {topic['id']} — {exc}")
                # Continue with remaining topics rather than aborting
                continue

    # Always regenerate indices (cheap, keeps nav fresh)
    print("\n📚 Writing category indices…")
    for cat in CATEGORY_ORDER:
        cat_topics = [t for t in TOPICS if t["category"] == cat]
        if cat_topics:
            write_category_index(cat, cat_topics, target_dir)
    write_home_index(target_dir)

    print(f"\n🎉 Done — {len(queue)} page(s) generated, indices updated.")


if __name__ == "__main__":
    main()
