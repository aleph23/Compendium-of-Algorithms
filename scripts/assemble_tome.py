#!/usr/bin/env python3

"""
assemble_tome.py
----------------
Uses an OpenAI-API-compatible provider to generate/refresh Starlight-ready Markdown pages.

Run modes
---------
  python assemble_tome.py           # incremental: only changed/new topics
  python assemble_tome.py --bootstrap   # start the compendium
  python assemble_tome.py --id resnet   # regenerate one topic by id
  python assemble_tome.py --target received-canon   # generate into the received canon

Environment
-----------
  LLM_API_KEY = OpenAI-API-compatible key (set in GitHub Actions secrets)
"""

import argparse
import hashlib
import json
import os
import re
import requests
import sys
import time
import logging
from datetime import datetime, timezone
from pathlib import Path

# Topic registry
sys.path.insert(0, str(Path(__file__).parent))
from topics_registry import TOPICS, TOPICS_BY_ID, CATEGORY_ORDER, CATEGORY_LABELS

API_KEY = os.environ.get("LLM_API_KEY") or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
if not API_KEY:
    sys.exit("ERROR: LLM_API_KEY environment variable not set.")

# Paths
ROOT = Path(__file__).parent.parent
CONTENT_DIR = ROOT / "src" / "content" / "docs"
STATE_FILE = ROOT / ".assembler_state.json"   # tracks content hashes
CONTEXT_FILE = ROOT / "research_context.json"   # written by research.py

# Arguments
pa = argparse.ArgumentParser(description="Compendium of Algorithms assembler")
directory = (lambda pa: (pa.add_argument("--target", choices=["frontier", "received-canon"], default="frontier", help="Target directory for output.  Corresponds to first (Established) or second (Emergent) book. (default: frontier)")), pa.parse_args()[1]) # type: ignore
target_dir = directory.target # type: ignore
topic = (lambda pa: (pa.add_argument("--id", metavar="ID", help="Generate a single topic by id")), pa.parse_args()[1]) # pyright: ignore[reportIndexIssue]
topic = topic.id # type: ignore
cap = (lambda pa: (pa.add_argument("--limit", type=int, default=0, help="Cap the number of topics generated this run")), pa.parse_args()[1]) # type: ignore
limit = cap.limit # type: ignore

bootstrap = "--bootstrap" in sys.argv
index_only = "--index" in sys.argv

inference_provider = "https://openrouter.ai"

AUTHOR = "anthropic/claude-opus-4-8"  # Use the best available model for quality
REVIEWER = "zai/glm-5-2"
MAX_TOKENS = 32000
RETRY_SLEEP = 45    # seconds between retries
MAX_RETRIES = 3

MIN_PARAGRAPHS = 5
MIN_MERMAID = 3
MIN_LINKS = 8
MIN_CODE_BLOCKS = 3

logging.basicConfig(level=logging.WARNING)
logging.getLogger("httpx").setLevel(logging.DEBUG)
print(f"[DEBUG] Loaded API Key: '{API_KEY[:8]}...{API_KEY[-4:]}' (Length: {len(API_KEY)})")

def is_page_complete(content: str):
    """
    Checks if a generated page is complete, first by quantitative structural
    checks, then by a qualitative review from an LLM.
    """

    def quantity(content: str) -> bool:
        """Structural check only. Does not judge quality, just presence."""
        paragraphs = [
            p for p in content.split("\n\n")
            if len(p.strip()) > 80 and not p.strips().startswith(("#", "|", "-", "```"))
        ]

        if len(paragraphs) < MIN_PARAGRAPHS:
            return False
        mermaid = re.findall(r"```mermaid", content)
        if len(mermaid) < MIN_MERMAID:
            return False
        if len(re.findall(r"\[[^\]]+\]\(https?://", content)) < MIN_LINKS:
            return False
        fences = re.findall(r"```[a-zA-Z]", content)
        if (len(fences) - len(mermaid)) < MIN_CODE_BLOCKS:
            return False
        return True

    def quality(content: str) -> tuple[bool, str]:
        """
        Submit entry to REVIEWER for subjective completion analysis.
        Returns a tuple (is_complete, reason).
        """
        review_prompt = f"""
        As an editor and publisher of technical reference material, decide if this chapter of *The Compendium of Algorithms* is suitable for publication. It is a Quick Reference, not Encyclopedic.
        It has either already passed a check of minimum quantitative requirements or has been granted an exception. Now it needs the thoughtful, subjective consideration of an editor.
        Perform a informed review of the material, unconcerned with checking simple boxes to find if the spirit of the specification has been met.
        Read the text for tone, flow, and completeness. Does it feel like a finished, authoritative, and polished piece of technical writing, but without the drone of a hypnosis script?
        Or is it more of a draft; thin facades masking missing content?

        The author was prompted with: {json.dumps(author_prompt)}

        Relying on your own knowledge and expertise, decide if it makes the cut. Respond with a single JSON object of two keys:
        - "complete": boolean
        - "reason": "string" If affirmative, a one-sentence explanation. If 'needs improvement', detail where and how you feel it falls short.

        Here is the chapter to review:
        ---
        {content}
        ---
        """
        try:
            response = requests.post(
                url=f"{inference_provider}/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer <OPENROUTER_API_KEY>",
                    "X-OpenRouter-Cache": "true",
                    "X-OpenRouter-Cache-TTL": "3600"
                },
                json={
                    "max_completion_tokens": 500,
                    "model": REVIEWER,
                    "response_format": {"type": "json_object"},
                    "temperature": 0.2,
                    "messages": [{
                        "role": "user",
                        "content": review_prompt}]
                }
            )
            response.raise_for_status()

            review_data = json.loads(response.choices[0].message.content)
            return bool(review_data.get("complete")), review_data.get("reason", "No reason provided.")
        except requests.exceptions.RequestException as e:
            logging.error(f"Quality review API request failed: {e}")
            return False, f"API request failed: {e}"
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            logging.error(f"Quality review API response parsing failed: {e}")
            return False, f"API response parsing failed: {e}"
        except Exception as e:
            logging.error(f"Quality review API call failed: {e}")
            return False, f"API call failed: {e}"

    if not quantity(content):
        complete = "Tests\nQuantitative: Failed\n"
    else:
        complete = "Tests\nQuantitative: Passed\n"
    subjective = quality(content)
    complete.append(f"Qualitative: {subjective}")

    return complete


# RESEARCH CONTEXT (produced by research.py, consumed here)
def load_research_context() -> dict:
    """Load research_context.json if present; return empty structure otherwise."""
    if CONTEXT_FILE.exists():
        ctx = json.loads(CONTEXT_FILE.read_text())
        print(f"Research context loaded: {ctx.get('coverage_from','?')} to {ctx.get('coverage_to','?')}")
        return ctx
    else:
        print("No research_context.json found—run research.py first for best results.\n"
              "Proceeding with model-knowledge only.")
        return {}

def format_research_block(topic: dict, ctx: dict):
    """
    Build a 'Recent Research Intelligence' section for the generation prompt.
    Returns an empty string if no relevant context exists.
    """
    if not ctx:
        return ""

    # Per-topic recent papers
    papers = ctx.get("per_topic", {}).get(topic["id"], [])
    lines = []
    if papers:
        lines.append("## Recent Research Intelligence")
        lines.append(
            f"These papers appeared in the last {ctx.get('coverage_days', 14)} days "
            "and may be relevant. Cite by URL where they genuinely inform the text."
        )
        for p in papers:
            lines.append(
                f"- {p['title']} ({p.get('published','')})\n"
                f"  Authors: {', '.join(p.get('authors', []))}\n"
                f"  URL: {p.get('url','')}\n"
                f"  Abstract: {p.get('abstract','')}"
            )

    summary = ctx.get("emergent", {}).get("summary", "")
    if summary:
        lines.append(f"\n:::note\n**Broader landscape note:** {summary}\n"
            "Consider a brief forward and possible reference cross-links.\n:::\n")

    return "\n".join(lines) if lines else ""


# GENERATION PROMPT OBJECT
def author_prompt(topic: dict, research_ctx: dict, bootstrap: bool):
    """Build the full generation prompt for one architecture topic."""
    deps = [
        TOPICS_BY_ID[d]["title"]
        for d in topic.get("depends_on", [])
        if d in TOPICS_BY_ID
    ]
    dep_str = ", ".join(deps) or "none"
    research_block = format_research_block(topic, research_ctx)

    existing_instruction = ""
    existing_path = CONTENT_DIR / target_dir / topic["category"] / f"{topic['id']}.md"
    if existing_path.exists():
        existing_content = existing_path.read_text(encoding="utf-8")
        existing_instruction = (
            "\n## Existing Page\n"
            "Below is the current version of this page. "
            "Preserve anything still accurate; update what has changed; add new sections where needed. "
            "Do NOT regress or lose existing content.\n\n"
            f"{existing_content}\n"
        )
    bootstrap_instruction = ""
    if bootstrap:
        bootstrap_instruction = (
            "This chapter is part of the foundational instantiation of the work. "
            "Write as a finished, authoritative reference. Do not mention that the "
            "book is new or in progress.\n"
        )

    prompt_body = f"""You are writing a chapter of the Compendium of Algorithms: The Formal Specification and Structural Composition of Probabilistic Computational Systems and Their Occasional Interdependence. It is a technical reference on neural network architectures.  Our goal is to impart new dexterity.  Readers range from scholastic neophyte wanting an overview to veteran white coat looking for a easy desk reference, to finger-pad-calloused coder crossing specialties. Assume as a low baseline level: College Senior reading comprehension, Algebra, Introductory Trigonometry, and a high-level definition of Calculus that is niave of its functional operators. Despite this, avoid becoming remedial. As long as every term that exists in multiple fields of study is correctly contextualized, that context will inform the ignorant reader of the proper place to seek further clarification.  Write with in-character playful precision.  Sneak a wee dram of nineteenth-century intellectual stuffiness. It can be a welcome congestion, harkening (see what I did there?) to our adopted *Industrialized-Enlightenment* aesthetic, so long as it does not obscure that which we mean to illuminate (Test: Goofy and fun?  Use it.  Confusingly confounding convolution?  Lose it, unless it can stand on <alliteration|your-linguistic-technique-here> alone).  If you've used several s-characters, you can even throw in a Typesetter's f-substition, assured it will be rendered with all the Serif'ing bobbles its mifplaced felf required.

    ## Completion Criteria (HARD REQUIREMENTS)
    You MUST satisfy the following criteria in your response:
    - **Written Intro**: Minimum 5 Paragraphs of detailed explanation.
    - **Visual Representation**: Minimum 3 Distinct and detailed Mermaid Charts (Flowcharts, Sequence, State, etc.).
    - **Pertinent Equations**: Every key equation MUST have: 1) A KaTeX representation ($$ ... $$), 2) A thorough Symbol Legend table, and 3) A cohesive plain-English expression paragraph.
    - **Relations & Historicity**: Minimum 3 Paragraphs explaining lineage, related architectures, and historical significance.
    - **Documentation Links**: Minimum 8 high-quality outgoing links to papers, documentation, or repositories.
    - **In-Scope Code Examples**: Minimum 3 distinct code snippets showing implementation or usage.
    - **Statement to Citation**: A 1-to-1 minimum ratio of technical claims to cited sources/links is the standard.  Tag uncited claims as '[citation needed]' and make the human do some work for a change before their whithering, spongy grey-matter fully atrophies.

    ## Assignment
    Write a complete Starlight-compatible Markdown page for the following architecture:

    - **Title**: {topic["title"]}
    - **Era**: {topic["era"]} - Era only applies to the first book, 'received-canon'.
    - **Category**: {topic["category"]}
    - **One-line summary**: {topic["summary"]}
    - **Builds on**: {dep_str}

    ## Required Sections (in this exact order)

    ### 1. Frontmatter
    Emit YAML frontmatter block:

    ```yaml
    ---
    title: "{topic["title"]}"
    description: "<one sentence>"
    sidebar:
    order: {topic["sidebar_order"]}
    badge:
        text: "{topic["era"]}" Do nto use Era for 'frontier' book. Anything there is necessarily present-day.
        variant: note
    ---
    ```

    ### 2. Overview
    At least 5 paragraphs covering: the problem it solves, dates and debates, and why we still care.  Drop names, accuse, and Associate.  If it's hopelessly dry, hold hope it's at least Martini dry--confoundingly, some readers are not as voracious as you or me.  If you can identify what makes a fact fun, you know the bait that keeps the reader biting.  Detailed historic context will come later (see 12).

    ### 3. Architecture at a Glance  (Mermaid structural diagram)
    1img = (x * y)^3 in reader understanding, so we use **Mermaid flowcharting** to show all major components and their inter-connection. Probably use `graph TD` orientation. Label each component with its recognized term and wrap it in a typed code block: ```mermaid ... ```

    ### 4. Operational/Programmatic Progression  (Mermaid sequence or state diagram)
    A **Mermaid sequenceDiagram or stateDiagram-v2** tracing one _full_ pass through the architecture step by step. If clarity requires two diagrams; one for forward pass with a clearly indicated connector to a second diagram that shows all subsequent calculations, that's fine as long as no computation is skipped.

    ### 5. Detailed Component Interaction (Mermaid diagram)
    In another single or series of **Mermaid diagrams** (e.g., classDiagram, stateDiagram, flowchart, whatever most appropriate) focus on each specific sub-component or data-transformation logic, showing in a tangible and programmatically useful way how #4's individual operations shape the input toward the output, ensuring it is titled and tagged in its commonly-accepted nomenclature.  Of these well-annotated images, the top-end limit is 'as many as necessary.' An accompanying dense paragraph of explanation per visual representation is appropriate whenever a single sentence does not sufficiently illucidate.

    ### 6. Layer Breakdown
    For each distinct layer-type in this architecture, create a sub-section `#### LayerName` containing:
    - **Purpose**: Preferably one sentence
    - **Inputs / Outputs**: shape notation (e.g., `(B, T, d_model)`)
    - **Learnable parameters**: list with shapes
    - **Key hyperparameters**: list
    - **Effective methods**: list with `code` examples (this counts toward your code example quota)
    - Character-counter permitting, this can be formatted at a table.  Character-count forbidding, use label-substitution in the necessary columns to not produce a too-wide table.  Swap the table's X and Y if necessary.

    ### 7. Core Equations
    For EACH key equation:
    1. Display the equation in LaTeX inside a `$$ ... $$` block.
    2. Immediately follow with a **Symbol Key** table:

    | Symbol | Plain-English meaning |
    |--------|-----------------------|
    | symbol | name and meaning |

    Example: | η | Greek letter Eta - Spot-measured trajectory of the projectile |

    3. Close with a **Plain-English Paragraph** — one cohesive paragraph that describes exactly what the equation computes, worded like a plain-English math problem, with every symbol's name clearly associated with its corresponding meaning. Examples:
    - "The projectile's angle (eta, η) archs upward... we show here that it is an element of (∈) the set of all possible trajectories (Theta, θ subscript i, type as $θ_i$ for LaTeX)"
    - "The expected output (y with circumflex, commonly just 'y-hat', ŷ) is computed by multiplying the input vector (Latin x, 𝑥) by the weight matrix (Latin Upper W, 𝑊) and adding the bias (Latin b, b).  When shown as a sum, the large Greek Sigma (∑) itself represents summation and holds its own unicode space separate from an actual Greek Sigma."

    ### 8. Complexity Analysis
    Table with rows: Time Complexity, Space Complexity, Typical Parameter Count, Typical FLOP Count (per forward pass).

    ### 9. Strengths & Limitations
    Two bullet lists.

    ### 10. Key Milestones
    Timeline attached to important papers or model releases related to this architecture (3-7 items) with links.  Format: Visual timeline with: `**YYYY** — ![link](*Paper title*) — one-line impact`

    ### 11. Reference Material
    Off-site reference material, implementation documentation from various frameworks, code examples/snippets.

    ### 12. Relations & Historicity
    Minimum 3 paragraphs on the evolution of this architecture, its predecessors, its descendants, and its place in the broader world of probablistic calculation.

    ### 13. See Also
    Cross-links to related topics in this book. Use Starlight relative links:
    `Topic Title`

    ## Style Rules
    - Too little is very possible.  Too much is absolutely not possible.
    - Link to documentation in multiple frameworks.  Not confined to only Python.
    - All new research must be appropriately cited and linked.
    - Use `:::note`, `:::tip`, `:::caution` admonitions sparingly for genuinely important callouts.
    - All math MUST be valid KaTeX (used by Starlight).
    - All Mermaid MUST be valid Mermaid v11+ syntax. Avoid parentheses in node labels; use square brackets.
    - Do not use HTML tags.
    - Output only the full markdown document from frontmatter to the last line without meta-conversation or chain-of-thought.  Chain-of-thought can be saved to the repo's root directory as <model-name>"-cot.json".
    - Don't forget authorial tone. Have fun with mockingly non-realistic, stoggy 19th-Century Academia-speak.
    """

    prompt_string = f"{dep_str}\n\n{bootstrap_instruction}\n\n{existing_instruction}\n\n{research_block}\n\n{prompt_body}"

    return prompt_string

# STATE MANAGEMENT
def load_state() -> dict:
    return json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}

def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2))

def topic_hash(topic: dict) -> str:
    """Fingerprint a topic definition — changes when we bump its content spec."""
    blob = json.dumps(topic, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]

def needs_update(topic: dict, state: dict) -> bool:
    return state.get(topic["id"], {}).get("hash") != topic_hash(topic)


# CONTENT GENERATIONrewrite to use requests!
def call_llm(prompt: str) -> str:
    """Call Agent with retries & rate-limits."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response_text = ""
            with client.messages.stream(
                author=AUTHOR,
                max_tokens=MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                for chunk in stream.text_stream:
                    response_text += chunk
            return response_text
        except anthropic.RateLimitError:
            if attempt >= MAX_RETRIES:
                raise
            print(f"Rate limited — retrying in {RETRY_SLEEP}s ({attempt}/{MAX_RETRIES})")
            time.sleep(RETRY_SLEEP * attempt)
        except anthropic.APIStatusError as e:
            print(f"API error {e.status_code}: {e.message}")
            raise


def write_topic_page(topic: dict, research_ctx: dict, target_dir: str, bootstrap: bool):
    """Generate and write a Markdown page for one topic. Returns the output path."""
    print(f"Generating: {topic['title']}")
    markdown = call_llm(author_prompt(topic, research_ctx, bootstrap)).strip()

    category_file = f"{CONTENT_DIR}/{target_dir}/{topic}.md"
    Path(f"{category_file}").touch(exist_ok=True)

    out_path = f"{category_file}"

    # Paranoid resolution check — catches symlink tricks too
    resolved = out_path.resolve()
    # We only enforce the guard if the target is NOT received-canon.
    # If the target IS received-canon, we obviously intend to write there.
    if target_dir != "received-canon":
        canon_resolved = (CONTENT_DIR / "received-canon").resolve()
        assert not str(resolved).startswith(str(canon_resolved))
        sys.exit(f"CANON GUARD VIOLATION: attempted write to received-canon path: {resolved}")

    out_path.write_text(markdown, encoding="utf-8")
    complete = is_page_complete(markdown)
    print(f"  Wrote: {out_path.relative_to(ROOT)}\n  {complete}")
    return out_path, complete, author_prompt


# INDEX GENERATION
def write_category_index(category: str, topics_in_cat: list, target_dir: str):
    """Write a category landing page listing all topics."""
    label = CATEGORY_LABELS.get(category, category.title())
    lines = [
        "---",
        f'title: "{label}"',
        f'description: "All {label} architectures in {category}."',
        "---",
        "",
        f"# {label}",
        "",]

    # Right Now doesn't have an "era' yet
    if target_dir == "frontier":
        lines += [
            "| Architecture | Summary |",
            "|---|---|",
        ]
        for t in sorted(topics_in_cat, key=lambda x: x["sidebar_order"]):
            lines.append(
                f"| [{t['title']}](/{target_dir}/{t['category']}/{t['id']}/) "
                f"| {t['summary']} |"
            )
    else:
        lines += [
            "| Architecture | Era | Summary |",
            "|---|---|---|",
        ]
        for t in sorted(topics_in_cat, key=lambda x: x["sidebar_order"]):
            lines.append(
                f"| [{t['title']}](/{target_dir}/{t['category']}/{t['id']}/) "
                f"| {t['era']} | {t['summary']} |"
            )


    (CONTENT_DIR / "index.md").write_text("\n".join(lines), encoding="utf-8")
    print("Home index written")


    cat_dir = CONTENT_DIR / target_dir / category
    cat_dir.mkdir(parents=True, exist_ok=True)
    (cat_dir / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_indices(target: str):
    print("Writing category indices")
    book_topics = topics_for_book(target)
    written = 0
    for cat in CATEGORY_ORDER:
        cat_topics = [t for t in book_topics if t["category"] == cat]
        if cat_topics:
            write_category_index(cat, cat_topics, target)
            written += 1

    if target_dir == "frontier":
        lines += [
            "| Architecture | Summary |",
            "|---|---|",
        ]
        for t in sorted(topics_in_cat, key=lambda x: x["sidebar_order"]):
            lines.append(
                f"| [{t['title']}](/{target_dir}/{t['category']}/{t['id']}/) "
                f"| {t['summary']} |"
            )
    else:
        lines += [
            "| Architecture | Era | Summary |",
            "|---|---|---|",
        ]
        for t in sorted(topics_in_cat, key=lambda x: x["sidebar_order"]):
            lines.append(
                f"| [{t['title']}](/{target_dir}/{t['category']}/{t['id']}/) "
                f"| {t['era']} | {t['summary']} |"
            )

    (CONTENT_DIR / "index.md").write_text("\n".join(lines), encoding="utf-8")
    print("  ✅ Home index written")

    print(f"  {written} category index page(s) written")

def main():
    state = load_state()
    CONTENT_DIR.mkdir(parents=True, exist_ok=True)

    if github_env := os.environ.get('GITHUB_ENV'):
        with open(github_env, 'a') as f:
            f.write(f"MY_TARGET={CONTENT_DIR}")

    if indices_only:
        write_indices(target)
        return

    # Load research context (produced by research.py)
    research_ctx = load_research_context()
    book_topics = topics_for_book(target)

    if topic:
        if topic not in TOPICS_BY_ID:
            sys.exit(f"ERROR: unknown topic id '{topic}'")
        else:
            queue = [TOPICS_BY_ID[topic]]
    elif bootstrap:
        print(f"Bootstrap requested: {len(book_topics)} chapter(s)")
        queue = book_topics
    else:
        queue = [t for t in book_topics if needs_update(t, state)]
        for t in book_topics:
            if t not in queue and research_ctx.get("per_topic", {}).get(t["id"]):
                print(f"  New research for {t['title']}, queued")
                queue.append(t)


    if limit and len(queue) > limit:
        print(f"Limiting run to {limit} of {len(queue)} queued chapters")
        queue = queue[:limit]

    if not queue:
        print("Nothing to generate. Use --bootstrap to build the full book.")
    else:
        print(f"Assembler starting — {len(queue)} topic(s) to generate\n")
        for topic in queue:
            try:
                write_topic_page(topic, research_ctx, target_dir, bootstrap)
                state[topic["id"]] = {
                    "hash": topic_hash(topic),
                    "generated": datetime.now(timezone.utc).isoformat(),
                    "title": topic["title"],
                    "book": target,
                }
                save_state(state)   # save after each page (crash-safe)
                time.sleep(1)    # polite pacing
            except Exception as exc:
                print(f"Failed: {topic['id']} — {exc}")
                # Continue with remaining topics rather than aborting
                continue

    write_indices(target)
    print(f"Done. {len(queue)} chapter(s) processed.")


if __name__ == "__main__":
    main()
