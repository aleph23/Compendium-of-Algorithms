#!/usr/bin/env python3
"""
version_manager.py
------------------
Tracks Frontier content mass by character count and manages semantic versioning.

Version file: frontier_version.json
  {
    "major": 1,
    "minor": 0,
    "char_count_baseline": 0, # baseline for next minor-bump check
    "char_count_last_major": 0, # baseline for major-bump accumulation
    "last_updated": "ISO datetime",
    "history": [
      {"version": "1.0", "date": "...", "chars": 0, "event": "init"}
    ]
  }

Exit codes
----------
  0 — normal (no archive needed)
  2 — archive triggered (caller should create GitHub Release)
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
FRONTIER_DIR = ROOT / "src" / "content" / "docs" / "frontier"
VERSION_FILE = ROOT / "frontier_version.json"

MINOR_THRESHOLD = 0.15   # 15% change triggers minor bump
MAJOR_THRESHOLD = 0.75   # 75% cumulative change triggers major bump

ARCHIVE_TRIGGERED = False   # set to True if this run needs a release


# Helpers

def count_chars() -> int:
    """Total character count of all .md files under frontier/."""
    return sum(
        len(md.read_text(encoding="utf-8", errors="replace"))
        for md in FRONTIER_DIR.rglob("*.md")
    )


def load_version() -> dict:
    if VERSION_FILE.exists():
        return json.loads(VERSION_FILE.read_text())
    # First ever run
    return {
        "major": 0,
        "minor": 0,
        "char_count_baseline": 0,
        "char_count_last_major": 0,
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "history": [],
    }

def save_version(v: dict):
    VERSION_FILE.write_text(json.dumps(v, indent=2))

def version_str(v: dict) -> str:
    return f"{v['major']}.{v['minor']}"

def log_event(v: dict, chars: int, event: str):
    v["history"].append({
        "version": version_str(v),
        "date": datetime.now(timezone.utc).isoformat()[:10],
        "chars": chars,
        "event": event,
    })
    v["last_updated"] = datetime.now(timezone.utc).isoformat()


# Main logic

def run():
    global ARCHIVE_TRIGGERED

    current = count_chars()
    v = load_version()

    print("\nFrontier Version Manager")
    print(f"Current version: {version_str(v)}")
    print(f"Current chars: {current:,}")
    print(f"Baseline chars: {v['char_count_baseline']:,}")
    print(f"Last-major chars: {v['char_count_last_major']:,}")

    # Bootstrap first run
    if v["char_count_baseline"] == 0:
        v["char_count_baseline"] = current
        v["char_count_last_major"] = current
        log_event(v, current, "init")
        save_version(v)
        print("   → First run: baseline established.")
        return

    baseline = v["char_count_baseline"]
    last_major = v["char_count_last_major"]

    delta_pct = 0.0 if baseline == 0 else (current - baseline) / baseline
    if last_major == 0:
        cumulative_pct = 0.0
    else:
        # Cumulative drift (unsigned) since last major
        cumulative_pct = abs(current - last_major) / last_major

    print(f" Delta vs baseline: {delta_pct:+.1%}")
    print(f" Cumulative vs last major: {cumulative_pct:.1%}")

    # Check major version first (takes priority)
    if cumulative_pct >= MAJOR_THRESHOLD:
        v["major"] += 1
        v["minor"] = 0
        v["char_count_last_major"] = current
        v["char_count_baseline"] = current
        log_event(v, current, f"major bump — {cumulative_pct:.1%} cumulative drift")
        save_version(v)
        print(f"\nMAJOR VERSION → {version_str(v)}  ({cumulative_pct:.1%} cumulative drift)")
        return

    # Check minor / archive
    if abs(delta_pct) >= MINOR_THRESHOLD:
        v["minor"] += 1
        v["char_count_baseline"]  = current
        event_label = f"minor bump — {delta_pct:+.1%} delta"

        if delta_pct <= -MINOR_THRESHOLD:
            # Significant content reduction → archive this version
            ARCHIVE_TRIGGERED = True
            event_label += " [ARCHIVE TRIGGERED]"
            print(f"\nARCHIVE TRIGGERED — {abs(delta_pct):.1%} content reduction")

        log_event(v, current, event_label)
        save_version(v)
        print(f" → Minor bump → {version_str(v)}")
    else:
        print(f" → No version change (delta {delta_pct:+.1%}, threshold ±{MINOR_THRESHOLD:.0%})")
    save_version(v)


if __name__ == "__main__":
    run()
    if ARCHIVE_TRIGGERED:
        sys.exit(2)   # signal to workflow: create a GitHub Release
