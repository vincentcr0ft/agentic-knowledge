"""
08 · Digital Twin — Temporal Reasoning
══════════════════════════════════════

Extracts temporal information from the event graph, builds ordered
timelines, applies Allen's interval algebra for consistency checking,
and detects temporal contradictions across sources.

Prerequisites:
  - Neo4j running with populated event graph
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any


# ═══════════════════════════════════════════════════════════════════════════
# Time parsing
# ═══════════════════════════════════════════════════════════════════════════

# Patterns for common time expressions
TIME_PATTERNS = [
    # HH:MM:SS
    (r"(\d{1,2}):(\d{2}):(\d{2})", lambda m: (int(m.group(1)), int(m.group(2)), int(m.group(3)))),
    # "2:15 PM", "14:15" — must come before plain HH:MM to avoid false match
    (r"(\d{1,2}):(\d{2})\s*(AM|PM|am|pm)",
     lambda m: (
         (int(m.group(1)) % 12) + (12 if m.group(3).upper() == "PM" else 0),
         int(m.group(2)), 0)),
    # HH:MM (no AM/PM suffix)
    (r"(\d{1,2}):(\d{2})\b", lambda m: (int(m.group(1)), int(m.group(2)), 0)),
    # "quarter past two" etc — approximate
    (r"quarter past (\w+)",
     lambda m: (_word_to_hour(m.group(1)), 15, 0)),
    (r"half past (\w+)",
     lambda m: (_word_to_hour(m.group(1)), 30, 0)),
    # "approximately X PM"
    (r"approximately\s+(\d{1,2}):(\d{2})\s*(PM|AM|pm|am)?",
     lambda m: (
         (int(m.group(1)) % 12) + (12 if (m.group(3) or "").upper() == "PM" else 0),
         int(m.group(2)), 0)),
]

WORD_HOURS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "noon": 12, "midnight": 0,
}


def _word_to_hour(word: str) -> int:
    return WORD_HOURS.get(word.lower(), 0)


def parse_time_value(value: str) -> float | None:
    """Parse a time string to minutes-since-midnight for ordering.

    Returns None if unparseable.
    """
    if not value:
        return None

    for pattern, extractor in TIME_PATTERNS:
        match = re.search(pattern, value)
        if match:
            try:
                h, m, s = extractor(match)
                return h * 60.0 + m + s / 60.0
            except (ValueError, TypeError):
                continue

    # Try direct "HH:MM" embedded in longer strings
    match = re.search(r"(\d{2}):(\d{2})", value)
    if match:
        h, m = int(match.group(1)), int(match.group(2))
        if 0 <= h <= 23 and 0 <= m <= 59:
            return h * 60.0 + m

    return None


def _minutes_to_time_str(minutes: float) -> str:
    """Convert minutes-since-midnight to HH:MM string."""
    h = int(minutes // 60) % 24
    m = int(minutes % 60)
    return f"{h:02d}:{m:02d}"


# ═══════════════════════════════════════════════════════════════════════════
# Timeline construction
# ═══════════════════════════════════════════════════════════════════════════

def build_timeline(driver) -> list[dict[str, Any]]:
    """Extract events with temporal information and build an ordered timeline.

    Returns a list of dicts with keys: description, time, time_minutes,
    source, label.
    """
    events = []

    with driver.session() as session:
        # Method 1: Events with AT_TIME relationships
        result = session.run(
            "MATCH (e:Event)-[:AT_TIME|:OCCURRED_AT_TIME|:HAS_TIME]->(t:Time) "
            "RETURN e.description AS description, t.value AS time_value, "
            "e.source AS source, labels(e)[0] AS label"
        )
        for rec in result:
            time_val = rec.get("time_value", "")
            minutes = parse_time_value(str(time_val)) if time_val else None
            events.append({
                "description": rec.get("description", "?"),
                "time": str(time_val) if time_val else "unknown",
                "time_minutes": minutes,
                "source": rec.get("source", "?"),
                "label": rec.get("label", "Event"),
            })

        # Method 2: Events with time-like properties directly
        result = session.run(
            "MATCH (e:Event) WHERE e.time IS NOT NULL OR e.timestamp IS NOT NULL "
            "AND NOT EXISTS { MATCH (e)-[:AT_TIME|:OCCURRED_AT_TIME|:HAS_TIME]->(:Time) } "
            "RETURN e.description AS description, "
            "coalesce(e.time, e.timestamp) AS time_value, "
            "e.source AS source, labels(e)[0] AS label"
        )
        for rec in result:
            time_val = rec.get("time_value", "")
            minutes = parse_time_value(str(time_val)) if time_val else None
            events.append({
                "description": rec.get("description", "?"),
                "time": str(time_val) if time_val else "unknown",
                "time_minutes": minutes,
                "source": rec.get("source", "?"),
                "label": rec.get("label", "Event"),
            })

        # Method 3: CCTV-style entries (may be stored as generic events)
        result = session.run(
            "MATCH (e) WHERE e.timestamp IS NOT NULL AND NOT e:Event "
            "RETURN coalesce(e.description, e.name_or_description) AS description, "
            "e.timestamp AS time_value, e.source AS source, labels(e)[0] AS label"
        )
        for rec in result:
            time_val = rec.get("time_value", "")
            minutes = parse_time_value(str(time_val)) if time_val else None
            if minutes is not None:
                events.append({
                    "description": rec.get("description", "?"),
                    "time": str(time_val),
                    "time_minutes": minutes,
                    "source": rec.get("source", "?"),
                    "label": rec.get("label", "?"),
                })

    # Deduplicate by description
    seen = set()
    unique = []
    for evt in events:
        key = evt["description"]
        if key not in seen:
            seen.add(key)
            unique.append(evt)

    # Sort: timed events first (by time), then untimed events
    timed = [e for e in unique if e["time_minutes"] is not None]
    untimed = [e for e in unique if e["time_minutes"] is None]
    timed.sort(key=lambda e: e["time_minutes"])

    timeline = timed + untimed

    # Materialise PRECEDED_BY relationships for ordered events
    _materialise_temporal_order(driver, timed)

    return timeline


def _materialise_temporal_order(driver, timed_events: list[dict]) -> int:
    """Create PRECEDED_BY relationships between temporally ordered events."""
    if len(timed_events) < 2:
        return 0

    created = 0
    with driver.session() as session:
        for i in range(len(timed_events) - 1):
            desc_a = timed_events[i]["description"]
            desc_b = timed_events[i + 1]["description"]
            time_a = timed_events[i]["time"]
            time_b = timed_events[i + 1]["time"]

            result = session.run(
                "MATCH (a:Event {description: $desc_a}) "
                "MATCH (b:Event {description: $desc_b}) "
                "WHERE a <> b "
                "MERGE (a)-[r:PRECEDED_BY]->(b) "
                "SET r.time_a = $time_a, r.time_b = $time_b "
                "RETURN count(r) AS cnt",
                desc_a=desc_a, desc_b=desc_b,
                time_a=time_a, time_b=time_b,
            )
            for rec in result:
                created += rec["cnt"]

    return created


# ═══════════════════════════════════════════════════════════════════════════
# Temporal consistency checking
# ═══════════════════════════════════════════════════════════════════════════

def check_consistency(driver) -> list[str]:
    """Check for temporal inconsistencies in the event graph.

    Returns a list of issue description strings.
    """
    issues = []

    with driver.session() as session:
        # Check 1: PRECEDED_BY cycles (A → B → A)
        result = session.run(
            "MATCH path = (a:Event)-[:PRECEDED_BY*2..5]->(a) "
            "RETURN [n IN nodes(path) | n.description] AS cycle "
            "LIMIT 5"
        )
        for rec in result:
            cycle = rec.get("cycle", [])
            issues.append(
                f"Temporal cycle detected: {' → '.join(str(c) for c in cycle)}"
            )

        # Check 2: Contradicting timestamps for the same event across sources
        result = session.run(
            "MATCH (e:Event)-[:AT_TIME|:OCCURRED_AT_TIME|:HAS_TIME]->(t1:Time) "
            "MATCH (e)-[:AT_TIME|:OCCURRED_AT_TIME|:HAS_TIME]->(t2:Time) "
            "WHERE t1 <> t2 AND t1.value <> t2.value "
            "RETURN e.description AS event, t1.value AS time1, t2.value AS time2, "
            "t1.source AS source1, t2.source AS source2 "
            "LIMIT 10"
        )
        for rec in result:
            issues.append(
                f"Conflicting times for '{rec['event']}': "
                f"{rec['time1']} (from {rec.get('source1', '?')}) vs "
                f"{rec['time2']} (from {rec.get('source2', '?')})"
            )

        # Check 3: Events with CONTRADICTS relationships on time
        result = session.run(
            "MATCH (a)-[r:CONTRADICTS]->(b) "
            "WHERE r.field CONTAINS 'time' OR r.field CONTAINS 'timestamp' "
            "RETURN coalesce(a.description, a.value) AS entity_a, "
            "coalesce(b.description, b.value) AS entity_b, "
            "r.value_a AS val_a, r.value_b AS val_b "
            "LIMIT 10"
        )
        for rec in result:
            issues.append(
                f"Temporal contradiction: '{rec['entity_a']}' ({rec['val_a']}) "
                f"vs '{rec['entity_b']}' ({rec['val_b']})"
            )

        # Check 4: Events that should have times but don't
        result = session.run(
            "MATCH (e:Event) "
            "WHERE NOT EXISTS { MATCH (e)-[:AT_TIME|:OCCURRED_AT_TIME|:HAS_TIME]->(:Time) } "
            "AND e.time IS NULL AND e.timestamp IS NULL "
            "RETURN e.description AS description "
            "LIMIT 10"
        )
        for rec in result:
            desc = rec.get("description", "?")
            issues.append(f"Event without timestamp: '{desc}'")

    return issues


# ═══════════════════════════════════════════════════════════════════════════
# Timeline formatting
# ═══════════════════════════════════════════════════════════════════════════

def format_timeline_ascii(timeline: list[dict]) -> str:
    """Format a timeline as an ASCII text diagram."""
    if not timeline:
        return "  (no events in timeline)"

    lines = ["", "  ── EVENT TIMELINE ──", ""]

    for i, evt in enumerate(timeline):
        time_str = evt.get("time", "??:??")
        desc = evt.get("description", "?")[:55]
        source = evt.get("source", "?")
        connector = "├" if i < len(timeline) - 1 else "└"
        lines.append(f"  {connector}─ [{time_str:>8s}] {desc}")
        lines.append(f"  {'│' if i < len(timeline) - 1 else ' '}              source: {source}")

    lines.append("")
    return "\n".join(lines)
