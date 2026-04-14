"""
08 · Digital Twin — Multi-Source Fusion
═══════════════════════════════════════

Cross-source entity resolution, corroboration detection, and
contradiction identification. Merges facts from multiple ingested
sources into a unified event graph with provenance tracking.

Prerequisites:
  - Neo4j running with graph already populated by multiple ingest runs
  - Ollama running with qwen2.5:7b
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from neo4j import GraphDatabase

from schema import (
    get_active_spec,
    linearise_graph,
    get_ontology_id,
)


llm = ChatOllama(model="qwen2.5:7b", temperature=0)


# ═══════════════════════════════════════════════════════════════════════════
# JSON helpers
# ═══════════════════════════════════════════════════════════════════════════

def _parse_json(content: str) -> dict | list | None:
    content = content.strip()
    content = re.sub(r"^```(?:json)?\s*", "", content)
    content = re.sub(r"\s*```$", "", content)
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        for start_c, end_c in [("{", "}"), ("[", "]")]:
            start = content.find(start_c)
            end = content.rfind(end_c) + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(content[start:end])
                except json.JSONDecodeError:
                    continue
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Step 1 — Discover sources in the graph
# ═══════════════════════════════════════════════════════════════════════════

def _get_sources(driver) -> list[str]:
    """Return distinct source_id values present in the graph."""
    with driver.session() as session:
        result = session.run(
            "MATCH (n) WHERE n.source IS NOT NULL "
            "RETURN DISTINCT n.source AS source_id"
        )
        return [r["source_id"] for r in result]


def _get_entities_by_source(driver, source_id: str) -> list[dict]:
    """Return all entities from a specific source."""
    with driver.session() as session:
        result = session.run(
            "MATCH (n) WHERE n.source = $sid AND NOT n:GraphVersion "
            "AND NOT n:Observation "
            "RETURN labels(n)[0] AS label, "
            "coalesce(n.description, n.name_or_description, n.name, "
            "n.value, n.summary) AS desc, "
            "properties(n) AS props",
            sid=source_id,
        )
        return [dict(r) for r in result]


# ═══════════════════════════════════════════════════════════════════════════
# Step 2 — Cross-source entity resolution
# ═══════════════════════════════════════════════════════════════════════════

CROSS_RESOLUTION_PROMPT = """\
You are performing cross-source entity resolution for an event investigation.
Two different sources have described entities. Determine which entities from
SOURCE B are the SAME real-world thing as entities in SOURCE A.

SOURCE A ({source_a}):
{entities_a}

SOURCE B ({source_b}):
{entities_b}

Return a JSON object:
{{
  "matches": [
    {{
      "entity_a_desc": "description from source A",
      "entity_b_desc": "description from source B",
      "confidence": 0.9,
      "reasoning": "why these are the same"
    }}
  ],
  "contradictions": [
    {{
      "entity_a_desc": "...",
      "entity_b_desc": "...",
      "field": "the property that conflicts",
      "value_a": "value in source A",
      "value_b": "value in source B",
      "reasoning": "why this is a contradiction"
    }}
  ]
}}

RULES:
- Only match entities with the SAME label type
- Require confidence >= 0.7 to match
- Flag contradictions even between matched entities
- Different timestamps for the same event are contradictions
- Different descriptions of the same person are NOT contradictions
"""


def _resolve_across_sources(
    driver,
    source_a: str,
    source_b: str,
) -> dict[str, Any]:
    """Run LLM-assisted entity resolution between two sources."""
    entities_a = _get_entities_by_source(driver, source_a)
    entities_b = _get_entities_by_source(driver, source_b)

    if not entities_a or not entities_b:
        return {"matches": [], "contradictions": []}

    # Summarise for the LLM (strip internal props)
    def _summarise(entities):
        lines = []
        for e in entities:
            label = e.get("label", "?")
            desc = e.get("desc", "?")
            props = {k: v for k, v in e.get("props", {}).items()
                     if k not in ("source", "source_type", "extracted_at",
                                  "ontology_id", "confidence")}
            lines.append(f"  [{label}] {desc} — {json.dumps(props)}")
        return "\n".join(lines)

    prompt = CROSS_RESOLUTION_PROMPT.format(
        source_a=source_a,
        source_b=source_b,
        entities_a=_summarise(entities_a),
        entities_b=_summarise(entities_b),
    )

    result = llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content="Resolve entities across these two sources now."),
    ])
    parsed = _parse_json(result.content)
    if not parsed or not isinstance(parsed, dict):
        return {"matches": [], "contradictions": []}
    return parsed


# ═══════════════════════════════════════════════════════════════════════════
# Step 3 — Apply merges, corroborations, and contradictions
# ═══════════════════════════════════════════════════════════════════════════

def _apply_merges(driver, matches: list[dict]) -> int:
    """Create POSSIBLY_SAME_AS relationships for matched entities."""
    count = 0
    with driver.session() as session:
        for match in matches:
            desc_a = match.get("entity_a_desc", "")
            desc_b = match.get("entity_b_desc", "")
            conf = match.get("confidence", 0.7)

            if not desc_a or not desc_b or desc_a == desc_b:
                continue

            result = session.run(
                "MATCH (a) WHERE coalesce(a.description, a.name_or_description, "
                "a.name, a.value, a.summary) = $desc_a "
                "MATCH (b) WHERE coalesce(b.description, b.name_or_description, "
                "b.name, b.value, b.summary) = $desc_b "
                "AND a <> b "
                "MERGE (a)-[r:POSSIBLY_SAME_AS]->(b) "
                "SET r.confidence = $conf, "
                "    r.reasoning = $reasoning "
                "RETURN count(r) AS cnt",
                desc_a=desc_a, desc_b=desc_b,
                conf=conf,
                reasoning=match.get("reasoning", ""),
            )
            for rec in result:
                count += rec["cnt"]

            # Boost confidence on both entities when corroborated
            if conf >= 0.8:
                session.run(
                    "MATCH (a) WHERE coalesce(a.description, a.name_or_description, "
                    "a.name, a.value, a.summary) = $desc_a "
                    "SET a.confidence = CASE "
                    "  WHEN a.confidence IS NULL THEN 0.9 "
                    "  WHEN toFloat(a.confidence) < 0.95 "
                    "    THEN toFloat(a.confidence) + (1.0 - toFloat(a.confidence)) * 0.3 "
                    "  ELSE a.confidence END",
                    desc_a=desc_a,
                )
                session.run(
                    "MATCH (b) WHERE coalesce(b.description, b.name_or_description, "
                    "b.name, b.value, b.summary) = $desc_b "
                    "SET b.confidence = CASE "
                    "  WHEN b.confidence IS NULL THEN 0.9 "
                    "  WHEN toFloat(b.confidence) < 0.95 "
                    "    THEN toFloat(b.confidence) + (1.0 - toFloat(b.confidence)) * 0.3 "
                    "  ELSE b.confidence END",
                    desc_b=desc_b,
                )

    return count


def _apply_corroborations(driver, matches: list[dict]) -> int:
    """Create CORROBORATED_BY relationships between source Observations."""
    count = 0
    with driver.session() as session:
        # Find where both sources reported the same entity
        for match in matches:
            if match.get("confidence", 0) < 0.8:
                continue
            desc_a = match.get("entity_a_desc", "")
            desc_b = match.get("entity_b_desc", "")

            result = session.run(
                "MATCH (a) WHERE coalesce(a.description, a.name_or_description, "
                "a.name, a.value, a.summary) = $desc_a "
                "MATCH (b) WHERE coalesce(b.description, b.name_or_description, "
                "b.name, b.value, b.summary) = $desc_b "
                "AND a <> b "
                "WITH a, b "
                "MATCH (obs_a:Observation) WHERE obs_a.description CONTAINS a.source "
                "MATCH (obs_b:Observation) WHERE obs_b.description CONTAINS b.source "
                "AND obs_a <> obs_b "
                "MERGE (obs_a)-[r:CORROBORATED_BY]->(obs_b) "
                "SET r.entity = $desc_a "
                "RETURN count(r) AS cnt",
                desc_a=desc_a, desc_b=desc_b,
            )
            for rec in result:
                count += rec["cnt"]
    return count


def _apply_contradictions(driver, contradictions: list[dict]) -> int:
    """Create CONTRADICTS relationships between conflicting entities."""
    count = 0
    with driver.session() as session:
        for contradiction in contradictions:
            desc_a = contradiction.get("entity_a_desc", "")
            desc_b = contradiction.get("entity_b_desc", "")
            field = contradiction.get("field", "")
            val_a = contradiction.get("value_a", "")
            val_b = contradiction.get("value_b", "")

            if not desc_a or not desc_b:
                continue

            result = session.run(
                "MATCH (a) WHERE coalesce(a.description, a.name_or_description, "
                "a.name, a.value, a.summary) = $desc_a "
                "MATCH (b) WHERE coalesce(b.description, b.name_or_description, "
                "b.name, b.value, b.summary) = $desc_b "
                "AND a <> b "
                "MERGE (a)-[r:CONTRADICTS]->(b) "
                "SET r.field = $field, "
                "    r.value_a = $val_a, "
                "    r.value_b = $val_b, "
                "    r.reasoning = $reasoning "
                "RETURN count(r) AS cnt",
                desc_a=desc_a, desc_b=desc_b,
                field=field, val_a=str(val_a), val_b=str(val_b),
                reasoning=contradiction.get("reasoning", ""),
            )
            for rec in result:
                count += rec["cnt"]

            # Lower confidence on contradicted entities
            for desc in (desc_a, desc_b):
                session.run(
                    "MATCH (n) WHERE coalesce(n.description, n.name_or_description, "
                    "n.name, n.value, n.summary) = $desc "
                    "SET n.confidence = CASE "
                    "  WHEN n.confidence IS NULL THEN 0.5 "
                    "  WHEN toFloat(n.confidence) > 0.3 "
                    "    THEN toFloat(n.confidence) * 0.7 "
                    "  ELSE n.confidence END",
                    desc=desc,
                )

    return count


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

def run_fusion(driver) -> dict[str, int]:
    """Run full multi-source fusion pipeline.

    Returns a summary dict with counts of merges, corroborations, and
    contradictions.
    """
    sources = _get_sources(driver)
    print(f"  ▸ Found {len(sources)} sources: {sources}")

    if len(sources) < 2:
        print("  ▸ Fewer than 2 sources — nothing to fuse")
        return {"merges": 0, "corroborations": 0, "contradictions": 0}

    total_merges = 0
    total_corroborations = 0
    total_contradictions = 0

    # Compare each pair of sources
    for i, src_a in enumerate(sources):
        for src_b in sources[i + 1:]:
            print(f"  ▸ Resolving: {src_a} ↔ {src_b}")
            resolution = _resolve_across_sources(driver, src_a, src_b)

            matches = resolution.get("matches", [])
            contradictions = resolution.get("contradictions", [])

            if matches:
                merges = _apply_merges(driver, matches)
                total_merges += merges
                print(f"    ✓ {merges} entity matches linked")

                corroborations = _apply_corroborations(driver, matches)
                total_corroborations += corroborations
                print(f"    ✓ {corroborations} corroboration links")

            if contradictions:
                contras = _apply_contradictions(driver, contradictions)
                total_contradictions += contras
                print(f"    ⚠ {contras} contradictions flagged")

    return {
        "merges": total_merges,
        "corroborations": total_corroborations,
        "contradictions": total_contradictions,
    }
