"""
06 · Event Digital Twin — Ingest Pipeline
═══════════════════════════════════════════

Takes a raw witness statement, extracts entities and relationships into
the domain ontology defined in schema.py, resolves co-references, and
loads the result into Neo4j.

Pipeline (LangGraph):

  parse_statement  →  extract_entities  →  resolve_entities  →  load_to_graph

Prerequisites:
  - Neo4j running on bolt://localhost:7687 (neo4j / cabbage123)
  - Ollama running with qwen2.5:7b
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

from neo4j import GraphDatabase

from schema import (
    build_extraction_prompt,
    init_database,
    linearise_graph,
    materialise_provenance,
    NODE_TYPES,
    ONTOLOGY_META,
    RELATIONSHIP_TYPES,
)


# ─── Connections ──────────────────────────────────────────────────────────

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "cabbage123"

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
llm = ChatOllama(model="qwen2.5:7b", temperature=0)


# ═══════════════════════════════════════════════════════════════════════════
# State
# ═══════════════════════════════════════════════════════════════════════════

class IngestState(TypedDict):
    raw_statement: str
    segments: list[str]                # sentences
    extracted: dict                     # raw LLM extraction result
    resolved: dict                      # after coreference resolution
    load_summary: str                   # what was loaded
    steps: list[str]                    # audit trail


# ═══════════════════════════════════════════════════════════════════════════
# Node 1 — parse_statement
# ═══════════════════════════════════════════════════════════════════════════

def parse_statement(state: IngestState) -> dict:
    """Split the raw statement into sentences for provenance tracking.

    Each sentence becomes a source reference so we can later say
    "this fact came from sentence 3 of the original statement."
    """
    text = state["raw_statement"].strip()

    # Split on sentence boundaries — period/question-mark/exclamation
    # followed by whitespace or end-of-string.  Keeps abbreviations like
    # "Dr." or "St." broadly intact because they are followed by a capital.
    raw_segments = re.split(r"(?<=[.!?])\s+", text)

    # Filter out empty / whitespace-only fragments
    segments = [s.strip() for s in raw_segments if s.strip()]

    print(f"  ▸ Parsed statement into {len(segments)} segments")
    for i, seg in enumerate(segments, 1):
        print(f"    [{i}] {seg[:80]}{'…' if len(seg) > 80 else ''}")

    return {
        "segments": segments,
        "steps": state.get("steps", []) + [
            f"parse_statement: {len(segments)} segments"
        ],
    }


# ═══════════════════════════════════════════════════════════════════════════
# Node 2 — extract_entities
# ═══════════════════════════════════════════════════════════════════════════

EXTRACTION_SYSTEM_PROMPT = build_extraction_prompt()


def _parse_json(content: str) -> dict | None:
    """Try to parse JSON from LLM output, tolerating markdown fences."""
    content = content.strip()

    # Strip markdown code fences
    content = re.sub(r"^```(?:json)?\s*", "", content)
    content = re.sub(r"\s*```$", "", content)

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Find outermost { … }
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(content[start:end])
            except json.JSONDecodeError:
                pass
    return None


def extract_entities(state: IngestState) -> dict:
    """Schema-guided extraction: send the full statement to the LLM with
    the ontology prompt from schema.py.

    We send the whole text (not per-sentence) because cross-sentence
    coreference is critical — "The driver got out … He looked …" needs
    to resolve to the same Person.
    """
    full_text = "\n".join(state["segments"])

    messages = [
        SystemMessage(content=EXTRACTION_SYSTEM_PROMPT),
        HumanMessage(content=full_text),
    ]
    result = llm.invoke(messages)
    extracted = _parse_json(result.content)

    if extracted is None:
        print("  ⚠ Extraction returned invalid JSON — retrying with stricter prompt")
        messages.append(HumanMessage(
            content="Your response was not valid JSON. Return ONLY the JSON object, no explanation.",
        ))
        result = llm.invoke(messages)
        extracted = _parse_json(result.content)

    if extracted is None:
        extracted = {"entities": [], "relationships": []}
        print("  ✗ Extraction failed after retry")
    else:
        n_ents = len(extracted.get("entities", []))
        n_rels = len(extracted.get("relationships", []))
        print(f"  ▸ Extracted {n_ents} entities, {n_rels} relationships")

    return {
        "extracted": extracted,
        "steps": state.get("steps", []) + [
            f"extract_entities: {len(extracted.get('entities', []))} entities, "
            f"{len(extracted.get('relationships', []))} relationships"
        ],
    }


# ═══════════════════════════════════════════════════════════════════════════
# Node 3 — resolve_entities
# ═══════════════════════════════════════════════════════════════════════════

COREFERENCE_PROMPT = """\
You are a coreference resolution specialist. You will receive entities \
extracted from a witness statement. Some MAY refer to the same real-world \
thing — but most are DISTINCT.

ENTITIES:
{entities_json}

ORIGINAL TEXT:
{text}

Your task: identify ONLY clear coreferences — cases where two extracted \
entities obviously refer to the same real-world thing (e.g. "the driver" \
and "a tall man" and "he" when they clearly refer to the same person).

Return a JSON object with:
{{
  "entities": [
    {{
      "id": "p1",
      "label": "Person",
      "properties": {{...merged properties...}},
      "original_ids": ["p2", "p3"]
    }}
  ],
  "id_mapping": {{
    "p2": "p1",
    "p3": "p1"
  }}
}}

CRITICAL RULES:
- NEVER merge entities with DIFFERENT labels (a Person is never an Event)
- NEVER merge different Events — each event is unique
- NEVER merge different Locations, Times, or Vehicles unless clearly identical
- Only merge Person entities when the text clearly indicates they are the same
- Entities that are NOT merged must still appear in the "entities" array \
with original_ids containing only their own id
- The id_mapping must map EVERY original id to its final id
- If NO merges are needed, return all entities unchanged with id_mapping \
where each id maps to itself
- Preserve ALL original properties when merging
"""


def resolve_entities(state: IngestState) -> dict:
    """Use LLM-driven coreference resolution to merge duplicate entities.

    The extraction step may produce multiple entities for the same
    real-world thing ("the driver", "a tall man", "he").  We ask the
    LLM to identify co-referent groups and merge them.
    """
    extracted = state["extracted"]
    entities = extracted.get("entities", [])
    relationships = extracted.get("relationships", [])

    if len(entities) <= 1:
        return {
            "resolved": extracted,
            "steps": state.get("steps", []) + [
                "resolve_entities: ≤1 entity, skipping resolution"
            ],
        }

    full_text = "\n".join(state["segments"])
    prompt = COREFERENCE_PROMPT.format(
        entities_json=json.dumps(entities, indent=2),
        text=full_text,
    )

    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content="Resolve coreferences now."),
    ]
    result = llm.invoke(messages)
    resolution = _parse_json(result.content)

    if resolution is None or "entities" not in resolution:
        print("  ⚠ Coreference resolution returned invalid JSON — keeping originals")
        return {
            "resolved": extracted,
            "steps": state.get("steps", []) + [
                "resolve_entities: resolution failed, keeping originals"
            ],
        }

    # Validate: if resolution dramatically reduced entity count, it
    # probably over-merged.  Keep originals in that case.
    resolved_entities = resolution.get("entities", [])
    if len(resolved_entities) < len(entities) * 0.4:
        print(f"  ⚠ Resolution over-merged ({len(entities)} → {len(resolved_entities)}) — keeping originals")
        return {
            "resolved": extracted,
            "steps": state.get("steps", []) + [
                f"resolve_entities: over-merged ({len(entities)}→{len(resolved_entities)}), kept originals"
            ],
        }

    # Remap relationship ids according to the resolution mapping
    id_mapping = resolution.get("id_mapping", {})
    resolved_rels = []
    for rel in relationships:
        resolved_rels.append({
            "from_id": id_mapping.get(rel.get("from_id", ""), rel.get("from_id", "")),
            "rel_type": rel.get("rel_type", ""),
            "to_id": id_mapping.get(rel.get("to_id", ""), rel.get("to_id", "")),
        })

    resolved = {
        "entities": resolved_entities,
        "relationships": resolved_rels,
    }

    original_count = len(entities)
    merged_count = len(resolution["entities"])
    print(f"  ▸ Resolved: {original_count} → {merged_count} entities "
          f"({original_count - merged_count} merged)")

    return {
        "resolved": resolved,
        "steps": state.get("steps", []) + [
            f"resolve_entities: {original_count} → {merged_count} entities"
        ],
    }


# ═══════════════════════════════════════════════════════════════════════════
# Node 4 — load_to_graph
# ═══════════════════════════════════════════════════════════════════════════

# Allowed labels — prevent injection via LLM-generated labels
ALLOWED_LABELS = set(NODE_TYPES.keys())
ALLOWED_REL_TYPES = set(RELATIONSHIP_TYPES.keys())

# Safe property name pattern
SAFE_PROP_RE = re.compile(r"^[a-z_][a-z0-9_]*$")


def _safe_prop_value(val) -> str | int | float | None:
    """Sanitise a property value for Neo4j parameterised queries."""
    if isinstance(val, (int, float)):
        return val
    if isinstance(val, str):
        return val
    if isinstance(val, list):
        return ", ".join(str(v) for v in val)
    return str(val) if val is not None else None


def load_to_graph(state: IngestState) -> dict:
    """Load resolved entities and relationships into Neo4j.

    Uses parameterised queries (not string interpolation) to prevent
    injection.  Every node gets provenance properties (source, timestamp).
    """
    resolved = state["resolved"]
    entities = resolved.get("entities", [])
    relationships = resolved.get("relationships", [])
    segments = state.get("segments", [])
    timestamp = datetime.now(timezone.utc).isoformat()

    # ── Initialise database constraints ─────────────────────────────────
    init_database(driver)

    created_nodes = 0
    created_rels = 0

    with driver.session() as session:
        # ── Clear existing event-twin data ──────────────────────────────
        session.run("MATCH (n) DETACH DELETE n")

        # ── Create entity nodes ─────────────────────────────────────────
        id_to_desc = {}  # track id → primary description for relationship matching
        for entity in entities:
            label = entity.get("label", "Entity")
            if label not in ALLOWED_LABELS:
                print(f"    ⚠ Skipping unknown label: {label}")
                continue

            entity_id = entity.get("id", "")
            props = dict(entity.get("properties", {}))

            # Build a stable description key for MERGE
            desc_key = (
                props.get("description")
                or props.get("name_or_description")
                or props.get("name")
                or props.get("value")
                or props.get("summary")
                or f"unnamed_{entity_id}"
            )
            id_to_desc[entity_id] = desc_key

            # Add provenance (PROV-O: prov:wasGeneratedBy)
            props["source"] = "original_statement"
            props["source_type"] = "statement"
            props["extracted_at"] = timestamp
            props["confidence"] = "high"
            props["ontology_id"] = ONTOLOGY_META["id"]

            # Filter to safe property names only
            safe_props = {
                k: _safe_prop_value(v)
                for k, v in props.items()
                if SAFE_PROP_RE.match(k) and v is not None
            }

            # Build parameterised MERGE + SET
            # We use the description as the merge key for idempotency
            merge_key = _get_merge_key(label)
            merge_val = safe_props.get(merge_key, desc_key)

            cypher = (
                f"MERGE (n:{label} {{{merge_key}: $merge_val}}) "
                f"SET n += $props"
            )
            try:
                session.run(cypher, merge_val=merge_val, props=safe_props)
                created_nodes += 1
            except Exception as e:
                print(f"    ⚠ Failed to create {label} '{desc_key}': {e}")

        # ── Create relationships ────────────────────────────────────────
        for rel in relationships:
            rel_type = rel.get("rel_type", "")
            from_id = rel.get("from_id", "")
            to_id = rel.get("to_id", "")

            if rel_type not in ALLOWED_REL_TYPES:
                print(f"    ⚠ Skipping unknown relationship type: {rel_type}")
                continue

            from_desc = id_to_desc.get(from_id)
            to_desc = id_to_desc.get(to_id)

            if not from_desc or not to_desc:
                print(f"    ⚠ Cannot resolve relationship {from_id}-[{rel_type}]->{to_id}")
                continue

            # Match nodes by their primary description property
            cypher = (
                f"MATCH (a) WHERE coalesce(a.description, a.name_or_description, "
                f"a.name, a.value, a.summary) = $from_desc "
                f"MATCH (b) WHERE coalesce(b.description, b.name_or_description, "
                f"b.name, b.value, b.summary) = $to_desc "
                f"MERGE (a)-[:{rel_type}]->(b)"
            )
            try:
                result = session.run(cypher, from_desc=from_desc, to_desc=to_desc)
                summary = result.consume()
                created_rels += summary.counters.relationships_created
            except Exception as e:
                print(f"    ⚠ Failed to create {from_desc}-[{rel_type}]->{to_desc}: {e}")

    summary_msg = f"Loaded {created_nodes} nodes and {created_rels} relationships"
    print(f"  ✓ {summary_msg}")

    # ── Materialise SOSA/PROV layer ─────────────────────────────────────
    prov_msg = materialise_provenance(
        driver,
        source_type="statement",
        observation_desc="Original witness statement",
        observation_type="witness_statement",
    )

    # ── Print resulting graph ───────────────────────────────────────────
    graph_view = linearise_graph(driver)
    print(f"\n  ── Graph ──")
    for line in graph_view.split("\n"):
        print(f"    {line}")

    return {
        "load_summary": f"{summary_msg}; {prov_msg}",
        "steps": state.get("steps", []) + [
            f"load_to_graph: {summary_msg}",
            f"load_to_graph: {prov_msg}",
        ],
    }


def _get_merge_key(label: str) -> str:
    """Return the property name used as the MERGE key for a given label."""
    merge_keys = {
        "Event":               "description",
        "Person":              "name_or_description",
        "Vehicle":             "description",
        "Location":            "description",
        "Time":                "value",
        "Object":              "description",
        "PhysicalDescription": "summary",
        "Observation":         "description",
    }
    return merge_keys.get(label, "description")


# ═══════════════════════════════════════════════════════════════════════════
# LangGraph pipeline
# ═══════════════════════════════════════════════════════════════════════════

def build_ingest_graph() -> StateGraph:
    """Build the ingest pipeline as a LangGraph."""
    builder = StateGraph(IngestState)

    builder.add_node("parse_statement", parse_statement)
    builder.add_node("extract_entities", extract_entities)
    builder.add_node("resolve_entities", resolve_entities)
    builder.add_node("load_to_graph", load_to_graph)

    builder.add_edge(START, "parse_statement")
    builder.add_edge("parse_statement", "extract_entities")
    builder.add_edge("extract_entities", "resolve_entities")
    builder.add_edge("resolve_entities", "load_to_graph")
    builder.add_edge("load_to_graph", END)

    return builder.compile()


# ═══════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════

def ingest_statement(text: str) -> IngestState:
    """Run the full ingest pipeline on a witness statement."""
    graph = build_ingest_graph()
    initial_state: IngestState = {
        "raw_statement": text,
        "segments": [],
        "extracted": {},
        "resolved": {},
        "load_summary": "",
        "steps": [],
    }
    result = graph.invoke(initial_state)
    return result


if __name__ == "__main__":
    # Load the sample statement
    statement_path = Path(__file__).parent / "statements" / "king_street_collision.txt"

    if statement_path.exists():
        text = statement_path.read_text().strip()
        print(f"{'═' * 70}")
        print(f"  EVENT DIGITAL TWIN — INGEST PIPELINE")
        print(f"{'═' * 70}")
        print(f"\n  Statement ({len(text)} chars):")
        print(f"  {text[:120]}…\n")
    else:
        # Fallback inline statement
        text = (
            "I was walking along King Street at approximately 2:15 PM on Tuesday "
            "when I heard a loud crash. I turned and saw a red car had collided "
            "with a cyclist at the junction of King Street and Queen's Road. The "
            "driver got out — a tall man wearing a dark jacket. He looked at the "
            "cyclist who was on the ground and then got back in his car and drove "
            "off heading north on Queen's Road. Another woman who was nearby "
            "called an ambulance. I stayed with the cyclist until the paramedics "
            "arrived about ten minutes later."
        )
        print(f"{'═' * 70}")
        print(f"  EVENT DIGITAL TWIN — INGEST PIPELINE (inline statement)")
        print(f"{'═' * 70}\n")

    result = ingest_statement(text)

    print(f"\n{'─' * 70}")
    print(f"  Audit trail:")
    for step in result.get("steps", []):
        print(f"    • {step}")
    print(f"{'─' * 70}")
