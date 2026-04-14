"""
08 · Digital Twin — Ingest Pipeline
════════════════════════════════════

Takes a raw witness statement, extracts entities and relationships using
the active ontology from 06-ontologies, resolves co-references, and
loads the result into Neo4j.

Pipeline (LangGraph):
  parse_statement → extract_entities → resolve_entities → load_to_graph

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
    get_extraction_prompt,
    get_active_spec,
    get_node_types,
    get_relationship_types,
    get_ontology_id,
    init_database,
    linearise_graph,
    materialise_provenance,
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
    source_id: str
    source_type: str
    segments: list[str]
    extracted: dict
    resolved: dict
    load_summary: str
    steps: list[str]


# ═══════════════════════════════════════════════════════════════════════════
# Node 1 — parse_statement
# ═══════════════════════════════════════════════════════════════════════════

def parse_statement(state: IngestState) -> dict:
    text = state["raw_statement"].strip()
    raw_segments = re.split(r"(?<=[.!?])\s+", text)
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

def _parse_json(content: str) -> dict | None:
    content = content.strip()
    content = re.sub(r"^```(?:json)?\s*", "", content)
    content = re.sub(r"\s*```$", "", content)
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(content[start:end])
            except json.JSONDecodeError:
                pass
    return None


def extract_entities(state: IngestState) -> dict:
    """Schema-guided extraction using the active ontology's prompt."""
    full_text = "\n".join(state["segments"])
    extraction_prompt = get_extraction_prompt()

    messages = [
        SystemMessage(content=extraction_prompt),
        HumanMessage(content=full_text),
    ]
    result = llm.invoke(messages)
    extracted = _parse_json(result.content)

    if extracted is None:
        print("  ⚠ Extraction returned invalid JSON — retrying")
        messages.append(HumanMessage(
            content="Your response was not valid JSON. Return ONLY the JSON object.",
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
entities obviously refer to the same real-world thing.

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
- NEVER merge entities with DIFFERENT labels
- NEVER merge different Events
- Only merge when the text clearly indicates identity
- Preserve ALL original properties when merging
- The id_mapping must map EVERY original id to its final id
"""


def resolve_entities(state: IngestState) -> dict:
    extracted = state["extracted"]
    entities = extracted.get("entities", [])
    relationships = extracted.get("relationships", [])

    if len(entities) <= 1:
        return {
            "resolved": extracted,
            "steps": state.get("steps", []) + [
                "resolve_entities: ≤1 entity, skipping"
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
        print("  ⚠ Coreference resolution failed — keeping originals")
        return {
            "resolved": extracted,
            "steps": state.get("steps", []) + [
                "resolve_entities: resolution failed, keeping originals"
            ],
        }

    resolved_entities = resolution.get("entities", [])
    if len(resolved_entities) < len(entities) * 0.4:
        print(f"  ⚠ Over-merged ({len(entities)} → {len(resolved_entities)}) — keeping originals")
        return {
            "resolved": extracted,
            "steps": state.get("steps", []) + [
                f"resolve_entities: over-merged, kept originals"
            ],
        }

    id_mapping = resolution.get("id_mapping", {})
    resolved_rels = []
    for rel in relationships:
        resolved_rels.append({
            "from_id": id_mapping.get(rel.get("from_id", ""), rel.get("from_id", "")),
            "rel_type": rel.get("rel_type", ""),
            "to_id": id_mapping.get(rel.get("to_id", ""), rel.get("to_id", "")),
        })

    resolved = {"entities": resolved_entities, "relationships": resolved_rels}
    print(f"  ▸ Resolved: {len(entities)} → {len(resolved_entities)} entities")

    return {
        "resolved": resolved,
        "steps": state.get("steps", []) + [
            f"resolve_entities: {len(entities)} → {len(resolved_entities)}"
        ],
    }


# ═══════════════════════════════════════════════════════════════════════════
# Node 4 — load_to_graph
# ═══════════════════════════════════════════════════════════════════════════

SAFE_PROP_RE = re.compile(r"^[a-z_][a-z0-9_]*$")


def _get_merge_key(label: str) -> str:
    merge_keys = {
        "Event": "description", "Process": "description",
        "Person": "name_or_description", "Actor": "name_or_description",
        "Agent": "name_or_description",
        "Vehicle": "description", "Location": "description",
        "Place": "description", "SpatialRegion": "description",
        "Time": "value", "TemporalRegion": "value",
        "TemporalInterval": "value",
        "Object": "description",
        "Role": "role_type", "AgentRole": "role_type",
        "PhysicalDescription": "summary",
        "DescriptiveICE": "summary",
        "Observation": "description",
        "InformationContentEntity": "description",
        "MaterialEntity": "description",
        "Site": "description",
        "Act": "description",
    }
    return merge_keys.get(label, "description")


def load_to_graph(state: IngestState) -> dict:
    """Load resolved entities and relationships into Neo4j (additive)."""
    resolved = state["resolved"]
    entities = resolved.get("entities", [])
    relationships = resolved.get("relationships", [])
    timestamp = datetime.now(timezone.utc).isoformat()
    source_id = state.get("source_id", "unknown")
    source_type = state.get("source_type", "statement")

    # Get allowed labels/rels from active ontology
    allowed_labels = set(get_node_types().keys())
    allowed_rels = set(get_relationship_types().keys())
    ontology_id = get_ontology_id()

    init_database(driver)

    created_nodes = 0
    created_rels = 0

    with driver.session() as session:
        # Record this ingestion as a GraphVersion node
        session.run(
            "CREATE (v:GraphVersion {source_id: $sid, source_type: $stype, "
            "timestamp: $ts, ontology_id: $ont})",
            sid=source_id, stype=source_type, ts=timestamp, ont=ontology_id,
        )

        id_to_desc = {}
        for entity in entities:
            label = entity.get("label", "Entity")
            if label not in allowed_labels:
                print(f"    ⚠ Skipping unknown label: {label}")
                continue

            entity_id = entity.get("id", "")
            props = dict(entity.get("properties", {}))

            desc_key = (
                props.get("description")
                or props.get("name_or_description")
                or props.get("name")
                or props.get("value")
                or props.get("summary")
                or f"unnamed_{entity_id}"
            )
            id_to_desc[entity_id] = desc_key

            props["source"] = source_id
            props["source_type"] = source_type
            props["extracted_at"] = timestamp
            props["confidence"] = 0.8
            props["ontology_id"] = ontology_id

            safe_props = {
                k: (", ".join(str(x) for x in v) if isinstance(v, list) else v)
                for k, v in props.items()
                if SAFE_PROP_RE.match(k) and v is not None
            }

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

        for rel in relationships:
            rel_type = rel.get("rel_type", "")
            from_id = rel.get("from_id", "")
            to_id = rel.get("to_id", "")

            if rel_type not in allowed_rels:
                print(f"    ⚠ Skipping unknown rel type: {rel_type}")
                continue

            from_desc = id_to_desc.get(from_id)
            to_desc = id_to_desc.get(to_id)

            if not from_desc or not to_desc:
                print(f"    ⚠ Cannot resolve {from_id}-[{rel_type}]->{to_id}")
                continue

            cypher = (
                f"MATCH (a) WHERE coalesce(a.description, a.name_or_description, "
                f"a.name, a.value, a.summary) = $from_desc "
                f"MATCH (b) WHERE coalesce(b.description, b.name_or_description, "
                f"b.name, b.value, b.summary) = $to_desc "
                f"MERGE (a)-[:{rel_type}]->(b)"
            )
            try:
                result = session.run(cypher, from_desc=from_desc, to_desc=to_desc)
                created_rels += result.consume().counters.relationships_created
            except Exception as e:
                print(f"    ⚠ Failed to create rel: {e}")

    summary_msg = f"Loaded {created_nodes} nodes and {created_rels} relationships"
    print(f"  ✓ {summary_msg}")

    prov_msg = materialise_provenance(
        driver,
        source_type=source_type,
        observation_desc=f"Source: {source_id}",
        observation_type=source_type,
    )

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


# ═══════════════════════════════════════════════════════════════════════════
# LangGraph pipeline
# ═══════════════════════════════════════════════════════════════════════════

def build_ingest_graph() -> StateGraph:
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


def ingest_statement(
    text: str,
    source_id: str = "statement_1",
    source_type: str = "statement",
    clear: bool = False,
) -> IngestState:
    """Ingest a statement into the graph.

    Args:
        text: The statement text.
        source_id: Unique identifier for this source.
        source_type: Type of source (statement, cctv, forensic, etc.).
        clear: If True, wipe the graph before ingesting.
    """
    if clear:
        with driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("  ✓ Graph cleared")

    graph = build_ingest_graph()
    initial_state: IngestState = {
        "raw_statement": text,
        "source_id": source_id,
        "source_type": source_type,
        "segments": [],
        "extracted": {},
        "resolved": {},
        "load_summary": "",
        "steps": [],
    }
    return graph.invoke(initial_state)


if __name__ == "__main__":
    statement_path = Path(__file__).parent / "statements" / "king_street_collision.txt"
    if statement_path.exists():
        text = statement_path.read_text().strip()
    else:
        text = (
            "I was walking along King Street at approximately 2:15 PM on Tuesday "
            "when I heard a loud crash. I turned and saw a red car had collided "
            "with a cyclist at the junction of King Street and Queen's Road."
        )
    print(f"{'═' * 70}")
    print(f"  DIGITAL TWIN — INGEST (ontology: {get_active_spec().name})")
    print(f"{'═' * 70}")
    ingest_statement(text)
