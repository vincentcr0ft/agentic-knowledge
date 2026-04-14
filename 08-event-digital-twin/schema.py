"""
08 · Digital Twin — Schema Bridge
══════════════════════════════════

Thin bridge that imports the ontology spec from 06-ontologies and exposes
the same interface the original ingest/interview/query pipeline expects.

This lets the witness-statement pipeline work with any pluggable ontology
from the ontology module while keeping backward compatibility.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add the ontology module to the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "06-ontologies"))

from ontology_spec import (      # noqa: E402
    OntologySpec,
    NodeDef,
    RelDef,
    ONTOLOGY_REGISTRY,
    build_extraction_prompt,
    build_shacl_shapes,
    linearise_for_spec,
    init_database_for_spec,
)

from neo4j import GraphDatabase  # noqa: E402

# ═══════════════════════════════════════════════════════════════════════════════
# Default ontology — can be overridden via select_ontology()
# ═══════════════════════════════════════════════════════════════════════════════

_active_spec: OntologySpec | None = None


def select_ontology(spec_id: str) -> OntologySpec:
    """Set the active ontology for this pipeline run."""
    global _active_spec
    if spec_id not in ONTOLOGY_REGISTRY:
        available = ", ".join(ONTOLOGY_REGISTRY.keys())
        raise ValueError(f"Unknown ontology '{spec_id}'. Available: {available}")
    _active_spec = ONTOLOGY_REGISTRY[spec_id]
    print(f"  ✓ Active ontology: {_active_spec.name} ({_active_spec.id})")
    return _active_spec


def get_active_spec() -> OntologySpec:
    """Return the active ontology, defaulting to schema-org-event-v1."""
    global _active_spec
    if _active_spec is None:
        _active_spec = ONTOLOGY_REGISTRY.get(
            "schema-org-event-v1",
            next(iter(ONTOLOGY_REGISTRY.values())),
        )
    return _active_spec


# ═══════════════════════════════════════════════════════════════════════════════
# Compatibility shims — functions the pipeline expects from the old schema.py
# ═══════════════════════════════════════════════════════════════════════════════

def get_node_types() -> dict[str, NodeDef]:
    return get_active_spec().node_types


def get_relationship_types() -> dict[str, RelDef]:
    return get_active_spec().relationship_types


def get_extraction_prompt() -> str:
    return build_extraction_prompt(get_active_spec())


def get_shacl_shapes() -> str:
    return build_shacl_shapes(get_active_spec())


def linearise_graph(driver) -> str:
    return linearise_for_spec(driver, get_active_spec())


def init_database(driver) -> None:
    init_database_for_spec(driver, get_active_spec())


def get_ontology_id() -> str:
    return get_active_spec().id


# ═══════════════════════════════════════════════════════════════════════════════
# Gap analysis — carried forward from the original schema.py
# ═══════════════════════════════════════════════════════════════════════════════

from dataclasses import dataclass  # noqa: E402


@dataclass
class Gap:
    """A single gap identified by completeness analysis."""
    rule_id: str
    priority: str
    entity_label: str
    entity_desc: str
    gap_description: str
    cypher_query: str = ""


def run_schema_completeness(driver) -> list[Gap]:
    """Run completeness rules from the active ontology spec."""
    spec = get_active_spec()
    gaps: list[Gap] = []

    with driver.session() as session:
        for rule in spec.completeness_rules:
            cypher = rule.get("cypher", "")
            if not cypher:
                continue
            records = session.run(cypher)
            for record in records:
                entity_desc = record.get("entity_desc", "unknown")
                label = record.get("label", "unknown")
                gaps.append(Gap(
                    rule_id=rule["rule_id"],
                    priority=rule["priority"],
                    entity_label=label,
                    entity_desc=str(entity_desc),
                    gap_description=rule["gap_template"].format(
                        entity_desc=entity_desc, label=label,
                    ),
                    cypher_query=cypher,
                ))
    return gaps


def prioritise_gaps(gaps: list[Gap]) -> list[Gap]:
    order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    return sorted(gaps, key=lambda g: order.get(g.priority, 99))


# ═══════════════════════════════════════════════════════════════════════════════
# Provenance materialisation (SOSA/PROV layer)
# ═══════════════════════════════════════════════════════════════════════════════

def materialise_provenance(
    driver,
    source_type: str,
    observation_desc: str,
    observation_type: str = "witness_statement",
    witness_desc: str | None = None,
) -> str:
    """Create SOSA Observation node and PROV-O lineage edges."""
    from datetime import datetime, timezone

    timestamp = datetime.now(timezone.utc).isoformat()
    ontology_id = get_ontology_id()
    created_rels = 0

    with driver.session() as session:
        session.run(
            "MERGE (obs:Observation {description: $desc}) "
            "SET obs.observation_type = $obs_type, "
            "    obs.source_type     = $src_type, "
            "    obs.extracted_at    = $ts, "
            "    obs.confidence      = 'high', "
            "    obs.ontology_id     = $ont_id",
            desc=observation_desc,
            obs_type=observation_type,
            src_type=source_type,
            ts=timestamp,
            ont_id=ontology_id,
        )

        if witness_desc:
            result = session.run(
                "MATCH (obs:Observation {description: $obs_desc}) "
                "MATCH (w:Person {name_or_description: $w_desc}) "
                "MERGE (obs)-[:MADE_BY]->(w)",
                obs_desc=observation_desc, w_desc=witness_desc,
            )
            created_rels += result.consume().counters.relationships_created
        else:
            result = session.run(
                "MATCH (obs:Observation {description: $obs_desc}) "
                "MATCH (w:Person {role: 'witness'}) "
                "MERGE (obs)-[:MADE_BY]->(w)",
                obs_desc=observation_desc,
            )
            created_rels += result.consume().counters.relationships_created

        if source_type == "statement":
            result = session.run(
                "MATCH (obs:Observation {description: $obs_desc}) "
                "MATCH (e:Event) MERGE (obs)-[:OBSERVED]->(e)",
                obs_desc=observation_desc,
            )
        else:
            result = session.run(
                "MATCH (obs:Observation {description: $obs_desc}) "
                "MATCH (e:Event) "
                "WHERE e.source_type = $src_type "
                "   OR EXISTS { MATCH (e)-[]-(n) WHERE n.source_type = $src_type } "
                "MERGE (obs)-[:OBSERVED]->(e)",
                obs_desc=observation_desc, src_type=source_type,
            )
        created_rels += result.consume().counters.relationships_created

        if source_type == "statement":
            result = session.run(
                "MATCH (obs:Observation {description: $obs_desc}) "
                "MATCH (e:Event) MERGE (e)-[:DERIVED_FROM]->(obs)",
                obs_desc=observation_desc,
            )
        else:
            result = session.run(
                "MATCH (obs:Observation {description: $obs_desc}) "
                "MATCH (e:Event) "
                "WHERE e.source_type = $src_type "
                "   OR EXISTS { MATCH (e)-[]-(n) WHERE n.source_type = $src_type } "
                "MERGE (e)-[:DERIVED_FROM]->(obs)",
                obs_desc=observation_desc, src_type=source_type,
            )
        created_rels += result.consume().counters.relationships_created

    msg = f"Provenance: created Observation '{observation_desc}' with {created_rels} SOSA/PROV rels"
    print(f"  ✓ {msg}")
    return msg


# ═══════════════════════════════════════════════════════════════════════════════
# Question generation prompt (for interview phase)
# ═══════════════════════════════════════════════════════════════════════════════

QUESTION_GENERATION_PROMPT = """\
You are a careful investigator conducting a follow-up interview with a witness.
Based on the gaps identified in the knowledge graph, generate follow-up questions.

GAPS (ordered by priority):
{gaps}

EXISTING GRAPH (for context — reference what IS known):
{triples}

RULES:
- Generate {max_questions} questions maximum
- Be SPECIFIC — reference known details
- Be NON-LEADING — ask open questions
- Group related gaps into a single question where natural
- Focus on the highest-priority gaps first
- Do NOT ask about things already captured in the graph

Return a JSON array of question objects:
[{{"question": "...", "targets_gaps": ["rule_id_1", "rule_id_2"]}}]
"""


# ═══════════════════════════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Available ontologies:")
    for sid, spec in ONTOLOGY_REGISTRY.items():
        print(f"  {sid:25s}  {spec.name}")

    print(f"\nActive: {get_active_spec().name}")
    print(f"\nExtraction prompt (first 10 lines):")
    for line in get_extraction_prompt().split("\n")[:10]:
        print(f"  {line}")
