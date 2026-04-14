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

from ontology_spec import OntologySpec, NodeSpec, RelSpec  # noqa: E402
from schema_org_event import SCHEMA_ORG_EVENT  # noqa: E402
from sem_event import SEM_EVENT  # noqa: E402
from bfo_cco_event import BFO_CCO_EVENT  # noqa: E402

from neo4j import GraphDatabase  # noqa: E402

# ── Compatibility aliases & registry ────────────────────────────────────
NodeDef = NodeSpec
RelDef = RelSpec

ONTOLOGY_REGISTRY: dict[str, OntologySpec] = {
    spec.id: spec for spec in [SCHEMA_ORG_EVENT, SEM_EVENT, BFO_CCO_EVENT]
}


def build_extraction_prompt(spec: OntologySpec) -> str:
    """Standalone wrapper around spec.build_extraction_prompt()."""
    return spec.build_extraction_prompt()


def build_shacl_shapes(spec: OntologySpec) -> str:
    """Standalone wrapper around spec.build_shacl_shapes()."""
    return spec.build_shacl_shapes()


def linearise_for_spec(driver, spec: OntologySpec) -> str:
    """Linearise graph nodes/rels into human-readable triples."""
    lines = []
    with driver.session() as session:
        result = session.run(
            "MATCH (a)-[r]->(b) "
            "RETURN labels(a)[0] AS a_label, "
            "  coalesce(a.description, a.name_or_description, a.value, "
            "    a.summary, toString(id(a))) AS a_desc, "
            "  type(r) AS rel, "
            "  labels(b)[0] AS b_label, "
            "  coalesce(b.description, b.name_or_description, b.value, "
            "    b.summary, toString(id(b))) AS b_desc"
        )
        for rec in result:
            lines.append(
                f"({rec['a_label']}: {rec['a_desc']}) "
                f"-[{rec['rel']}]-> "
                f"({rec['b_label']}: {rec['b_desc']})"
            )
    return "\n".join(lines) if lines else "(empty graph)"


def init_database_for_spec(driver, spec: OntologySpec) -> None:
    """Create uniqueness constraints from the ontology spec."""
    stmts = spec.get_constraint_cypher()
    with driver.session() as session:
        for stmt in stmts:
            session.run(stmt)

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
