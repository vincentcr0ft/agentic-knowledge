"""
06 · Event Digital Twin — Domain Ontology & Completeness Rules
═══════════════════════════════════════════════════════════════

Composites three standard ontologies into a witness-statement schema:

  PROV-O   (W3C)         — provenance: who said what, when, derived from where
  SOSA/SSN (W3C)         — observation model: witness-as-sensor, statement-as-observation
  Schema.org Event       — event-centric modelling with temporal/spatial properties

The schema is expressed as plain Python data structures so that extraction
prompts, gap-analysis Cypher, and graph-loading code can all import and
reference a single source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from neo4j import GraphDatabase


# ═══════════════════════════════════════════════════════════════════════════════
# Ontology namespace mappings (informational — used in comments & docs)
# ═══════════════════════════════════════════════════════════════════════════════

NAMESPACES = {
    "prov":   "http://www.w3.org/ns/prov#",
    "sosa":   "http://www.w3.org/ns/sosa/",
    "ssn":    "http://www.w3.org/ns/ssn/",
    "schema": "https://schema.org/",
    "evt":    "http://example.org/event-twin#",   # local schema namespace
}


# ═══════════════════════════════════════════════════════════════════════════════
# Ontology registry — identifies the active ontology for comparison/testing
# ═══════════════════════════════════════════════════════════════════════════════

ONTOLOGY_META = {
    "id": "witness-statement-v1",
    "version": "1.0",
    "name": "Witness Statement Event Ontology",
    "layers": [
        {"standard": "PROV-O",           "role": "Provenance tracking and lineage"},
        {"standard": "SOSA/SSN",         "role": "Observation model — witness as sensor"},
        {"standard": "Schema.org Event", "role": "Event structure with temporal/spatial anchoring"},
    ],
    "description": (
        "Composites PROV-O, SOSA/SSN, and Schema.org Event into a "
        "three-layer ontology for witness statement analysis. Every node "
        "carries provenance properties (PROV-O), the witness statement is "
        "modelled as a structured Observation (SOSA), and events are "
        "first-class nodes with temporal/spatial/participant anchoring "
        "(Schema.org Event)."
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# Node type definitions
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class NodeDef:
    """Definition of a node type in the event ontology."""
    label: str
    required_props: tuple[str, ...]
    optional_props: tuple[str, ...] = ()
    description: str = ""
    ontology_mapping: str = ""          # e.g. "schema:Event + prov:Entity"


NODE_TYPES: dict[str, NodeDef] = {
    # ── Core event node ─────────────────────────────────────────────────────
    # Maps to schema:Event (structure) + prov:Entity (provenance tracking)
    "Event": NodeDef(
        label="Event",
        required_props=("description", "type"),
        optional_props=("severity", "duration"),
        description="Something that happened — the central node of the graph",
        ontology_mapping="schema:Event + prov:Entity",
    ),

    # ── Participants ────────────────────────────────────────────────────────
    # A Person who is a witness maps additionally to sosa:Sensor (human observer)
    "Person": NodeDef(
        label="Person",
        required_props=("name_or_description", "role"),
        optional_props=("name", "age_estimate", "gender"),
        description="Human participant: witness, suspect, victim, or bystander",
        ontology_mapping="schema:Person + prov:Agent  (witness → sosa:Sensor)",
    ),

    # ── Vehicles ────────────────────────────────────────────────────────────
    "Vehicle": NodeDef(
        label="Vehicle",
        required_props=("description",),
        optional_props=("colour", "make", "model", "registration", "type"),
        description="Vehicle involved in or mentioned during events",
        ontology_mapping="schema:Vehicle + prov:Entity",
    ),

    # ── Locations ───────────────────────────────────────────────────────────
    # Maps to schema:Place with spatial qualifiers
    "Location": NodeDef(
        label="Location",
        required_props=("description",),
        optional_props=("type", "address", "latitude", "longitude"),
        description="Where something happened — street, building, area",
        ontology_mapping="schema:Place",
    ),

    # ── Temporal anchors ────────────────────────────────────────────────────
    "Time": NodeDef(
        label="Time",
        required_props=("value",),
        optional_props=("precision", "date", "day_of_week"),
        description="When something happened — exact, approximate, or relative",
        ontology_mapping="schema:DateTime",
    ),

    # ── Physical objects ────────────────────────────────────────────────────
    "Object": NodeDef(
        label="Object",
        required_props=("description",),
        optional_props=("type", "colour", "size"),
        description="Physical object mentioned: weapon, clothing item, etc.",
        ontology_mapping="prov:Entity",
    ),

    # ── Appearance descriptors ──────────────────────────────────────────────
    "PhysicalDescription": NodeDef(
        label="PhysicalDescription",
        required_props=("summary",),
        optional_props=(
            "height", "build", "hair_colour", "hair_style",
            "clothing", "distinguishing_features", "ethnicity_estimate",
        ),
        description="Appearance of a person — physical and clothing details",
        ontology_mapping="evt:PhysicalDescription",
    ),

    # ── Observation (SOSA layer) ────────────────────────────────────────────
    # The witness statement itself as a structured observation
    "Observation": NodeDef(
        label="Observation",
        required_props=("description", "observation_type"),
        optional_props=("confidence", "conditions"),
        description=(
            "A witness observation — maps to sosa:Observation. "
            "Links the observer (sosa:Sensor/Person) to what was observed "
            "(sosa:FeatureOfInterest/Event) and the extracted result"
        ),
        ontology_mapping="sosa:Observation + prov:Activity",
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# Relationship type definitions
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class RelDef:
    """Definition of a relationship type."""
    rel_type: str
    from_label: str
    to_label: str
    description: str = ""
    ontology_mapping: str = ""


RELATIONSHIP_TYPES: dict[str, RelDef] = {
    # ── Event participation ─────────────────────────────────────────────────
    "PARTICIPATED_IN": RelDef(
        "PARTICIPATED_IN", "Person", "Event",
        "Person was directly involved in the event",
        "schema:participant (inverse)",
    ),
    "WITNESSED": RelDef(
        "WITNESSED", "Person", "Event",
        "Person observed the event (witness-as-sensor)",
        "sosa:madeBySensor (inverse) — sensor observes feature-of-interest",
    ),

    # ── Spatio-temporal anchoring ───────────────────────────────────────────
    "OCCURRED_AT": RelDef(
        "OCCURRED_AT", "Event", "Location",
        "Where the event happened",
        "schema:location",
    ),
    "OCCURRED_AT_TIME": RelDef(
        "OCCURRED_AT_TIME", "Event", "Time",
        "When the event happened",
        "schema:startDate / schema:endDate",
    ),

    # ── Object/vehicle usage ────────────────────────────────────────────────
    "USED": RelDef(
        "USED", "Person", "Object",
        "Person used an object or vehicle",
        "prov:used",
    ),
    "DROVE": RelDef(
        "DROVE", "Person", "Vehicle",
        "Person was driving a vehicle",
        "evt:drove",
    ),

    # ── Descriptions ────────────────────────────────────────────────────────
    "DESCRIBED_AS": RelDef(
        "DESCRIBED_AS", "Person", "PhysicalDescription",
        "Links a person to their appearance description",
        "evt:describedAs",
    ),

    # ── Inter-event relationships ───────────────────────────────────────────
    "CAUSED": RelDef(
        "CAUSED", "Event", "Event",
        "One event caused another",
        "evt:caused",
    ),
    "PRECEDED": RelDef(
        "PRECEDED", "Event", "Event",
        "Temporal ordering — this event happened before the other",
        "evt:preceded",
    ),

    # ── Spatial relationships ───────────────────────────────────────────────
    "LOCATED_NEAR": RelDef(
        "LOCATED_NEAR", "Location", "Location",
        "Spatial proximity between locations",
        "schema:geo (qualitative)",
    ),

    # ── Ownership ───────────────────────────────────────────────────────────
    "OWNED_BY": RelDef(
        "OWNED_BY", "Vehicle", "Person",
        "Vehicle or object belongs to a person",
        "schema:ownedBy",
    ),

    # ── SOSA / provenance links ─────────────────────────────────────────────
    "OBSERVED": RelDef(
        "OBSERVED", "Observation", "Event",
        "What the observation is about (sosa:hasFeatureOfInterest)",
        "sosa:hasFeatureOfInterest",
    ),
    "MADE_BY": RelDef(
        "MADE_BY", "Observation", "Person",
        "Who made the observation (sosa:madeBySensor)",
        "sosa:madeBySensor",
    ),

    # ── Provenance ──────────────────────────────────────────────────────────
    "DERIVED_FROM": RelDef(
        "DERIVED_FROM", "Event", "Observation",
        "This graph entity was derived from a particular observation",
        "prov:wasDerivedFrom",
    ),
}


# Property attached to every node for provenance tracking (prov:wasGeneratedBy)
PROVENANCE_PROPS = {
    "source":        "Source text (sentence or answer) this fact was extracted from",
    "source_type":   "One of: statement | interview_round_N",
    "extracted_at":  "ISO-8601 timestamp of extraction",
    "confidence":    "Extraction confidence: high | medium | low",
    "ontology_id":   "Identifier of the ontology used to produce this node",
}


# ═══════════════════════════════════════════════════════════════════════════════
# SOSA/PROV materialisation — create Observation nodes and provenance edges
# ═══════════════════════════════════════════════════════════════════════════════

def materialise_provenance(
    driver: GraphDatabase.driver,
    source_type: str,
    observation_desc: str,
    observation_type: str = "witness_statement",
    witness_desc: str | None = None,
) -> str:
    """Create the SOSA Observation node and PROV-O lineage edges.

    This is the "system-managed" counterpart to LLM extraction — it
    programmatically instantiates the SOSA/PROV layer that the LLM
    extraction prompt intentionally skips.

    Creates:
      (Observation) — sosa:Observation + prov:Activity
      (Observation)-[:MADE_BY]->(Person{role:'witness'}) — sosa:madeBySensor
      (Observation)-[:OBSERVED]->(Event) — sosa:hasFeatureOfInterest
      (Event)-[:DERIVED_FROM]->(Observation) — prov:wasDerivedFrom

    Args:
        driver:           Neo4j driver.
        source_type:      Provenance tag to match nodes, e.g. "statement"
                          or "interview_round_1".
        observation_desc: Human-readable label for the Observation node.
        observation_type: One of "witness_statement" | "follow_up".
        witness_desc:     name_or_description of the witness Person. If
                          None, auto-detects the Person with role='witness'.

    Returns:
        Summary string of what was created.
    """
    from datetime import datetime, timezone

    timestamp = datetime.now(timezone.utc).isoformat()
    ontology_id = ONTOLOGY_META["id"]
    created_obs = 0
    created_rels = 0

    with driver.session() as session:
        # ── 1. Create the Observation node ──────────────────────────────
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
        created_obs = 1

        # ── 2. MADE_BY — link Observation to witness (sosa:madeBySensor)
        if witness_desc:
            result = session.run(
                "MATCH (obs:Observation {description: $obs_desc}) "
                "MATCH (w:Person {name_or_description: $w_desc}) "
                "MERGE (obs)-[:MADE_BY]->(w)",
                obs_desc=observation_desc,
                w_desc=witness_desc,
            )
            created_rels += result.consume().counters.relationships_created
        else:
            # Auto-detect witness
            result = session.run(
                "MATCH (obs:Observation {description: $obs_desc}) "
                "MATCH (w:Person {role: 'witness'}) "
                "MERGE (obs)-[:MADE_BY]->(w)",
                obs_desc=observation_desc,
            )
            created_rels += result.consume().counters.relationships_created

        # ── 3. OBSERVED — link Observation to each Event ────────────────
        #    For "statement" source, link to ALL events.
        #    For interview rounds, link to events touched in that round.
        if source_type == "statement":
            result = session.run(
                "MATCH (obs:Observation {description: $obs_desc}) "
                "MATCH (e:Event) "
                "MERGE (obs)-[:OBSERVED]->(e)",
                obs_desc=observation_desc,
            )
        else:
            # Link to events that have nodes connected in this round
            result = session.run(
                "MATCH (obs:Observation {description: $obs_desc}) "
                "MATCH (e:Event) "
                "WHERE e.source_type = $src_type "
                "   OR EXISTS { "
                "     MATCH (e)-[]-(n) WHERE n.source_type = $src_type "
                "   } "
                "MERGE (obs)-[:OBSERVED]->(e)",
                obs_desc=observation_desc,
                src_type=source_type,
            )
        created_rels += result.consume().counters.relationships_created

        # ── 4. DERIVED_FROM — Event back-links to Observation ───────────
        if source_type == "statement":
            result = session.run(
                "MATCH (obs:Observation {description: $obs_desc}) "
                "MATCH (e:Event) "
                "MERGE (e)-[:DERIVED_FROM]->(obs)",
                obs_desc=observation_desc,
            )
        else:
            result = session.run(
                "MATCH (obs:Observation {description: $obs_desc}) "
                "MATCH (e:Event) "
                "WHERE e.source_type = $src_type "
                "   OR EXISTS { "
                "     MATCH (e)-[]-(n) WHERE n.source_type = $src_type "
                "   } "
                "MERGE (e)-[:DERIVED_FROM]->(obs)",
                obs_desc=observation_desc,
                src_type=source_type,
            )
        created_rels += result.consume().counters.relationships_created

    msg = (
        f"Provenance: created Observation '{observation_desc}' "
        f"with {created_rels} SOSA/PROV relationships"
    )
    print(f"  ✓ {msg}")
    return msg


# ═══════════════════════════════════════════════════════════════════════════════
# Event type taxonomy
# ═══════════════════════════════════════════════════════════════════════════════

EVENT_TYPES = (
    "incident",         # the core event under investigation
    "observation",      # witness sees/hears something
    "action",           # a participant does something
    "arrival",          # someone/something arrives
    "departure",        # someone/something leaves
    "communication",    # a call, shout, conversation
    "environmental",    # weather, lighting, noise
)

PERSON_ROLES = (
    "witness",
    "suspect",
    "victim",
    "bystander",
    "first_responder",
    "reporting_officer",
)

TIME_PRECISIONS = (
    "exact",            # "at 14:15"
    "approximate",      # "about 2:15 PM"
    "relative",         # "ten minutes later"
    "vague",            # "in the afternoon"
)


# ═══════════════════════════════════════════════════════════════════════════════
# Completeness rules — each returns a list of gap dicts
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Gap:
    """A single gap identified by completeness analysis."""
    rule_id: str
    priority: str            # critical | high | medium | low
    entity_label: str        # e.g. "Event", "Person"
    entity_desc: str         # description of the specific entity
    gap_description: str     # human-readable description of what's missing
    cypher_query: str = ""   # the Cypher that detected it


# ── Level 1: Schema completeness (Cypher-driven) ───────────────────────────

SCHEMA_COMPLETENESS_RULES: list[dict] = [
    {
        "rule_id": "event_needs_time",
        "priority": "critical",
        "description": "Every Event must have at least one OCCURRED_AT_TIME",
        "cypher": (
            "MATCH (e:Event) "
            "WHERE NOT (e)-[:OCCURRED_AT_TIME]->(:Time) "
            "RETURN e.description AS entity_desc, labels(e)[0] AS label"
        ),
        "gap_template": "Event '{entity_desc}' has no timestamp",
    },
    {
        "rule_id": "event_needs_location",
        "priority": "critical",
        "description": "Every Event must have at least one OCCURRED_AT location",
        "cypher": (
            "MATCH (e:Event) "
            "WHERE NOT (e)-[:OCCURRED_AT]->(:Location) "
            "RETURN e.description AS entity_desc, labels(e)[0] AS label"
        ),
        "gap_template": "Event '{entity_desc}' has no location",
    },
    {
        "rule_id": "event_needs_participant",
        "priority": "critical",
        "description": "Every Event must have at least one participant or witness",
        "cypher": (
            "MATCH (e:Event) "
            "WHERE NOT (:Person)-[:PARTICIPATED_IN|WITNESSED]->(e) "
            "RETURN e.description AS entity_desc, labels(e)[0] AS label"
        ),
        "gap_template": "Event '{entity_desc}' has no participants or witnesses",
    },
    {
        "rule_id": "suspect_needs_description",
        "priority": "high",
        "description": "Every Person with role=suspect must have a PhysicalDescription",
        "cypher": (
            "MATCH (p:Person {role: 'suspect'}) "
            "WHERE NOT (p)-[:DESCRIBED_AS]->(:PhysicalDescription) "
            "RETURN p.name_or_description AS entity_desc, labels(p)[0] AS label"
        ),
        "gap_template": "Suspect '{entity_desc}' has no physical description",
    },
    {
        "rule_id": "vehicle_needs_identifiers",
        "priority": "high",
        "description": "Vehicles should have colour, make, and/or registration",
        "cypher": (
            "MATCH (v:Vehicle) "
            "WHERE v.colour IS NULL AND v.make IS NULL AND v.registration IS NULL "
            "RETURN v.description AS entity_desc, labels(v)[0] AS label"
        ),
        "gap_template": "Vehicle '{entity_desc}' has no identifying details (colour/make/registration)",
    },
    {
        "rule_id": "vehicle_no_registration",
        "priority": "high",
        "description": "Vehicles involved in incidents should have a registration if possible",
        "cypher": (
            "MATCH (v:Vehicle) WHERE v.registration IS NULL "
            "RETURN v.description AS entity_desc, labels(v)[0] AS label"
        ),
        "gap_template": "Vehicle '{entity_desc}' has no registration number",
    },
    {
        "rule_id": "events_need_ordering",
        "priority": "medium",
        "description": "Events should be temporally ordered (PRECEDED relationships)",
        "cypher": (
            "MATCH (e:Event) "
            "WHERE NOT (e)-[:PRECEDED|CAUSED]->(:Event) "
            "  AND NOT (:Event)-[:PRECEDED|CAUSED]->(e) "
            "RETURN e.description AS entity_desc, labels(e)[0] AS label"
        ),
        "gap_template": "Event '{entity_desc}' is not linked to any other event temporally or causally",
    },
    {
        "rule_id": "time_precision",
        "priority": "medium",
        "description": "Time nodes should be as precise as possible",
        "cypher": (
            "MATCH (t:Time) "
            "WHERE t.precision IN ['vague', 'relative'] "
            "RETURN t.value AS entity_desc, labels(t)[0] AS label"
        ),
        "gap_template": "Time '{entity_desc}' is imprecise — can the witness be more specific?",
    },
    {
        "rule_id": "orphan_entities",
        "priority": "low",
        "description": "No node should be disconnected from the graph",
        "cypher": (
            "MATCH (n) WHERE NOT (n)-[]-() "
            "RETURN coalesce(n.description, n.name_or_description, n.value, 'unknown') "
            "  AS entity_desc, labels(n)[0] AS label"
        ),
        "gap_template": "'{entity_desc}' ({label}) is disconnected from all other entities",
    },
    {
        "rule_id": "witness_links",
        "priority": "medium",
        "description": "The witness should have WITNESSED relationships to all events",
        "cypher": (
            "MATCH (w:Person {role: 'witness'}), (e:Event) "
            "WHERE NOT (w)-[:WITNESSED]->(e) "
            "  AND NOT (w)-[:PARTICIPATED_IN]->(e) "
            "RETURN w.name_or_description + ' → ' + e.description AS entity_desc, "
            "  'Person' AS label"
        ),
        "gap_template": "Witness not linked to event: {entity_desc}",
    },
]


# ── Level 2: Narrative coherence (LLM-driven — prompt templates) ───────────

NARRATIVE_COHERENCE_PROMPT = """\
You are an investigative analyst reviewing a knowledge graph built from a \
witness statement. The graph is represented as linearised triples below.

GRAPH TRIPLES:
{triples}

Analyse the graph for narrative coherence problems. Check for:

1. **Temporal consistency** — Are events ordered logically? Are there \
unexplained gaps in the timeline?
2. **Spatial consistency** — Does movement between locations make physical \
sense? Are distances/times plausible?
3. **Participant consistency** — Is the same person described differently in \
different events? Are roles contradictory?
4. **Causal plausibility** — Do the causal links make sense? Are there \
effects without causes?

Return a JSON array of gaps. Each gap:
{{"rule_id": "narrative_<type>", "priority": "medium", "entity_label": "<label>", \
"entity_desc": "<entity>", "gap_description": "<what is inconsistent or missing>"}}

If no coherence issues are found, return an empty array: []
"""


# ── Level 3: Investigative completeness (LLM-driven — prompt template) ─────

INVESTIGATIVE_COMPLETENESS_PROMPT = """\
You are an experienced police investigator reviewing the following knowledge \
graph built from a witness statement. The graph is represented as linearised \
triples.

GRAPH TRIPLES:
{triples}

From an investigative standpoint, identify information gaps that would matter \
for the investigation. Consider:

- Can each person mentioned be identified or located?
- Is the sequence of events clear enough to reconstruct what happened?
- Are there implied participants who aren't explicitly mentioned?
- Are environmental conditions (weather, lighting, visibility) captured?
- Is the witness's own position and line of sight established?
- Are directions of travel, speeds, and distances noted?
- Is there any physical evidence that should be looked for?

Return a JSON array of gaps. Each gap:
{{"rule_id": "investigative_<type>", "priority": "<critical|high|medium|low>", \
"entity_label": "<label>", "entity_desc": "<entity or 'general'>", \
"gap_description": "<what an investigator would want to know>"}}

If no investigative gaps are found, return an empty array: []
"""


# ═══════════════════════════════════════════════════════════════════════════════
# Gap analysis — schema completeness runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_schema_completeness(
    driver: GraphDatabase.driver,
) -> list[Gap]:
    """Execute all schema completeness rules and return gaps found."""
    gaps: list[Gap] = []
    with driver.session() as session:
        for rule in SCHEMA_COMPLETENESS_RULES:
            records = session.run(rule["cypher"])
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
                    cypher_query=rule["cypher"],
                ))
    return gaps


def prioritise_gaps(gaps: list[Gap]) -> list[Gap]:
    """Sort gaps by priority: critical → high → medium → low."""
    order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    return sorted(gaps, key=lambda g: order.get(g.priority, 99))


# ═══════════════════════════════════════════════════════════════════════════════
# Graph linearisation — convert Neo4j subgraph to triples for LLM consumption
# ═══════════════════════════════════════════════════════════════════════════════

def linearise_graph(driver: GraphDatabase.driver) -> str:
    """Dump the full graph as linearised triples for LLM prompts.

    Format per Dai et al. finding: (Subject, RELATIONSHIP, Object)
    with key properties inlined.
    """
    cypher = (
        "MATCH (a)-[r]->(b) "
        "RETURN labels(a)[0] AS a_label, "
        "  coalesce(a.description, a.name_or_description, a.name, a.value, "
        "    a.summary, 'unknown') AS a_desc, "
        "  type(r) AS rel, "
        "  labels(b)[0] AS b_label, "
        "  coalesce(b.description, b.name_or_description, b.name, b.value, "
        "    b.summary, 'unknown') AS b_desc"
    )
    lines = []
    with driver.session() as session:
        for rec in session.run(cypher):
            lines.append(
                f"({rec['a_label']}: {rec['a_desc']}) "
                f"-[{rec['rel']}]-> "
                f"({rec['b_label']}: {rec['b_desc']})"
            )
    return "\n".join(lines) if lines else "(empty graph)"


# ═══════════════════════════════════════════════════════════════════════════════
# Extraction prompt — schema-guided entity/relationship extraction
# ═══════════════════════════════════════════════════════════════════════════════

def build_extraction_prompt() -> str:
    """Generate the extraction system prompt from the schema definitions.

    This ensures the LLM only extracts entity and relationship types that
    exist in our ontology, and includes all required/optional properties.
    """
    lines = ["Extract entities and relationships from the witness statement text."]
    lines.append("")

    # ── Node types ──────────────────────────────────────────────────────────
    lines.append("TARGET ENTITY TYPES:")
    for ndef in NODE_TYPES.values():
        if ndef.label == "Observation":
            continue  # Observation nodes are system-managed, not extracted
        req = ", ".join(ndef.required_props)
        opt = ", ".join(ndef.optional_props) if ndef.optional_props else "none"
        lines.append(f"  - {ndef.label} (required: {req}; optional: {opt})")
    lines.append("")

    # ── Allowed event types ─────────────────────────────────────────────────
    lines.append(f"EVENT TYPES: {', '.join(EVENT_TYPES)}")
    lines.append(f"PERSON ROLES: {', '.join(PERSON_ROLES)}")
    lines.append(f"TIME PRECISIONS: {', '.join(TIME_PRECISIONS)}")
    lines.append("")

    # ── Relationship types ──────────────────────────────────────────────────
    lines.append("TARGET RELATIONSHIP TYPES:")
    for rdef in RELATIONSHIP_TYPES.values():
        if rdef.rel_type in ("OBSERVED", "MADE_BY", "DERIVED_FROM"):
            continue  # Provenance/SOSA links are system-managed
        lines.append(f"  - {rdef.rel_type} ({rdef.from_label} → {rdef.to_label}): {rdef.description}")
    lines.append("")

    # ── Output format ───────────────────────────────────────────────────────
    lines.append("Return valid JSON with this structure:")
    lines.append('{')
    lines.append('  "entities": [')
    lines.append('    {"id": "e1", "label": "Event", "properties": {"description": "...", "type": "incident"}}')
    lines.append('  ],')
    lines.append('  "relationships": [')
    lines.append('    {"from_id": "p1", "rel_type": "PARTICIPATED_IN", "to_id": "e1"}')
    lines.append('  ]')
    lines.append('}')
    lines.append("")
    lines.append("RULES:")
    lines.append("- Extract ALL entities and relationships mentioned or implied in the text")
    lines.append("- Use the entity types and relationship types listed above — do not invent new ones")
    lines.append("- Assign a unique id to each entity (e.g. e1, p1, v1, l1, t1, o1, pd1)")
    lines.append("- For Person entities, always include role (witness/suspect/victim/bystander)")
    lines.append("- For Time entities, include precision (exact/approximate/relative/vague)")
    lines.append("- Resolve pronouns: if 'he' clearly refers to a previously mentioned person, use the same id")
    lines.append("- The statement author is a Person with role='witness' — include them")
    lines.append("- Break compound events into separate Event nodes linked by PRECEDED or CAUSED")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Question generation — priority-ordered prompt templates
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
- Be SPECIFIC — reference known details: "You mentioned a red car — do you \
recall its make or model?"
- Be NON-LEADING — ask "What did the person look like?" not "Was the person \
tall?"
- Group related gaps into a single question where natural
- Focus on the highest-priority gaps first
- Do NOT ask about things already captured in the graph

Return a JSON array of question objects:
[{{"question": "...", "targets_gaps": ["rule_id_1", "rule_id_2"]}}]
"""


# ═══════════════════════════════════════════════════════════════════════════════
# Neo4j constraint setup — run once to initialise the database schema
# ═══════════════════════════════════════════════════════════════════════════════

CONSTRAINT_CYPHER = [
    # Uniqueness constraints for entity resolution / MERGE operations
    "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Event) REQUIRE e.description IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (l:Location) REQUIRE l.description IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Time) REQUIRE t.value IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (obs:Observation) REQUIRE obs.description IS UNIQUE",
]

INDEX_CYPHER = [
    "CREATE INDEX IF NOT EXISTS FOR (p:Person) ON (p.name_or_description)",
    "CREATE INDEX IF NOT EXISTS FOR (v:Vehicle) ON (v.description)",
    "CREATE INDEX IF NOT EXISTS FOR (o:Object) ON (o.description)",
    "CREATE INDEX IF NOT EXISTS FOR (pd:PhysicalDescription) ON (pd.summary)",
    # Ontology comparison index — allows filtering nodes by ontology version
    "CREATE INDEX IF NOT EXISTS FOR (n:Observation) ON (n.ontology_id)",
]


def init_database(driver: GraphDatabase.driver) -> None:
    """Create constraints and indexes in Neo4j."""
    with driver.session() as session:
        for stmt in CONSTRAINT_CYPHER + INDEX_CYPHER:
            session.run(stmt)


# ═══════════════════════════════════════════════════════════════════════════════
# Quick self-test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"Ontology: {ONTOLOGY_META['name']} (v{ONTOLOGY_META['version']})")
    print(f"  ID: {ONTOLOGY_META['id']}")
    for layer in ONTOLOGY_META["layers"]:
        print(f"  Layer: {layer['standard']:20s} — {layer['role']}")

    print("\nNode types:")
    for name, ndef in NODE_TYPES.items():
        print(f"  {name:25s}  ← {ndef.ontology_mapping}")

    print("\nRelationship types:")
    for name, rdef in RELATIONSHIP_TYPES.items():
        print(f"  ({rdef.from_label})-[{name}]->({rdef.to_label})  ← {rdef.ontology_mapping}")

    print("\nProvenance properties (stamped on every node):")
    for prop, desc in PROVENANCE_PROPS.items():
        print(f"  {prop:15s} — {desc}")

    print("\nSchema completeness rules:")
    for rule in SCHEMA_COMPLETENESS_RULES:
        print(f"  [{rule['priority']:8s}] {rule['description']}")

    print("\nExtraction prompt (first 20 lines):")
    prompt = build_extraction_prompt()
    for line in prompt.split("\n")[:20]:
        print(f"  {line}")

    print(f"\n  ... ({len(prompt.split(chr(10)))} lines total)")
