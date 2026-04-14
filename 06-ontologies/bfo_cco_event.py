"""
BFO / CCO — Basic Formal Ontology + Common Core Ontologies.

BFO (ISO/IEC 21838-2:2021) is the top-level ontology adopted by the
US Department of Defense and Intelligence Community as their baseline
standard for formal ontology work (January 2024 memorandum).

The Common Core Ontologies (CCO) extend BFO with mid-level modules:
  • Event Ontology      — processes, acts, stasis
  • Agent Ontology      — persons, organisations, roles
  • Information Entity  — documents, statements, claims
  • Geospatial Ontology — sites, spatial regions
  • Time Ontology       — temporal intervals and instants

This spec maps the CCO terms to a property-graph representation
suitable for Neo4j and LLM-driven extraction.

References:
  Smith, Barry et al. "Building Ontologies with BFO." MIT Press, 2015.
  DOD/IC Memorandum on BFO+CCO adoption, February 2024.
"""

from ontology_spec import OntologySpec, NodeSpec, RelSpec, CompletenessRule


BFO_CCO_EVENT = OntologySpec(
    id="bfo-cco-event-v1",
    version="1.0",
    name="BFO / CCO Event Ontology",
    description=(
        "Rigorous upper-ontology grounded in BFO (ISO 21838) with CCO "
        "extensions for events, agents, information entities, and "
        "geospatial regions.  Designed for defence/intelligence "
        "interoperability and formal reasoning."
    ),

    node_types={
        # ── Occurrents (things that happen) ─────────────────────────────
        "Process": NodeSpec(
            label="Process",
            required_props=("description", "process_type"),
            optional_props=("severity", "duration"),
            description=(
                "bfo:Process — an occurrent that unfolds in time. "
                "The core event node (replaces schema:Event)."
            ),
            ontology_mapping="bfo:Process",
            merge_key="description",
        ),
        "Act": NodeSpec(
            label="Act",
            required_props=("description", "act_type"),
            optional_props=(),
            description="cco:Act — a process performed intentionally by an agent",
            ontology_mapping="cco:Act (subclass of bfo:Process)",
            merge_key="description",
        ),

        # ── Continuants (things that persist) ───────────────────────────
        "Agent": NodeSpec(
            label="Agent",
            required_props=("name_or_description",),
            optional_props=("name", "age_estimate", "gender"),
            description="cco:Agent — a person or organisation that bears roles",
            ontology_mapping="cco:Agent",
            merge_key="name_or_description",
        ),
        "AgentRole": NodeSpec(
            label="AgentRole",
            required_props=("role_type",),
            optional_props=("description",),
            description=(
                "bfo:Role — a realizable entity inhering in an Agent "
                "(e.g. witness-role, suspect-role)"
            ),
            ontology_mapping="bfo:Role",
            merge_key="role_type",
        ),
        "MaterialEntity": NodeSpec(
            label="MaterialEntity",
            required_props=("description",),
            optional_props=("type", "colour", "make", "model", "registration"),
            description=(
                "bfo:MaterialEntity — a physical object: vehicle, weapon, etc."
            ),
            ontology_mapping="bfo:MaterialEntity",
            merge_key="description",
        ),
        "Site": NodeSpec(
            label="Site",
            required_props=("description",),
            optional_props=("type", "address", "latitude", "longitude"),
            description="bfo:Site — a spatial region: street, building, area",
            ontology_mapping="bfo:Site",
            merge_key="description",
        ),

        # ── Temporal regions ────────────────────────────────────────────
        "TemporalInterval": NodeSpec(
            label="TemporalInterval",
            required_props=("value",),
            optional_props=("precision", "start", "end", "date"),
            description="bfo:TemporalInterval — a connected region of time",
            ontology_mapping="bfo:TemporalInterval",
            merge_key="value",
        ),

        # ── Information entities ────────────────────────────────────────
        "InformationContentEntity": NodeSpec(
            label="InformationContentEntity",
            required_props=("description", "ice_type"),
            optional_props=("content",),
            description=(
                "cco:InformationContentEntity — a statement, claim, "
                "description, or observation record"
            ),
            ontology_mapping="cco:InformationContentEntity",
            merge_key="description",
        ),
        "DescriptiveICE": NodeSpec(
            label="DescriptiveICE",
            required_props=("summary",),
            optional_props=(
                "height", "build", "hair_colour",
                "clothing", "distinguishing_features",
            ),
            description=(
                "cco:DescriptiveInformationContentEntity — appearance "
                "description of an Agent"
            ),
            ontology_mapping="cco:DescriptiveInformationContentEntity",
            merge_key="summary",
        ),
    },

    relationship_types={
        # Agent ↔ Process
        "AGENT_IN": RelSpec(
            "AGENT_IN", "Agent", "Process",
            "Agent participated in the process",
            "cco:agent_in",
        ),
        "HAS_AGENT": RelSpec(
            "HAS_AGENT", "Process", "Agent",
            "Process involves this agent",
            "cco:has_agent",
        ),
        # Roles
        "BEARS_ROLE": RelSpec(
            "BEARS_ROLE", "Agent", "AgentRole",
            "Agent bears this role (bfo:bearer_of)",
            "bfo:bearer_of",
        ),
        "REALIZED_IN": RelSpec(
            "REALIZED_IN", "AgentRole", "Process",
            "Role is realized in this process",
            "bfo:realized_in",
        ),
        # Spatial
        "OCCURS_AT": RelSpec(
            "OCCURS_AT", "Process", "Site",
            "Process occurs at this site (bfo:occurs_in)",
            "bfo:occurs_in",
        ),
        # Temporal
        "OCCUPIES_TEMPORAL": RelSpec(
            "OCCUPIES_TEMPORAL", "Process", "TemporalInterval",
            "Process occupies this temporal region",
            "bfo:occupies_temporal_region",
        ),
        # Material entities
        "HAS_PARTICIPANT": RelSpec(
            "HAS_PARTICIPANT", "Process", "MaterialEntity",
            "Process involves this material entity",
            "bfo:has_participant",
        ),
        "OPERATED_BY": RelSpec(
            "OPERATED_BY", "MaterialEntity", "Agent",
            "Material entity operated/driven by agent",
            "cco:operated_by",
        ),
        # Process structure
        "HAS_PART": RelSpec(
            "HAS_PART", "Process", "Process",
            "Process mereological decomposition (bfo:has_part)",
            "bfo:has_part",
        ),
        "PRECEDES": RelSpec(
            "PRECEDES", "Process", "Process",
            "Temporal ordering (bfo:precedes)",
            "bfo:precedes",
        ),
        "CAUSED_BY": RelSpec(
            "CAUSED_BY", "Process", "Process",
            "Causal dependency between processes",
            "cco:caused_by",
        ),
        # Descriptions
        "DESCRIBED_BY": RelSpec(
            "DESCRIBED_BY", "Agent", "DescriptiveICE",
            "Agent described by this information entity",
            "cco:described_by",
        ),
        # Information provenance
        "IS_ABOUT": RelSpec(
            "IS_ABOUT", "InformationContentEntity", "Process",
            "ICE is about this process (cco:is_about)",
            "cco:is_about",
        ),
        "CREATED_BY": RelSpec(
            "CREATED_BY", "InformationContentEntity", "Agent",
            "ICE was created by this agent",
            "cco:created_by",
        ),
    },

    completeness_rules=[
        CompletenessRule(
            "process_needs_temporal", "critical",
            "Every Process must occupy a TemporalInterval",
            "MATCH (p:Process) WHERE NOT (p)-[:OCCUPIES_TEMPORAL]->(:TemporalInterval) "
            "RETURN p.description AS entity_desc, labels(p)[0] AS label",
            "Process '{entity_desc}' has no temporal region",
        ),
        CompletenessRule(
            "process_needs_site", "critical",
            "Every Process must occur at a Site",
            "MATCH (p:Process) WHERE NOT (p)-[:OCCURS_AT]->(:Site) "
            "RETURN p.description AS entity_desc, labels(p)[0] AS label",
            "Process '{entity_desc}' has no spatial anchor",
        ),
        CompletenessRule(
            "process_needs_agent", "critical",
            "Every Process must have at least one Agent",
            "MATCH (p:Process) WHERE NOT (p)-[:HAS_AGENT]->(:Agent) "
            "  AND NOT (:Agent)-[:AGENT_IN]->(p) "
            "RETURN p.description AS entity_desc, labels(p)[0] AS label",
            "Process '{entity_desc}' has no associated agent",
        ),
        CompletenessRule(
            "agent_needs_role", "high",
            "Every Agent should bear at least one Role",
            "MATCH (a:Agent) WHERE NOT (a)-[:BEARS_ROLE]->(:AgentRole) "
            "RETURN a.name_or_description AS entity_desc, labels(a)[0] AS label",
            "Agent '{entity_desc}' bears no formal role",
        ),
        CompletenessRule(
            "suspect_needs_description", "high",
            "Suspect-role agents should have a DescriptiveICE",
            "MATCH (a:Agent)-[:BEARS_ROLE]->(r:AgentRole {role_type: 'suspect'}) "
            "WHERE NOT (a)-[:DESCRIBED_BY]->(:DescriptiveICE) "
            "RETURN a.name_or_description AS entity_desc, labels(a)[0] AS label",
            "Suspect '{entity_desc}' has no physical description",
        ),
    ],

    provenance_props={
        "source": "Source text this fact was extracted from",
        "source_type": "statement | interview_round_N",
        "extracted_at": "ISO-8601 timestamp",
        "confidence": "high | medium | low",
        "ontology_id": "Identifier of the ontology used",
    },

    event_types=(
        "incident", "observation", "intentional_act", "arrival",
        "departure", "communication", "environmental_process",
    ),
    person_roles=(),  # BFO/CCO uses Role nodes
    time_precisions=("exact", "approximate", "relative", "vague"),
)
