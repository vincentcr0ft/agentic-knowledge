"""
Schema.org Event ontology — the lightweight baseline.

Maps to Schema.org Event + PROV-O + SOSA/SSN as used in the
original event-digital-twin pipeline.  This is the ontology you
already had — now expressed as a pluggable OntologySpec.
"""

from ontology_spec import OntologySpec, NodeSpec, RelSpec, CompletenessRule


SCHEMA_ORG_EVENT = OntologySpec(
    id="schema-org-event-v1",
    version="1.0",
    name="Schema.org Event Ontology",
    description=(
        "Lightweight event-centric ontology using Schema.org Event for "
        "structure, PROV-O for provenance, and SOSA/SSN for the "
        "witness-as-sensor observation model."
    ),

    node_types={
        "Event": NodeSpec(
            label="Event",
            required_props=("description", "type"),
            optional_props=("severity", "duration"),
            description="Something that happened — the central node",
            ontology_mapping="schema:Event + prov:Entity",
            merge_key="description",
        ),
        "Person": NodeSpec(
            label="Person",
            required_props=("name_or_description", "role"),
            optional_props=("name", "age_estimate", "gender"),
            description="Human participant: witness, suspect, victim, bystander",
            ontology_mapping="schema:Person + prov:Agent",
            merge_key="name_or_description",
        ),
        "Vehicle": NodeSpec(
            label="Vehicle",
            required_props=("description",),
            optional_props=("colour", "make", "model", "registration", "type"),
            description="Vehicle involved in or mentioned during events",
            ontology_mapping="schema:Vehicle + prov:Entity",
            merge_key="description",
        ),
        "Location": NodeSpec(
            label="Location",
            required_props=("description",),
            optional_props=("type", "address", "latitude", "longitude"),
            description="Where something happened",
            ontology_mapping="schema:Place",
            merge_key="description",
        ),
        "Time": NodeSpec(
            label="Time",
            required_props=("value",),
            optional_props=("precision", "date", "day_of_week"),
            description="When something happened",
            ontology_mapping="schema:DateTime",
            merge_key="value",
        ),
        "Object": NodeSpec(
            label="Object",
            required_props=("description",),
            optional_props=("type", "colour", "size"),
            description="Physical object mentioned",
            ontology_mapping="prov:Entity",
            merge_key="description",
        ),
        "PhysicalDescription": NodeSpec(
            label="PhysicalDescription",
            required_props=("summary",),
            optional_props=(
                "height", "build", "hair_colour", "hair_style",
                "clothing", "distinguishing_features",
            ),
            description="Appearance of a person",
            ontology_mapping="evt:PhysicalDescription",
            merge_key="summary",
        ),
        "Observation": NodeSpec(
            label="Observation",
            required_props=("description", "observation_type"),
            optional_props=("confidence", "conditions"),
            description="A witness observation (sosa:Observation)",
            ontology_mapping="sosa:Observation + prov:Activity",
            merge_key="description",
        ),
    },

    relationship_types={
        "PARTICIPATED_IN": RelSpec(
            "PARTICIPATED_IN", "Person", "Event",
            "Person was directly involved", "schema:participant (inverse)",
        ),
        "WITNESSED": RelSpec(
            "WITNESSED", "Person", "Event",
            "Person observed the event", "sosa:madeBySensor (inverse)",
        ),
        "OCCURRED_AT": RelSpec(
            "OCCURRED_AT", "Event", "Location",
            "Where the event happened", "schema:location",
        ),
        "OCCURRED_AT_TIME": RelSpec(
            "OCCURRED_AT_TIME", "Event", "Time",
            "When the event happened", "schema:startDate",
        ),
        "USED": RelSpec(
            "USED", "Person", "Object",
            "Person used an object", "prov:used",
        ),
        "DROVE": RelSpec(
            "DROVE", "Person", "Vehicle",
            "Person was driving a vehicle", "evt:drove",
        ),
        "DESCRIBED_AS": RelSpec(
            "DESCRIBED_AS", "Person", "PhysicalDescription",
            "Person's appearance", "evt:describedAs",
        ),
        "CAUSED": RelSpec(
            "CAUSED", "Event", "Event",
            "One event caused another", "evt:caused",
        ),
        "PRECEDED": RelSpec(
            "PRECEDED", "Event", "Event",
            "Temporal ordering", "evt:preceded",
        ),
        "LOCATED_NEAR": RelSpec(
            "LOCATED_NEAR", "Location", "Location",
            "Spatial proximity", "schema:geo",
        ),
        "OWNED_BY": RelSpec(
            "OWNED_BY", "Vehicle", "Person",
            "Vehicle belongs to person", "schema:ownedBy",
        ),
        # System-managed SOSA/PROV links
        "OBSERVED": RelSpec(
            "OBSERVED", "Observation", "Event",
            "sosa:hasFeatureOfInterest", "sosa:hasFeatureOfInterest",
        ),
        "MADE_BY": RelSpec(
            "MADE_BY", "Observation", "Person",
            "sosa:madeBySensor", "sosa:madeBySensor",
        ),
        "DERIVED_FROM": RelSpec(
            "DERIVED_FROM", "Event", "Observation",
            "prov:wasDerivedFrom", "prov:wasDerivedFrom",
        ),
    },

    completeness_rules=[
        CompletenessRule(
            "event_needs_time", "critical",
            "Every Event must have at least one OCCURRED_AT_TIME",
            "MATCH (e:Event) WHERE NOT (e)-[:OCCURRED_AT_TIME]->(:Time) "
            "RETURN e.description AS entity_desc, labels(e)[0] AS label",
            "Event '{entity_desc}' has no timestamp",
        ),
        CompletenessRule(
            "event_needs_location", "critical",
            "Every Event must have at least one OCCURRED_AT location",
            "MATCH (e:Event) WHERE NOT (e)-[:OCCURRED_AT]->(:Location) "
            "RETURN e.description AS entity_desc, labels(e)[0] AS label",
            "Event '{entity_desc}' has no location",
        ),
        CompletenessRule(
            "event_needs_participant", "critical",
            "Every Event must have at least one participant or witness",
            "MATCH (e:Event) WHERE NOT (:Person)-[:PARTICIPATED_IN|WITNESSED]->(e) "
            "RETURN e.description AS entity_desc, labels(e)[0] AS label",
            "Event '{entity_desc}' has no participants or witnesses",
        ),
        CompletenessRule(
            "suspect_needs_description", "high",
            "Suspects must have a PhysicalDescription",
            "MATCH (p:Person {role: 'suspect'}) "
            "WHERE NOT (p)-[:DESCRIBED_AS]->(:PhysicalDescription) "
            "RETURN p.name_or_description AS entity_desc, labels(p)[0] AS label",
            "Suspect '{entity_desc}' has no physical description",
        ),
        CompletenessRule(
            "vehicle_needs_identifiers", "high",
            "Vehicles should have colour, make, or registration",
            "MATCH (v:Vehicle) "
            "WHERE v.colour IS NULL AND v.make IS NULL AND v.registration IS NULL "
            "RETURN v.description AS entity_desc, labels(v)[0] AS label",
            "Vehicle '{entity_desc}' has no identifying details",
        ),
    ],

    provenance_props={
        "source": "Source text this fact was extracted from",
        "source_type": "statement | interview_round_N",
        "extracted_at": "ISO-8601 timestamp of extraction",
        "confidence": "high | medium | low",
        "ontology_id": "Identifier of the ontology used",
    },

    event_types=(
        "incident", "observation", "action", "arrival",
        "departure", "communication", "environmental",
    ),
    person_roles=(
        "witness", "suspect", "victim", "bystander",
        "first_responder", "reporting_officer",
    ),
    time_precisions=("exact", "approximate", "relative", "vague"),

    system_managed_rels=("OBSERVED", "MADE_BY", "DERIVED_FROM"),
    system_managed_nodes=("Observation",),
)
