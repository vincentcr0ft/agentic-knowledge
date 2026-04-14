"""
SEM — Simple Event Model ontology.

SEM (van Hage et al., 2011) was designed at the VU Amsterdam Semantic
Web group specifically for representing events found in text,
historical records, and multimedia.  Its key advantage over Schema.org
Event is:

  • First-class role modelling — the same Person can play different
    roles in different events via sem:RoleType on the participation edge.
  • Sub-event decomposition — sem:hasSubEvent allows event nesting.
  • Temporal model — sem:hasTimeStamp accepts either an instant or an
    interval (via sem:hasEarliestBeginTimeStamp / sem:hasLatestEndTimeStamp).

Reference:
  Willem Robert van Hage, Véronique Malaisé, Roxane Segers, Laura
  Hollink, and Guus Schreiber. "Design and use of the Simple Event
  Model (SEM)." *Web Semantics*, 9(2):128-136, 2011.
"""

from ontology_spec import OntologySpec, NodeSpec, RelSpec, CompletenessRule


SEM_EVENT = OntologySpec(
    id="sem-event-v1",
    version="1.0",
    name="Simple Event Model (SEM)",
    description=(
        "Purpose-built event ontology with first-class roles and sub-event "
        "decomposition.  Designed for narrative, historical, and multimedia events."
    ),

    node_types={
        "Event": NodeSpec(
            label="Event",
            required_props=("description", "event_type"),
            optional_props=("severity",),
            description="sem:Event — something that happened",
            ontology_mapping="sem:Event",
            merge_key="description",
        ),
        "Actor": NodeSpec(
            label="Actor",
            required_props=("name_or_description",),
            optional_props=("name", "actor_type", "age_estimate", "gender"),
            description="sem:Actor — human, organisation, or other agent",
            ontology_mapping="sem:Actor",
            merge_key="name_or_description",
        ),
        "Place": NodeSpec(
            label="Place",
            required_props=("description",),
            optional_props=("type", "address", "latitude", "longitude"),
            description="sem:Place — where events occur",
            ontology_mapping="sem:Place",
            merge_key="description",
        ),
        "Time": NodeSpec(
            label="Time",
            required_props=("value",),
            optional_props=("precision", "earliest", "latest", "date"),
            description="sem:Time — when events occur (instant or interval)",
            ontology_mapping="sem:Time",
            merge_key="value",
        ),
        "Role": NodeSpec(
            label="Role",
            required_props=("role_type",),
            optional_props=("description",),
            description="sem:RoleType — the capacity in which an Actor participates",
            ontology_mapping="sem:RoleType",
            merge_key="role_type",
        ),
        "Object": NodeSpec(
            label="Object",
            required_props=("description",),
            optional_props=("type", "colour", "size", "registration"),
            description="Physical object or vehicle mentioned",
            ontology_mapping="sem:Object",
            merge_key="description",
        ),
        "PhysicalDescription": NodeSpec(
            label="PhysicalDescription",
            required_props=("summary",),
            optional_props=(
                "height", "build", "hair_colour",
                "clothing", "distinguishing_features",
            ),
            description="Appearance snapshot of an Actor",
            ontology_mapping="evt:PhysicalDescription",
            merge_key="summary",
        ),
    },

    relationship_types={
        # Event ↔ Actor with role
        "HAS_ACTOR": RelSpec(
            "HAS_ACTOR", "Event", "Actor",
            "Actor participated in Event (sem:hasActor)",
            "sem:hasActor",
        ),
        "HAS_ROLE": RelSpec(
            "HAS_ROLE", "Actor", "Role",
            "Role the Actor played in this context",
            "sem:roleType",
        ),
        # Spatio-temporal
        "HAS_PLACE": RelSpec(
            "HAS_PLACE", "Event", "Place",
            "Where the Event occurred (sem:hasPlace)",
            "sem:hasPlace",
        ),
        "HAS_TIME": RelSpec(
            "HAS_TIME", "Event", "Time",
            "When the Event occurred (sem:hasTimeStamp)",
            "sem:hasTimeStamp",
        ),
        # Event structure
        "HAS_SUB_EVENT": RelSpec(
            "HAS_SUB_EVENT", "Event", "Event",
            "Event decomposition (sem:hasSubEvent)",
            "sem:hasSubEvent",
        ),
        "PRECEDED": RelSpec(
            "PRECEDED", "Event", "Event",
            "Temporal ordering between events",
            "sem:preceded",
        ),
        "CAUSED": RelSpec(
            "CAUSED", "Event", "Event",
            "Causal link between events",
            "sem:caused",
        ),
        # Object/vehicle usage
        "INVOLVED_OBJECT": RelSpec(
            "INVOLVED_OBJECT", "Event", "Object",
            "Object involved in the event",
            "sem:hasObject",
        ),
        "USED_BY": RelSpec(
            "USED_BY", "Object", "Actor",
            "Object was used/operated by actor",
            "sem:usedBy",
        ),
        # Descriptions
        "DESCRIBED_AS": RelSpec(
            "DESCRIBED_AS", "Actor", "PhysicalDescription",
            "Actor's appearance description",
            "evt:describedAs",
        ),
    },

    completeness_rules=[
        CompletenessRule(
            "event_needs_time", "critical",
            "Every Event must have at least one HAS_TIME",
            "MATCH (e:Event) WHERE NOT (e)-[:HAS_TIME]->(:Time) "
            "RETURN e.description AS entity_desc, labels(e)[0] AS label",
            "Event '{entity_desc}' has no temporal anchor",
        ),
        CompletenessRule(
            "event_needs_place", "critical",
            "Every Event must have at least one HAS_PLACE",
            "MATCH (e:Event) WHERE NOT (e)-[:HAS_PLACE]->(:Place) "
            "RETURN e.description AS entity_desc, labels(e)[0] AS label",
            "Event '{entity_desc}' has no spatial anchor",
        ),
        CompletenessRule(
            "event_needs_actor", "critical",
            "Every Event must have at least one HAS_ACTOR",
            "MATCH (e:Event) WHERE NOT (e)-[:HAS_ACTOR]->(:Actor) "
            "RETURN e.description AS entity_desc, labels(e)[0] AS label",
            "Event '{entity_desc}' has no associated actor",
        ),
        CompletenessRule(
            "actor_needs_role", "high",
            "Every Actor should have at least one Role",
            "MATCH (a:Actor) WHERE NOT (a)-[:HAS_ROLE]->(:Role) "
            "RETURN a.name_or_description AS entity_desc, labels(a)[0] AS label",
            "Actor '{entity_desc}' has no assigned role",
        ),
        CompletenessRule(
            "suspect_needs_description", "high",
            "Actors with suspect role should have a PhysicalDescription",
            "MATCH (a:Actor)-[:HAS_ROLE]->(r:Role {role_type: 'suspect'}) "
            "WHERE NOT (a)-[:DESCRIBED_AS]->(:PhysicalDescription) "
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
        "incident", "observation", "action", "arrival",
        "departure", "communication", "environmental",
    ),
    person_roles=(),  # SEM uses Role nodes instead of inline roles
    time_precisions=("exact", "approximate", "relative", "vague"),
)
