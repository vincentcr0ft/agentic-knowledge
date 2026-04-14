"""
06 · Ontology Comparison — Pluggable Ontology Specification
════════════════════════════════════════════════════════════

Defines the abstract OntologySpec data structure that every ontology
must implement.  The pipeline (extraction, loading, gap analysis,
SHACL validation) is driven entirely from the spec — swap the spec,
and the whole pipeline adapts.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ═══════════════════════════════════════════════════════════════════════════════
# Core spec types
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class NodeSpec:
    """Definition of a node type within an ontology."""
    label: str
    required_props: tuple[str, ...]
    optional_props: tuple[str, ...] = ()
    description: str = ""
    ontology_mapping: str = ""      # e.g. "schema:Event + prov:Entity"
    merge_key: str = ""             # property used for MERGE deduplication


@dataclass(frozen=True)
class RelSpec:
    """Definition of a relationship type."""
    rel_type: str
    from_label: str
    to_label: str
    description: str = ""
    ontology_mapping: str = ""


@dataclass(frozen=True)
class CompletenessRule:
    """A schema-level gap detection rule (Cypher-driven)."""
    rule_id: str
    priority: str               # critical | high | medium | low
    description: str
    cypher: str
    gap_template: str           # must contain {entity_desc}


@dataclass
class OntologySpec:
    """Complete specification of a domain ontology.

    Every piece of the pipeline — extraction prompts, graph loading,
    gap analysis, SHACL shapes — is generated from this single source
    of truth.

    To compare ontologies, create multiple OntologySpec instances and
    feed the same source text through each.
    """
    id: str
    version: str
    name: str
    description: str

    node_types: dict[str, NodeSpec]
    relationship_types: dict[str, RelSpec]
    completeness_rules: list[CompletenessRule] = field(default_factory=list)
    provenance_props: dict[str, str] = field(default_factory=dict)

    # Optional taxonomies
    event_types: tuple[str, ...] = ()
    person_roles: tuple[str, ...] = ()
    time_precisions: tuple[str, ...] = ()

    # System-managed rel types (excluded from LLM extraction prompt)
    system_managed_rels: tuple[str, ...] = ()
    # System-managed node types (excluded from LLM extraction prompt)
    system_managed_nodes: tuple[str, ...] = ()

    # ── Prompt generation ───────────────────────────────────────────────

    def build_extraction_prompt(self) -> str:
        """Generate an LLM extraction prompt from this ontology spec.

        The prompt tells the LLM which node types, relationship types,
        and properties to extract, and the output JSON schema to use.
        """
        lines = [
            "Extract entities and relationships from the text below.",
            "",
            f"ONTOLOGY: {self.name} (v{self.version})",
            "",
        ]

        # Node types
        lines.append("TARGET ENTITY TYPES:")
        for ndef in self.node_types.values():
            if ndef.label in self.system_managed_nodes:
                continue
            req = ", ".join(ndef.required_props)
            opt = ", ".join(ndef.optional_props) if ndef.optional_props else "none"
            lines.append(f"  - {ndef.label} (required: {req}; optional: {opt})")
            if ndef.description:
                lines.append(f"    → {ndef.description}")
        lines.append("")

        # Taxonomies
        if self.event_types:
            lines.append(f"EVENT TYPES: {', '.join(self.event_types)}")
        if self.person_roles:
            lines.append(f"PERSON ROLES: {', '.join(self.person_roles)}")
        if self.time_precisions:
            lines.append(f"TIME PRECISIONS: {', '.join(self.time_precisions)}")
        lines.append("")

        # Relationship types
        lines.append("TARGET RELATIONSHIP TYPES:")
        for rdef in self.relationship_types.values():
            if rdef.rel_type in self.system_managed_rels:
                continue
            lines.append(
                f"  - {rdef.rel_type} ({rdef.from_label} → {rdef.to_label})"
                f": {rdef.description}"
            )
        lines.append("")

        # Output format
        lines.extend([
            "Return valid JSON with this structure:",
            "{",
            '  "entities": [',
            '    {"id": "e1", "label": "Event", "properties": {"description": "...", "type": "incident"}}',
            "  ],",
            '  "relationships": [',
            '    {"from_id": "p1", "rel_type": "PARTICIPATED_IN", "to_id": "e1"}',
            "  ]",
            "}",
            "",
            "RULES:",
            "- Extract ALL entities and relationships mentioned or implied",
            "- Use ONLY the entity/relationship types listed above",
            "- Assign a unique id to each entity (e.g. e1, p1, v1, l1, t1)",
            "- Resolve pronouns to existing entity ids where clear",
            "- The statement author is a Person with role='witness'",
            "- Break compound events into separate nodes linked by temporal/causal rels",
        ])

        return "\n".join(lines)

    # ── SHACL shape generation ──────────────────────────────────────────

    def build_shacl_shapes(self) -> str:
        """Generate SHACL shapes in Turtle format from this spec."""
        lines = [
            '@prefix sh:     <http://www.w3.org/ns/shacl#> .',
            '@prefix xsd:    <http://www.w3.org/2001/XMLSchema#> .',
            '@prefix evt:    <http://example.org/event-twin#> .',
            "",
        ]

        for ndef in self.node_types.values():
            shape_name = f"evt:{ndef.label}Shape"
            lines.append(f"{shape_name} a sh:NodeShape ;")
            lines.append(f"    sh:targetClass evt:{ndef.label} ;")

            for prop in ndef.required_props:
                lines.append(f"    sh:property [")
                lines.append(f"        sh:path evt:{prop} ;")
                lines.append(f"        sh:minCount 1 ;")
                lines.append(f"        sh:datatype xsd:string ;")
                lines.append(
                    f'        sh:message "Every {ndef.label} must have a {prop}" ;'
                )
                lines.append(f"    ] ;")

            # Remove trailing " ;" and replace with " ."
            if lines[-1].endswith(" ;"):
                lines[-1] = lines[-1][:-2] + " ."
            lines.append("")

        return "\n".join(lines)

    # ── Neo4j constraint generation ─────────────────────────────────────

    def get_constraint_cypher(self) -> list[str]:
        """Generate CREATE CONSTRAINT statements for this ontology."""
        stmts = []
        for ndef in self.node_types.values():
            key = ndef.merge_key or (
                ndef.required_props[0] if ndef.required_props else None
            )
            if key:
                stmts.append(
                    f"CREATE CONSTRAINT IF NOT EXISTS "
                    f"FOR (n:{ndef.label}) REQUIRE n.{key} IS UNIQUE"
                )
        return stmts

    # ── Summary ─────────────────────────────────────────────────────────

    def summary(self) -> str:
        """One-line summary of the ontology."""
        return (
            f"{self.name} v{self.version}: "
            f"{len(self.node_types)} node types, "
            f"{len(self.relationship_types)} rel types, "
            f"{len(self.completeness_rules)} completeness rules"
        )
